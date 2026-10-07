"""Watch a GCS folder for new or updated JSONL files and trigger an Agent Search DataStore sync.

Supports two execution patterns:
1. Continuous / Single-Pass GCS Watcher (`watch_gcs_and_sync.py`):
   - Inspects `gs://{GCS_BUCKET}/{GCS_FOLDER}/*.jsonl` and tracks object metadata
     (`generation`, `md5Hash`, `updated`, `size`) in a local `.gcs_sync_state.json` file.
   - Whenever a newly uploaded or updated `.jsonl` object is detected:
     a) Triggers an immediate incremental GCS import (`POST .../documents:import` with `gcsSource`)
        for the changed `.jsonl` URI(s) so new records are indexed right away.
     b) Optionally triggers the DataConnector sync run (`POST .../dataConnector:startConnectorRun`).
     c) Polls the Long-Running Operation (LRO) until completion and records the synced generation.
2. Event-Driven Cloud Run / Cloud Functions Webhook (`handle_gcs_finalize_event`):
   - Can be deployed as an Eventarc (`google.cloud.storage.object.v1.finalized`) handler
     that automatically triggers `sync_gcs_uris_to_datastore()` whenever a `.jsonl` object
     is uploaded to `gs://{GCS_BUCKET}/{GCS_FOLDER}/`.

Usage:
  # Single scan of the GCS folder (syncs any new/modified .jsonl files and exits)
  python watch_gcs_and_sync.py --once

  # Continuous polling watcher (checks GCS every 15 seconds)
  python watch_gcs_and_sync.py --interval 15

  # End-to-end test: upload a new delta .jsonl file to GCS and watch it trigger DataStore sync
  python watch_gcs_and_sync.py --simulate-upload
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google.auth.transport.requests import AuthorizedSession

from config import (
    COLLECTION_ID,
    DATASTORE_ID,
    GCLOUD_ACCOUNT,
    GCS_BUCKET,
    GCS_FOLDER,
    LOCATION,
    PROJECT_ID,
)
from setup_gcs_datastore import get_discovery_session

STATE_FILE_PATH = Path(__file__).resolve().parent / ".gcs_sync_state.json"


# ==============================================================================
# 1. Local State Tracking (Persists last-synced GCS object generations)
# ==============================================================================
def load_sync_state(state_path: Path = STATE_FILE_PATH) -> Dict[str, Dict[str, Any]]:
    """Load map of `gs://...` URI -> last synced object metadata (`generation`, `updated`, `size`)."""
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_sync_state(
    state: Dict[str, Dict[str, Any]], state_path: Path = STATE_FILE_PATH
) -> None:
    """Persist synced GCS object metadata to disk."""
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


# ==============================================================================
# 2. List Objects in GCS Folder & Detect New / Modified Files
# ==============================================================================
def list_gcs_jsonl_objects(
    bucket: str = GCS_BUCKET, folder: str = GCS_FOLDER
) -> List[Dict[str, Any]]:
    """List all `.jsonl` objects under `gs://{bucket}/{folder}/` with generation & timestamp metadata."""
    prefix_uri = f"gs://{bucket}/{folder.strip('/')}/**"
    cmd = [
        "gcloud",
        "storage",
        "objects",
        "list",
        prefix_uri,
        f"--project={PROJECT_ID}",
        "--format=json(name,bucket,generation,metageneration,md5Hash,size,timeCreated,updated)",
    ]
    if GCLOUD_ACCOUNT:
        cmd.extend(["--account", GCLOUD_ACCOUNT])

    last_err = ""
    for attempt in range(3):
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            break
        if "matched no objects" in (result.stderr or ""):
            return []
        last_err = (result.stderr or "").strip()
        time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to list GCS objects: {last_err}")

    raw_items = json.loads(result.stdout or "[]")
    jsonl_objects: List[Dict[str, Any]] = []
    for item in raw_items:
        name = item.get("name", "")
        if not name.endswith(".jsonl"):
            continue
        uri = f"gs://{item.get('bucket', bucket)}/{name}"
        jsonl_objects.append(
            {
                "uri": uri,
                "name": name,
                "generation": str(item.get("generation", "")),
                "md5Hash": item.get("md5Hash", ""),
                "size": int(item.get("size", 0) or 0),
                "updated": item.get("updated", ""),
            }
        )
    return jsonl_objects


def detect_changed_gcs_files(
    current_objects: List[Dict[str, Any]],
    synced_state: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return only objects that are newly uploaded or have a newer `generation` / `md5Hash`."""
    changed: List[Dict[str, Any]] = []
    for obj in current_objects:
        uri = obj["uri"]
        prev = synced_state.get(uri)
        if prev is None:
            obj["change_type"] = "NEW_FILE"
            changed.append(obj)
        elif (
            prev.get("generation") != obj["generation"]
            or prev.get("md5Hash") != obj["md5Hash"]
        ):
            obj["change_type"] = "UPDATED_FILE"
            changed.append(obj)
    return changed


# ==============================================================================
# 3. Trigger Agent Search DataStore Sync (GCS Import + Connector Run)
# ==============================================================================
def sync_gcs_uris_to_datastore(
    session: AuthorizedSession,
    base_url: str,
    gcs_uris: List[str],
    reconciliation_mode: str = "INCREMENTAL",
    trigger_connector_run: bool = True,
    wait_for_lro: bool = True,
) -> Dict[str, Any]:
    """Trigger an immediate Agent Search DataStore sync for the given `gs://` JSONL URI(s).

    1. Calls `POST .../branches/default_branch/documents:import` with `gcsSource.inputUris`
       to immediately index the newly uploaded or modified `.jsonl` file(s).
    2. Optionally calls `POST .../collections/{COLLECTION_ID}/dataConnector:startConnectorRun`
       so the parent GCS DataConnector also records a fresh sync run.
    """
    import_url = (
        f"{base_url}/projects/{PROJECT_ID}/locations/{LOCATION}"
        f"/collections/default_collection/dataStores/{DATASTORE_ID}"
        f"/branches/default_branch/documents:import"
    )
    payload = {
        "gcsSource": {
            "inputUris": gcs_uris,
            "dataSchema": "custom",
        },
        "reconciliationMode": reconciliation_mode,
    }

    print(
        f"Triggering Agent Search GCS Import ({reconciliation_mode}) for {len(gcs_uris)} file(s):"
    )
    for u in gcs_uris:
        print(f"  -> {u}")

    resp = session.post(import_url, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(
            f"GCS import trigger failed (HTTP {resp.status_code}): {resp.text}"
        )

    lro_data = resp.json()
    lro_name = lro_data.get("name", "")
    print(f"Started Import LRO: {lro_name}")

    if trigger_connector_run:
        connector_url = (
            f"{base_url}/projects/{PROJECT_ID}/locations/{LOCATION}"
            f"/collections/{COLLECTION_ID}/dataConnector:startConnectorRun"
        )
        conn_resp = session.post(connector_url, json={})
        if conn_resp.status_code == 200:
            print(f"Triggered DataConnector sync run on '{COLLECTION_ID}' (HTTP 200).")
        else:
            # Connector may already have an active sync run in progress; log status non-fatally
            print(
                f"DataConnector startConnectorRun status: HTTP {conn_resp.status_code}"
            )

    if wait_for_lro and lro_name and not lro_data.get("done"):
        print("Polling Import LRO until completion...")
        t0 = time.perf_counter()
        for _ in range(60):
            op_resp = session.get(f"{base_url}/{lro_name}")
            if op_resp.status_code == 200:
                op_data = op_resp.json()
                if op_data.get("done"):
                    elapsed_s = time.perf_counter() - t0
                    meta = op_data.get("metadata", {})
                    print(
                        f"Import LRO completed in {elapsed_s:.1f}s | "
                        f"successCount={meta.get('successCount', 0)}, "
                        f"failureCount={meta.get('failureCount', 0)}"
                    )
                    return op_data
            time.sleep(2.0)

    return lro_data


# ==============================================================================
# 4. Eventarc / Cloud Functions Handler (Serverless Event-Driven Option)
# ==============================================================================
def handle_gcs_finalize_event(cloud_event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Cloud Run / Cloud Functions entrypoint for `google.cloud.storage.object.v1.finalized`.

    When deployed with an Eventarc GCS trigger on `gs://{GCS_BUCKET}`, this function
    is invoked automatically whenever any file is uploaded or overwritten.
    """
    bucket = cloud_event_data.get("bucket", "")
    name = cloud_event_data.get("name", "")
    expected_prefix = f"{GCS_FOLDER.strip('/')}/"

    if bucket != GCS_BUCKET or not name.startswith(expected_prefix) or not name.endswith(".jsonl"):
        print(f"Ignoring non-matching GCS object event: gs://{bucket}/{name}")
        return None

    gcs_uri = f"gs://{bucket}/{name}"
    session, base_url = get_discovery_session()
    return sync_gcs_uris_to_datastore(
        session=session,
        base_url=base_url,
        gcs_uris=[gcs_uri],
        reconciliation_mode="INCREMENTAL",
        trigger_connector_run=True,
        wait_for_lro=False,
    )


# ==============================================================================
# 5. Polling Watcher Loop & Simulation Demo
# ==============================================================================
def check_and_sync_once(
    session: AuthorizedSession,
    base_url: str,
    reconciliation_mode: str = "INCREMENTAL",
    baseline_only: bool = False,
) -> List[Dict[str, Any]]:
    """Scan `gs://{GCS_BUCKET}/{GCS_FOLDER}/` once and trigger DataStore sync if any new/updated files exist."""
    state = load_sync_state()
    current_objects = list_gcs_jsonl_objects(GCS_BUCKET, GCS_FOLDER)
    changed = detect_changed_gcs_files(current_objects, state)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not changed:
        print(
            f"[{now_str}] Checked gs://{GCS_BUCKET}/{GCS_FOLDER}/ "
            f"({len(current_objects)} .jsonl file(s)) — no new or modified files."
        )
        return []

    if baseline_only:
        for obj in changed:
            state[obj["uri"]] = {
                "generation": obj["generation"],
                "md5Hash": obj["md5Hash"],
                "size": obj["size"],
                "updated": obj["updated"],
                "last_synced_at": now_str,
            }
        save_sync_state(state)
        print(
            f"[{now_str}] Recorded baseline state for {len(changed)} existing GCS .jsonl file(s)."
        )
        return []

    print(
        f"\n[{now_str}] Detected {len(changed)} new/updated .jsonl file(s) in gs://{GCS_BUCKET}/{GCS_FOLDER}/:"
    )
    for obj in changed:
        print(
            f"  * [{obj['change_type']}] {obj['uri']} "
            f"(generation={obj['generation']}, size={obj['size']} bytes, updated={obj['updated']})"
        )

    uris_to_sync = [obj["uri"] for obj in changed]
    lro_result = sync_gcs_uris_to_datastore(
        session=session,
        base_url=base_url,
        gcs_uris=uris_to_sync,
        reconciliation_mode=reconciliation_mode,
        trigger_connector_run=True,
        wait_for_lro=True,
    )

    for obj in changed:
        state[obj["uri"]] = {
            "generation": obj["generation"],
            "md5Hash": obj["md5Hash"],
            "size": obj["size"],
            "updated": obj["updated"],
            "last_synced_at": now_str,
            "last_lro": lro_result.get("name", ""),
        }
    save_sync_state(state)
    return changed


def simulate_new_file_upload_and_sync() -> None:
    """End-to-end test: baseline existing GCS folder, upload a new `.jsonl` file, and watch it trigger sync."""
    session, base_url = get_discovery_session()
    print("=" * 95)
    print(f"GCS Watcher & Auto-Sync Verification — Watching gs://{GCS_BUCKET}/{GCS_FOLDER}/")
    print("=" * 95)

    # 1. Baseline existing files in GCS so only the newly uploaded file triggers sync
    check_and_sync_once(session, base_url, baseline_only=True)

    # 2. Create a local delta JSONL file with 2 new records and upload it to GCS
    local_delta_file = Path(__file__).resolve().parent / "delta_new_upload.jsonl"
    delta_records = [
        {
            "id": "MV00301",
            "_id": "MV00301",
            "record_type": "movie",
            "media_type": "movie",
            "defaultVariantCode": "ET00600301",
            "title": "Border 2: The Battalion",
            "categories": ["movies"],
            "slug": "border-2-the-battalion",
            "available_time": "2026-10-15T00:00:00+05:30",
            "release_date": "2026-10-15T00:00:00+05:30",
            "showtime_start": "2026-10-15T19:30:00+05:30",
            "showtime_end": "2026-10-15T22:30:00+05:30",
            "hash_tags": ["NEW_RELEASE", "BLOCKBUSTER"],
            "genres": ["Action", "War", "Drama"],
            "languages": ["Hindi"],
            "searchKeywords": ["movie", "border 2", "war", "action", "hindi"],
            "formats": ["2D", "IMAX 2D"],
            "artists": ["Sunny Deol", "Varun Dhawan", "Diljit Dosanjh"],
            "city": "Delhi NCR",
            "venue": "PVR Select Citywalk, Saket",
            "location_city": {"address": "Delhi NCR, Delhi, India"},
            "location_latlng": {"lat": 28.5286, "lng": 77.2193},
            "uri": "https://events.example.com/delhi-ncr/movies/border-2-the-battalion/ET00600301",
        },
        {
            "id": "etm100301z",
            "_id": "etm100301z",
            "record_type": "event",
            "eventGroupCode": "EG60000301",
            "title": "RAHUL DUA LIVE STANDUP TOUR - HYDERABAD",
            "slug": "rahul-dua-live-standup-tour-hyderabad",
            "persons": ["Rahul Dua"],
            "languages": ["Hindi", "English"],
            "duration": 90,
            "ageLimit": 16,
            "ageLimitDisplay": "16yrs +",
            "isActive": True,
            "isSearchable": True,
            "createdAt": "2026-10-07T10:00:00+05:30",
            "release_date": "2026-10-07T10:00:00+05:30",
            "showtime_start": "2026-10-14T20:00:00+05:30",
            "showtime_end": "2026-10-14T21:30:00+05:30",
            "hash_tags": ["STANDUP_COMEDY", "NEXT_WEEK"],
            "searchKeywords": ["stand up comedy", "rahul dua", "comedy", "hyderabad"],
            "city": "Hyderabad",
            "venue": "HITEX Exhibition Centre",
            "location_city": {"address": "Hyderabad, Telangana, India"},
            "location_latlng": {"lat": 17.4709, "lng": 78.3727},
        },
    ]

    with open(local_delta_file, "w", encoding="utf-8") as f:
        for rec in delta_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dest_uri = f"gs://{GCS_BUCKET}/{GCS_FOLDER}/{local_delta_file.name}"
    state = load_sync_state()
    state.pop(dest_uri, None)
    save_sync_state(state)

    print(f"\nUploading new file '{local_delta_file.name}' -> {dest_uri} ...")
    upload_cmd = [
        "gcloud",
        "storage",
        "cp",
        "--quiet",
        str(local_delta_file),
        dest_uri,
        f"--project={PROJECT_ID}",
    ]
    if GCLOUD_ACCOUNT:
        upload_cmd.extend(["--account", GCLOUD_ACCOUNT])
    for attempt in range(3):
        up_res = subprocess.run(upload_cmd, check=False)
        if up_res.returncode == 0:
            break
        time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to upload {local_delta_file} to {dest_uri}")

    # 3. Run the watcher scan — it detects the newly uploaded file and triggers DataStore sync
    changed_files = check_and_sync_once(session, base_url, reconciliation_mode="INCREMENTAL")

    # 4. Run a second scan immediately to verify idempotency (no duplicate sync when unchanged)
    print("\nRunning follow-up scan to confirm idempotency (already-synced generation skipped):")
    check_and_sync_once(session, base_url, reconciliation_mode="INCREMENTAL")

    # 5. Verify the newly synced documents exist in the DataStore
    docs_base = (
        f"{base_url}/projects/{PROJECT_ID}/locations/{LOCATION}"
        f"/collections/default_collection/dataStores/{DATASTORE_ID}"
        f"/branches/default_branch/documents"
    )
    print("\nVerifying synced documents in Agent Search DataStore:")
    for rec in delta_records:
        doc_id = rec["id"]
        doc_resp = session.get(f"{docs_base}/{doc_id}")
        if doc_resp.status_code == 200:
            sd = doc_resp.json().get("structData", {})
            print(
                f"  [VERIFIED HTTP 200] {doc_id:<12} | ({sd.get('record_type')}) "
                f"{sd.get('title')} — {sd.get('city')}"
            )
        else:
            print(f"  [HTTP {doc_resp.status_code}] {doc_id}")

    # Clean up local temporary delta file
    if local_delta_file.exists():
        local_delta_file.unlink()
    print("=" * 95)


def run_watcher_loop(interval_seconds: int = 15, reconciliation_mode: str = "INCREMENTAL") -> None:
    """Continuously poll `gs://{GCS_BUCKET}/{GCS_FOLDER}/` and sync new/updated `.jsonl` files."""
    session, base_url = get_discovery_session()
    print(
        f"Starting GCS Watcher on gs://{GCS_BUCKET}/{GCS_FOLDER}/ "
        f"(poll interval: {interval_seconds}s, mode: {reconciliation_mode})"
    )
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            check_and_sync_once(
                session=session,
                base_url=base_url,
                reconciliation_mode=reconciliation_mode,
            )
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped GCS watcher.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Watch GCS folder for new/updated JSONL uploads and trigger Agent Search DataStore sync."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scan the GCS folder once, trigger sync if new/modified files exist, and exit.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Polling interval in seconds for continuous watching (default: 15).",
    )
    parser.add_argument(
        "--mode",
        choices=["INCREMENTAL", "FULL"],
        default="INCREMENTAL",
        help="Reconciliation mode for Agent Search import (default: INCREMENTAL).",
    )
    parser.add_argument(
        "--simulate-upload",
        action="store_true",
        help="Upload a sample delta .jsonl file to GCS and verify the watcher triggers a live DataStore sync.",
    )
    args = parser.parse_args()

    if args.simulate_upload:
        simulate_new_file_upload_and_sync()
    elif args.once:
        sess, api_base = get_discovery_session()
        check_and_sync_once(sess, api_base, reconciliation_mode=args.mode)
    else:
        run_watcher_loop(interval_seconds=args.interval, reconciliation_mode=args.mode)

# ==============================================================================
# EXECUTION OUTPUT (`python watch_gcs_and_sync.py --simulate-upload`)
# ==============================================================================
# ===============================================================================================
# GCS Watcher & Auto-Sync Verification — Watching gs://your-gcs-bucket-name/sample_search_data/
# ===============================================================================================
# [2026-10-07T10:48:51Z] Recorded baseline state for 1 existing GCS .jsonl file(s).
#
# Uploading new file 'delta_new_upload.jsonl' -> gs://your-gcs-bucket-name/sample_search_data/delta_new_upload.jsonl ...
#
# [2026-10-07T10:48:58Z] Detected 1 new/updated .jsonl file(s) in gs://your-gcs-bucket-name/sample_search_data/:
#   * [NEW_FILE] gs://your-gcs-bucket-name/sample_search_data/delta_new_upload.jsonl (generation=1791370136421307, size=1755 bytes)
# Triggering Agent Search GCS Import (INCREMENTAL) for 1 file(s):
#   -> gs://your-gcs-bucket-name/sample_search_data/delta_new_upload.jsonl
# Started Import LRO: projects/000000000000/locations/global/collections/default_collection/dataStores/sample-data-connector_gcs_store/branches/0/operations/import-documents-9108705056319265354
# Triggered DataConnector sync run on 'sample-data-connector' (HTTP 200).
# Polling Import LRO until completion...
# Import LRO completed in 0.8s | successCount=2, failureCount=0
#
# Running follow-up scan to confirm idempotency (already-synced generation skipped):
# [2026-10-07T10:49:08Z] Checked gs://your-gcs-bucket-name/sample_search_data/ (2 .jsonl file(s)) — no new or modified files.
#
# Verifying synced documents in Agent Search DataStore:
#   [VERIFIED HTTP 200] MV00301      | (movie) Border 2: The Battalion — Delhi NCR
#   [VERIFIED HTTP 200] etm100301z   | (event) RAHUL DUA LIVE STANDUP TOUR - HYDERABAD — Hyderabad
# ===============================================================================================

