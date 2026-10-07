"""Google Cloud Function (2nd Gen) — Event-Driven GCS to Agent Search DataStore Sync.

Triggered automatically by Eventarc whenever a file is uploaded or overwritten in the
configured GCS bucket (`google.cloud.storage.object.v1.finalized`).

How it works:
1. Eventarc delivers a `CloudEvent` containing `bucket`, `name`, `generation`, and `size`.
2. The function filters for objects matching `{GCS_FOLDER}/*.jsonl` (ignoring other paths).
3. It triggers an immediate incremental import (`POST .../documents:import` with `gcsSource`)
   for the newly uploaded `gs://{bucket}/{name}` URI.
4. It also triggers `POST .../collections/{COLLECTION_ID}/dataConnector:startConnectorRun`
   so the GCS DataConnector state stays synchronized.

Local Testing:
  python cloud_function_gcs_sync/main.py --test-local
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import google.auth
from google.auth.transport.requests import AuthorizedSession

# Support optional functions_framework decorator (installed automatically in Cloud Functions runtime)
try:
    import functions_framework
    from cloudevents.http import CloudEvent
except ImportError:
    functions_framework = None  # type: ignore[assignment]
    CloudEvent = Any  # type: ignore[assignment,misc]

# Fallback to parent config.py when running locally in the repo
_PARENT_DIR = Path(__file__).resolve().parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

try:
    from config import (
        COLLECTION_ID as _DEFAULT_COLLECTION_ID,
        DATASTORE_ID as _DEFAULT_DATASTORE_ID,
        GCS_BUCKET as _DEFAULT_GCS_BUCKET,
        GCS_FOLDER as _DEFAULT_GCS_FOLDER,
        LOCATION as _DEFAULT_LOCATION,
        PROJECT_ID as _DEFAULT_PROJECT_ID,
    )
except ImportError:
    _DEFAULT_PROJECT_ID = "your-gcp-project-id"
    _DEFAULT_LOCATION = "global"
    _DEFAULT_GCS_BUCKET = "your-gcs-bucket-name"
    _DEFAULT_GCS_FOLDER = "sample_search_data"
    _DEFAULT_COLLECTION_ID = "sample-data-connector"
    _DEFAULT_DATASTORE_ID = "sample-data-connector_gcs_store"


PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", _DEFAULT_PROJECT_ID)
LOCATION: str = os.getenv("DISCOVERY_ENGINE_LOCATION", _DEFAULT_LOCATION)
GCS_BUCKET: str = os.getenv("GCS_BUCKET", _DEFAULT_GCS_BUCKET)
GCS_FOLDER: str = os.getenv("GCS_FOLDER", _DEFAULT_GCS_FOLDER)
COLLECTION_ID: str = os.getenv("COLLECTION_ID", _DEFAULT_COLLECTION_ID)
DATASTORE_ID: str = os.getenv("DATASTORE_ID", _DEFAULT_DATASTORE_ID)
RECONCILIATION_MODE: str = os.getenv("RECONCILIATION_MODE", "INCREMENTAL")


def _get_session_and_base_url() -> Tuple[AuthorizedSession, str]:
    """Return an authenticated Discovery Engine session (Cloud Run ADC or local mTLS fallback)."""
    try:
        from setup_gcs_datastore import get_discovery_session

        return get_discovery_session()
    except Exception:
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
            quota_project_id=PROJECT_ID,
        )
        return (
            AuthorizedSession(creds),
            "https://discoveryengine.googleapis.com/v1alpha",
        )


def process_gcs_event_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Core logic executed when a GCS `object.v1.finalized` event is received."""
    bucket = data.get("bucket", "")
    name = data.get("name", "")
    generation = data.get("generation", "")
    size = data.get("size", "0")

    expected_prefix = f"{GCS_FOLDER.strip('/')}/"
    if bucket != GCS_BUCKET or not name.startswith(expected_prefix) or not name.endswith(".jsonl"):
        msg = (
            f"Ignored GCS event for gs://{bucket}/{name} "
            f"(does not match gs://{GCS_BUCKET}/{expected_prefix}*.jsonl)"
        )
        print(msg)
        return {"status": "ignored", "reason": msg}

    gcs_uri = f"gs://{bucket}/{name}"
    print(
        f"Received GCS finalize event for {gcs_uri} "
        f"(generation={generation}, size={size} bytes)"
    )

    session, base_url = _get_session_and_base_url()

    # 1. Trigger immediate incremental import of the newly uploaded JSONL file
    import_url = (
        f"{base_url}/projects/{PROJECT_ID}/locations/{LOCATION}"
        f"/collections/default_collection/dataStores/{DATASTORE_ID}"
        f"/branches/default_branch/documents:import"
    )
    import_payload = {
        "gcsSource": {
            "inputUris": [gcs_uri],
            "dataSchema": "custom",
        },
        "reconciliationMode": RECONCILIATION_MODE,
    }
    import_resp = session.post(import_url, json=import_payload)
    if import_resp.status_code != 200:
        raise RuntimeError(
            f"Discovery Engine documents:import failed (HTTP {import_resp.status_code}): {import_resp.text}"
        )
    import_lro = import_resp.json().get("name", "")
    print(f"Started Agent Search Import LRO: {import_lro}")

    # 2. Trigger parent DataConnector sync run (`startConnectorRun`)
    connector_url = (
        f"{base_url}/projects/{PROJECT_ID}/locations/{LOCATION}"
        f"/collections/{COLLECTION_ID}/dataConnector:startConnectorRun"
    )
    conn_resp = session.post(connector_url, json={})
    print(
        f"DataConnector startConnectorRun ({COLLECTION_ID}) status: HTTP {conn_resp.status_code}"
    )

    return {
        "status": "triggered",
        "gcs_uri": gcs_uri,
        "generation": generation,
        "reconciliation_mode": RECONCILIATION_MODE,
        "import_lro": import_lro,
        "connector_sync_http_status": conn_resp.status_code,
    }


def _cloud_event_decorator(func):
    """Apply `@functions_framework.cloud_event` when `functions-framework` is installed."""
    if functions_framework is not None:
        return functions_framework.cloud_event(func)
    return func


@_cloud_event_decorator
def sync_gcs_to_datastore(cloud_event: CloudEvent) -> Dict[str, Any]:
    """Cloud Run Functions (2nd Gen) Eventarc entrypoint (`--entry-point=sync_gcs_to_datastore`)."""
    data = cloud_event.data if hasattr(cloud_event, "data") else dict(cloud_event)
    return process_gcs_event_payload(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the GCS-to-Vertex-AI-Search Cloud Function locally with a simulated CloudEvent."
    )
    parser.add_argument(
        "--test-local",
        action="store_true",
        help="Simulate an Eventarc `google.cloud.storage.object.v1.finalized` event and invoke the handler.",
    )
    parser.add_argument(
        "--object-name",
        default=f"{GCS_FOLDER.strip('/')}/delta_new_upload.jsonl",
        help="GCS object path inside the bucket to simulate (default: <GCS_FOLDER>/delta_new_upload.jsonl).",
    )
    args = parser.parse_args()

    if args.test_local:
        class _MockCloudEvent:
            def __init__(self, payload: Dict[str, Any]) -> None:
                self.data = payload

        simulated_event = _MockCloudEvent(
            {
                "bucket": GCS_BUCKET,
                "name": args.object_name,
                "generation": "1791370136421307",
                "size": "1755",
            }
        )
        print("Simulating Eventarc GCS `object.v1.finalized` CloudEvent:")
        result = sync_gcs_to_datastore(simulated_event)
        print("\nCloud Function Return Payload:")
        print(json.dumps(result, indent=2))
