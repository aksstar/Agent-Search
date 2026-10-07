"""Upload sample entertainment data to GCS and create a Discovery Engine DataStore with 4-hour periodic GCS sync."""

import json
import os
import subprocess
import time
import google.auth
from google.auth.transport import mtls
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials

from config import (
    COLLECTION_DISPLAY_NAME,
    COLLECTION_ID,
    ENGINE_DISPLAY_NAME,
    ENGINE_ID,
    GCLOUD_ACCOUNT,
    GCS_BUCKET,
    GCS_FOLDER,
    LOCAL_JSON_FILE,
    LOCAL_JSONL_FILE,
    LOCATION,
    PROJECT_ID,
    REFRESH_INTERVAL_SECONDS,
)


def get_discovery_session() -> tuple[AuthorizedSession, str]:
    """Create an authenticated session (supporting CBA/mTLS when using gcloud tokens)."""
    try:
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
            quota_project_id=PROJECT_ID,
        )
        session = AuthorizedSession(creds)
        if mtls.has_default_client_cert_source():
            session.configure_mtls_channel(mtls.default_client_cert_source())
            base_url = "https://discoveryengine.mtls.googleapis.com/v1alpha"
        else:
            base_url = "https://discoveryengine.googleapis.com/v1alpha"
        # Quick probe to verify ADC works
        probe = session.get(
            f"{base_url}/projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/dataStores"
        )
        if probe.status_code == 200:
            return session, base_url
    except Exception:
        pass

    # Fallback to active gcloud account token with mTLS
    cmd = ["gcloud", "auth", "print-access-token"]
    if GCLOUD_ACCOUNT:
        cmd.extend(["--account", GCLOUD_ACCOUNT])
    token = subprocess.check_output(cmd, text=True).strip()
    creds = Credentials(token=token, quota_project_id=PROJECT_ID)
    session = AuthorizedSession(creds)
    if mtls.has_default_client_cert_source():
        session.configure_mtls_channel(mtls.default_client_cert_source())
        base_url = "https://discoveryengine.mtls.googleapis.com/v1alpha"
    else:
        base_url = "https://discoveryengine.googleapis.com/v1alpha"
    return session, base_url


def convert_to_jsonl(input_json_path: str, output_jsonl_path: str) -> int:
    """Convert sample metadata JSON (movies + events) into newline-delimited JSONL for Vertex AI Search."""
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    movies = data.get("movies", [])
    events = data.get("events", [])

    records = []
    for item in movies:
        rec = dict(item)
        rec_id = rec.get("id") or rec.get("_id")
        rec["id"] = rec_id
        rec["_id"] = rec_id
        rec["record_type"] = "movie"
        records.append(rec)

    for item in events:
        rec = dict(item)
        rec_id = rec.get("_id") or rec.get("id")
        rec["id"] = rec_id
        rec["_id"] = rec_id
        rec["record_type"] = "event"
        records.append(rec)

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Prepared {len(records)} JSONL records in {output_jsonl_path}")
    return len(records)


def upload_to_gcs(local_files: list[str]) -> list[str]:
    """Upload local data files to configured GCS bucket and folder."""
    uploaded_uris = []
    for local_file in local_files:
        dest_uri = f"gs://{GCS_BUCKET}/{GCS_FOLDER}/{os.path.basename(local_file)}"
        print(f"Uploading {local_file} -> {dest_uri} ...")
        cmd = [
            "gcloud",
            "storage",
            "cp",
            local_file,
            dest_uri,
            f"--project={PROJECT_ID}",
        ]
        if GCLOUD_ACCOUNT:
            cmd.extend(["--account", GCLOUD_ACCOUNT])
        subprocess.run(cmd, check=True)
        uploaded_uris.append(dest_uri)
    return uploaded_uris


def setup_gcs_datastore_with_periodic_sync(
    session: AuthorizedSession, base_url: str
) -> dict:
    """Create a Discovery Engine GCS DataConnector & DataStore with 4-hour periodic sync (14400s)."""
    parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
    connector_url = f"{base_url}/{parent}/collections/{COLLECTION_ID}/dataConnector"

    # Check if connector already exists
    existing = session.get(connector_url)
    if existing.status_code == 200:
        print(f"DataConnector '{COLLECTION_ID}' already exists:")
        connector_data = existing.json()
        print(json.dumps(connector_data, indent=2))
        return connector_data

    setup_url = f"{base_url}/{parent}:setUpDataConnector"
    payload = {
        "collectionId": COLLECTION_ID,
        "collectionDisplayName": COLLECTION_DISPLAY_NAME,
        "dataConnector": {
            "dataSource": "gcs",
            "refreshInterval": REFRESH_INTERVAL_SECONDS,
            "syncMode": "PERIODIC",
            "params": {
                "instance_uris": [
                    f"gs://{GCS_BUCKET}/{GCS_FOLDER}/*.jsonl",
                ]
            },
            "entities": [
                {
                    "entityName": "gcs_store",
                    "params": {
                        "data_schema": "custom",
                        "auto_generate_ids": False,
                        "content_config": "content_config_unspecified",
                        "industry_vertical": "GENERIC",
                    },
                }
            ],
        },
    }

    print(
        f"Creating GCS DataStore & DataConnector '{COLLECTION_ID}' with {REFRESH_INTERVAL_SECONDS} (4-hour) periodic sync..."
    )
    resp = session.post(setup_url, json=payload)
    print(f"setUpDataConnector status: {resp.status_code}")
    print(resp.text)
    resp.raise_for_status()
    lro = resp.json()
    lro_name = lro.get("name", "")

    # Poll LRO only if not already marked done
    if lro_name and not lro.get("done"):
        for _ in range(30):
            op_resp = session.get(f"{base_url}/{lro_name}")
            if op_resp.status_code == 200:
                op_data = op_resp.json()
                if op_data.get("done"):
                    print("DataConnector setup LRO completed:")
                    print(json.dumps(op_data, indent=2))
                    break
            time.sleep(3)

    dc_resp = session.get(connector_url)
    if dc_resp.status_code == 200:
        connector_data = dc_resp.json()
        if not connector_data.get("lastSyncTime"):
            print("Triggering immediate initial GCS sync run via startConnectorRun...")
            run_resp = session.post(f"{connector_url}:startConnectorRun", json={})
            print(f"startConnectorRun status: {run_resp.status_code} {run_resp.text}")
        return connector_data
    return lro


def ensure_search_engine(
    session: AuthorizedSession, base_url: str, data_store_id: str
) -> str:
    """Ensure a Search Engine (App) is attached to the DataStore so indexing and search are enabled."""
    parent = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection"
    engine_id = ENGINE_ID
    engine_url = f"{base_url}/{parent}/engines/{engine_id}"

    existing = session.get(engine_url)
    if existing.status_code == 200:
        return existing.json().get("name", f"{parent}/engines/{engine_id}")

    payload = {
        "displayName": ENGINE_DISPLAY_NAME,
        "solutionType": "SOLUTION_TYPE_SEARCH",
        "industryVertical": "GENERIC",
        "dataStoreIds": [data_store_id],
        "searchEngineConfig": {
            "searchTier": "SEARCH_TIER_ENTERPRISE",
            "searchAddOns": ["SEARCH_ADD_ON_LLM"],
        },
    }
    print(f"Attaching Search Engine '{engine_id}' to DataStore '{data_store_id}'...")
    resp = session.post(f"{base_url}/{parent}/engines?engineId={engine_id}", json=payload)
    resp.raise_for_status()
    return f"{parent}/engines/{engine_id}"


def verify_datastore_and_sync(
    session: AuthorizedSession, base_url: str, connector_info: dict
) -> None:
    """Verify the created DataStore, inspect its GCS import operations, and run live search queries."""
    entities = connector_info.get("entities", [])
    if not entities:
        print("No entities found on connector.")
        return

    data_store_name = entities[0].get("dataStore", "")
    data_store_id = data_store_name.split("/")[-1]
    print(f"\nDataStore Resource Name: {data_store_name}")

    engine_name = ensure_search_engine(session, base_url, data_store_id)
    print(f"Search Engine Resource Name: {engine_name}")

    ds_resp = session.get(f"{base_url}/{data_store_name}")
    if ds_resp.status_code == 200:
        print("DataStore Details:")
        print(json.dumps(ds_resp.json(), indent=2))

    ops_resp = session.get(f"{base_url}/{data_store_name}/branches/0/operations")
    if ops_resp.status_code == 200:
        ops = ops_resp.json().get("operations", [])
        print(f"\nFound {len(ops)} import operation(s) on DataStore:")
        for op in ops[:3]:
            print(
                json.dumps(
                    {
                        "name": op.get("name"),
                        "done": op.get("done"),
                        "metadata": op.get("metadata"),
                        "error": op.get("error"),
                    },
                    indent=2,
                )
            )

    # Run sample live search queries
    search_url = f"{base_url}/{engine_name}/servingConfigs/default_search:search"
    for query in ["Coldplay Mumbai", "Drishyam", "Bengaluru"]:
        s_resp = session.post(search_url, json={"query": query, "pageSize": 3})
        if s_resp.status_code == 200:
            s_data = s_resp.json()
            results = s_data.get("results", [])
            print(
                f"\nSearch Query: {query!r} -> Total Matches: {s_data.get('totalSize', len(results))}"
            )
            for item in results:
                doc = item.get("document", {}).get("structData", {})
                print(
                    f"  - [{doc.get('id')}] ({doc.get('record_type')}) {doc.get('title')} | {doc.get('city')} | {doc.get('venue')}"
                )


if __name__ == "__main__":
    convert_to_jsonl(LOCAL_JSON_FILE, LOCAL_JSONL_FILE)
    upload_to_gcs([LOCAL_JSONL_FILE])

    sess, api_base = get_discovery_session()
    connector_info = setup_gcs_datastore_with_periodic_sync(sess, api_base)
    print("\nFinal Connector & DataStore Info:")
    print(json.dumps(connector_info, indent=2))
    verify_datastore_and_sync(sess, api_base, connector_info)

