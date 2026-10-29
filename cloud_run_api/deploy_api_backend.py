#!/usr/bin/env python3
"""Build and deploy the headless FastAPI backend (`cloud_run_api/api_server.py`) to Cloud Run (`asia-south1`).

The UI (`search_studio/ui.html`) remains strictly local (`http://127.0.0.1:8080`), while all API and
benchmark requests execute directly inside GCP `asia-south1`.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote
import zipfile

import requests

API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    BQ_5STORES_DATASET,
    BQ_TABLE_V2_RUNTIME,
    BQ_TABLE_V3_RUNTIME,
    CROWDING_FIELD,
    DATASTORE_ID_V2_RUNTIME,
    DATASTORE_ID_V3_RUNTIME,
    ENGINE_ID_V2_RUNTIME,
    ENGINE_ID_V3_RUNTIME,
    GEMINI_PROJECT_ID,
    LEGACY_V2_LANGUAGES_FIELD,
    LEGACY_V2_REGIONS_FIELD,
    LEGACY_V2_SHOW_DATES_FIELD,
    LEGACY_V2_SOLD_OUT_FIELD,
    SEARCH_5STORES_PROJECT_ID as PROJECT_ID,
    SEARCH_5STORES_PROJECT_NUMBER as PROJECT_NUMBER,
    VERTEX_GEMMA_1B_ROUTER_ENDPOINT_ID,
    VERTEX_GEMMA_1B_ROUTER_MODEL_ID,
    VERTEX_GEMMA_ROUTER_ENDPOINT_ID,
    VERTEX_GEMMA_ROUTER_MODEL_ID,
    VERTEX_GEMMA_ROUTER_PROJECT_ID,
    VERTEX_GEMMA_ROUTER_PROJECT_NUMBER,
    VERTEX_GEMMA_ROUTER_REGION,
)
from setup_5stores_bq_and_vertex import EcpClient

REGION = "asia-south1"
AR_REPO_ID = "gcf-artifacts"
SERVICE_ID = "ent-agent-search-api"
SOURCE_BUCKET = f"{PROJECT_ID}-agent-search-code"
SOURCE_OBJECT = "cloud_build/ent-agent-search-api-source.tar.gz"
COMPUTE_SA = f"{PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

REQUIRED_ROOT_FILES = (
    "config.py",
    "setup_gcs_datastore.py",
    "setup_5stores_bq_and_vertex.py",
    "evaluate_all_queries.py",
    "eval_5stores_nlp_collapse_html.py",
)

SEARCH_STUDIO_PY_FILES = (
    "__init__.py",
    "client.py",
    "prompts.py",
    "taxonomy.py",
    "compiler.py",
    "intent_extractor.py",
    "router_api.py",
    "search_service.py",
    "server.py",
    "setfit_minilm_router_weights.npz",
)


def build_source_tarball() -> bytes:
    """Create an in-memory `.tar.gz` containing only the headless FastAPI backend (excluding `ui.html`)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(API_DIR / "Dockerfile", arcname="Dockerfile")
        tf.add(API_DIR / "requirements.txt", arcname="requirements.txt")
        tf.add(API_DIR / "api_server.py", arcname="api_server.py")
        for fname in REQUIRED_ROOT_FILES:
            fpath = ROOT_DIR / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required root Python module: {fpath}")
            tf.add(fpath, arcname=fname)
        for sfname in SEARCH_STUDIO_PY_FILES:
            sfpath = ROOT_DIR / "search_studio" / sfname
            if not sfpath.exists():
                raise FileNotFoundError(f"Missing required search_studio module: {sfpath}")
            tf.add(sfpath, arcname=f"search_studio/{sfname}")
    return buf.getvalue()


def upload_tarball_to_gcs(client: EcpClient, tar_bytes: bytes) -> None:
    """Upload the Cloud Build source tarball and SetFit-MiniLM weights artifact to GCS over mTLS."""
    print(f"1. Uploading source tarball ({len(tar_bytes):,} bytes) to gs://{SOURCE_BUCKET}/{SOURCE_OBJECT}...", flush=True)
    client._refresh_token()
    encoded_obj = quote(SOURCE_OBJECT, safe="")
    url = f"http://127.0.0.1:{client.port}/upload/storage/v1/b/{SOURCE_BUCKET}/o?uploadType=media&name={encoded_obj}"
    headers = {
        "Authorization": f"Bearer {client.token}",
        "x-goog-user-project": PROJECT_ID,
        "x-goog-ecpproxy-target-host": "storage.mtls.googleapis.com:443",
        "Content-Type": "application/gzip",
    }
    resp = requests.post(url, headers=headers, data=tar_bytes, timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"GCS tarball upload failed ({resp.status_code}): {resp.text[:400]}")

    weights_path = ROOT_DIR / "search_studio" / "setfit_minilm_router_weights.npz"
    if weights_path.exists():
        w_obj = quote("models/setfit_minilm_router/setfit_minilm_router_weights.npz", safe="")
        w_url = f"http://127.0.0.1:{client.port}/upload/storage/v1/b/{SOURCE_BUCKET}/o?uploadType=media&name={w_obj}"
        w_headers = dict(headers, **{"Content-Type": "application/octet-stream"})
        requests.post(w_url, headers=w_headers, data=weights_path.read_bytes(), timeout=60)
        print(f"   Uploaded SetFit-MiniLM weights to gs://{SOURCE_BUCKET}/models/setfit_minilm_router/setfit_minilm_router_weights.npz", flush=True)
    print("   Uploaded source tarball cleanly.", flush=True)


def run_cloud_build(client: EcpClient, image_uri: str) -> None:
    """Build the FastAPI container image via Cloud Build and push to Artifact Registry in `asia-south1`."""
    print(f"2. Building FastAPI container image `{image_uri}` via Cloud Build...", flush=True)
    latest_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO_ID}/{SERVICE_ID}:latest"
    build_body: Dict[str, Any] = {
        "source": {
            "storageSource": {
                "bucket": SOURCE_BUCKET,
                "object": SOURCE_OBJECT,
            }
        },
        "steps": [
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": [
                    "build",
                    "-t",
                    image_uri,
                    "-t",
                    latest_uri,
                    ".",
                ],
            }
        ],
        "images": [image_uri, latest_uri],
        "serviceAccount": f"projects/{PROJECT_ID}/serviceAccounts/{COMPUTE_SA}",
        "options": {
            "logging": "CLOUD_LOGGING_ONLY",
        },
        "timeout": "600s",
    }
    resp = client.request(
        "cloudbuild.mtls.googleapis.com",
        "POST",
        f"/v1/projects/{PROJECT_ID}/locations/{REGION}/builds",
        build_body,
    )
    if resp.status_code != 200:
        resp = client.request(
            "cloudbuild.mtls.googleapis.com",
            "POST",
            f"/v1/projects/{PROJECT_ID}/builds",
            build_body,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Cloud Build trigger failed ({resp.status_code}): {resp.text[:500]}")

    op_data = resp.json()
    build_meta = op_data.get("metadata", {}).get("build", {})
    build_id = build_meta.get("id", "")
    poll_path = (
        f"/v1/projects/{PROJECT_ID}/locations/{REGION}/builds/{build_id}"
        if build_id
        else f"/v1/{op_data.get('name', '')}"
    )
    print(f"   Cloud Build started (build_id={build_id})...", flush=True)

    for i in range(90):
        poll = client.request("cloudbuild.mtls.googleapis.com", "GET", poll_path)
        if poll.status_code == 200:
            pdata = poll.json()
            b_status = pdata.get("status", "")
            if b_status == "SUCCESS" or (pdata.get("done") and "error" not in pdata):
                print(f"   Cloud Build SUCCEEDED -> {image_uri}", flush=True)
                return
            if b_status in ("FAILURE", "INTERNAL_ERROR", "TIMEOUT", "CANCELLED", "EXPIRED") or "error" in pdata:
                raise RuntimeError(f"Cloud Build failed ({b_status}): {json.dumps(pdata)[:600]}")
        if i > 0 and i % 4 == 0:
            print(f"   ... building image ({i * 4}s elapsed)...", flush=True)
        time.sleep(4)
    raise RuntimeError(f"Timed out waiting for Cloud Build {build_id}")


def ensure_cloudfunctions_v2_shell(client: EcpClient) -> None:
    """Ensure the Cloud Run service is backed by a Gen 2 Cloud Function resource so `roles/editor`
    automatically grants `cloudfunctions.functions.invoke` -> Cloud Run invoker access.
    """
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    fn_name = f"{parent}/functions/{SERVICE_ID}"
    svc_name = f"{parent}/services/{SERVICE_ID}"

    chk_fn = client.request("cloudfunctions.mtls.googleapis.com", "GET", f"/v2/{fn_name}")
    if chk_fn.status_code == 200:
        print("   Gen 2 Cloud Function shell already exists.", flush=True)
        return

    # If a raw Cloud Run service without GCF parent exists, remove it first so GCF v2 can create it with invoker IAM
    chk_svc = client.request("run.mtls.googleapis.com", "GET", f"/v2/{svc_name}")
    if chk_svc.status_code == 200:
        print("   Removing raw Cloud Run stub so Gen 2 Cloud Function can provision Cloud Run route with invoker access...", flush=True)
        del_r = client.request("run.mtls.googleapis.com", "DELETE", f"/v2/{svc_name}")
        if del_r.status_code == 200:
            del_op = del_r.json().get("name", "")
            for _ in range(30):
                p = client.request("run.mtls.googleapis.com", "GET", f"/v2/{del_op}")
                if p.status_code == 200 and p.json().get("done"):
                    break
                time.sleep(2)

    print("   Creating Gen 2 Cloud Function shell to bind Cloud Run invoker access...", flush=True)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.py", "def app_entry(request):\n    return ('OK', 200)\n")
        zf.writestr("requirements.txt", "functions-framework==3.*\n")
    zip_bytes = zbuf.getvalue()

    gen_resp = client.request("cloudfunctions.mtls.googleapis.com", "POST", f"/v2/{parent}/functions:generateUploadUrl", {})
    upload_info = gen_resp.json()
    requests.put(upload_info["uploadUrl"], data=zip_bytes, headers={"Content-Type": "application/zip"}, timeout=60)

    v2_body = {
        "name": fn_name,
        "description": "Headless FastAPI Backend for Entertainment Agent Search & Gemma 3 Decision Router (asia-south1)",
        "buildConfig": {
            "runtime": "python312",
            "entryPoint": "app_entry",
            "source": {"storageSource": upload_info["storageSource"]},
        },
        "serviceConfig": {
            "maxInstanceCount": 4,
            "availableMemory": "1Gi",
            "timeoutSeconds": 180,
            "serviceAccountEmail": COMPUTE_SA,
        },
    }
    cr = client.request("cloudfunctions.mtls.googleapis.com", "POST", f"/v2/{parent}/functions?functionId={SERVICE_ID}", v2_body)
    if cr.status_code != 200:
        raise RuntimeError(f"GCF v2 shell creation failed ({cr.status_code}): {cr.text[:400]}")
    op_name = cr.json()["name"]
    for i in range(90):
        op_r = client.request("cloudfunctions.mtls.googleapis.com", "GET", f"/v2/{op_name}")
        if op_r.status_code == 200 and op_r.json().get("done"):
            print("   Gen 2 Cloud Function shell ACTIVE!", flush=True)
            return
        if i > 0 and i % 4 == 0:
            print(f"   ... provisioning Gen 2 shell ({i * 4}s elapsed)...", flush=True)
        time.sleep(4)


def deploy_cloud_run_service(client: EcpClient, image_uri: str) -> Dict[str, Any]:
    """Patch the Cloud Run service in `asia-south1` to run the FastAPI + Uvicorn (`uvloop` + `httptools`) container."""
    ensure_cloudfunctions_v2_shell(client)
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    svc_name = f"{parent}/services/{SERVICE_ID}"
    print(f"3. Updating Cloud Run service `{svc_name}` with FastAPI + Uvicorn image `{image_uri}`...", flush=True)

    env_vars = [
        {"name": "USE_ADC", "value": "true"},
        {"name": "GCP_PROJECT_ID", "value": PROJECT_ID},
        {"name": "GCP_PROJECT_NUMBER", "value": PROJECT_NUMBER},
        {"name": "GEMINI_PROJECT_ID", "value": PROJECT_ID},
        {"name": "SEARCH_5STORES_PROJECT_ID", "value": PROJECT_ID},
        {"name": "SEARCH_5STORES_PROJECT_NUMBER", "value": PROJECT_NUMBER},
        {"name": "BQ_5STORES_DATASET", "value": BQ_5STORES_DATASET},
        {"name": "BQ_TABLE_V3", "value": BQ_TABLE_V3_RUNTIME},
        {"name": "BQ_TABLE_V2", "value": BQ_TABLE_V2_RUNTIME},
        {"name": "ENGINE_ID_V3", "value": ENGINE_ID_V3_RUNTIME},
        {"name": "DATASTORE_ID_V3", "value": DATASTORE_ID_V3_RUNTIME},
        {"name": "ENGINE_ID_V2", "value": ENGINE_ID_V2_RUNTIME},
        {"name": "DATASTORE_ID_V2", "value": DATASTORE_ID_V2_RUNTIME},
        {"name": "LEGACY_V2_LANGUAGES_FIELD", "value": LEGACY_V2_LANGUAGES_FIELD},
        {"name": "LEGACY_V2_SOLD_OUT_FIELD", "value": LEGACY_V2_SOLD_OUT_FIELD},
        {"name": "CROWDING_FIELD", "value": CROWDING_FIELD},
        {"name": "LEGACY_V2_REGIONS_FIELD", "value": LEGACY_V2_REGIONS_FIELD},
        {"name": "LEGACY_V2_SHOW_DATES_FIELD", "value": LEGACY_V2_SHOW_DATES_FIELD},
        {"name": "VERTEX_GEMMA_ROUTER_PROJECT_ID", "value": VERTEX_GEMMA_ROUTER_PROJECT_ID},
        {"name": "VERTEX_GEMMA_ROUTER_PROJECT_NUMBER", "value": VERTEX_GEMMA_ROUTER_PROJECT_NUMBER},
        {"name": "VERTEX_GEMMA_ROUTER_REGION", "value": VERTEX_GEMMA_ROUTER_REGION},
        {"name": "VERTEX_GEMMA_ROUTER_ENDPOINT_ID", "value": VERTEX_GEMMA_ROUTER_ENDPOINT_ID},
        {"name": "VERTEX_GEMMA_ROUTER_MODEL_ID", "value": VERTEX_GEMMA_ROUTER_MODEL_ID},
        {"name": "VERTEX_GEMMA_1B_ROUTER_ENDPOINT_ID", "value": VERTEX_GEMMA_1B_ROUTER_ENDPOINT_ID},
        {"name": "VERTEX_GEMMA_1B_ROUTER_MODEL_ID", "value": VERTEX_GEMMA_1B_ROUTER_MODEL_ID},
    ]

    cur_svc = client.request("run.mtls.googleapis.com", "GET", f"/v2/{svc_name}").json()
    cur_svc["annotations"] = {"cloudfunctions.googleapis.com/function-id": SERVICE_ID}
    cur_svc["template"] = {
        "serviceAccount": COMPUTE_SA,
        "timeout": "180s",
        "maxInstanceRequestConcurrency": 80,
        "scaling": {
            "minInstanceCount": 1,
            "maxInstanceCount": 4,
        },
        "containers": [
            {
                "image": image_uri,
                "ports": [{"containerPort": 8080}],
                "resources": {
                    "limits": {
                        "cpu": "2",
                        "memory": "1Gi",
                    },
                    "cpuIdle": False,
                    "startupCpuBoost": True,
                },
                "env": env_vars,
            }
        ],
    }

    resp = client.request("run.mtls.googleapis.com", "PATCH", f"/v2/{svc_name}?updateMask=template,annotations", cur_svc)
    if resp.status_code != 200:
        raise RuntimeError(f"Cloud Run deploy failed ({resp.status_code}): {resp.text[:500]}")

    op_name = resp.json().get("name", "")
    print(f"   Waiting for Cloud Run revision readiness ({op_name})...", flush=True)
    for _ in range(60):
        poll = client.request("run.mtls.googleapis.com", "GET", f"/v2/{op_name}")
        if poll.status_code == 200:
            pdata = poll.json()
            if pdata.get("done"):
                if "error" in pdata:
                    raise RuntimeError(f"Cloud Run LRO failed: {json.dumps(pdata['error'])}")
                break
        time.sleep(3)

    final_svc = client.request("run.mtls.googleapis.com", "GET", f"/v2/{svc_name}")
    if final_svc.status_code != 200:
        raise RuntimeError(f"Failed fetching deployed service: {final_svc.text[:400]}")
    return final_svc.json()


def main() -> None:
    """Build and deploy the headless FastAPI backend to Cloud Run (`asia-south1`)."""
    client = EcpClient(port=18962)
    try:
        tar_bytes = build_source_tarball()
        upload_tarball_to_gcs(client, tar_bytes)
        tag = f"v{int(time.time())}"
        image_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO_ID}/{SERVICE_ID}:{tag}"
        run_cloud_build(client, image_uri)
        svc = deploy_cloud_run_service(client, image_uri)
        print("\n=== DEPLOYED HEADLESS FASTAPI CLOUD RUN BACKEND ===", flush=True)
        print(
            json.dumps(
                {
                    "service": svc.get("name"),
                    "uri": svc.get("uri"),
                    "urls": svc.get("urls"),
                    "latestReadyRevision": svc.get("latestReadyRevision"),
                    "region": REGION,
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
