"""FastAPI + Uvicorn (`uvloop` + `httptools` + `ORJSONResponse`) Local UI & Proxy Server (`http://127.0.0.1:8080`).

Serves the Search Studio UI (`search_studio/ui.html`) locally on `http://127.0.0.1:8080`.
API requests (`/api/*`) can either:
1. Execute locally on Apple Metal GPU (`mlx-lm`) + ECP mTLS (`backend="local"`), OR
2. Proxy to the headless FastAPI Cloud Run Backend in `asia-south1` (`backend="cloud_run"`), so all
   Vertex AI calls (`google/gemma-3-270m-it` L4 vLLM, `gemini-3.5-flash-lite`, and Vertex AI Search)
   originate directly inside GCP `asia-south1`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, ORJSONResponse
import requests
import uvicorn

import config
from config import GCLOUD_ACCOUNT
from search_studio.client import (
    ALL_QUERIES,
    DEFAULT_BOOST_MODEL,
    HAS_MLX,
    PORT,
    STARTUP_PREWARM_LOCAL_MODELS,
    _MLX_GPU_LOCK,
    get_ecp_client,
)
from search_studio.intent_extractor import get_or_init_local_mlx_runtime
from search_studio.router_api import (
    get_or_init_local_mlx_router_runtime,
    handle_route_decision_request,
    handle_router_benchmark_request,
)
from search_studio.search_service import (
    explain_live_hit,
    run_compare_v2_vs_v3,
    run_live_search,
    run_side_by_side_nlqu_vs_llm,
)

UI_HTML_PATH = Path(__file__).resolve().parent / "ui.html"

_ID_TOKEN_LOCK = threading.Lock()
_CACHED_ID_TOKEN = ""
_ID_TOKEN_TS = 0.0
_CLOUD_RUN_SESSION = requests.Session()


def load_ui_html() -> str:
    """Load the frontend Search Studio UI HTML from `search_studio/ui.html`."""
    return UI_HTML_PATH.read_text(encoding="utf-8")


def get_cloud_run_id_token(force: bool = False) -> str:
    """Fetch and cache a Google OIDC identity token (`gcloud auth print-identity-token`) for Cloud Run."""
    global _CACHED_ID_TOKEN, _ID_TOKEN_TS
    now = time.time()
    with _ID_TOKEN_LOCK:
        if not force and _CACHED_ID_TOKEN and (now - _ID_TOKEN_TS) < 2400:
            return _CACHED_ID_TOKEN
        cmd = ["gcloud", "auth", "print-identity-token"]
        if GCLOUD_ACCOUNT:
            cmd.append(f"--account={GCLOUD_ACCOUNT}")
        _CACHED_ID_TOKEN = subprocess.check_output(cmd, text=True).strip()
        _ID_TOKEN_TS = now
        return _CACHED_ID_TOKEN


def proxy_to_cloud_run_api(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Forward an `/api/*` request to the headless Cloud Run FastAPI backend in `asia-south1`."""
    base_url = (config._load_local_config().get("cloud_run_api_url") or config.CLOUD_RUN_API_URL or "").rstrip("/")
    if not base_url:
        raise RuntimeError("`cloud_run_api_url` is not configured in `config.local.json` or `CLOUD_RUN_API_URL`.")
    url = f"{base_url}{path}"
    id_token = get_cloud_run_id_token()
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    resp = _CLOUD_RUN_SESSION.request(method, url, params=params, json=json_body, headers=headers, timeout=180)
    if resp.status_code == 401:
        id_token = get_cloud_run_id_token(force=True)
        headers["Authorization"] = f"Bearer {id_token}"
        resp = _CLOUD_RUN_SESSION.request(method, url, params=params, json=json_body, headers=headers, timeout=180)
    wan_rtt_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    if resp.status_code != 200:
        raise RuntimeError(f"Cloud Run API ({url}) returned HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    if isinstance(data, dict):
        data["cloud_run_url"] = base_url
        data["client_to_cloud_run_total_ms"] = wan_rtt_ms
    return data


def _should_use_cloud_run(payload: Dict[str, Any]) -> bool:
    """Check whether the request explicitly asks to execute on the Cloud Run (`asia-south1`) backend."""
    backend = str(payload.get("backend") or payload.get("origin") or "").strip().lower()
    return backend in ("cloud_run", "cloudrun", "gcp", "asia-south1")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Pre-warm local ECP mTLS proxy and Apple Metal GPU KV caches on startup."""
    get_ecp_client()
    if HAS_MLX:
        with _MLX_GPU_LOCK:
            for local_key in STARTUP_PREWARM_LOCAL_MODELS:
                try:
                    get_or_init_local_mlx_runtime(local_key)
                except Exception as exc:
                    print(f"[MLX] Skipping startup pre-warm for {local_key}: {exc}", flush=True)
            try:
                get_or_init_local_mlx_router_runtime("local:gemma-3-1b-it-qat-4bit")
            except Exception as exc:
                print(f"[MLX Router] Skipping startup pre-warm for Gemma 3 1B router: {exc}", flush=True)
    yield


app = FastAPI(
    title="Entertainment Search Studio (Local UI + FastAPI)",
    version="4.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def serve_ui() -> HTMLResponse:
    """Serve the Search Studio UI (`search_studio/ui.html`) on local port `8080`."""
    return HTMLResponse(content=load_ui_html())


@app.get("/api/presets")
def api_presets() -> Dict[str, Any]:
    """Return the 30 preset benchmark queries."""
    return {"queries": ALL_QUERIES}


@app.get("/api/route_decision")
def api_get_route_decision(request: Request) -> Dict[str, Any]:
    """Handle `GET /api/route_decision` (locally or proxied to Cloud Run when `backend=cloud_run`)."""
    qs = dict(request.query_params)
    try:
        if _should_use_cloud_run(qs):
            return proxy_to_cloud_run_api("GET", "/api/route_decision", params=qs)
        return handle_route_decision_request(qs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/route_decision")
async def api_post_route_decision(request: Request) -> Dict[str, Any]:
    """Handle `POST /api/route_decision` (locally or proxied to Cloud Run when `backend=cloud_run`)."""
    body = await request.json()
    try:
        if _should_use_cloud_run(body):
            return proxy_to_cloud_run_api("POST", "/api/route_decision", json_body=body)
        return handle_route_decision_request(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/benchmark_router")
def api_get_benchmark_router(request: Request) -> Dict[str, Any]:
    """Handle `GET /api/benchmark_router` (locally or proxied to Cloud Run when `backend=cloud_run`)."""
    qs = dict(request.query_params)
    try:
        if _should_use_cloud_run(qs):
            return proxy_to_cloud_run_api("GET", "/api/benchmark_router", params=qs)
        return handle_router_benchmark_request(qs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/benchmark_router")
async def api_post_benchmark_router(request: Request) -> Dict[str, Any]:
    """Handle `POST /api/benchmark_router` (locally or proxied to Cloud Run when `backend=cloud_run`)."""
    body = await request.json()
    try:
        if _should_use_cloud_run(body):
            return proxy_to_cloud_run_api("POST", "/api/benchmark_router", json_body=body)
        return handle_router_benchmark_request(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/search")
async def api_post_search(request: Request) -> Dict[str, Any]:
    """Handle `POST /api/search` (locally or proxied to Cloud Run when `backend=cloud_run`)."""
    body = await request.json()
    try:
        if _should_use_cloud_run(body):
            return proxy_to_cloud_run_api("POST", "/api/search", json_body=body)
        return run_live_search(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/side_by_side")
async def api_post_side_by_side(request: Request) -> Dict[str, Any]:
    """Handle `POST /api/side_by_side` (locally or proxied to Cloud Run when `backend=cloud_run`)."""
    body = await request.json()
    try:
        if _should_use_cloud_run(body):
            return proxy_to_cloud_run_api("POST", "/api/side_by_side", json_body=body)
        return run_side_by_side_nlqu_vs_llm(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/compare_apps")
async def api_post_compare_apps(request: Request) -> Dict[str, Any]:
    """Handle `POST /api/compare_apps` (locally or proxied to Cloud Run when `backend=cloud_run`)."""
    body = await request.json()
    try:
        if _should_use_cloud_run(body):
            return proxy_to_cloud_run_api("POST", "/api/compare_apps", json_body=body)
        return run_compare_v2_vs_v3(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/explain")
async def api_post_explain(request: Request) -> Dict[str, Any]:
    """Handle `POST /api/explain` (locally or proxied to Cloud Run when `backend=cloud_run`)."""
    body = await request.json()
    try:
        if _should_use_cloud_run(body):
            return proxy_to_cloud_run_api("POST", "/api/explain", json_body=body)
        return explain_live_hit(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class RealtimeTestHandler(BaseHTTPRequestHandler):
    """Legacy compatibility handler retained for callers importing `RealtimeTestHandler`."""


def main() -> None:
    """Start the local Search Studio UI + FastAPI server on `http://127.0.0.1:PORT`."""
    print(
        f"Search Studio Local UI + FastAPI ({DEFAULT_BOOST_MODEL}) running at http://127.0.0.1:{PORT}",
        flush=True,
    )
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        loop="uvloop",
        http="httptools",
        access_log=False,
    )


if __name__ == "__main__":
    main()
