"""High-Performance FastAPI + Uvicorn (`uvloop` + `httptools` + `ORJSONResponse`) Headless API Backend (`asia-south1`).

Separated from the local Search Studio UI (`search_studio/ui.html`), which remains on `http://127.0.0.1:8080`.
All Vertex AI calls (`google/gemma-3-270m-it` vLLM L4 endpoint, `gemini-3.5-flash-lite`, and Vertex AI Search)
originate directly inside GCP `asia-south1` over pre-warmed persistent HTTP keep-alive pools.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
import uvicorn

from search_studio.client import (
    ALL_QUERIES,
    PORT,
    USE_ADC,
    _get_adc_token,
    get_ecp_client,
    get_thread_host_session,
)
from search_studio.router_api import (
    handle_route_decision_request,
    handle_router_benchmark_request,
)
from search_studio.search_service import (
    explain_live_hit,
    run_compare_v2_vs_v3,
    run_live_search,
    run_side_by_side_nlqu_vs_llm,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Pre-warm GCP ADC token or local ECP proxy and initialize HTTP keep-alive sessions."""
    if USE_ADC:
        try:
            _get_adc_token()
            get_thread_host_session("asia-south1-aiplatform.googleapis.com")
            get_thread_host_session("discoveryengine.googleapis.com")
        except Exception as exc:
            print(f"[FastAPI Startup] ADC pre-warm warning: {exc}", flush=True)
    else:
        get_ecp_client()
    yield


app = FastAPI(
    title="Entertainment Agent Search & Gemma 3 Decision Router API",
    description=(
        "High-speed headless FastAPI backend (`asia-south1`) for Vertex AI Model Garden "
        "`google/gemma-3-270m-it` (L4 vLLM `--enable-prefix-caching`), `gemini-3.5-flash-lite`, "
        "and Vertex AI Agent Search (`v3` / `v2`)."
    ),
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


def _origin_tag() -> str:
    """Return an identifier describing whether the request executed on Cloud Run (`asia-south1`) or locally."""
    return "cloud_run_asia_south1" if os.getenv("K_SERVICE") else "local_fastapi"


@app.get("/")
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    """Health and service metadata endpoint."""
    return {
        "status": "OK",
        "framework": "FastAPI + Uvicorn (uvloop + httptools + orjson)",
        "mode": "headless_api_backend",
        "region": "asia-south1",
        "origin_env": _origin_tag(),
        "k_service": os.getenv("K_SERVICE", "local"),
        "k_revision": os.getenv("K_REVISION", "local"),
        "docs_url": "/docs",
        "endpoints": [
            "/healthz",
            "/docs",
            "/api/presets",
            "/api/route_decision",
            "/api/benchmark_router",
            "/api/search",
            "/api/side_by_side",
            "/api/compare_apps",
            "/api/explain",
        ],
    }


@app.get("/api/presets")
def get_presets() -> Dict[str, Any]:
    """Return the 30 benchmark search queries."""
    return {"queries": ALL_QUERIES, "origin_env": _origin_tag()}


@app.get("/api/route_decision")
def get_route_decision(
    query: str = Query(..., description="User search query to classify"),
    model: Optional[str] = Query(None, description="Router model (`vertex:gemma-3-270m-it` or `gemini-3.5-flash-lite`)"),
    include_reason: bool = Query(True, description="Include `reason` field (`false` for fast 6-token mode)"),
    use_cache: bool = Query(False, description="Enable in-memory LRU cache"),
    execute_search: bool = Query(False, description="Also execute downstream Vertex AI Search"),
) -> Dict[str, Any]:
    """Classify a query into `DIRECT_AGENT_SEARCH` or `CALL_LLM_INTENT` via GET query parameters."""
    try:
        res = handle_route_decision_request(
            {
                "query": query,
                "model": model,
                "include_reason": include_reason,
                "use_cache": use_cache,
                "execute_search": execute_search,
            }
        )
        res["origin_env"] = _origin_tag()
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/route_decision")
async def post_route_decision(request: Request) -> Dict[str, Any]:
    """Classify a query into `DIRECT_AGENT_SEARCH` or `CALL_LLM_INTENT` via POST JSON body."""
    try:
        body = await request.json()
        res = handle_route_decision_request(body)
        res["origin_env"] = _origin_tag()
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/benchmark_router")
def get_benchmark_router(
    model: str = Query("vertex:gemma-3-270m-it", description="Router model to benchmark (`vertex:gemma-3-270m-it` or `gemini-3.5-flash-lite`)"),
    include_reason: Optional[bool] = Query(None, description="Optional: benchmark only `true` or `false` mode (default runs both)"),
) -> Dict[str, Any]:
    """Run the 30-query Decision Router benchmark directly inside the Cloud Run (`asia-south1`) container."""
    try:
        params: Dict[str, Any] = {"model": model}
        if include_reason is not None:
            params["include_reason"] = include_reason
        return handle_router_benchmark_request(params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/benchmark_router")
async def post_benchmark_router(request: Request) -> Dict[str, Any]:
    """Run the 30-query Decision Router benchmark via POST JSON body."""
    try:
        body = await request.json()
        return handle_router_benchmark_request(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/search")
async def post_search(request: Request) -> Dict[str, Any]:
    """Execute a live single-engine search (`v3` or `v2`) from `asia-south1`."""
    try:
        body = await request.json()
        res = run_live_search(body)
        res["origin_env"] = _origin_tag()
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/side_by_side")
async def post_side_by_side(request: Request) -> Dict[str, Any]:
    """Execute side-by-side Managed NLQU vs LLM Boost search from `asia-south1`."""
    try:
        body = await request.json()
        res = run_side_by_side_nlqu_vs_llm(body)
        res["origin_env"] = _origin_tag()
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/compare_apps")
async def post_compare_apps(request: Request) -> Dict[str, Any]:
    """Compare `v2` vs `v3` search apps side-by-side from `asia-south1`."""
    try:
        body = await request.json()
        res = run_compare_v2_vs_v3(body)
        res["origin_env"] = _origin_tag()
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/explain")
async def post_explain(request: Request) -> Dict[str, Any]:
    """Explain a single ranked search hit using `gemini-3.5-flash-lite` in `asia-south1`."""
    try:
        body = await request.json()
        res = explain_live_hit(body)
        res["origin_env"] = _origin_tag()
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        loop="uvloop",
        http="httptools",
        access_log=False,
    )
