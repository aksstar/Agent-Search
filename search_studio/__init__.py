"""Modular Entertainment Search Studio Package.

Production Pipeline:
1. `router_api`: SetFit-MiniLM Query Decision Classifier (`GET / POST /api/route_decision` — `setfit:minilm-l6-v2` in-process ONNX & `vertex:setfit-minilm-l6-v2` GCP Vertex AI Endpoint).
2. `intent_extractor`: Conditional 8-Key LLM Slot Extractor (invoked only when `call_llm_for_intent == True`).
3. `compiler` & `taxonomy`: Deterministic Intent-to-Boost & Geolocation compiler (`serialize_llm_slots_to_boost_specs`).
4. `search_service`: Live Vertex AI Agent Search execution, Concentric Lat/Long `GEO_DISTANCE` rings, and comparisons.
5. `server`: FastAPI + Uvicorn HTTP server serving `search_studio/ui.html`.
"""

from search_studio.compiler import serialize_llm_slots_to_boost_specs
from search_studio.intent_extractor import (
    extract_boosts_cloud_gemini,
    extract_boosts_llm,
    extract_boosts_local_mlx,
)
from search_studio.prompts import (
    GEMINI_INTENT_SYSTEM_INSTRUCTION,
    LOCAL_LLM_SYSTEM_PROMPT,
    build_explain_hit_prompt,
    build_gemma_chat_messages,
)
from search_studio.router_api import (
    DEFAULT_ROUTER_MODEL,
    decide_query_route,
    decide_query_route_gemma3,
    decide_query_route_setfit_minilm,
    decide_query_route_vertex_setfit,
    handle_route_decision_request,
    handle_router_benchmark_request,
)
from search_studio.search_service import (
    build_curl_command,
    explain_live_hit,
    run_compare_v2_vs_v3,
    run_live_search,
    run_side_by_side_nlqu_vs_llm,
)
from search_studio.server import RealtimeTestHandler, load_ui_html, main

__all__ = [
    "DEFAULT_ROUTER_MODEL",
    "GEMINI_INTENT_SYSTEM_INSTRUCTION",
    "LOCAL_LLM_SYSTEM_PROMPT",
    "RealtimeTestHandler",
    "build_curl_command",
    "build_explain_hit_prompt",
    "build_gemma_chat_messages",
    "decide_query_route",
    "decide_query_route_gemma3",
    "decide_query_route_setfit_minilm",
    "decide_query_route_vertex_setfit",
    "explain_live_hit",
    "extract_boosts_cloud_gemini",
    "extract_boosts_llm",
    "extract_boosts_local_mlx",
    "handle_route_decision_request",
    "handle_router_benchmark_request",
    "load_ui_html",
    "main",
    "run_compare_v2_vs_v3",
    "run_live_search",
    "run_side_by_side_nlqu_vs_llm",
    "serialize_llm_slots_to_boost_specs",
]
