"""Modular Entertainment Search Studio Package.

Modules:
- `client`: Search App configs (`v3` vs `v2`), ECP/mTLS HTTP client, and runtime locks/caches.
- `prompts`: LLM System Prompts & multi-turn few-shot chat templates (Intent, Router, Explainer).
- `taxonomy`: Canonical city/locality/lat-lng maps, formats, genres, categories, and grounding sets.
- `compiler`: Deterministic Intent-to-Boost & Geolocation compiler (`serialize_llm_slots_to_boost_specs`).
- `intent_extractor`: Local Apple Metal GPU (`mlx-lm`) & Cloud Vertex Gemini Intent Extractor.
- `router_api`: Local Gemma 3 Query Decision Router API (`/api/route_decision`).
- `search_service`: Live Vertex AI Search execution, Concentric Lat/Long `GEO_DISTANCE` rings, and comparisons.
- `server`: HTTP server and request handler serving `search_studio/ui.html`.
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
    build_gemma_router_chat_messages,
)
from search_studio.router_api import (
    decide_query_route_cloud_gemini,
    decide_query_route_gemma3,
    decide_query_route_setfit_minilm,
    decide_query_route_vertex_vllm,
    handle_route_decision_request,
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
    "GEMINI_INTENT_SYSTEM_INSTRUCTION",
    "LOCAL_LLM_SYSTEM_PROMPT",
    "RealtimeTestHandler",
    "build_curl_command",
    "build_explain_hit_prompt",
    "build_gemma_chat_messages",
    "build_gemma_router_chat_messages",
    "decide_query_route_cloud_gemini",
    "decide_query_route_gemma3",
    "decide_query_route_setfit_minilm",
    "decide_query_route_vertex_vllm",
    "explain_live_hit",
    "extract_boosts_cloud_gemini",
    "extract_boosts_llm",
    "extract_boosts_local_mlx",
    "handle_route_decision_request",
    "load_ui_html",
    "main",
    "run_compare_v2_vs_v3",
    "run_live_search",
    "run_side_by_side_nlqu_vs_llm",
    "serialize_llm_slots_to_boost_specs",
]
