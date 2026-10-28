"""Entrypoint for the Real-Time Entertainment Search Studio Web Application.

The application is modularized under the `search_studio/` package:
  - `search_studio/prompts.py`          : LLM System Prompts & Few-Shot Chat Templates
  - `search_studio/router_api.py`       : Local Gemma 3 Decision Router API (`/api/route_decision`)
  - `search_studio/intent_extractor.py` : Local Metal GPU (`mlx-lm`) & Cloud Gemini Intent Extractor
  - `search_studio/compiler.py`         : Deterministic Intent-to-Boost & Geolocation Compiler
  - `search_studio/taxonomy.py`         : City, Sub-Locality, Lat/Long, Format, Genre & Category Maps
  - `search_studio/search_service.py`   : Vertex AI Agent Search Dispatcher, Lat/Long Rings & Comparisons
  - `search_studio/client.py`           : Search App Configs (`v3` vs `v2`) & ECP/mTLS Client
  - `search_studio/server.py`           : HTTP API Server (`http://127.0.0.1:8080`)
  - `search_studio/ui.html`             : Clean Dark-Mode Search Studio Frontend UI

Run (with `.venv` containing `mlx-lm`):
    ./.venv/bin/python realtime_test_webapp.py
Then open:
    http://127.0.0.1:8080
"""

from __future__ import annotations

from search_studio import (
    GEMINI_INTENT_SYSTEM_INSTRUCTION,
    LOCAL_LLM_SYSTEM_PROMPT,
    RealtimeTestHandler,
    build_curl_command,
    build_explain_hit_prompt,
    build_gemma_chat_messages,
    build_gemma_router_chat_messages,
    decide_query_route_gemma3,
    explain_live_hit,
    extract_boosts_cloud_gemini,
    extract_boosts_llm,
    extract_boosts_local_mlx,
    handle_route_decision_request,
    load_ui_html,
    main,
    run_compare_v2_vs_v3,
    run_live_search,
    run_side_by_side_nlqu_vs_llm,
    serialize_llm_slots_to_boost_specs,
)

if __name__ == "__main__":
    main()
