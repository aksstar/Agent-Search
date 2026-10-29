"""Live Vertex AI Agent Search Execution, Concentric Lat/Long Boosting & Comparison Engine.

Provides:
- `run_live_search()`: Single-engine search with `auto_route`, `llm_boost`, `vertex_native`, or `none`,
  3 concentric Lat/Long `GEO_DISTANCE` proximity boost rings, and automatic 0-hit sub-locality fallback.
- `run_side_by_side_nlqu_vs_llm()`: Concurrent Managed Vertex NLQU vs. LLM Boost + Gemma 3 Route Decision.
- `run_compare_v2_vs_v3()`: Concurrent head-to-head comparison between `v2` (Venue-Grain) and `v3` (Session-Grain).
- `explain_live_hit()`: Live hit diagnosis via `gemini-3.5-flash-lite` in `asia-south1`.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from typing import Any, Dict, List, Optional

from config import (
    CROWDING_FIELD,
    ENGINE_ID_V2_DISPLAY,
    ENGINE_ID_V2_RUNTIME,
    ENGINE_ID_V3_DISPLAY,
    ENGINE_ID_V3_RUNTIME,
    LEGACY_V2_LANGUAGES_FIELD,
    LEGACY_V2_SOLD_OUT_FIELD,
    SEARCH_5STORES_DISPLAY_PROJECT_ID,
)
from search_studio.client import (
    ALL_QUERIES,
    DEFAULT_BOOST_MODEL,
    GEMINI_PROJECT,
    PROJECT_ID,
    PROJECT_NUMBER,
    SEARCH_MTLS_HOST,
    SEARCH_PATH,
    SEARCH_PUBLIC_HOST,
    ecp_post,
    evaluate_hit,
    extract_doc_summary,
    format_nlp_filters,
    get_search_app_config,
)
from search_studio.intent_extractor import (
    extract_boosts_llm,
    extract_non_thought_json_text,
    resolve_gemini_endpoint_and_config,
)
from search_studio.prompts import build_explain_hit_prompt
from search_studio.router_api import DEFAULT_ROUTER_MODEL, decide_query_route_gemma3

def build_curl_command(payload: Dict[str, Any], search_path: str = SEARCH_PATH) -> str:
    """Build a copy-pasteable, sanitized `curl` command for the Search Studio UI inspector drawer."""
    payload_str = json.dumps(payload, indent=2).replace(LEGACY_V2_LANGUAGES_FIELD, "languages").replace(CROWDING_FIELD, "entity_code")
    display_path = (
        search_path.replace(str(PROJECT_NUMBER), "<YOUR_PROJECT_NUMBER>")
        .replace(ENGINE_ID_V3_RUNTIME, ENGINE_ID_V3_DISPLAY)
        .replace(ENGINE_ID_V2_RUNTIME, ENGINE_ID_V2_DISPLAY)
    )
    return (
        f'curl -X POST "https://{SEARCH_PUBLIC_HOST}{display_path}" \\\n'
        f'  -H "Authorization: Bearer $(gcloud auth print-access-token)" \\\n'
        f'  -H "x-goog-user-project: {SEARCH_5STORES_DISPLAY_PROJECT_ID}" \\\n'
        f'  -H "Content-Type: application/json" \\\n'
        f"  -d '{payload_str}'"
    )


def _compute_hit_distance_km(
    user_lat: Optional[float],
    user_lng: Optional[float],
    sd: Dict[str, Any],
) -> Optional[float]:
    """Compute Haversine distance in km between (user_lat, user_lng) and document location."""
    if user_lat is None or user_lng is None:
        return None
    import math

    doc_lat: Optional[float] = None
    doc_lng: Optional[float] = None
    loc_obj = sd.get("location")
    if isinstance(loc_obj, dict):
        if loc_obj.get("latitude") is not None and loc_obj.get("longitude") is not None:
            try:
                doc_lat = float(loc_obj["latitude"])
                doc_lng = float(loc_obj["longitude"])
            except (TypeError, ValueError):
                pass
    elif isinstance(loc_obj, str) and "," in loc_obj:
        parts = loc_obj.split(",", 1)
        try:
            doc_lat = float(parts[0].strip())
            doc_lng = float(parts[1].strip())
        except (TypeError, ValueError):
            pass

    if doc_lat is None or doc_lng is None:
        if sd.get("latitude") is not None and sd.get("longitude") is not None:
            try:
                doc_lat = float(sd["latitude"])
                doc_lng = float(sd["longitude"])
            except (TypeError, ValueError):
                pass

    if doc_lat is None or doc_lng is None:
        return None

    r_earth = 6371.0
    d_lat = math.radians(doc_lat - user_lat)
    d_lng = math.radians(doc_lng - user_lng)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(user_lat))
        * math.cos(math.radians(doc_lat))
        * (math.sin(d_lng / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(r_earth * c, 1)


def run_live_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a live Vertex AI Agent Search request supporting `auto_route` (Gemma 3 Router),
    `llm_boost` (Local MLX / Cloud Gemini), `vertex_native` (Managed NLQU), or `none` (Pure BM25+Hybrid),
    along with 3-ring Concentric Lat/Long `GEO_DISTANCE` proximity boosting and 0-hit fallback.
    """
    query = str(params.get("query") or "").strip()
    if not query:
        raise ValueError("Query must not be empty.")

    app_cfg = get_search_app_config(params.get("search_app"))
    app_key = app_cfg["key"]
    target_search_path = app_cfg["search_path"]

    nlu_mode = str(params.get("nlu_mode") or "llm_boost").strip().lower()
    if nlu_mode in ("gemini_boost", "asia_south1_llm", "gemini", "argon", "local_mlx"):
        nlu_mode = "llm_boost"
    elif nlu_mode not in ("llm_boost", "vertex_native", "none", "auto_route"):
        nlu_mode = "llm_boost"

    selected_model = str(params.get("gemini_model") or DEFAULT_BOOST_MODEL).strip()
    router_model = str(params.get("router_model") or DEFAULT_ROUTER_MODEL).strip()
    use_llm_cache = bool(params.get("use_llm_cache", False))
    page_size = max(1, min(50, int(params.get("page_size", 10))))
    raw_behavior = str(params.get("filter_behavior") or "SOFT_BOOST").strip().upper()
    filter_behavior = "HARD_FILTER" if raw_behavior == "HARD_FILTER" else "SOFT_BOOST"
    geo_enabled = bool(params.get("geo_enabled", True))
    allowed_fields_enabled = bool(params.get("allowed_fields_enabled", True))
    collapse_enabled = bool(params.get("collapse_enabled", True))
    custom_filter = str(params.get("custom_filter") or "").strip()

    user_lat: Optional[float] = None
    user_lng: Optional[float] = None
    raw_lat = params.get("user_lat")
    raw_lng = params.get("user_lng")
    if raw_lat not in (None, "") and raw_lng not in (None, ""):
        try:
            lat_f = float(raw_lat)
            lng_f = float(raw_lng)
            if -90.0 <= lat_f <= 90.0 and -180.0 <= lng_f <= 180.0:
                user_lat = round(lat_f, 6)
                user_lng = round(lng_f, 6)
        except (TypeError, ValueError):
            pass

    try:
        user_radius_km = float(params.get("user_radius_km") or 0)
    except (TypeError, ValueError):
        user_radius_km = 0.0

    route_decision: Optional[Dict[str, Any]] = params.get("_precomputed_route_decision")
    effective_nlu_mode = nlu_mode
    if nlu_mode == "auto_route":
        if not route_decision:
            route_decision = decide_query_route_gemma3(
                query,
                model_key=router_model,
                use_cache=use_llm_cache,
                include_reason=True,
            )
        effective_nlu_mode = "llm_boost" if route_decision.get("call_llm_for_intent") else "none"

    nlu_parse_ms = 0.0
    nlu_engine_label = "⚪ Direct Agent Search (No LLM Intent)"
    parsed_nlu: Optional[Dict[str, Any]] = None
    active_model_name = selected_model
    display_model_name = selected_model

    if effective_nlu_mode == "llm_boost":
        parsed_nlu = dict(
            extract_boosts_llm(
                query,
                model_name=selected_model,
                app_key=app_key,
                use_cache=use_llm_cache,
            )
        )
        parsed_nlu["boost_specs"] = list(parsed_nlu.get("boost_specs") or [])
        active_model_name = parsed_nlu["model"]
        display_model_name = parsed_nlu["display_model"]
        nlu_parse_ms = parsed_nlu["parse_ms"]
        cache_tag = " [cached]" if parsed_nlu.get("cached") else ""
        icon = "🟢" if active_model_name.startswith("local:") else "☁️"
        if route_decision:
            r_ms = float(route_decision.get("router_latency_ms") or 0.0)
            nlu_parse_ms = round(nlu_parse_ms + r_ms, 1)
            nlu_engine_label = f"🧭 Auto-Routed → {icon} {display_model_name} ({r_ms}ms router + {parsed_nlu['parse_ms']}ms intent)"
        else:
            nlu_engine_label = f"{icon} {display_model_name} ({nlu_parse_ms} ms{cache_tag})"
    elif effective_nlu_mode == "vertex_native":
        nlu_engine_label = "🧠 Managed Vertex NLQU"
    elif route_decision:
        r_ms = float(route_decision.get("router_latency_ms") or 0.0)
        nlu_parse_ms = r_ms
        nlu_engine_label = f"⚡ Auto-Routed → Direct Agent Search ({r_ms} ms Gemma 3 router, 0 ms LLM intent)"

    effective_query = parsed_nlu["rewritten_query"] if parsed_nlu else query

    payload: Dict[str, Any] = {
        "query": effective_query,
        "pageSize": page_size,
        "queryExpansionSpec": {"condition": "AUTO"},
        "spellCorrectionSpec": {"mode": "AUTO"},
    }

    filter_clauses: List[str] = []
    if custom_filter:
        filter_clauses.append(custom_filter if "GEO_DISTANCE" in custom_filter else f"({custom_filter})")

    active_geo_filter = ""
    if effective_nlu_mode == "vertex_native":
        nlqu_spec: Dict[str, Any] = {
            "filterExtractionCondition": "ENABLED",
            "extractedFilterBehavior": filter_behavior,
        }
        if geo_enabled:
            nlqu_spec["geoSearchQueryDetectionFieldNames"] = ["location"]
        if allowed_fields_enabled:
            nlqu_spec["allowedFieldNames"] = app_cfg["nlp_allowed_fields"]
        payload["naturalLanguageQueryUnderstandingSpec"] = nlqu_spec
    else:
        payload["naturalLanguageQueryUnderstandingSpec"] = {
            "filterExtractionCondition": "DISABLED",
        }
        active_boost_specs: List[Dict[str, Any]] = []
        if parsed_nlu:
            if filter_behavior == "HARD_FILTER" and parsed_nlu.get("hard_filter"):
                filter_clauses.append(parsed_nlu["hard_filter"])
                if geo_enabled and parsed_nlu.get("geo_filter") and user_lat is None:
                    active_geo_filter = parsed_nlu["geo_filter"]
            else:
                if geo_enabled and parsed_nlu.get("geo_filter") and user_lat is None:
                    active_geo_filter = parsed_nlu["geo_filter"]
                    filter_clauses.append(active_geo_filter)
                if parsed_nlu.get("boost_specs"):
                    active_boost_specs.extend(parsed_nlu["boost_specs"])

        # Lat/Long Concentric Proximity Boosting
        if user_lat is not None and user_lng is not None:
            guard_meters = int(user_radius_km * 1000) if user_radius_km > 0 else 3000000
            active_geo_filter = f"location:GEO_DISTANCE({user_lat}, {user_lng}, {guard_meters})"
            filter_clauses.append(active_geo_filter)
            lat_lng_rings = [
                {"condition": f"location:GEO_DISTANCE({user_lat}, {user_lng}, 3000)", "boost": 0.95},
                {"condition": f"location:GEO_DISTANCE({user_lat}, {user_lng}, 10000)", "boost": 0.80},
                {"condition": f"location:GEO_DISTANCE({user_lat}, {user_lng}, 25000)", "boost": 0.55},
            ]
            active_boost_specs.extend(lat_lng_rings)
            if parsed_nlu is not None:
                parsed_nlu["boost_specs"] = active_boost_specs

        if active_boost_specs:
            payload["boostSpec"] = {"conditionBoostSpecs": active_boost_specs}

    if filter_clauses:
        payload["filter"] = " AND ".join(filter_clauses)

    if collapse_enabled:
        payload["crowdingSpecs"] = [
            {
                "field": CROWDING_FIELD,
                "maxCount": 1,
                "mode": "DROP_CROWDED_RESULTS",
            }
        ]

    t_search0 = time.perf_counter()
    resp = ecp_post(SEARCH_MTLS_HOST, target_search_path, payload, user_project=PROJECT_ID)
    search_ms = round((time.perf_counter() - t_search0) * 1000.0, 1)

    # Automatic fallback if sub-locality GEO_DISTANCE returned 0 results (e.g. unindexed suburb)
    geo_fallback_note: Optional[str] = None
    if (
        resp.status_code == 200
        and active_geo_filter
        and filter_behavior != "HARD_FILTER"
        and not (resp.json().get("results") or [])
    ):
        geo_fallback_note = f"0 hits within radius for {active_geo_filter}; fell back to city/all-India boost"
        if user_lat is not None and user_lng is not None and user_radius_km > 0:
            active_geo_filter = f"location:GEO_DISTANCE({user_lat}, {user_lng}, 3000000)"
            payload["filter"] = f"({custom_filter}) AND {active_geo_filter}" if custom_filter else active_geo_filter
        else:
            active_geo_filter = ""
            if custom_filter:
                payload["filter"] = f"({custom_filter})"
            else:
                payload.pop("filter", None)
        t_fb = time.perf_counter()
        resp = ecp_post(SEARCH_MTLS_HOST, target_search_path, payload, user_project=PROJECT_ID)
        search_ms = round(search_ms + (time.perf_counter() - t_fb) * 1000.0, 1)

    boost_syntax_warning: Optional[str] = None
    if resp.status_code == 400 and effective_nlu_mode == "llm_boost" and ("boostSpec" in payload or "filter" in payload):
        boost_syntax_warning = resp.text[:300]
        valid_specs: List[Dict[str, Any]] = []
        probe_filter = f"location:GEO_DISTANCE({user_lat}, {user_lng}, 3000000)" if (user_lat is not None and user_lng is not None) else None
        for spec in (parsed_nlu or {}).get("boost_specs", []):
            probe_payload: Dict[str, Any] = {
                "query": effective_query,
                "pageSize": 1,
                "naturalLanguageQueryUnderstandingSpec": {"filterExtractionCondition": "DISABLED"},
                "boostSpec": {"conditionBoostSpecs": [spec]},
            }
            if probe_filter:
                probe_payload["filter"] = probe_filter
            probe_resp = ecp_post(SEARCH_MTLS_HOST, target_search_path, probe_payload, user_project=PROJECT_ID)
            if probe_resp.status_code == 200:
                valid_specs.append(spec)
        payload.pop("filter", None)
        fallback_filters: List[str] = []
        if custom_filter:
            fallback_filters.append(f"({custom_filter})")
        if probe_filter:
            fallback_filters.append(probe_filter)
            active_geo_filter = probe_filter
        else:
            active_geo_filter = ""
        if fallback_filters:
            payload["filter"] = " AND ".join(fallback_filters)
        if valid_specs:
            payload["boostSpec"] = {"conditionBoostSpecs": valid_specs}
        else:
            payload.pop("boostSpec", None)
        if parsed_nlu is not None:
            parsed_nlu["boost_specs"] = valid_specs
        t_retry = time.perf_counter()
        resp = ecp_post(SEARCH_MTLS_HOST, target_search_path, payload, user_project=PROJECT_ID)
        search_ms = round((time.perf_counter() - t_retry) * 1000.0, 1)

    total_latency_ms = round(nlu_parse_ms + search_ms, 1)

    if resp.status_code != 200:
        return {
            "status": "ERROR",
            "status_code": resp.status_code,
            "error": resp.text[:1000],
            "search_app": app_key,
            "engine_id": app_cfg["engine_id"],
            "app_short_label": app_cfg["short_label"],
            "latency_ms": total_latency_ms,
            "nlu_parse_ms": nlu_parse_ms,
            "search_ms": search_ms,
            "nlu_mode": nlu_mode,
            "effective_nlu_mode": effective_nlu_mode,
            "route_decision": route_decision,
            "gemini_model": active_model_name,
            "display_model": display_model_name,
            "nlu_engine_label": nlu_engine_label,
            "request_payload": payload,
            "curl_command": build_curl_command(payload, search_path=target_search_path),
        }

    data = resp.json()
    nlu_info = data.get("naturalLanguageQueryUnderstandingInfo", {}) or {}
    boost_specs_out: List[Dict[str, Any]] = []
    llm_raw_out: Optional[Dict[str, Any]] = None

    if effective_nlu_mode == "vertex_native":
        extracted_filters = format_nlp_filters(nlu_info)
        rewritten_query = nlu_info.get("rewrittenQuery", "") or query
    elif parsed_nlu:
        rewritten_query = effective_query
        boost_specs_out = parsed_nlu.get("boost_specs", [])
        eff_summary_items: List[str] = []
        if active_geo_filter:
            eff_summary_items.append(f"[GEO] {active_geo_filter}")
        eff_summary_items.extend(
            f"[{'+' if float(s['boost']) >= 0 else ''}{s['boost']}] {s['condition']}"
            for s in boost_specs_out
        )
        extracted_filters = " | ".join(eff_summary_items) if eff_summary_items else "(no explicit facet constraints)"
        llm_raw_out = parsed_nlu.get("llm_raw")
        nlu_info = {
            "searchApp": app_cfg["engine_id"],
            "engine": nlu_engine_label,
            "model": active_model_name,
            "llmLatencyMs": nlu_parse_ms,
            "vertexSearchLatencyMs": search_ms,
            "rewrittenQuery": rewritten_query,
            "geoFilter": active_geo_filter,
            "userLat": user_lat,
            "userLng": user_lng,
            "userRadiusKm": user_radius_km,
            "conditionBoostSpecs": boost_specs_out,
            "llmRawOutput": llm_raw_out,
        }
        if route_decision:
            nlu_info["routeDecision"] = route_decision
        if geo_fallback_note:
            nlu_info["geoFallbackNote"] = geo_fallback_note
        if boost_syntax_warning:
            nlu_info["droppedInvalidFilterWarning"] = boost_syntax_warning
    else:
        boost_specs_out = (payload.get("boostSpec") or {}).get("conditionBoostSpecs", [])
        if route_decision:
            extracted_filters = f"[DIRECT_AGENT_SEARCH] Skipped LLM Intent ({route_decision.get('reason')})"
            nlu_info = {
                "searchApp": app_cfg["engine_id"],
                "engine": nlu_engine_label,
                "routeDecision": route_decision,
                "vertexSearchLatencyMs": search_ms,
            }
        else:
            extracted_filters = f"[GEO] {active_geo_filter}" if active_geo_filter else "(NLU disabled)"
        rewritten_query = query

    preset_match = next(
        (q for q in ALL_QUERIES if q["query"].strip().lower() == query.lower()),
        {
            "id": "LIVE",
            "category": "Live Custom Query",
            "query": query,
            "what_query": query,
            "intent": "User real-time search query",
        },
    )

    raw_results = data.get("results", []) or []
    hits: List[Dict[str, Any]] = []
    seen_entities: set[str] = set()
    dup_entities = 0

    for rank, r in enumerate(raw_results, start=1):
        doc = r.get("document", {}) or {}
        sd = doc.get("structData", {}) or {}
        h = extract_doc_summary(doc)
        h["rank"] = rank
        h["search_app"] = app_key
        if (h.get("entity_code") or ""):
            if (h.get("entity_code") or "") in seen_entities:
                dup_entities += 1
            seen_entities.add((h.get("entity_code") or ""))
        is_pass, explanation = evaluate_hit(preset_match, h, extracted_filters)
        h["pass"] = is_pass
        h["explanation"] = explanation
        h["address"] = sd.get("address") or ""
        h["location"] = sd.get("location")
        h["distance_km"] = _compute_hit_distance_km(user_lat, user_lng, sd)
        h["top_level_formats"] = sd.get("formats") or []
        h["session_formats"] = sorted(
            list({str(s.get("format")) for s in (sd.get("sessions") or []) if isinstance(s, dict) and s.get("format")})
        )
        h["is_sold_out"] = sd.get("is_sold_out") if "is_sold_out" in sd else sd.get(LEGACY_V2_SOLD_OUT_FIELD)
        h["raw_struct_data"] = sd
        hits.append(h)

    pass_count = sum(1 for h in hits if h["pass"])
    fail_count = len(hits) - pass_count

    return {
        "status": "OK",
        "search_app": app_key,
        "engine_id": app_cfg["engine_id"],
        "datastore_id": app_cfg["datastore_id"],
        "bq_table": app_cfg["bq_table"],
        "app_short_label": app_cfg["short_label"],
        "query": query,
        "nlu_mode": nlu_mode,
        "effective_nlu_mode": effective_nlu_mode,
        "route_decision": route_decision,
        "gemini_model": active_model_name,
        "display_model": display_model_name,
        "nlu_engine_label": nlu_engine_label,
        "preset_id": preset_match["id"],
        "category": preset_match["category"],
        "latency_ms": total_latency_ms,
        "nlu_parse_ms": nlu_parse_ms,
        "search_ms": search_ms,
        "total_size": data.get("totalSize", len(hits)),
        "hit_count": len(hits),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "top1_pass": bool(hits and hits[0]["pass"]),
        "duplicate_entities": dup_entities,
        "extracted_filters": extracted_filters,
        "rewritten_query": rewritten_query,
        "geo_filter": active_geo_filter,
        "boost_specs": boost_specs_out,
        "llm_raw": llm_raw_out,
        "nlu_info": nlu_info,
        "hits": hits,
        "request_payload": payload,
        "curl_command": build_curl_command(payload, search_path=target_search_path),
    }


def run_side_by_side_nlqu_vs_llm(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Managed Vertex NLQU and the selected Boost LLM (Local MLX Metal GPU or Vertex Cloud Gemini)
    concurrently against the selected Search App (`v3` or `v2`), plus Gemma 3 Route Decision, and return ranked hit lists & deltas.
    """
    app_cfg = get_search_app_config(params.get("search_app"))
    query = str(params.get("query") or "").strip()
    use_llm_cache = bool(params.get("use_llm_cache", False))
    router_model = str(params.get("router_model") or DEFAULT_ROUTER_MODEL).strip()

    nlqu_params = dict(params)
    nlqu_params["nlu_mode"] = "vertex_native"

    llm_params = dict(params)
    llm_params["nlu_mode"] = "llm_boost"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_nlqu = pool.submit(run_live_search, nlqu_params)
        f_llm = pool.submit(run_live_search, llm_params)
        f_route = pool.submit(
            decide_query_route_gemma3,
            query,
            router_model,
            use_llm_cache,
            True,
        )
        res_nlqu = f_nlqu.result()
        res_llm = f_llm.result()
        try:
            route_decision = f_route.result()
        except Exception as exc:
            route_decision = {"status": "ERROR", "error": str(exc)}

    nlqu_rank_by_key: Dict[str, int] = {}
    for h in res_nlqu.get("hits") or []:
        key = h.get("entity_code") or h.get("id")
        if key and key not in nlqu_rank_by_key:
            nlqu_rank_by_key[key] = h["rank"]

    for h in res_llm.get("hits") or []:
        key = h.get("entity_code") or h.get("id")
        prev_rank = nlqu_rank_by_key.get(key)
        if prev_rank is None:
            h["rank_delta_label"] = "NEW in Top-N"
            h["rank_delta_type"] = "new"
        elif prev_rank > h["rank"]:
            h["rank_delta_label"] = f"▲ +{prev_rank - h['rank']} vs NLQU (#{prev_rank})"
            h["rank_delta_type"] = "up"
        elif prev_rank < h["rank"]:
            h["rank_delta_label"] = f"▼ -{h['rank'] - prev_rank} vs NLQU (#{prev_rank})"
            h["rank_delta_type"] = "down"
        else:
            h["rank_delta_label"] = f"= Same rank (#{prev_rank})"
            h["rank_delta_type"] = "same"

    nlqu_ms = float(res_nlqu.get("latency_ms", 0.0))
    gem_ms = float(res_llm.get("latency_ms", 0.0))
    latency_diff_ms = round(nlqu_ms - gem_ms, 1)

    return {
        "status": "OK",
        "search_app": app_cfg["key"],
        "engine_id": app_cfg["engine_id"],
        "app_short_label": app_cfg["short_label"],
        "query": params.get("query"),
        "gemini_model": res_llm.get("display_model") or res_llm.get("gemini_model"),
        "latency_diff_ms": latency_diff_ms,
        "route_decision": route_decision,
        "vertex_nlqu": res_nlqu,
        "gemini_boost": res_llm,
    }


def run_compare_v2_vs_v3(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run the exact same query & NLU mode concurrently against `v2` (`ent-bq-5stores-v2-app`)
    and `v3` (`ent-bq-5stores-master-v3-app`) for direct head-to-head app comparison.
    """
    v2_params = dict(params)
    v2_params["search_app"] = "v2"

    v3_params = dict(params)
    v3_params["search_app"] = "v3"

    # Note: if using a local Metal GPU model without cache, _MLX_GPU_LOCK serializes only the ~80ms
    # GPU inference while both Vertex AI Search RPCs overlap concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_v2 = pool.submit(run_live_search, v2_params)
        f_v3 = pool.submit(run_live_search, v3_params)
        res_v2 = f_v2.result()
        res_v3 = f_v3.result()

    v2_rank_by_key: Dict[str, int] = {}
    for h in res_v2.get("hits") or []:
        key = h.get("entity_code") or h.get("id")
        if key and key not in v2_rank_by_key:
            v2_rank_by_key[key] = h["rank"]

    for h in res_v3.get("hits") or []:
        key = h.get("entity_code") or h.get("id")
        prev_rank = v2_rank_by_key.get(key)
        if prev_rank is None:
            h["rank_delta_label"] = "NEW in v3 Top-N"
            h["rank_delta_type"] = "new"
        elif prev_rank > h["rank"]:
            h["rank_delta_label"] = f"▲ +{prev_rank - h['rank']} vs v2 (#{prev_rank})"
            h["rank_delta_type"] = "up"
        elif prev_rank < h["rank"]:
            h["rank_delta_label"] = f"▼ -{h['rank'] - prev_rank} vs v2 (#{prev_rank})"
            h["rank_delta_type"] = "down"
        else:
            h["rank_delta_label"] = f"= Same rank (#{prev_rank})"
            h["rank_delta_type"] = "same"

    v2_ms = float(res_v2.get("search_ms", res_v2.get("latency_ms", 0.0)))
    v3_ms = float(res_v3.get("search_ms", res_v3.get("latency_ms", 0.0)))

    return {
        "status": "OK",
        "query": params.get("query"),
        "nlu_mode": res_v3.get("nlu_mode") or params.get("nlu_mode"),
        "search_latency_diff_ms": round(v2_ms - v3_ms, 1),
        "app_v2": res_v2,
        "app_v3": res_v3,
    }




def explain_live_hit(params: Dict[str, Any]) -> Dict[str, Any]:
    """Use `gemini-3.5-flash-lite` in `asia-south1` (fast cloud explainer) to diagnose a single ranked hit."""
    query = params.get("query", "")
    extracted_filters = params.get("extracted_filters", "")
    rewritten_query = params.get("rewritten_query", "")
    hit = params.get("hit", {}) or {}
    app_cfg = get_search_app_config(params.get("search_app") or hit.get("search_app"))
    resolved_model, gemini_region, gemini_host, gemini_path, gen_cfg = resolve_gemini_endpoint_and_config(
        "gemini-3.5-flash-lite", max_tokens_default=420, temperature=0.1
    )
    prompt = build_explain_hit_prompt(
        query=query,
        extracted_filters=extracted_filters,
        rewritten_query=rewritten_query,
        hit=hit,
        app_cfg=app_cfg,
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }
    t0 = time.perf_counter()
    resp = ecp_post(gemini_host, gemini_path, body, user_project=GEMINI_PROJECT)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    if resp.status_code != 200:
        return {
            "status": "ERROR",
            "error": f"{resolved_model} ({gemini_region}) returned {resp.status_code}: {resp.text[:300]}",
        }

    raw_txt = extract_non_thought_json_text(resp.json())
    parsed = json.loads(raw_txt)
    parsed["model"] = f"{resolved_model} ({gemini_region})"
    parsed["latency_ms"] = elapsed_ms
    return parsed
