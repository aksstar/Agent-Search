"""Local Gemma 3 Query Decision Router API (`GET / POST /api/route_decision`).

Uses `mlx-community/gemma-3-1b-it-qat-4bit` on Apple Metal GPU with a pre-warmed
367-token KV cache (reusing loaded model weights so 0 extra GPU RAM is consumed)
to classify whether an incoming query should:
  - `DIRECT_AGENT_SEARCH` (`call_llm_for_intent = False`): Go directly to Vertex AI Agent Search
  - `CALL_LLM_INTENT` (`call_llm_for_intent = True`): Call the LLM Intent/Boost Extractor first
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Dict

from config import (
    VERTEX_GEMMA_1B_ROUTER_ENDPOINT_ID,
    VERTEX_GEMMA_ROUTER_ENDPOINT_ID,
    VERTEX_GEMMA_ROUTER_PROJECT_ID,
    VERTEX_GEMMA_ROUTER_PROJECT_NUMBER,
    VERTEX_GEMMA_ROUTER_REGION,
)
from search_studio.client import (
    CLOUD_GEMINI_MODELS,
    HAS_MLX,
    LOCAL_MODELS,
    _MAX_NLU_CACHE,
    _MLX_GPU_LOCK,
    _MLX_ROUTER_RUNTIMES,
    _NLU_CACHE_LOCK,
    _ROUTER_DECISION_CACHE,
    ecp_post,
)
from search_studio.intent_extractor import (
    _extract_first_json_object,
    extract_non_thought_json_text,
    get_or_init_local_mlx_runtime,
    resolve_gemini_endpoint_and_config,
)
from search_studio.prompts import build_gemma_router_chat_messages

if HAS_MLX:
    import mlx.core as mx
    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

VERTEX_VLLM_ROUTER_MODELS = (
    "vertex:gemma-3-270m-it",
    "google/gemma-3-270m-it",
    "gemma-3-270m-it",
    "vertex:gemma-3-1b-it",
    "google/gemma-3-1b-it",
    "gemma-3-1b-it",
)

ROUTER_CLOUD_SYSTEM_INSTRUCTION = (
    "You are the Entertainment Search Query Router.\n"
    "Classify whether the user query should go directly to Vertex AI Agent Search (`DIRECT_AGENT_SEARCH`) "
    "or call the LLM Intent/Boost Extractor first (`CALL_LLM_INTENT`).\n"
    "- `DIRECT_AGENT_SEARCH`: Pure movie/show/event title, artist/actor/comedian name, theatre/venue name, "
    "or bank/card offer lookup with ZERO city, locality, price, date, time-of-day, format, language, genre, "
    "category, or negation filters.\n"
    "- `CALL_LLM_INTENT`: Query contains ANY city/locality/near me, price/under/between/free/cheapest, "
    "date/time/today/tonight/tomorrow/friday/weekend/late night, format (IMAX/4DX/3D/Dolby/recliner), "
    "language, genre/category, negation (without/not), or occasion.\n\n"
    "Examples:\n"
    'Query: Spider-Man: No Way Home -> {"route":"DIRECT_AGENT_SEARCH","reason":"Direct movie title lookup with no facet filters"}\n'
    'Query: Zakir Khan show -> {"route":"DIRECT_AGENT_SEARCH","reason":"Direct artist show lookup with no city or date filter"}\n'
    'Query: IMAX 3D movies in Mumbai -> {"route":"CALL_LLM_INTENT","reason":"Has format (IMAX 3D) and city (Mumbai) filters"}\n'
    'Query: concerts between 1000 and 3000 in delhi -> {"route":"CALL_LLM_INTENT","reason":"Has category (concerts), price range (1000-3000), and city (Delhi)"}\n'
    'Query: funny movie without any horror -> {"route":"CALL_LLM_INTENT","reason":"Has genre (Comedy) and negative genre exclusion (without horror)"}'
)


def build_gemma_vllm_router_prompt(query: str, seed_prefix: str = '{"route":"') -> str:
    """Format the 367-token multi-turn Gemma 3 chat prompt for Vertex AI vLLM (`--enable-prefix-caching`)."""
    messages = build_gemma_router_chat_messages()
    parts = ["<bos>"]
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"].replace("PLACEHOLDER_QUERY", query)
        parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
    parts.append(f"<start_of_turn>model\n{seed_prefix}")
    return "".join(parts)


def get_or_init_local_mlx_router_runtime(
    model_key: str = "local:gemma-3-1b-it-qat-4bit",
) -> Dict[str, Any]:
    """Initialize and pre-warm the Gemma 3 Router KV cache (reusing the loaded model weights
    in `_MLX_RUNTIMES` so zero extra GPU RAM is consumed). Must be called while holding `_MLX_GPU_LOCK`.
    """
    if not HAS_MLX:
        raise RuntimeError("`mlx-lm` is not installed in this Python interpreter. Run with `./.venv/bin/python`.")
    if model_key not in LOCAL_MODELS:
        model_key = "local:gemma-3-1b-it-qat-4bit"

    router_rt = _MLX_ROUTER_RUNTIMES.get(model_key)
    if router_rt is not None:
        return router_rt

    # Reuse already-loaded model & tokenizer from _MLX_RUNTIMES
    base_rt = get_or_init_local_mlx_runtime(model_key)
    model = base_rt["model"]
    tokenizer = base_rt["tokenizer"]
    hf_repo = base_rt["hf_repo"]

    messages = build_gemma_router_chat_messages()
    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_text, suffix_template = templated.split("PLACEHOLDER_QUERY")
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

    base_router_cache = make_prompt_cache(model)
    t_pre = time.perf_counter()
    model(mx.array([prefix_tokens]), cache=base_router_cache)
    mx.eval([c.state for c in base_router_cache])
    prefix_len = len(prefix_tokens)
    print(
        f"[MLX Router] Pre-warmed Gemma 3 Router KV cache ({prefix_len} tokens) for `{hf_repo}` in {(time.perf_counter() - t_pre) * 1000.0:.1f} ms",
        flush=True,
    )

    router_rt = {
        "model_key": model_key,
        "hf_repo": hf_repo,
        "model": model,
        "tokenizer": tokenizer,
        "base_prompt_cache": base_router_cache,
        "prefix_len": prefix_len,
        "suffix_template": suffix_template,
        "sampler": make_sampler(temp=0.0),
    }
    _MLX_ROUTER_RUNTIMES[model_key] = router_rt
    return router_rt


def decide_query_route_vertex_vllm(
    query: str,
    model_key: str = "vertex:gemma-3-1b-it",
    endpoint_id: str = "",
    region: str = VERTEX_GEMMA_ROUTER_REGION,
    project_id: str = VERTEX_GEMMA_ROUTER_PROJECT_ID,
    project_number: str = VERTEX_GEMMA_ROUTER_PROJECT_NUMBER,
    use_cache: bool = False,
    include_reason: bool = True,
) -> Dict[str, Any]:
    """Use `google/gemma-3-1b-it` deployed on a dedicated Vertex AI Endpoint in `asia-south1`
    with vLLM to classify whether `query` should go directly to Agent Search (`DIRECT_AGENT_SEARCH`)
    or call the LLM Intent Extractor (`CALL_LLM_INTENT`).
    """
    q_clean = str(query or "").strip()
    if not q_clean:
        raise ValueError("Query must not be empty.")

    canonical_key = "vertex:gemma-3-1b-it"
    hf_name = "google/gemma-3-1b-it"
    resolved_ep = endpoint_id or VERTEX_GEMMA_1B_ROUTER_ENDPOINT_ID or VERTEX_GEMMA_ROUTER_ENDPOINT_ID

    cache_key = f"router::{canonical_key}::{include_reason}::{q_clean.lower()}"
    if use_cache:
        with _NLU_CACHE_LOCK:
            if cache_key in _ROUTER_DECISION_CACHE:
                cached = dict(_ROUTER_DECISION_CACHE[cache_key])
                cached["router_latency_ms"] = 0.0
                cached["cached"] = True
                return cached

    seed_prefix = '{"route":"'
    prompt_str = build_gemma_vllm_router_prompt(q_clean, seed_prefix=seed_prefix)
    max_toks = 44 if include_reason else 8
    stop_seqs = ["}", "<end_of_turn>"] if include_reason else ['",', "}", "<end_of_turn>"]

    host = f"{region}-aiplatform.mtls.googleapis.com"
    path = f"/v1beta1/projects/{project_number}/locations/{region}/endpoints/{resolved_ep}:predict"
    body = {
        "instances": [
            {
                "prompt": prompt_str,
                "max_tokens": max_toks,
                "temperature": 0.0,
                "top_p": 1.0,
                "stop": stop_seqs,
                "raw_response": True,
            }
        ]
    }

    t0 = time.perf_counter()
    resp = ecp_post(host, path, body, user_project=project_id)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Vertex vLLM endpoint {resolved_ep} ({hf_name} in {region}) returned HTTP {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()
    preds = data.get("predictions") or []
    raw_pred = preds[0] if preds else ""
    if isinstance(raw_pred, dict):
        gen_text = str(raw_pred.get("text") or raw_pred.get("output") or raw_pred.get("content") or "")
    else:
        gen_text = str(raw_pred)
    if "Output:\n" in gen_text:
        gen_text = gen_text.split("Output:\n", 1)[1]

    raw_out = (seed_prefix + gen_text.lstrip()).strip()
    if not raw_out.endswith("}"):
        if '"reason"' not in raw_out:
            raw_out = raw_out.rstrip(',"') + f'","reason":"Classified by Vertex {hf_name} vLLM Router"}}'
        else:
            raw_out = raw_out.rstrip('"') + '"}'

    try:
        parsed = json.loads(raw_out)
    except Exception:
        parsed = _extract_first_json_object(raw_out)

    route_str = str(parsed.get("route") or "CALL_LLM_INTENT").strip().upper()
    if "DIRECT" in route_str:
        route_str = "DIRECT_AGENT_SEARCH"
        call_llm = False
    else:
        route_str = "CALL_LLM_INTENT"
        call_llm = True

    reason_str = str(
        parsed.get("reason")
        or ("Requires structured filter/boost extraction" if call_llm else "Direct entity lookup")
    ).strip()

    decision = {
        "status": "OK",
        "query": q_clean,
        "call_llm_for_intent": call_llm,
        "route": route_str,
        "recommended_nlu_mode": "llm_boost" if call_llm else "vertex_native",
        "reason": reason_str,
        "router_model": canonical_key,
        "router_display_model": f"{hf_name} (Vertex L4 vLLM • {region})",
        "router_latency_ms": elapsed_ms,
        "cached": False,
    }

    with _NLU_CACHE_LOCK:
        _ROUTER_DECISION_CACHE[cache_key] = dict(decision)
        while len(_ROUTER_DECISION_CACHE) > _MAX_NLU_CACHE:
            _ROUTER_DECISION_CACHE.popitem(last=False)

    return decision


def decide_query_route_cloud_gemini(
    query: str,
    model_name: str = "gemini-3.5-flash-lite",
    use_cache: bool = False,
    include_reason: bool = True,
) -> Dict[str, Any]:
    """Use Cloud Vertex AI Gemini (`gemini-3.5-flash-lite` in `asia-south1`) to classify
    whether `query` should go directly to Agent Search (`DIRECT_AGENT_SEARCH`) or call
    the LLM Intent Extractor (`CALL_LLM_INTENT`).
    """
    q_clean = str(query or "").strip()
    if not q_clean:
        raise ValueError("Query must not be empty.")

    resolved_model, gemini_region, gemini_host, gemini_path, gen_cfg = resolve_gemini_endpoint_and_config(
        model_name, max_tokens_default=96 if include_reason else 32, temperature=0.0
    )
    cache_key = f"router::{resolved_model}::{include_reason}::{q_clean.lower()}"
    if use_cache:
        with _NLU_CACHE_LOCK:
            if cache_key in _ROUTER_DECISION_CACHE:
                cached = dict(_ROUTER_DECISION_CACHE[cache_key])
                cached["router_latency_ms"] = 0.0
                cached["cached"] = True
                return cached

    if include_reason:
        gen_cfg["responseSchema"] = {
            "type": "OBJECT",
            "properties": {
                "route": {"type": "STRING", "enum": ["DIRECT_AGENT_SEARCH", "CALL_LLM_INTENT"]},
                "reason": {"type": "STRING"},
            },
            "required": ["route", "reason"],
        }
    else:
        gen_cfg["responseSchema"] = {
            "type": "OBJECT",
            "properties": {
                "route": {"type": "STRING", "enum": ["DIRECT_AGENT_SEARCH", "CALL_LLM_INTENT"]},
            },
            "required": ["route"],
        }

    body = {
        "systemInstruction": {"parts": [{"text": ROUTER_CLOUD_SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": f"Query: {q_clean}"}]}],
        "generationConfig": gen_cfg,
    }

    t0 = time.perf_counter()
    resp = ecp_post(gemini_host, gemini_path, body)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    if resp.status_code != 200:
        raise RuntimeError(
            f"{resolved_model} ({gemini_region}) router returned HTTP {resp.status_code}: {resp.text[:400]}"
        )

    raw_txt = extract_non_thought_json_text(resp.json())
    parsed = _extract_first_json_object(raw_txt)

    route_str = str(parsed.get("route") or "CALL_LLM_INTENT").strip().upper()
    if "DIRECT" in route_str:
        route_str = "DIRECT_AGENT_SEARCH"
        call_llm = False
    else:
        route_str = "CALL_LLM_INTENT"
        call_llm = True

    reason_str = str(
        parsed.get("reason")
        or ("Requires structured filter/boost extraction" if call_llm else "Direct entity lookup")
    ).strip()

    decision = {
        "status": "OK",
        "query": q_clean,
        "call_llm_for_intent": call_llm,
        "route": route_str,
        "recommended_nlu_mode": "llm_boost" if call_llm else "vertex_native",
        "reason": reason_str,
        "router_model": resolved_model,
        "router_display_model": f"{resolved_model} (Vertex Cloud • {gemini_region})",
        "router_latency_ms": elapsed_ms,
        "cached": False,
    }

    with _NLU_CACHE_LOCK:
        _ROUTER_DECISION_CACHE[cache_key] = dict(decision)
        while len(_ROUTER_DECISION_CACHE) > _MAX_NLU_CACHE:
            _ROUTER_DECISION_CACHE.popitem(last=False)

    return decision


def decide_query_route_gemma3(
    query: str,
    model_key: str = "local:gemma-3-1b-it-qat-4bit",
    use_cache: bool = False,
    include_reason: bool = True,
) -> Dict[str, Any]:
    """Use Local Gemma 3 on Apple Metal GPU, Vertex AI Model Garden Gemma 3 270M/1B vLLM,
    or Cloud Gemini to decide whether `query` should call the LLM Intent/Boost Extractor
    (`CALL_LLM_INTENT`, `call_llm_for_intent=True`) or go directly to Agent Search
    (`DIRECT_AGENT_SEARCH`, `call_llm_for_intent=False`).
    """
    q_clean = str(query or "").strip()
    if not q_clean:
        raise ValueError("Query must not be empty.")
    if model_key in VERTEX_VLLM_ROUTER_MODELS or model_key.startswith("vertex:"):
        return decide_query_route_vertex_vllm(
            query=q_clean,
            model_key=model_key,
            use_cache=use_cache,
            include_reason=include_reason,
        )
    if model_key in CLOUD_GEMINI_MODELS or model_key.startswith("gemini-"):
        return decide_query_route_cloud_gemini(
            query=q_clean,
            model_name=model_key,
            use_cache=use_cache,
            include_reason=include_reason,
        )
    if not HAS_MLX:
        return decide_query_route_vertex_vllm(
            query=q_clean,
            model_key=model_key,
            use_cache=use_cache,
            include_reason=include_reason,
        )
    if model_key not in LOCAL_MODELS:
        model_key = "local:gemma-3-1b-it-qat-4bit"

    cache_key = f"router::{model_key}::{include_reason}::{q_clean.lower()}"
    if use_cache:
        with _NLU_CACHE_LOCK:
            if cache_key in _ROUTER_DECISION_CACHE:
                cached = dict(_ROUTER_DECISION_CACHE[cache_key])
                cached["router_latency_ms"] = 0.0
                cached["cached"] = True
                return cached

    with _MLX_GPU_LOCK:
        rt = get_or_init_local_mlx_router_runtime(model_key)
        t0 = time.perf_counter()
        prompt_cache = copy.deepcopy(rt["base_prompt_cache"])
        seed_prefix = '{"route":"'
        q_tokens = rt["tokenizer"].encode(
            q_clean + rt["suffix_template"] + seed_prefix,
            add_special_tokens=False,
        )
        max_toks = 44 if include_reason else 8
        buf = [seed_prefix]
        for step in mlx_stream_generate(
            rt["model"],
            rt["tokenizer"],
            prompt=q_tokens,
            max_tokens=max_toks,
            sampler=rt["sampler"],
            prompt_cache=prompt_cache,
        ):
            buf.append(step.text)
            if not include_reason and '",' in "".join(buf):
                break
            if "}" in step.text:
                break
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    raw_out = "".join(buf).strip()
    if not raw_out.endswith("}"):
        if '"reason"' not in raw_out:
            raw_out = raw_out.rstrip(',"') + '","reason":"Classified by Gemma 3 Router"}'
        else:
            raw_out = raw_out.rstrip('"') + '"}'

    try:
        parsed = json.loads(raw_out)
    except Exception:
        parsed = _extract_first_json_object(raw_out)

    route_str = str(parsed.get("route") or "CALL_LLM_INTENT").strip().upper()
    if "DIRECT" in route_str:
        route_str = "DIRECT_AGENT_SEARCH"
        call_llm = False
    else:
        route_str = "CALL_LLM_INTENT"
        call_llm = True

    reason_str = str(parsed.get("reason") or ("Requires structured filter/boost extraction" if call_llm else "Direct entity lookup")).strip()

    short_name = model_key.replace("local:", "")
    decision = {
        "status": "OK",
        "query": q_clean,
        "call_llm_for_intent": call_llm,
        "route": route_str,
        "recommended_nlu_mode": "llm_boost" if call_llm else "vertex_native",
        "reason": reason_str,
        "router_model": model_key,
        "router_display_model": f"{short_name} (Local Metal GPU)",
        "router_latency_ms": elapsed_ms,
        "cached": False,
    }

    with _NLU_CACHE_LOCK:
        _ROUTER_DECISION_CACHE[cache_key] = dict(decision)
        while len(_ROUTER_DECISION_CACHE) > _MAX_NLU_CACHE:
            _ROUTER_DECISION_CACHE.popitem(last=False)

    return decision


def handle_route_decision_request(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle `GET /api/route_decision` or `POST /api/route_decision` powered by local Gemma 3 (`mlx-lm`), Vertex vLLM, or Cloud Gemini."""
    query = str(params.get("query") or "").strip()
    if not query:
        raise ValueError("Missing required parameter 'query'.")
    default_model = "local:gemma-3-1b-it-qat-4bit" if HAS_MLX else "vertex:gemma-3-1b-it"
    model_key = str(params.get("model") or params.get("router_model") or default_model).strip()
    use_cache = bool(params.get("use_cache", False))
    raw_inc = params.get("include_reason", True)
    include_reason = str(raw_inc).lower() not in ("false", "0", "no") if isinstance(raw_inc, str) else bool(raw_inc)
    raw_exec = params.get("execute_search", False)
    execute_search = str(raw_exec).lower() in ("true", "1", "yes") if isinstance(raw_exec, str) else bool(raw_exec)

    decision = decide_query_route_gemma3(
        query=query,
        model_key=model_key,
        use_cache=use_cache,
        include_reason=include_reason,
    )
    if execute_search:
        from search_studio.search_service import run_live_search

        search_params = dict(params)
        search_params["nlu_mode"] = "auto_route"
        search_params["_precomputed_route_decision"] = decision
        decision = dict(decision)
        decision["search_result"] = run_live_search(search_params)
    return decision


ROUTER_BENCHMARK_30 = (
    (1, "Spider-Man: No Way Home", False),
    (2, "Pushpa 2: The Rule", False),
    (3, "Singham Again", False),
    (4, "Bhool Bhulaiyaa 3", False),
    (5, "Coldplay Music of the Spheres", False),
    (6, "Zakir Khan show", False),
    (7, "Arijit Singh", False),
    (8, "Diljit Dosanjh Dil-Luminati", False),
    (9, "Sunburn Arena", False),
    (10, "Prithvi Theatre", False),
    (11, "Nita Mukesh Ambani Cultural Centre", False),
    (12, "ICICI credit card movie offer", False),
    (13, "HDFC bank Buy 1 Get 1 offer", False),
    (14, "Gladiator II", False),
    (15, "Kantara", False),
    (16, "IMAX 3D movies in Mumbai", True),
    (17, "hindi horror movie in imax this friday in pune", True),
    (18, "concerts between 1000 and 3000 in delhi", True),
    (19, "standup comedy in koramangala tonight", True),
    (20, "couple garba passes in ahmedabad under 1500", True),
    (21, "cinemas near phoenix marketcity whitefield", True),
    (22, "funny movie without any horror", True),
    (23, "something after 9 pm near hitec city that isnt a movie", True),
    (24, "malayalam movies with english subtitles", True),
    (25, "recliner seats movie in gurgaon tomorrow evening", True),
    (26, "kids magic show this weekend in chennai under 500", True),
    (27, "telugu action movie in 4dx this saturday in hyderabad", True),
    (28, "late night movie shows in andheri tonight", True),
    (29, "plays in kolkata this sunday", True),
    (30, "free workshops in bangalore this weekend", True),
)


def handle_router_benchmark_request(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run the 30-query Decision Router benchmark inside the current runtime (e.g. Cloud Run `asia-south1`)."""
    import os
    import statistics

    model_key = str(params.get("model") or "vertex:gemma-3-1b-it").strip()
    raw_inc = params.get("include_reason")

    if raw_inc is None:
        modes = [False, True]
    else:
        inc_val = str(raw_inc).lower() not in ("false", "0", "no") if isinstance(raw_inc, str) else bool(raw_inc)
        modes = [inc_val]

    # Warm-up 2 calls so HTTP keep-alive session and vLLM prefix cache are hot
    for _ in range(2):
        decide_query_route_gemma3("Spider-Man: No Way Home", model_key=model_key, use_cache=False, include_reason=False)
        decide_query_route_gemma3("IMAX 3D movies in Mumbai", model_key=model_key, use_cache=False, include_reason=True)

    results: Dict[str, Any] = {}
    for inc in modes:
        mode_name = "full_with_reason" if inc else "fast_route_only"
        lats = []
        correct = 0
        tp = 0
        total_intent = 0
        per_q = []
        for q_id, q_str, expected_call_llm in ROUTER_BENCHMARK_30:
            dec = decide_query_route_gemma3(q_str, model_key=model_key, use_cache=False, include_reason=inc)
            lat = float(dec["router_latency_ms"])
            lats.append(lat)
            actual = bool(dec["call_llm_for_intent"])
            ok = actual == expected_call_llm
            if ok:
                correct += 1
            if expected_call_llm:
                total_intent += 1
                if actual:
                    tp += 1
            per_q.append({
                "id": q_id,
                "query": q_str,
                "route": dec["route"],
                "reason": dec.get("reason", ""),
                "latency_ms": lat,
                "correct": ok,
            })
        s_lat = sorted(lats)
        n = len(s_lat)
        results[mode_name] = {
            "mode": mode_name,
            "model": model_key,
            "include_reason": inc,
            "count": n,
            "min_ms": round(min(s_lat), 1),
            "mean_ms": round(statistics.mean(s_lat), 1),
            "p50_ms": round(s_lat[n // 2], 1),
            "p90_ms": round(s_lat[min(n - 1, int(n * 0.90))], 1),
            "p95_ms": round(s_lat[min(n - 1, int(n * 0.95))], 1),
            "max_ms": round(max(s_lat), 1),
            "accuracy": f"{correct}/{n} ({round(100.0 * correct / n, 1)}%)",
            "call_llm_intent_recall": f"{tp}/{total_intent} ({round(100.0 * tp / max(1, total_intent), 1)}%)",
            "queries": per_q,
        }
    return {
        "status": "OK",
        "origin_env": "cloud_run_asia_south1" if os.getenv("K_SERVICE") else "local_mac_mtls",
        "k_service": os.getenv("K_SERVICE", ""),
        "k_revision": os.getenv("K_REVISION", ""),
        "model": model_key,
        "results": results,
    }


