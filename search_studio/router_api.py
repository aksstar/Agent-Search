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

from search_studio.client import (
    HAS_MLX,
    LOCAL_MODELS,
    _MAX_NLU_CACHE,
    _MLX_GPU_LOCK,
    _MLX_ROUTER_RUNTIMES,
    _NLU_CACHE_LOCK,
    _ROUTER_DECISION_CACHE,
)
from search_studio.intent_extractor import (
    _extract_first_json_object,
    get_or_init_local_mlx_runtime,
)
from search_studio.prompts import build_gemma_router_chat_messages

if HAS_MLX:
    import mlx.core as mx
    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

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


def decide_query_route_gemma3(
    query: str,
    model_key: str = "local:gemma-3-1b-it-qat-4bit",
    use_cache: bool = False,
    include_reason: bool = True,
) -> Dict[str, Any]:
    """Use Local Gemma 3 on Apple Metal GPU to decide whether `query` should call the
    LLM Intent/Boost Extractor (`CALL_LLM_INTENT`, `call_llm_for_intent=True`) or go
    directly to Agent Search (`DIRECT_AGENT_SEARCH`, `call_llm_for_intent=False`).
    """
    q_clean = str(query or "").strip()
    if not q_clean:
        raise ValueError("Query must not be empty.")
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
    """Handle `GET /api/route_decision` or `POST /api/route_decision` powered by local Gemma 3 (`mlx-lm`)."""
    query = str(params.get("query") or "").strip()
    if not query:
        raise ValueError("Missing required parameter 'query'.")
    model_key = str(params.get("model") or params.get("router_model") or "local:gemma-3-1b-it-qat-4bit").strip()
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
