"""LLM-Based Intent Extractor (Local Apple Metal GPU `mlx-lm` & Cloud Vertex AI Gemini).

Extracts compact 8-key JSON intent slots from natural-language search queries and compiles
them into Vertex AI Agent Search `conditionBoostSpecs` via `search_studio.compiler`.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from search_studio.client import (
    CLOUD_GEMINI_MODELS,
    DEFAULT_BOOST_MODEL,
    DEFAULT_SEARCH_APP,
    GEMINI_MODEL_REGIONS,
    GEMINI_PROJECT,
    GEMINI_REGION,
    HAS_MLX,
    LOCAL_MODELS,
    _LLM_BOOST_CACHE,
    _MAX_NLU_CACHE,
    _MLX_GPU_LOCK,
    _MLX_RUNTIMES,
    _NLU_CACHE_LOCK,
    _date_context,
    ecp_post,
    get_search_app_config,
)
from search_studio.compiler import (
    _sanitize_geo_filter_expr,
    serialize_llm_slots_to_boost_specs,
)
from search_studio.prompts import (
    GEMINI_INTENT_SYSTEM_INSTRUCTION,
    LOCAL_LLM_SYSTEM_PROMPT,
    build_gemma_chat_messages,
)

if HAS_MLX:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

def _salvage_json_slots(raw_text: str) -> Dict[str, Any]:
    """Resilient key-value + string-array salvage when a tiny 270M model emits a minor JSON syntax glitch."""
    import re

    out: Dict[str, Any] = {}
    for m_arr in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*\[([^\]]*)\]', raw_text):
        arr_key, arr_body = m_arr.group(1), m_arr.group(2)
        items = [s.group(1) for s in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', arr_body)]
        if items:
            out[arr_key] = items
    for m in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*("([^"\\]*(?:\\.[^"\\]*)*)"|true|false|-?\d+)', raw_text):
        key, val_raw, str_inner = m.group(1), m.group(2), m.group(3)
        if str_inner is not None:
            out[key] = str_inner
        elif val_raw == "true":
            out[key] = True
        elif val_raw == "false":
            out[key] = False
        else:
            try:
                out[key] = int(val_raw)
            except ValueError:
                pass
    return out


def _extract_first_json_object(raw_text: str) -> Dict[str, Any]:
    """Extract the first balanced `{...}` JSON object from generated text (with resilient salvage)."""
    start = raw_text.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    esc = False
    candidate = raw_text[start:]
    for idx in range(start, len(raw_text)):
        ch = raw_text[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw_text[start : idx + 1]
                    break
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return _salvage_json_slots(candidate)



def get_or_init_local_mlx_runtime(model_key: str) -> Dict[str, Any]:
    """Load the 4-bit MLX model onto Apple M4 Pro Metal GPU and pre-warm its system prompt KV cache.
    Must be called while holding `_MLX_GPU_LOCK`.
    """
    if not HAS_MLX:
        raise RuntimeError("`mlx-lm` is not installed in this Python interpreter. Run with `./.venv/bin/python`.")
    hf_repo = LOCAL_MODELS[model_key]
    dctx = _date_context()
    date_key = dctx["today_iso"]

    rt = _MLX_RUNTIMES.get(model_key)
    if rt is not None and rt.get("date_key") == date_key:
        return rt

    if rt is not None:
        model = rt["model"]
        tokenizer = rt["tokenizer"]
    else:
        print(f"[MLX] Loading local model `{hf_repo}` onto Metal GPU...", flush=True)
        t_load = time.perf_counter()
        model, tokenizer = mlx_load(hf_repo)
        print(f"[MLX] Loaded `{hf_repo}` in {(time.perf_counter() - t_load) * 1000.0:.1f} ms", flush=True)

    is_gemma = "gemma" in model_key.lower()
    if is_gemma:
        messages = build_gemma_chat_messages(dctx)
    else:
        sys_prompt = LOCAL_LLM_SYSTEM_PROMPT.format(**dctx)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "PLACEHOLDER_QUERY"},
        ]

    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_text, suffix_template = templated.split("PLACEHOLDER_QUERY")
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

    base_prompt_cache = make_prompt_cache(model)
    t_pre = time.perf_counter()
    model(mx.array([prefix_tokens]), cache=base_prompt_cache)
    mx.eval([c.state for c in base_prompt_cache])
    prefix_len = len(prefix_tokens)
    print(
        f"[MLX] Pre-warmed system prompt KV cache ({prefix_len} tokens) for `{hf_repo}` in {(time.perf_counter() - t_pre) * 1000.0:.1f} ms",
        flush=True,
    )

    rt = {
        "model_key": model_key,
        "hf_repo": hf_repo,
        "model": model,
        "tokenizer": tokenizer,
        "base_prompt_cache": base_prompt_cache,
        "prefix_len": prefix_len,
        "suffix_template": suffix_template,
        "sampler": make_sampler(temp=0.0),
        "date_key": date_key,
    }
    _MLX_RUNTIMES[model_key] = rt
    return rt


def _stream_first_json_object_mlx(
    model: Any,
    tokenizer: Any,
    q_tokens: List[int],
    sampler: Any,
    prompt_cache: Any,
    max_tokens: int = 110,
) -> str:
    """Stream tokens on Metal GPU and stop immediately when the root `{...}` JSON closes."""
    buf = ['{"']
    depth = 1
    in_str = True
    esc = False
    for step in mlx_stream_generate(
        model,
        tokenizer,
        prompt=q_tokens,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
    ):
        piece = step.text
        buf.append(piece)
        for ch in piece:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return "".join(buf)
    return "".join(buf)


def extract_boosts_local_mlx(
    query: str,
    model_key: str,
    app_key: str = DEFAULT_SEARCH_APP,
    use_cache: bool = False,
) -> Dict[str, Any]:
    """Run local Metal GPU (`mlx-lm`) LLM extraction using the pre-warmed KV prompt cache."""
    cache_key = f"{app_key}::{model_key}::{query.strip().lower()}"
    if use_cache:
        with _NLU_CACHE_LOCK:
            if cache_key in _LLM_BOOST_CACHE:
                cached = dict(_LLM_BOOST_CACHE[cache_key])
                cached["parse_ms"] = 0.0
                cached["cached"] = True
                return cached

    with _MLX_GPU_LOCK:
        rt = get_or_init_local_mlx_runtime(model_key)
        t0 = time.perf_counter()
        # Clone pre-warmed cache via copy-on-write MLX array handles (safe for Gemma RotatingKVCache & Qwen KVCache)
        prompt_cache = copy.deepcopy(rt["base_prompt_cache"])
        q_tokens = rt["tokenizer"].encode(
            query.strip() + rt["suffix_template"] + '{"',
            add_special_tokens=False,
        )
        raw_json_str = _stream_first_json_object_mlx(
            rt["model"],
            rt["tokenizer"],
            q_tokens,
            rt["sampler"],
            prompt_cache,
            max_tokens=110,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    try:
        slots = _extract_first_json_object(raw_json_str)
    except Exception:
        slots = {"q": query.strip()}
    trust_sem = any(tag in model_key.lower() for tag in ("3b", "4b", "e2b"))
    rewritten, boost_specs, geo_filter, hard_filter, summary_display = serialize_llm_slots_to_boost_specs(
        slots, query, app_key=app_key, trust_semantic_slots=trust_sem
    )

    short_name = model_key.replace("local:", "")
    result = {
        "model": model_key,
        "display_model": f"{short_name} (Local Metal GPU)",
        "engine": f"{short_name} (Apple M4 Pro MLX)",
        "parse_ms": elapsed_ms,
        "cached": False,
        "llm_raw": slots,
        "rewritten_query": rewritten,
        "boost_specs": boost_specs,
        "geo_filter": geo_filter,
        "hard_filter": hard_filter,
        "extracted_filters_display": summary_display,
    }

    with _NLU_CACHE_LOCK:
        _LLM_BOOST_CACHE[cache_key] = dict(result)
        while len(_LLM_BOOST_CACHE) > _MAX_NLU_CACHE:
            _LLM_BOOST_CACHE.popitem(last=False)

    return result




def resolve_gemini_endpoint_and_config(
    model_name: Optional[str],
    max_tokens_default: int = 320,
    temperature: float = 0.0,
) -> Tuple[str, str, str, str, Dict[str, Any]]:
    model = (model_name or "gemini-4-argon").strip()
    if model not in CLOUD_GEMINI_MODELS:
        model = "gemini-4-argon"
    region = GEMINI_MODEL_REGIONS.get(model, GEMINI_REGION)
    host = (
        "aiplatform.mtls.googleapis.com"
        if region == "global"
        else f"{region}-aiplatform.mtls.googleapis.com"
    )
    path = f"/v1/projects/{GEMINI_PROJECT}/locations/{region}/publishers/google/models/{model}:generateContent"
    if model.startswith("gemini-4"):
        gen_cfg: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max(768, max_tokens_default),
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "LOW"},
        }
    else:
        gen_cfg = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens_default,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        }
    return model, region, host, path, gen_cfg


def extract_non_thought_json_text(resp_json: Dict[str, Any]) -> str:
    parts = resp_json.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text_parts = [str(p.get("text", "")) for p in parts if isinstance(p, dict) and not p.get("thought") and p.get("text")]
    return text_parts[-1] if text_parts else "{}"


def extract_boosts_cloud_gemini(
    query: str,
    model_name: str,
    app_key: str = DEFAULT_SEARCH_APP,
    use_cache: bool = False,
) -> Dict[str, Any]:
    resolved_model, gemini_region, gemini_host, gemini_path, gen_cfg = resolve_gemini_endpoint_and_config(
        model_name, max_tokens_default=320
    )
    cache_key = f"{app_key}::{resolved_model}::{query.strip().lower()}"
    if use_cache:
        with _NLU_CACHE_LOCK:
            if cache_key in _LLM_BOOST_CACHE:
                cached = dict(_LLM_BOOST_CACHE[cache_key])
                cached["parse_ms"] = 0.0
                cached["cached"] = True
                return cached

    dctx = _date_context()
    sys_inst = GEMINI_INTENT_SYSTEM_INSTRUCTION.format(**dctx)
    body = {
        "systemInstruction": {"parts": [{"text": sys_inst}]},
        "contents": [{"role": "user", "parts": [{"text": query.strip()}]}],
        "generationConfig": gen_cfg,
    }

    t0 = time.perf_counter()
    resp = ecp_post(gemini_host, gemini_path, body, user_project=GEMINI_PROJECT)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    if resp.status_code != 200:
        raise RuntimeError(
            f"{resolved_model} ({gemini_region}) returned HTTP {resp.status_code}: {resp.text[:400]}"
        )

    raw_txt = extract_non_thought_json_text(resp.json())
    llm_obj = _extract_first_json_object(raw_txt)

    rewritten, boost_specs, geo_filter, hard_filter, summary_display = serialize_llm_slots_to_boost_specs(
        llm_obj, query, app_key=app_key, trust_semantic_slots=True
    )

    result = {
        "model": resolved_model,
        "display_model": f"{resolved_model} (Vertex Cloud • {gemini_region})",
        "engine": f"{resolved_model} ({gemini_region})",
        "parse_ms": elapsed_ms,
        "cached": False,
        "llm_raw": llm_obj,
        "rewritten_query": rewritten,
        "boost_specs": boost_specs,
        "geo_filter": geo_filter,
        "hard_filter": hard_filter,
        "extracted_filters_display": summary_display,
    }

    with _NLU_CACHE_LOCK:
        _LLM_BOOST_CACHE[cache_key] = dict(result)
        while len(_LLM_BOOST_CACHE) > _MAX_NLU_CACHE:
            _LLM_BOOST_CACHE.popitem(last=False)

    return result


def extract_boosts_llm(
    query: str,
    model_name: Optional[str] = None,
    app_key: str = DEFAULT_SEARCH_APP,
    use_cache: bool = False,
) -> Dict[str, Any]:
    """Dispatch boost extraction to either Local Apple Metal GPU (`mlx-lm`) or Vertex Cloud Gemini."""
    chosen = (model_name or DEFAULT_BOOST_MODEL).strip()
    if chosen in LOCAL_MODELS:
        return extract_boosts_local_mlx(query, chosen, app_key=app_key, use_cache=use_cache)
    return extract_boosts_cloud_gemini(query, chosen, app_key=app_key, use_cache=use_cache)

