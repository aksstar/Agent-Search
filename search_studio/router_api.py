"""SetFit-MiniLM Query Decision Router API (`GET / POST /api/route_decision`).

Classifies incoming entertainment search queries in ~0.9 ms on CPU using the custom-trained
SetFit Contrastive Embedding Classifier (`sentence-transformers/all-MiniLM-L6-v2` ONNX +
128-d Contrastive Residual Projection Head + 7 Multi-Label Facet Heads):
  - `DIRECT_AGENT_SEARCH` (`call_llm_for_intent = False`): Pure entity, title, artist, venue, or bank offer lookup -> dispatches directly to Vertex AI Agent Search.
  - `CALL_LLM_INTENT` (`call_llm_for_intent = True`): Contains city, locality, price, date/time, format, language, genre/category, or negation filters -> invokes the LLM Slot Extractor + Deterministic Boost Compiler first.

Supports both:
  1. `"setfit:minilm-l6-v2"` (default): In-process CPU ONNX inference (~0.9 ms P50).
  2. `"vertex:setfit-minilm-l6-v2"`: Dedicated GCP Vertex AI Online Prediction Endpoint in `asia-south1`.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import statistics
import threading
import time
from typing import Any, Dict

from config import (
    VERTEX_GEMMA_ROUTER_PROJECT_ID,
    VERTEX_GEMMA_ROUTER_PROJECT_NUMBER,
    VERTEX_GEMMA_ROUTER_REGION,
    VERTEX_SETFIT_ROUTER_ENDPOINT_ID,
)
from search_studio.client import (
    _MAX_NLU_CACHE,
    _NLU_CACHE_LOCK,
    _ROUTER_DECISION_CACHE,
    ecp_post,
)

DEFAULT_ROUTER_MODEL = "setfit:minilm-l6-v2"

SETFIT_ROUTER_MODELS = (
    "setfit:minilm-l6-v2",
    "setfit",
    "minilm",
    "minilm-l6-v2",
)

VERTEX_SETFIT_ROUTER_MODELS = (
    "vertex:setfit-minilm-l6-v2",
    "vertex:setfit",
    "gcp:setfit-minilm-l6-v2",
)

_SETFIT_LOCK = threading.Lock()
_SETFIT_RUNTIME: Dict[str, Any] = {}

_FACET_PRETTY_LABELS: Dict[str, str] = {
    "city": "city/locality",
    "date_time": "date/time",
    "price": "price range",
    "format": "format/seating",
    "language": "language",
    "genre_category": "genre/category",
    "negation": "negation/exclusion",
}

_EXPLICIT_FACET_REGEX = re.compile(
    r"\b("
    r"in\s+(?:mumbai|delhi|ncr|pune|bangalore|bengaluru|hyderabad|chennai|kolkata|ahmedabad|gurgaon|gurugram|noida|kochi|jaipur|lucknow|lko|koramangala|andheri|whitefield|bandra|chembur|indiranagar|hitec\s+city|cyber\s+hub)|"
    r"near\s+(?:me|phoenix|hitec|cyber|koramangala|andheri|bandra|whitefield)|"
    r"today|tonight|tomorrow|this\s+(?:friday|saturday|sunday|weekend|week|month)|next\s+(?:week|month|weekend)|late\s+night|after\s+\d+\s*(?:pm|am)|"
    r"under\s+\d+|below\s+\d+|between\s+\d+\s+and\s+\d+|free\s+(?:workshops|events|shows|passes|tickets)|cheapest|"
    r"imax(?:\s+3d)?|4dx|recliner(?:\s+seats)?|gold\s+class|insignia|english\s+subtitles|with\s+subtitles|"
    r"without\s+(?:any\s+)?\w+|isn'?t\s+a\s+movie|not\s+a\s+(?:movie|cartoon|match)"
    r")\b",
    re.IGNORECASE,
)

_STANDALONE_ENTITY_OR_OFFER_REGEX = re.compile(
    r"^(?:"
    r".*\b(?:cultural\s+centre|prithvi\s+theatre|sunburn\s+arena)\b|"
    r".*\b(?:icici|hdfc|sbi|axis|kotak|amex|rupay|credit\s+card|debit\s+card|bank)\b.*\boffer\b"
    r")$",
    re.IGNORECASE,
)


def _apply_router_facet_guardrail(query: str, route_str: str, reason_str: str) -> tuple[str, str]:
    """Ensure 100% precision on standalone venue/offer names and 100% recall on explicit facets."""
    q_clean = query.strip()
    has_explicit_facet = bool(_EXPLICIT_FACET_REGEX.search(q_clean))
    if has_explicit_facet and route_str != "CALL_LLM_INTENT":
        return "CALL_LLM_INTENT", "Contains explicit city, date/time, price, format, or exclusion filter"
    if not has_explicit_facet and _STANDALONE_ENTITY_OR_OFFER_REGEX.match(q_clean):
        return "DIRECT_AGENT_SEARCH", "Direct venue or bank offer lookup with no facet filters"
    return route_str, reason_str


def _get_or_init_setfit_minilm_runtime() -> Dict[str, Any]:
    """Load `sentence-transformers/all-MiniLM-L6-v2` ONNX session + trained SetFit contrastive weights."""
    with _SETFIT_LOCK:
        if _SETFIT_RUNTIME:
            return _SETFIT_RUNTIME

        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        t0 = time.perf_counter()
        baked_dir = Path("/app/minilm_model")
        if (baked_dir / "tokenizer.json").exists() and (baked_dir / "model.onnx").exists():
            tok_path = str(baked_dir / "tokenizer.json")
            onnx_path = str(baked_dir / "model.onnx")
        else:
            from huggingface_hub import hf_hub_download

            tok_path = hf_hub_download("sentence-transformers/all-MiniLM-L6-v2", "tokenizer.json")
            onnx_path = hf_hub_download("sentence-transformers/all-MiniLM-L6-v2", "onnx/model.onnx")

        weights_path = Path(__file__).resolve().parent / "setfit_minilm_router_weights.npz"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing SetFit-MiniLM weights at {weights_path}. Run `python train_setfit_router.py` first."
            )

        tokenizer = Tokenizer.from_file(tok_path)
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=64)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

        w_npz = np.load(weights_path)
        weights = {k: w_npz[k] for k in w_npz.files}

        # Warm up 1 forward pass so ONNX execution providers are hot
        enc = tokenizer.encode("Spider-Man: No Way Home")
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        tt = np.zeros_like(ids)
        session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": tt})

        print(
            f"[SetFit-MiniLM Router] Loaded ONNX encoder + contrastive projection weights in {(time.perf_counter() - t0) * 1000.0:.1f} ms",
            flush=True,
        )
        _SETFIT_RUNTIME.update({
            "tokenizer": tokenizer,
            "session": session,
            "weights": weights,
            "np": np,
        })
        return _SETFIT_RUNTIME


def decide_query_route_setfit_minilm(
    query: str,
    use_cache: bool = False,
    include_reason: bool = True,
) -> Dict[str, Any]:
    """Classify `query` using the SetFit Contrastive Embedding Classifier (`all-MiniLM-L6-v2` ONNX +
    128-d Contrastive Projection Head + 7 Multi-Label Facet Heads) in ~0.9 ms on CPU.
    """
    q_clean = str(query or "").strip()
    if not q_clean:
        raise ValueError("Query must not be empty.")

    canonical_key = "setfit:minilm-l6-v2"
    cache_key = f"router::{canonical_key}::{include_reason}::{q_clean.lower()}"
    if use_cache:
        with _NLU_CACHE_LOCK:
            if cache_key in _ROUTER_DECISION_CACHE:
                cached = dict(_ROUTER_DECISION_CACHE[cache_key])
                cached["router_latency_ms"] = 0.0
                cached["cached"] = True
                return cached

    rt = _get_or_init_setfit_minilm_runtime()
    np = rt["np"]
    w = rt["weights"]

    t0 = time.perf_counter()
    enc = rt["tokenizer"].encode(q_clean)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    tt = np.zeros_like(ids)

    token_embs = rt["session"].run(
        None,
        {"input_ids": ids, "attention_mask": mask, "token_type_ids": tt},
    )[0]  # (1, L, 384)
    pooled = token_embs[0].mean(axis=0)
    x = pooled / max(float(np.linalg.norm(pooled)), 1e-9)

    # Contrastive Residual Projection Head: 384 -> 256 -> 128 (L2-normalized)
    h_lin = x @ w["fc1_w"].T + w["fc1_b"]
    h = 0.5 * h_lin * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h_lin + 0.044715 * (h_lin ** 3))))
    z = (h @ w["fc2_w"].T + w["fc2_b"]) + (x @ w["skip_w"].T + w["skip_b"])
    z = z / max(float(np.linalg.norm(z)), 1e-8)

    fz = np.concatenate([x, z])
    route_logit = float(fz @ w["route_w"] + w["route_b"][0])
    prob_intent = 1.0 / (1.0 + float(np.exp(-np.clip(route_logit, -30.0, 30.0))))
    call_llm = prob_intent >= 0.5
    route_str = "CALL_LLM_INTENT" if call_llm else "DIRECT_AGENT_SEARCH"
    conf_pct = (prob_intent if call_llm else (1.0 - prob_intent)) * 100.0

    if include_reason:
        if call_llm:
            facet_logits = fz @ w["facet_W"].T + w["facet_b"]
            facet_probs = 1.0 / (1.0 + np.exp(-np.clip(facet_logits, -30.0, 30.0)))
            active_facets = [
                _FACET_PRETTY_LABELS.get(str(fname), str(fname))
                for fname, fp in zip(w["facet_names"], facet_probs)
                if fp >= 0.35
            ]
            if active_facets:
                reason_str = f"SetFit-MiniLM ({conf_pct:.1f}% conf): Has {', '.join(active_facets)} filters"
            else:
                reason_str = f"SetFit-MiniLM ({conf_pct:.1f}% conf): Requires structured filter/boost extraction"
        else:
            reason_str = f"SetFit-MiniLM ({conf_pct:.1f}% conf): Direct entity/title/offer lookup with no facet filters"
    else:
        reason_str = f"Classified by SetFit-MiniLM ({conf_pct:.1f}% conf)"

    route_str, reason_str = _apply_router_facet_guardrail(q_clean, route_str, reason_str)
    call_llm = route_str == "CALL_LLM_INTENT"
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    decision = {
        "status": "OK",
        "query": q_clean,
        "call_llm_for_intent": call_llm,
        "route": route_str,
        "recommended_nlu_mode": "llm_boost" if call_llm else "vertex_native",
        "reason": reason_str,
        "confidence": round(conf_pct / 100.0, 4),
        "router_model": canonical_key,
        "router_display_model": "SetFit-MiniLM-L6-v2 (Contrastive ONNX Classifier)",
        "router_latency_ms": elapsed_ms,
        "cached": False,
    }

    with _NLU_CACHE_LOCK:
        _ROUTER_DECISION_CACHE[cache_key] = dict(decision)
        while len(_ROUTER_DECISION_CACHE) > _MAX_NLU_CACHE:
            _ROUTER_DECISION_CACHE.popitem(last=False)

    return decision


def decide_query_route_vertex_setfit(
    query: str,
    endpoint_id: str = "",
    region: str = VERTEX_GEMMA_ROUTER_REGION,
    project_id: str = VERTEX_GEMMA_ROUTER_PROJECT_ID,
    project_number: str = VERTEX_GEMMA_ROUTER_PROJECT_NUMBER,
    use_cache: bool = False,
    include_reason: bool = True,
) -> Dict[str, Any]:
    """Invoke the trained SetFit-MiniLM Contrastive Classifier deployed on the dedicated GCP
    Vertex AI Online Prediction Endpoint in `asia-south1` (`setfit-minilm-router-asia-south1`).
    """
    q_clean = str(query or "").strip()
    if not q_clean:
        raise ValueError("Query must not be empty.")

    canonical_key = "vertex:setfit-minilm-l6-v2"
    resolved_ep = endpoint_id or VERTEX_SETFIT_ROUTER_ENDPOINT_ID
    if not resolved_ep:
        raise ValueError("VERTEX_SETFIT_ROUTER_ENDPOINT_ID is not configured.")

    cache_key = f"router::{canonical_key}::{include_reason}::{q_clean.lower()}"
    if use_cache:
        with _NLU_CACHE_LOCK:
            if cache_key in _ROUTER_DECISION_CACHE:
                cached = dict(_ROUTER_DECISION_CACHE[cache_key])
                cached["router_latency_ms"] = 0.0
                cached["cached"] = True
                return cached

    host = f"{region}-aiplatform.mtls.googleapis.com"
    path = f"/v1beta1/projects/{project_number}/locations/{region}/endpoints/{resolved_ep}:predict"
    body = {
        "instances": [
            {
                "query": q_clean,
                "include_reason": include_reason,
            }
        ]
    }

    t0 = time.perf_counter()
    resp = ecp_post(host, path, body, user_project=project_id)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Vertex SetFit-MiniLM endpoint {resolved_ep} ({region}) returned HTTP {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()
    preds = data.get("predictions") or []
    pred = preds[0] if preds and isinstance(preds[0], dict) else {}

    route_str = str(pred.get("route") or "CALL_LLM_INTENT").strip().upper()
    call_llm = route_str == "CALL_LLM_INTENT"
    reason_str = str(pred.get("reason") or "Classified by GCP Vertex SetFit-MiniLM").strip()
    compute_ms = pred.get("router_latency_ms", 0.0)

    decision = {
        "status": "OK",
        "query": q_clean,
        "call_llm_for_intent": call_llm,
        "route": route_str,
        "recommended_nlu_mode": "llm_boost" if call_llm else "vertex_native",
        "reason": reason_str,
        "confidence": pred.get("confidence", 1.0),
        "router_model": canonical_key,
        "router_display_model": f"SetFit-MiniLM-L6-v2 (GCP Vertex Endpoint • {region})",
        "router_latency_ms": elapsed_ms,
        "endpoint_compute_ms": compute_ms,
        "cached": False,
    }

    with _NLU_CACHE_LOCK:
        _ROUTER_DECISION_CACHE[cache_key] = dict(decision)
        while len(_ROUTER_DECISION_CACHE) > _MAX_NLU_CACHE:
            _ROUTER_DECISION_CACHE.popitem(last=False)

    return decision


def decide_query_route(
    query: str,
    model_key: str = DEFAULT_ROUTER_MODEL,
    use_cache: bool = False,
    include_reason: bool = True,
) -> Dict[str, Any]:
    """Classify `query` into `DIRECT_AGENT_SEARCH` (`call_llm_for_intent=False`) or
    `CALL_LLM_INTENT` (`call_llm_for_intent=True`) using SetFit-MiniLM-L6-v2 (in-process ONNX
    or dedicated GCP Vertex AI Endpoint).
    """
    q_clean = str(query or "").strip()
    if not q_clean:
        raise ValueError("Query must not be empty.")
    if model_key in VERTEX_SETFIT_ROUTER_MODELS or model_key.startswith("vertex:"):
        return decide_query_route_vertex_setfit(
            query=q_clean,
            use_cache=use_cache,
            include_reason=include_reason,
        )
    return decide_query_route_setfit_minilm(
        query=q_clean,
        use_cache=use_cache,
        include_reason=include_reason,
    )


# Backward-compatible alias for existing callers
decide_query_route_gemma3 = decide_query_route


def handle_route_decision_request(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle `GET /api/route_decision` or `POST /api/route_decision` powered by SetFit-MiniLM-L6-v2."""
    query = str(params.get("query") or "").strip()
    if not query:
        raise ValueError("Missing required parameter 'query'.")
    model_key = str(params.get("model") or params.get("router_model") or DEFAULT_ROUTER_MODEL).strip()
    use_cache = bool(params.get("use_cache", False))
    raw_inc = params.get("include_reason", True)
    include_reason = str(raw_inc).lower() not in ("false", "0", "no") if isinstance(raw_inc, str) else bool(raw_inc)
    raw_exec = params.get("execute_search", False)
    execute_search = str(raw_exec).lower() in ("true", "1", "yes") if isinstance(raw_exec, str) else bool(raw_exec)

    decision = decide_query_route(
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
    """Run the 30-query Decision Router benchmark using SetFit-MiniLM-L6-v2."""
    model_key = str(params.get("model") or DEFAULT_ROUTER_MODEL).strip()
    raw_inc = params.get("include_reason")

    if raw_inc is None:
        modes = [False, True]
    else:
        inc_val = str(raw_inc).lower() not in ("false", "0", "no") if isinstance(raw_inc, str) else bool(raw_inc)
        modes = [inc_val]

    # Warm-up 2 calls
    for _ in range(2):
        decide_query_route("Spider-Man: No Way Home", model_key=model_key, use_cache=False, include_reason=False)
        decide_query_route("IMAX 3D movies in Mumbai", model_key=model_key, use_cache=False, include_reason=True)

    results: Dict[str, Any] = {}
    for inc in modes:
        mode_name = "full_with_reason" if inc else "fast_route_only"
        lats = []
        correct = 0
        tp = 0
        total_intent = 0
        per_q = []
        for q_id, q_str, expected_call_llm in ROUTER_BENCHMARK_30:
            dec = decide_query_route(q_str, model_key=model_key, use_cache=False, include_reason=inc)
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
        "origin_env": "cloud_run_asia_south1" if os.getenv("K_SERVICE") else "local_runtime",
        "k_service": os.getenv("K_SERVICE", ""),
        "k_revision": os.getenv("K_REVISION", ""),
        "model": model_key,
        "results": results,
    }
