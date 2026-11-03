"""Deploy `google/gemma-3-270m-it` from Vertex AI Model Garden to `asia-south1` (`g2-standard-4`, `1x NVIDIA_L4`)
using the official Model Garden vLLM container with `--enable-prefix-caching`, and benchmark the Decision Router API.

Usage:
  ./.venv/bin/python deploy_model_garden_gemma3_router.py --status
  ./.venv/bin/python deploy_model_garden_gemma3_router.py --wait-and-benchmark
  ./.venv/bin/python deploy_model_garden_gemma3_router.py --deploy
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Dict, List, Optional

import requests

from config import (
    VERTEX_GEMMA_ROUTER_ENDPOINT_ID,
    VERTEX_GEMMA_ROUTER_MODEL_ID,
    VERTEX_GEMMA_ROUTER_PROJECT_ID,
    VERTEX_GEMMA_ROUTER_PROJECT_NUMBER,
    VERTEX_GEMMA_ROUTER_REGION,
)
from search_studio.client import ALL_QUERIES, get_ecp_client
from search_studio.router_api import decide_query_route_vertex_vllm

MODEL_GARDEN_PUBLISHER_MODEL = "publishers/google/models/gemma3@gemma-3-270m-it"
MODEL_GARDEN_VLLM_IMAGE = (
    "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250430_0916_RC00_maas"
)
MODEL_GARDEN_WEIGHTS_URI = "gs://vertex-model-garden-restricted-us/gemma3/gemma-3-270m-it"


def _ecp_request(
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    region: str = VERTEX_GEMMA_ROUTER_REGION,
    project_id: str = VERTEX_GEMMA_ROUTER_PROJECT_ID,
) -> requests.Response:
    """Execute an mTLS / ECP request against `{region}-aiplatform.mtls.googleapis.com`."""
    client = get_ecp_client()
    client._refresh_token()
    host = f"{region}-aiplatform.mtls.googleapis.com"
    url = f"http://127.0.0.1:{client.port}{path}"
    headers = {
        "Authorization": f"Bearer {client.token}",
        "x-goog-user-project": project_id,
        "x-goog-ecpproxy-target-host": f"{host}:443",
        "Content-Type": "application/json",
    }
    return requests.request(method, url, headers=headers, json=body, timeout=65)


def check_endpoint_status(
    endpoint_id: str = VERTEX_GEMMA_ROUTER_ENDPOINT_ID,
    project_number: str = VERTEX_GEMMA_ROUTER_PROJECT_NUMBER,
    region: str = VERTEX_GEMMA_ROUTER_REGION,
) -> Dict[str, Any]:
    """Fetch the Vertex AI Endpoint resource and check whether `deployedModels` is populated."""
    path = f"/v1beta1/projects/{project_number}/locations/{region}/endpoints/{endpoint_id}"
    resp = _ecp_request("GET", path)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {path} returned {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def wait_for_endpoint_deployed(
    endpoint_id: str = VERTEX_GEMMA_ROUTER_ENDPOINT_ID,
    project_number: str = VERTEX_GEMMA_ROUTER_PROJECT_NUMBER,
    region: str = VERTEX_GEMMA_ROUTER_REGION,
    poll_seconds: int = 20,
    max_wait_seconds: int = 1800,
) -> Dict[str, Any]:
    """Poll the Vertex AI Endpoint until `deployedModels` has an active model deployment."""
    t_start = time.time()
    while True:
        ep = check_endpoint_status(endpoint_id=endpoint_id, project_number=project_number, region=region)
        deployed = ep.get("deployedModels") or []
        elapsed = int(time.time() - t_start)
        if deployed:
            print(
                f"[{elapsed}s] Endpoint {endpoint_id} is DEPLOYED! DeployedModel ID: {deployed[0].get('id')}",
                flush=True,
            )
            return ep
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"Timed out after {elapsed}s waiting for endpoint {endpoint_id} to deploy.")
        print(f"[{elapsed}s] Waiting for g2-standard-4 (1x NVIDIA_L4) vLLM container to finish deploying...", flush=True)
        time.sleep(poll_seconds)


def run_benchmark_30_queries() -> Dict[str, Any]:
    """Run the 30-query Decision Router benchmark against the deployed Vertex AI vLLM endpoint."""
    # Warm up prefix cache + HTTP connection pool with 2 warmup calls
    print("Warming up vLLM `--enable-prefix-caching` KV blocks and HTTP keep-alive session...", flush=True)
    for _ in range(2):
        decide_query_route_vertex_vllm("Spider-Man: No Way Home", use_cache=False, include_reason=False)
        decide_query_route_vertex_vllm("IMAX 3D movies in Mumbai", use_cache=False, include_reason=True)

    results: Dict[str, Any] = {}
    for include_reason in (False, True):
        mode_key = "full_with_reason" if include_reason else "fast_route_only"
        latencies: List[float] = []
        correct = 0
        intent_tp = 0
        intent_total = 0
        per_query: List[Dict[str, Any]] = []

        for q_item in ALL_QUERIES:
            q_str = q_item["query"]
            # Ground truth: Q3 ("Spider-Man: No Way Home") and Q15 ("ICICI bank credit card offer") are DIRECT_AGENT_SEARCH
            expected_call_llm = q_item["id"] not in (3, 15)
            dec = decide_query_route_vertex_vllm(q_str, use_cache=False, include_reason=include_reason)
            lat = float(dec["router_latency_ms"])
            latencies.append(lat)
            actual_call_llm = bool(dec["call_llm_for_intent"])
            is_ok = actual_call_llm == expected_call_llm
            if is_ok:
                correct += 1
            if expected_call_llm:
                intent_total += 1
                if actual_call_llm:
                    intent_tp += 1
            per_query.append(
                {
                    "id": q_item["id"],
                    "query": q_str,
                    "route": dec["route"],
                    "reason": dec.get("reason", ""),
                    "latency_ms": lat,
                    "correct": is_ok,
                }
            )

        s_lat = sorted(latencies)
        n = len(s_lat)
        p50 = s_lat[n // 2]
        p90 = s_lat[min(n - 1, int(n * 0.90))]
        p95 = s_lat[min(n - 1, int(n * 0.95))]
        summary = {
            "mode": mode_key,
            "include_reason": include_reason,
            "count": n,
            "min_ms": round(min(s_lat), 1),
            "mean_ms": round(statistics.mean(s_lat), 1),
            "p50_ms": round(p50, 1),
            "p90_ms": round(p90, 1),
            "p95_ms": round(p95, 1),
            "max_ms": round(max(s_lat), 1),
            "accuracy": f"{correct}/{n} ({round(100.0 * correct / n, 1)}%)",
            "call_llm_intent_recall": f"{intent_tp}/{intent_total} ({round(100.0 * intent_tp / max(1, intent_total), 1)}%)",
            "queries": per_query,
        }
        results[mode_key] = summary
        print(f"\n=== Benchmark: {mode_key} (include_reason={include_reason}) ===", flush=True)
        print(json.dumps({k: v for k, v in summary.items() if k != "queries"}, indent=2), flush=True)

    return results


def main() -> None:
    """CLI entrypoint for inspecting, waiting on, or benchmarking the Vertex AI Gemma 3 270M vLLM router."""
    parser = argparse.ArgumentParser(description="Vertex AI Model Garden Gemma 3 270M vLLM Router Manager")
    parser.add_argument("--status", action="store_true", help="Check current endpoint deployment status")
    parser.add_argument("--wait-and-benchmark", action="store_true", help="Wait until deployed, then run 30-query benchmark")
    args = parser.parse_args()

    if args.status:
        ep = check_endpoint_status()
        print(json.dumps(ep, indent=2))
        return

    if args.wait_and_benchmark:
        wait_for_endpoint_deployed()
        run_benchmark_30_queries()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
