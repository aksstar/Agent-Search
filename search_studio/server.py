"""HTTP API & UI Server for Entertainment Search Studio (`http://127.0.0.1:8080`)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from search_studio.client import (
    ALL_QUERIES,
    DEFAULT_BOOST_MODEL,
    HAS_MLX,
    PORT,
    STARTUP_PREWARM_LOCAL_MODELS,
    _MLX_GPU_LOCK,
    get_ecp_client,
)
from search_studio.intent_extractor import get_or_init_local_mlx_runtime
from search_studio.router_api import (
    get_or_init_local_mlx_router_runtime,
    handle_route_decision_request,
)
from search_studio.search_service import (
    explain_live_hit,
    run_compare_v2_vs_v3,
    run_live_search,
    run_side_by_side_nlqu_vs_llm,
)

UI_HTML_PATH = Path(__file__).resolve().parent / "ui.html"


def load_ui_html() -> str:
    """Load the frontend Search Studio UI HTML from `search_studio/ui.html`."""
    return UI_HTML_PATH.read_text(encoding="utf-8")


class RealtimeTestHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving the Search Studio UI (`GET /`) and JSON REST APIs (`/api/*`)."""

    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        """Serialize `data` as UTF-8 JSON and write the HTTP response with CORS headers."""
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html_str: str) -> None:
        """Write an HTML response (`200 OK`) to the client."""
        raw = html_str.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight `OPTIONS` requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle `GET /` (UI), `GET /api/presets`, and `GET /api/route_decision?query=...`."""
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_html(load_ui_html())
            return
        if parsed.path == "/api/presets":
            self._send_json(200, {"queries": ALL_QUERIES})
            return
        if parsed.path == "/api/route_decision":
            try:
                qs = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}
                res = handle_route_decision_request(qs)
                self._send_json(200, res)
            except Exception as exc:
                self._send_json(400, {"status": "ERROR", "error": str(exc)})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        """Handle `POST /api/route_decision`, `/api/search`, `/api/side_by_side`, `/api/compare_apps`, and `/api/explain`."""
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        try:
            if parsed.path == "/api/route_decision":
                res = handle_route_decision_request(body)
                self._send_json(200, res)
                return
            if parsed.path == "/api/search":
                res = run_live_search(body)
                self._send_json(200, res)
                return
            if parsed.path == "/api/side_by_side":
                res = run_side_by_side_nlqu_vs_llm(body)
                self._send_json(200, res)
                return
            if parsed.path == "/api/compare_apps":
                res = run_compare_v2_vs_v3(body)
                self._send_json(200, res)
                return
            if parsed.path == "/api/explain":
                res = explain_live_hit(body)
                self._send_json(200, res)
                return
            self._send_json(404, {"error": "Not found"})
        except Exception as exc:
            self._send_json(500, {"status": "ERROR", "error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        """Suppress default per-request stderr access logs to keep console output clean."""
        return


def main() -> None:
    """Pre-warm the ECP proxy and local Metal GPU KV caches, then start `ThreadingHTTPServer` on `PORT`."""
    get_ecp_client()
    if HAS_MLX:
        with _MLX_GPU_LOCK:
            for local_key in STARTUP_PREWARM_LOCAL_MODELS:
                try:
                    get_or_init_local_mlx_runtime(local_key)
                except Exception as exc:
                    print(f"[MLX] Skipping startup pre-warm for {local_key}: {exc}", flush=True)
            try:
                get_or_init_local_mlx_router_runtime("local:gemma-3-1b-it-qat-4bit")
            except Exception as exc:
                print(f"[MLX Router] Skipping startup pre-warm for Gemma 3 1B router: {exc}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), RealtimeTestHandler)
    print(
        f"Managed NLQU vs Local Metal LLM ({DEFAULT_BOOST_MODEL}) + Gemma 3 Decision API running at http://127.0.0.1:{PORT}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
