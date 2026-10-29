"""Search App Registry, ECP / mTLS HTTP Proxy Client, and Runtime State Locks."""

from __future__ import annotations

from collections import OrderedDict
import datetime
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import (
    BQ_TABLE_V2_DISPLAY,
    BQ_TABLE_V3_DISPLAY,
    DATASTORE_ID_V2_DISPLAY,
    DATASTORE_ID_V3_DISPLAY,
    ECP_HTTP_PROXY_BIN,
    ENGINE_ID_V2_DISPLAY,
    ENGINE_ID_V2_RUNTIME,
    ENGINE_ID_V3_DISPLAY,
    ENGINE_ID_V3_RUNTIME,
    GCLOUD_ACCOUNT,
    GEMINI_PROJECT_ID,
    CROWDING_FIELD,
    LEGACY_V2_LANGUAGES_FIELD,
    LEGACY_V2_REGIONS_FIELD,
    LEGACY_V2_SHOW_DATES_FIELD,
    LEGACY_V2_SOLD_OUT_FIELD,
    SEARCH_5STORES_DISPLAY_PROJECT_ID,
    SEARCH_5STORES_PROJECT_ID as PROJECT_ID,
    SEARCH_5STORES_PROJECT_NUMBER as PROJECT_NUMBER,
)

try:
    from eval_5stores_nlp_collapse_html import (
        ALL_QUERIES,
        NLP_ALLOWED_FIELDS,
        NLP_ALLOWED_FIELDS_V3,
        SEARCH_PATH,
        evaluate_hit,
        extract_doc_summary,
        format_nlp_filters,
    )
    from setup_5stores_bq_and_vertex import EcpClient
except ImportError:
    import subprocess

    SEARCH_PATH = (
        f"/v1alpha/projects/{PROJECT_NUMBER}/locations/global/collections/default_collection"
        f"/engines/{ENGINE_ID_V2_RUNTIME}/servingConfigs/default_search:search"
    )
    NLP_ALLOWED_FIELDS = [
        "record_type", "regionName", "venue_name", "address", "location",
        LEGACY_V2_LANGUAGES_FIELD, "genres", "formats", "categories", "artists",
        "sessions.format", "sessions.languages", "sessions.min_price",
        "sessions.max_price", "sessions.showtime_start", "sessions.time_slot",
    ]
    NLP_ALLOWED_FIELDS_V3 = [
        "record_type", "categories", "genres", "languages", "formats",
        "regions", "location", "artists", "show_dates", "release_date",
        "time_slots", "min_price", "is_sold_out",
    ]
    ALL_QUERIES = [
        {"id": 1, "category": "Movie + Format + City", "query": "hindi horror movie in imax this friday in pune", "what_query": "hindi horror movie"},
        {"id": 2, "category": "Event + Price Range + City", "query": "concerts between 1000 and 3000 in delhi", "what_query": "concerts"},
        {"id": 3, "category": "Direct Title Lookup", "query": "Spider-Man: No Way Home", "what_query": "Spider-Man"},
    ]

    class EcpClient:  # type: ignore[no-redef]
        """Fallback HTTP client managing a local `ecp_http_proxy` subprocess and OAuth token refresh."""

        def __init__(self, port: int = 18895) -> None:
            """Start local `ecp_http_proxy` on `port` and fetch initial OAuth access token."""
            self.port = port
            self._last_token_time = 0.0
            self.token = ""
            self.proc: Optional[subprocess.Popen[bytes]] = None
            self._refresh_token()
            self._start_proxy()

        def _start_proxy(self) -> None:
            """Spawn `ecp_http_proxy` and poll `/readyz` until the local mTLS tunnel is ready."""
            self.close()
            self.proc = subprocess.Popen(
                [ECP_HTTP_PROXY_BIN, "-enterprise_certificate_file_path", "/etc/certificate_config.json", "-nonce_token", "ent-setup-nonce", "-port", str(self.port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(30):
                try:
                    if requests.get(f"http://127.0.0.1:{self.port}/readyz", timeout=2).status_code == 200:
                        return
                except Exception:
                    time.sleep(0.1)

        def _refresh_token(self, force: bool = False) -> None:
            """Refresh the cached `gcloud` OAuth bearer token every 4 minutes (or when forced on 401)."""
            now = time.time()
            if force or not self.token or (now - self._last_token_time) > 240:
                cmd = ["gcloud", "auth", "print-access-token"]
                if GCLOUD_ACCOUNT:
                    cmd.append(f"--account={GCLOUD_ACCOUNT}")
                self.token = subprocess.check_output(cmd, text=True).strip()
                self._last_token_time = now

        def close(self) -> None:
            """Terminate the background `ecp_http_proxy` subprocess cleanly."""
            if self.proc:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=3)
                except Exception:
                    pass
                self.proc = None

    def extract_doc_summary(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw Discovery Engine `document` payload into a flat UI-ready hit dictionary."""
        sd = doc.get("structData", {}) or {}
        return {
            "id": doc.get("id", ""),
            "entity_code": sd.get(CROWDING_FIELD) or sd.get("entity_code") or "",
            "session_id": str(sd.get("session_id") or ""),
            "time_slots": [str(x) for x in (sd.get("time_slots") or []) if x],
            "record_type": sd.get("record_type") or "unknown",
            "title": sd.get("title") or "Untitled",
            "regions": [str(x) for x in (sd.get("regions") or []) if x],
            "venues": [str(sd["venue_name"])] if sd.get("venue_name") else [],
            "address": str(sd.get("address") or ""),
            "tags": [str(x) for x in (sd.get("tags") or [])[:30] if x],
            "languages": [str(x) for x in (sd.get("languages") or []) if x],
            "genres": [str(x) for x in (sd.get("genres") or []) if x],
            "formats": [str(x) for x in (sd.get("formats") or []) if x],
            "categories": [str(x) for x in (sd.get("categories") or []) if x],
            "artists": [str(x) for x in (sd.get("artists") or []) if x],
            "min_price": float(sd["min_price"]) if sd.get("min_price") is not None else None,
            "max_price": float(sd["max_price"]) if sd.get("max_price") is not None else None,
            "date_display": "—",
            "description": str(sd.get("description") or "")[:300],
        }

    def format_nlp_filters(nlp_info: Dict[str, Any]) -> str:
        """Extract the human-readable filter string from `naturalLanguageQueryUnderstandingInfo`."""
        return (nlp_info.get("extractedFilters") or "").strip()

    def evaluate_hit(query_item: Dict[str, Any], hit: Dict[str, Any], extracted_filters: str = "") -> Tuple[bool, str]:
        """Evaluate whether a returned search hit satisfies the query constraints."""
        return True, f"Matched {hit.get('record_type', 'record')}"

import copy

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

PORT = int(os.getenv("PORT", "8080"))
GEMINI_PROJECT = os.getenv("GEMINI_PROJECT", GEMINI_PROJECT_ID)
GEMINI_REGION = "asia-south1"
GEMINI_HOST = "asia-south1-aiplatform.mtls.googleapis.com"
GEMINI_MODEL_REGIONS: Dict[str, str] = {
    "gemini-3.5-flash-lite": "asia-south1",
    "gemini-4-argon": "global",
    "gemini-2.5-flash-lite": "global",
}
SEARCH_MTLS_HOST = "discoveryengine.mtls.googleapis.com"
SEARCH_PUBLIC_HOST = "discoveryengine.googleapis.com"

SEARCH_APPS: Dict[str, Dict[str, Any]] = {
    "v3": {
        "key": "v3",
        "engine_id": ENGINE_ID_V3_DISPLAY,
        "datastore_id": DATASTORE_ID_V3_DISPLAY,
        "bq_table": BQ_TABLE_V3_DISPLAY,
        "short_label": "v3 Master (Session-Grain • 390,870 docs)",
        "header_pill": f"🚀 App: {ENGINE_ID_V3_DISPLAY} (390,870 docs • Session-Grain)",
        "search_path": (
            f"/v1alpha/projects/{PROJECT_NUMBER}/locations/global/collections/default_collection"
            f"/engines/{ENGINE_ID_V3_RUNTIME}/servingConfigs/default_search:search"
        ),
        "nlp_allowed_fields": NLP_ALLOWED_FIELDS_V3,
    },
    "v2": {
        "key": "v2",
        "engine_id": ENGINE_ID_V2_DISPLAY,
        "datastore_id": DATASTORE_ID_V2_DISPLAY,
        "bq_table": BQ_TABLE_V2_DISPLAY,
        "short_label": "v2 Unified (Venue-Grain • 262,086 docs)",
        "header_pill": f"📦 App: {ENGINE_ID_V2_DISPLAY} (262,086 docs • Venue-Grain)",
        "search_path": SEARCH_PATH,
        "nlp_allowed_fields": NLP_ALLOWED_FIELDS,
    },
}
DEFAULT_SEARCH_APP = "v3"


def get_search_app_config(app_key: Optional[str]) -> Dict[str, Any]:
    """Resolve `app_key` (`v3`, `v2`, or engine ID) to its `SEARCH_APPS` configuration dictionary."""
    key = str(app_key or DEFAULT_SEARCH_APP).strip().lower()
    if key in (ENGINE_ID_V2_RUNTIME.lower(), ENGINE_ID_V2_DISPLAY.lower(), "v2"):
        return SEARCH_APPS["v2"]
    if key in (ENGINE_ID_V3_RUNTIME.lower(), ENGINE_ID_V3_DISPLAY.lower(), "v3"):
        return SEARCH_APPS["v3"]
    return SEARCH_APPS[DEFAULT_SEARCH_APP]


LOCAL_MODELS: Dict[str, str] = {
    "local:gemma-3-1b-it-qat-4bit": "mlx-community/gemma-3-1b-it-qat-4bit",
    "local:gemma-3-270m-it-qat-4bit": "mlx-community/gemma-3-270m-it-qat-4bit",
    "local:gemma-4-e2b-it-4bit": "mlx-community/gemma-4-e2b-it-4bit",
    "local:gemma-3-4b-it-qat-4bit": "mlx-community/gemma-3-4b-it-qat-4bit",
    "local:Qwen2.5-1.5B-Instruct-4bit": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "local:Qwen2.5-3B-Instruct-4bit": "mlx-community/Qwen2.5-3B-Instruct-4bit",
}
STARTUP_PREWARM_LOCAL_MODELS = (
    "local:gemma-3-1b-it-qat-4bit",
    "local:gemma-3-270m-it-qat-4bit",
    "local:Qwen2.5-1.5B-Instruct-4bit",
    "local:Qwen2.5-3B-Instruct-4bit",
)
CLOUD_GEMINI_MODELS = (
    "gemini-4-argon",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
)
DEFAULT_BOOST_MODEL = (
    "local:gemma-3-1b-it-qat-4bit" if HAS_MLX else "gemini-4-argon"
)

_CLIENT_LOCK = threading.Lock()
_ECP_CLIENT: Optional[EcpClient] = None
_THREAD_LOCAL = threading.local()

# Metal GPU lock + pre-warmed KV-cached model states
_MLX_GPU_LOCK = threading.Lock()
_MLX_RUNTIMES: Dict[str, Dict[str, Any]] = {}
_MLX_ROUTER_RUNTIMES: Dict[str, Dict[str, Any]] = {}

# Optional LRU cache for LLM boost extraction & router decisions
_NLU_CACHE_LOCK = threading.Lock()
_LLM_BOOST_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_ROUTER_DECISION_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_MAX_NLU_CACHE = 512
USE_ADC = os.getenv("USE_ADC", "").lower() == "true" or bool(os.getenv("K_SERVICE"))
_ADC_TOKEN = ""
_ADC_TOKEN_EXPIRY = 0.0


def _get_adc_token(force: bool = False) -> str:
    """Fetch and cache the GCP Service Account OAuth token from the Cloud Run / GCE Metadata Server."""
    global _ADC_TOKEN, _ADC_TOKEN_EXPIRY
    now = time.time()
    with _CLIENT_LOCK:
        if not force and _ADC_TOKEN and now < _ADC_TOKEN_EXPIRY:
            return _ADC_TOKEN
        try:
            r = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"},
                timeout=5,
            )
            if r.status_code == 200:
                data = r.json()
                _ADC_TOKEN = str(data.get("access_token") or "")
                _ADC_TOKEN_EXPIRY = now + max(60.0, float(data.get("expires_in", 300)) - 60.0)
                return _ADC_TOKEN
        except Exception:
            pass
        try:
            import google.auth
            import google.auth.transport.requests

            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(google.auth.transport.requests.Request())
            _ADC_TOKEN = str(creds.token or "")
            _ADC_TOKEN_EXPIRY = now + 240.0
            return _ADC_TOKEN
        except Exception as exc:
            raise RuntimeError(f"Failed to obtain GCP ADC token in Cloud Run: {exc}") from exc


def get_ecp_client() -> Optional[EcpClient]:
    """Return the singleton `EcpClient` instance protected by `_CLIENT_LOCK` (or `None` on Cloud Run / ADC)."""
    if USE_ADC:
        return None
    global _ECP_CLIENT
    with _CLIENT_LOCK:
        if _ECP_CLIENT is None:
            _ECP_CLIENT = EcpClient(port=18991)
        return _ECP_CLIENT


def get_thread_host_session(target_host: str, force_new: bool = False) -> requests.Session:
    """Dedicated keep-alive `requests.Session` per (thread, target_host)."""
    sessions: Optional[Dict[str, requests.Session]] = getattr(_THREAD_LOCAL, "sessions", None)
    if sessions is None:
        sessions = {}
        _THREAD_LOCAL.sessions = sessions
    if force_new and target_host in sessions:
        try:
            sessions[target_host].close()
        except Exception:
            pass
        sessions.pop(target_host, None)
    sess = sessions.get(target_host)
    if sess is None:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8)
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
        sessions[target_host] = sess
    return sess


def ecp_post(host: str, path: str, body: Dict[str, Any], user_project: str = PROJECT_ID) -> requests.Response:
    """Thread-safe keep-alive HTTP POST (direct intra-GCP HTTPS on Cloud Run, or local `ecp_http_proxy` mTLS)."""
    if USE_ADC:
        public_host = host.replace(".mtls.googleapis.com", ".googleapis.com")
        token = _get_adc_token()
        session = get_thread_host_session(public_host)
        url = f"https://{public_host}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": user_project,
            "Content-Type": "application/json",
        }
        resp = session.post(url, headers=headers, json=body, timeout=65)
        if resp.status_code == 401:
            token = _get_adc_token(force=True)
            headers["Authorization"] = f"Bearer {token}"
            resp = session.post(url, headers=headers, json=body, timeout=65)
        return resp

    client = get_ecp_client()
    assert client is not None
    with _CLIENT_LOCK:
        client._refresh_token()
        token = client.token
        port = client.port

    session = get_thread_host_session(host)
    url = f"http://127.0.0.1:{port}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "x-goog-user-project": user_project,
        "x-goog-ecpproxy-target-host": f"{host}:443",
        "Content-Type": "application/json",
    }
    try:
        resp = session.post(url, headers=headers, json=body, timeout=65)
    except requests.exceptions.RequestException:
        session = get_thread_host_session(host, force_new=True)
        with _CLIENT_LOCK:
            client._start_proxy()
            client._refresh_token(force=True)
            headers["Authorization"] = f"Bearer {client.token}"
            url = f"http://127.0.0.1:{client.port}{path}"
        resp = session.post(url, headers=headers, json=body, timeout=65)

    for retry_idx in range(6):
        if resp.status_code == 429:
            time.sleep(min(20.0, 1.5 * (2 ** retry_idx)))
            resp = session.post(url, headers=headers, json=body, timeout=65)
            continue
        if resp.status_code not in (502, 503, 504) or retry_idx >= 4:
            break
        session = get_thread_host_session(host, force_new=True)
        if retry_idx >= 2:
            with _CLIENT_LOCK:
                client._start_proxy()
                client._refresh_token(force=True)
                headers["Authorization"] = f"Bearer {client.token}"
                url = f"http://127.0.0.1:{client.port}{path}"
        retry_headers = dict(headers)
        retry_headers["Connection"] = "close"
        resp = session.post(url, headers=retry_headers, json=body, timeout=65)

    if resp.status_code == 401:
        with _CLIENT_LOCK:
            client._refresh_token(force=True)
            headers["Authorization"] = f"Bearer {client.token}"
        resp = session.post(url, headers=headers, json=body, timeout=65)
    return resp



def _date_context() -> Dict[str, str]:
    """Compute relative ISO-8601 date anchors (today, tomorrow, upcoming Friday/weekend, next week/month)."""
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    days_until_friday = (4 - today.weekday()) % 7
    upcoming_friday = today + datetime.timedelta(days=days_until_friday)
    days_until_saturday = (5 - today.weekday()) % 7
    weekend_sat = today + datetime.timedelta(days=days_until_saturday)
    if today.weekday() == 6:
        weekend_sat = today
    friday_sun = upcoming_friday + datetime.timedelta(days=2)
    week_end = today + datetime.timedelta(days=6)
    next_week_start = today + datetime.timedelta(days=7)
    next_week_end = today + datetime.timedelta(days=13)
    month_end = today + datetime.timedelta(days=30)
    first_next_month = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    last_next_month = (first_next_month.replace(day=28) + datetime.timedelta(days=4)).replace(day=1) - datetime.timedelta(days=1)
    return {
        "today_iso": today.isoformat(),
        "weekday_name": today.strftime("%A"),
        "tomorrow_iso": tomorrow.isoformat(),
        "upcoming_friday_iso": upcoming_friday.isoformat(),
        "weekend_sat_iso": weekend_sat.isoformat(),
        "weekend_sun_iso": friday_sun.isoformat(),
        "week_end_iso": week_end.isoformat(),
        "next_week_start_iso": next_week_start.isoformat(),
        "next_week_end_iso": next_week_end.isoformat(),
        "month_end_iso": month_end.isoformat(),
        "next_month_start_iso": first_next_month.isoformat(),
        "next_month_end_iso": last_next_month.isoformat(),
    }

