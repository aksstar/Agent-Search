#!/usr/bin/env python3
"""Real-Time Testing Web App for 5-Store Entertainment Agent Search (`ent-bq-5stores-master-v3-app` & `ent-bq-5stores-v2-app`).

Compares **Managed Vertex AI Search NLQU** vs **LLM-Based Boost Condition Extraction** (running **locally on Apple M4 Pro Metal GPU via `mlx-lm`** in ~95–350 ms OR via Vertex Cloud `gemini-4-argon` / `gemini-3.5-flash-lite`) — plus a **Local Gemma 3 Decision Router (`/api/route_decision`)**.

Features:
- **Local Open-Source Metal GPU Inference (`mlx-lm`)**:
  - `mlx-community/gemma-3-1b-it-qat-4bit` (Default — ~97 ms with pre-warmed Metal KV prompt cache)
  - `mlx-community/gemma-3-270m-it-qat-4bit` (Ultra-light — ~55 ms with pre-warmed Metal KV prompt cache)
  - `mlx-community/Qwen2.5-3B-Instruct-4bit` (~180 ms with pre-warmed Metal KV prompt cache)
  - `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (~115 ms with pre-warmed Metal KV prompt cache)
- **Cloud Vertex AI Models (1-click switcher in UI)**:
  - `gemini-4-argon` (`thinkingLevel: "LOW"`)
  - `gemini-3.5-flash-lite` / `gemini-2.5-flash-lite` (`thinkingBudget: 0`)
- **Tab 1 (`⚖️ Side-by-Side: Managed NLQU vs Selected Boost LLM`)**:
  Runs Managed Vertex NLQU and the selected Boost LLM concurrently, showing latency breakdown (`LLM ms + Search ms`),
  extracted filters vs `conditionBoostSpecs` pills, `curl` & JSON drawers, and rank-movement badges.
- **Tab 2 (`🔍 Single-Engine Inspector`)**:
  Test any single engine (`Auto-Route`, `LLM Boost`, `Managed Vertex NLQU`, or `Direct Agent Search`) across all 135 benchmark presets.

Run (with `.venv` containing `mlx-lm`):
    ./.venv/bin/python realtime_test_webapp.py
Then open:
    http://127.0.0.1:8080
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

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
        def __init__(self, port: int = 18895) -> None:
            self.port = port
            self._last_token_time = 0.0
            self.token = ""
            self.proc: Optional[subprocess.Popen[bytes]] = None
            self._refresh_token()
            self._start_proxy()

        def _start_proxy(self) -> None:
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
            now = time.time()
            if force or not self.token or (now - self._last_token_time) > 240:
                cmd = ["gcloud", "auth", "print-access-token"]
                if GCLOUD_ACCOUNT:
                    cmd.append(f"--account={GCLOUD_ACCOUNT}")
                self.token = subprocess.check_output(cmd, text=True).strip()
                self._last_token_time = now

        def close(self) -> None:
            if self.proc:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=3)
                except Exception:
                    pass
                self.proc = None

    def extract_doc_summary(doc: Dict[str, Any]) -> Dict[str, Any]:
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
        return (nlp_info.get("extractedFilters") or "").strip()

    def evaluate_hit(query_item: Dict[str, Any], hit: Dict[str, Any], extracted_filters: str = "") -> Tuple[bool, str]:
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


def get_ecp_client() -> EcpClient:
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
    """Thread-safe keep-alive HTTP POST through local `ecp_http_proxy` with auto-401 & 502-EOF recovery."""
    client = get_ecp_client()
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


# ==============================================================================
# 1. Local Open-Source Metal GPU (`mlx-lm`) Intent & Boost Compiler
#    (100% LLM NLU — zero regexes on user query; compact JSON -> Vertex Boosts)
# ==============================================================================

LOCAL_LLM_SYSTEM_PROMPT = """You are Entertainment Search's Intent Extractor grounded on BigQuery `ent_5stores_master_v3`. Today={today_iso} ({weekday_name}), Tomorrow={tomorrow_iso}, UpcomingFri={upcoming_friday_iso}, UpcomingSat={weekend_sat_iso}, FriWeekendEnd={weekend_sun_iso}, ThisWeekEnd={week_end_iso}, NextMonth={next_month_start_iso}..{next_month_end_iso}.
Extract ONLY constraints explicitly mentioned or clearly implied by the user query into minified single-line JSON. NEVER invent price, city, date, time slot, format, language, or category if not mentioned!

Allowed JSON keys (include ONLY keys present in the query; ALWAYS include "q"):
- "type": "movie" | "event" | "venue" | "artist" | "offer"
  * "movie": films, movie formats (4D/4DX/IMAX/IMAX 3D/3D/Dolby/ScreenX), cinema seating/experiences (recliner, gold class, insignia, couple seats), showtimes, "trending/popular/top movies", "releasing...", "movies directed by / starring X", "rajini padam", "cricket pe bani movie, match nahi", "cheapest movie tickets..."
  * "venue": cinema halls/theatres/multiplexes/venues ("cinemas near...", "theatres in...", "multiplexes in...") or cinema chains (PVR, INOX, Cinepolis, MovieMax, Miraj) ONLY when NO movie title, format (IMAX/4DX/3D/Dolby), language, genre, date, or seating tier is requested
  * "event": stand-up comedy, comedians like X, concerts, live music, gigs, artist + city/month ("anuv jain kochi november"), workshops, pottery/art, plays/natak/natok, exhibitions, college fest/festivals/carnivals, garba/dandiya, Durga Puja, NYE/ladies night/DJ parties, sports/cricket/T20/marathon/trekking, kids shows/magic, bhajan clubbing, date night/parents anniversary outings, "not a movie" / "isnt a movie"
  * "offer": bank/credit/debit card deals, discounts, coupons, cashback, BOGO ONLY (NEVER for "cheapest", "passes", "free", or "under 1500")
  * "artist": static person bio profile ONLY when asking purely about a person with NO city, date, movie, concert, or "like X"
- "no_movie": true (ONLY if query explicitly negates movies: "not a movie", "isnt a movie", "other than movies")
- "no_event": true (ONLY if query explicitly negates live events/matches: "match nahi", "not a match")
- "city": exact Title Case Indian city name ONLY — never include neighborhoods, formats, or dates in "city" (e.g. "Mumbai"|"Bengaluru"|"National Capital Region (NCR)"|"Hyderabad"|"Pune"|"Chennai"|"Kolkata"|"Ahmedabad"|"Kochi"|"Goa"|"Jaipur"|"Chandigarh"|"Lucknow"|"Bhubaneswar"|"Patna"|"Indore"|"Surat"|"Bhopal"|"Nagpur"|"Coimbatore"|"Guwahati"|"Ranchi"|"Raipur"|"Vadodara"|"Vizag (Visakhapatnam)"|"Mysuru (Mysore)"|"Thiruvananthapuram (Trivandrum)"|"Mangaluru (Mangalore)"; map lko->"Lucknow", blr/banglore->"Bengaluru", hyd->"Hyderabad", Bandra/Chembur/Andheri/BKC/NMACC/Phoenix Palladium/Lower Parel/Juhu/Thane/Navi Mumbai->"Mumbai", Koramangala/Indiranagar/Whitefield/Cubbon Park->"Bengaluru", Anna Nagar/Velachery/T. Nagar->"Chennai", Wakad/Koregaon Park/Hinjewadi->"Pune", Hitec City/Gachibowli/Banjara Hills->"Hyderabad", South City Mall/Salt Lake/Park Street->"Kolkata", Delhi/NCR/Gurgaon/Gurugram/Noida/Cyber Hub->"National Capital Region (NCR)", "near me" with no city->"Mumbai")
- "fmt": string or array from ["4DX","IMAX 3D","IMAX","DOLBY","SCREENX","ICE","EPIQ","3D","2D"] (use "IMAX 3D" when query says "imax 3d"; use ["IMAX","4DX"] when comparing "imax or 4dx")
- "lang": string or array from ["Hindi","English","Tamil","Telugu","Kannada","Malayalam","Marathi","Bengali","Punjabi","Gujarati","Odia","Assamese","Bhojpuri","Korean","Japanese"] (for "malayalam movies with english subtitles" use ["Malayalam","English"])
- "genres": subset of exact BQ genres ["Comedy","Action","Horror","Thriller","Suspense","Romantic","Drama","Sci-Fi","Animation","Adventure","Crime","Mystery","Musical","Family","Sufi","EDM","Fantasy","Devotional","Biography","Historical","Sports","Psychological","Supernatural","Mythological","Anime","Bollywood","Indie","Folk","Rock","Pop","Hip Hop","Techno","Classical","Ghazal","Retro","Tribute","Acoustic"] (ALWAYS use "Romantic", NEVER "Romance"!)
- "no_genre": negated genre(s) e.g. "Horror" ("without any horror") or "Animation" ("not a cartoon")
- "cats": subset of exact BQ categories ["stand-up-comedy","comedy-shows","open-mic-comedy","open-mic-music-shows","concerts","music-shows","music-festivals","festivals","Carnivals","workshops","arts-and-crafts-workshops","hobby-classes","plays","theatre-plays","club-gigs","dj-night","nightlife","parties","ladies-night","nye-parties","navratri","durga-puja","trekking","camping","adventure","activities","kids","entertainment-kids","kids-workshop","magic","sports","cricket","t20","running","marathon","screening","films-screening"]
- "artist": exact Title Case full person name (expand nicknames: "rajini"->"Rajinikanth", "srk"->"Shah Rukh Khan", "big b"->"Amitabh Bachchan", "arjit"->"Arijit Singh"; OMIT "artist" on similarity queries like "comedians like zakir khan")
- "min_price": integer INR floor (>0 ONLY when query specifies a lower price bound: "between M and N", "from M to N", "above M", "over M")
- "max_price" (or "price"): integer INR ceiling (>0 for "between M and N", "under N", "below N", "N ke andar", 250 for "cheapest"/"budget"/"sasta", 0 ONLY if query literally says "free")
- "avail": true ONLY if query asks for available/unsold seats or tickets ("seats still available", "tickets left", "not sold out")
- "when": "today"|"tonight"|"tomorrow"|"this friday"|"saturday"|"sunday"|"this weekend"|"this week"|"next week"|"this month"|"next month"|"november"|"december"|"nye" ONLY when query mentions a date/time window
- "rel_range": ["YYYY-MM-DD","YYYY-MM-DD"] ONLY when query asks for movie release dates ("releasing this friday" -> ["{upcoming_friday_iso}","{weekend_sun_iso}"])
- "show_range": ["YYYY-MM-DD","YYYY-MM-DD"] ONLY when query mentions active show dates ("today"/"aaj"/"ivattu" -> ["{today_iso}","{today_iso}"], "tomorrow"/"kal"/"udya" -> ["{tomorrow_iso}","{tomorrow_iso}"], "this weekend"/"ee weekend" -> ["{weekend_sat_iso}","{weekend_sun_iso}"], "sunday" -> ["{weekend_sun_iso}","{weekend_sun_iso}"], "this week" -> ["{today_iso}","{week_end_iso}"])
- "slots": subset of ["morning","afternoon","evening","night"] ONLY when query mentions time of day ("tonight"/"shaam" -> ["evening","night"], "after 9 pm"/"late night"/"raat" -> ["night"], "subah" -> ["morning"])
- "q": concise 1-5 word core search keywords:
  * KEEP movie/show/artist names, theatre chains (PVR/INOX/Cinepolis/MovieMax), explicit languages, and explicit formats (IMAX/4DX/3D/Dolby).
  * Translate code-mixed/slang/typos/abbreviations into clean English keywords ("horor"->"horror movie", "ind vs wi tkt lko"->"India vs West Indies cricket", "rajini padam"->"Rajinikanth tamil movie", "gujarati natak"->"Gujarati natak play", "college fest"->"college festival concert carnival", "recliner seats"->"movie cinema", "gold class or insignia"->"PVR INOX movie", "comedians like zakir khan"->"stand up comedy", "parents anniversary"->"musical sufi theatre concert", "watch with friends"->"comedy action movie", "movies releasing this friday"->"movies releasing friday").
  * STRIP negated clauses ("without any horror", "but not a cartoon", "that isnt a movie", "match nahi"), strip price ranges ("between 1000 and 3000", "under 1500"), strip "for date night" on romantic movie queries, strip availability filler ("seats still available for"), and strip city words when "city" is set on movie/event.

Examples:
User: movies directed by lokesh kanagaraj
Assistant: {{"type":"movie","artist":"Lokesh Kanagaraj","q":"Lokesh Kanagaraj movies"}}
User: cinemas near phoenix marketcity whitefield
Assistant: {{"type":"venue","city":"Bengaluru","q":"cinemas phoenix marketcity whitefield"}}
User: couple garba passes in ahmedabad under 1500
Assistant: {{"type":"event","city":"Ahmedabad","max_price":1500,"cats":["navratri"],"q":"couple garba passes"}}
User: concerts between 1000 and 3000 in delhi
Assistant: {{"type":"event","city":"National Capital Region (NCR)","min_price":1000,"max_price":3000,"cats":["concerts"],"q":"concerts"}}
User: hindi horror movie in imax this friday in pune
Assistant: {{"type":"movie","city":"Pune","lang":"Hindi","genres":["Horror"],"fmt":"IMAX","when":"this friday","show_range":["{upcoming_friday_iso}","{weekend_sun_iso}"],"q":"hindi horror IMAX movie"}}
User: malayalam movies with english subtitles
Assistant: {{"type":"movie","lang":["Malayalam","English"],"q":"Malayalam English movies"}}
User: gold class or insignia shows tonight
Assistant: {{"type":"movie","when":"tonight","show_range":["{today_iso}","{today_iso}"],"slots":["evening","night"],"q":"PVR INOX movie"}}
User: something to take my parents to on their anniversary
Assistant: {{"type":"event","genres":["Musical","Sufi","Classical"],"cats":["theatre-plays","plays","music-shows","concerts"],"q":"musical sufi theatre concert"}}
User: funny movie without any horror
Assistant: {{"type":"movie","genres":["Comedy"],"no_genre":"Horror","q":"funny comedy movie"}}
User: imax or 4dx which is better for spider man
Assistant: {{"type":"movie","fmt":["IMAX","4DX"],"q":"Spider-Man IMAX 4DX"}}
User: something after 9 pm near hitec city that isnt a movie
Assistant: {{"type":"event","no_movie":true,"city":"Hyderabad","slots":["night"],"q":"live events"}}
User: ind vs wi tkt lko
Assistant: {{"type":"event","city":"Lucknow","cats":["cricket","t20","sports"],"q":"India vs West Indies cricket"}}
User: icici credit card movie offer
Assistant: {{"type":"offer","q":"ICICI credit card movie offer"}}"""

# Exact BigQuery (`ent_5stores_master_v3` & `ent_5stores_unified`) facet & geography mappings
CANONICAL_CITY_MAP: Dict[str, str] = {
    "delhi": "National Capital Region (NCR)",
    "new delhi": "National Capital Region (NCR)",
    "delhi-ncr": "National Capital Region (NCR)",
    "delhi ncr": "National Capital Region (NCR)",
    "ncr": "National Capital Region (NCR)",
    "del": "National Capital Region (NCR)",
    "gurgaon": "National Capital Region (NCR)",
    "gurugram": "National Capital Region (NCR)",
    "gurugram (gurgaon)": "National Capital Region (NCR)",
    "ggn": "National Capital Region (NCR)",
    "noida": "National Capital Region (NCR)",
    "greater noida": "National Capital Region (NCR)",
    "faridabad": "National Capital Region (NCR)",
    "ghaziabad": "National Capital Region (NCR)",
    "national capital region (ncr)": "National Capital Region (NCR)",
    "bangalore": "Bengaluru",
    "banglore": "Bengaluru",
    "bengalooru": "Bengaluru",
    "bengaluru": "Bengaluru",
    "blr": "Bengaluru",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "mum": "Mumbai",
    "navi mumbai": "Mumbai",
    "thane": "Mumbai",
    "hyderabad": "Hyderabad",
    "hyderbad": "Hyderabad",
    "hyd": "Hyderabad",
    "ahmedabad": "Ahmedabad",
    "ahemdabad": "Ahmedabad",
    "ahmedbad": "Ahmedabad",
    "amd": "Ahmedabad",
    "lucknow": "Lucknow",
    "lko": "Lucknow",
    "kolkata": "Kolkata",
    "kolkatta": "Kolkata",
    "calcutta": "Kolkata",
    "kol": "Kolkata",
    "chennai": "Chennai",
    "madras": "Chennai",
    "pune": "Pune",
    "kochi": "Kochi",
    "cochin": "Kochi",
    "ernakulam": "Kochi",
    "chandigarh": "Chandigarh",
    "chd": "Chandigarh",
    "guwahati": "Guwahati",
    "gauhati": "Guwahati",
    "vizag": "Vizag (Visakhapatnam)",
    "visakhapatnam": "Vizag (Visakhapatnam)",
    "mysore": "Mysuru (Mysore)",
    "mysuru": "Mysuru (Mysore)",
    "bhubaneswar": "Bhubaneswar",
    "bhubaneshwar": "Bhubaneswar",
    "bbsr": "Bhubaneswar",
    "trivandrum": "Thiruvananthapuram (Trivandrum)",
    "thiruvananthapuram": "Thiruvananthapuram (Trivandrum)",
    "mangalore": "Mangaluru (Mangalore)",
    "mangaluru": "Mangaluru (Mangalore)",
    "hubli": "Hubballi (Hubli)",
    "hubballi": "Hubballi (Hubli)",
    "allahabad": "Prayagraj (Allahabad)",
    "prayagraj": "Prayagraj (Allahabad)",
    "belgaum": "Belagavi (Belgaum)",
    "belagavi": "Belagavi (Belgaum)",
    "rajahmundry": "Rajamahendravaram (Rajahmundry)",
    "gulbarga": "Kalaburagi (Gulbarga)",
    "tumkur": "Tumakuru (Tumkur)",
    "trichy": "Tiruchirappalli",
    "pondicherry": "Puducherry",
    "pondy": "Puducherry",
    "calicut": "Kozhikode",
    "bellary": "Ballari",
    "shimoga": "Shivamogga",
    "baroda": "Vadodara",
    "banaras": "Varanasi",
    "kashi": "Varanasi",
    "near me": "Mumbai",
}

# Unambiguous city tokens safe for fallback scan inside raw query text (no common English word collisions)
SAFE_QUERY_CITY_TOKENS: Dict[str, str] = {
    k: v
    for k, v in CANONICAL_CITY_MAP.items()
    if k not in ("near me", "del", "mum", "kol")
} | {
    "pune": "Pune",
    "jaipur": "Jaipur",
    "goa": "Goa",
    "indore": "Indore",
    "surat": "Surat",
    "bhopal": "Bhopal",
    "nagpur": "Nagpur",
    "coimbatore": "Coimbatore",
    "patna": "Patna",
    "ranchi": "Ranchi",
    "raipur": "Raipur",
    "vadodara": "Vadodara",
    "amritsar": "Amritsar",
    "ludhiana": "Ludhiana",
    "dehradun": "Dehradun",
    "varanasi": "Varanasi",
    "kanpur": "Kanpur",
    "nashik": "Nashik",
    "madurai": "Madurai",
    "vijayawada": "Vijayawada",
    "udaipur": "Udaipur",
    "jodhpur": "Jodhpur",
    "rajkot": "Rajkot",
    "jabalpur": "Jabalpur",
    "gwalior": "Gwalior",
    "siliguri": "Siliguri",
    "shillong": "Shillong",
}

GEO_CITY_NAME_MAP: Dict[str, str] = {
    "National Capital Region (NCR)": "New Delhi",
    "Vizag (Visakhapatnam)": "Visakhapatnam",
    "Mysuru (Mysore)": "Mysuru",
    "Thiruvananthapuram (Trivandrum)": "Thiruvananthapuram",
    "Mangaluru (Mangalore)": "Mangaluru",
    "Hubballi (Hubli)": "Hubballi",
    "Prayagraj (Allahabad)": "Prayagraj",
    "Belagavi (Belgaum)": "Belagavi",
    "Rajamahendravaram (Rajahmundry)": "Rajahmundry",
    "Kalaburagi (Gulbarga)": "Kalaburagi",
    "Tumakuru (Tumkur)": "Tumakuru",
}

WHOLE_CITY_NAMES = set(CANONICAL_CITY_MAP.keys()) | {
    "mumbai", "bengaluru", "national capital region (ncr)", "new delhi", "faridabad", "ghaziabad",
    "hyderabad", "pune", "chennai", "kolkata", "howrah", "ahmedabad", "kochi", "goa", "panaji", "margao",
    "jaipur", "chandigarh", "mohali", "panchkula", "lucknow", "vizag (visakhapatnam)", "mysuru (mysore)",
    "thiruvananthapuram (trivandrum)", "mangaluru (mangalore)", "hubballi (hubli)", "prayagraj (allahabad)",
    "belagavi (belgaum)", "rajamahendravaram (rajahmundry)", "kalaburagi (gulbarga)", "tumakuru (tumkur)",
    "surat", "vadodara", "rajkot", "gandhinagar", "bhavnagar", "jamnagar",
    "indore", "bhopal", "jabalpur", "gwalior", "ujjain",
    "nagpur", "nashik", "aurangabad", "chhatrapati sambhajinagar", "kolhapur", "solapur", "amravati", "sangli", "satara",
    "coimbatore", "madurai", "tiruchirappalli", "salem", "tirunelveli", "vellore", "erode", "tiruppur", "puducherry",
    "vijayawada", "guntur", "nellore", "kurnool", "tirupati", "kakinada", "warangal", "nizamabad", "karimnagar", "khammam",
    "davanagere", "ballari", "shivamogga", "udupi", "manipal",
    "kozhikode", "thrissur", "kollam", "kannur", "alappuzha", "palakkad", "kottayam",
    "bhubaneswar", "cuttack", "puri", "rourkela", "sambalpur",
    "patna", "ranchi", "jamshedpur", "dhanbad", "bokaro",
    "guwahati", "siliguri", "durgapur", "asansol", "agartala", "shillong", "imphal", "aizawl", "gangtok",
    "dehradun", "haridwar", "roorkee", "haldwani", "nainital", "rishikesh",
    "amritsar", "ludhiana", "jalandhar", "patiala", "bathinda", "pathankot", "shimla", "dharamshala", "jammu", "srinagar",
    "varanasi", "kanpur", "agra", "meerut", "bareilly", "aligarh", "moradabad", "gorakhpur", "jhansi", "mathura", "ayodhya",
    "jodhpur", "udaipur", "kota", "bikaner", "ajmer", "alwar", "bhilwara", "sikar",
    "raipur", "bilaspur", "bhilai", "durg", "karnal", "panipat", "ambala", "hisar", "rohtak", "sonipat",
    "near me",
}

FORMAT_VALUES_V3: Dict[str, List[str]] = {
    "4DX": ["4DX", "4DX 3D", "MX4D", "MX4D 3D"],
    "4D": ["4DX", "4DX 3D", "MX4D", "MX4D 3D"],
    "MX4D": ["MX4D", "MX4D 3D"],
    "IMAX 3D": ["IMAX 3D"],
    "IMAX": ["IMAX 2D", "IMAX 3D"],
    "DOLBY": ["DOLBY CINEMA 2D", "DOLBY CINEMA 3D"],
    "DOLBY ATMOS": ["DOLBY CINEMA 2D", "DOLBY CINEMA 3D"],
    "SCREENX": ["2D SCREEN X", "3D SCREEN X"],
    "SCREEN X": ["2D SCREEN X", "3D SCREEN X"],
    "ICE": ["ICE", "ICE 3D"],
    "EPIQ": ["EPIQ", "EPIQ 3D"],
    "3D": ["3D", "IMAX 3D", "4DX 3D", "MX4D 3D", "ICE 3D", "3D SCREEN X", "DOLBY CINEMA 3D"],
    "2D": ["2D", "IMAX 2D", "DOLBY CINEMA 2D", "2D SCREEN X"],
}

FORMAT_SPEC_MAP_V2: Dict[str, str] = {
    "4DX": 'formats: ANY("4DX","4DX 3D","MX4D","MX4D 3D") OR sessions.format: ANY("4DX","4DX-3D","MX4D")',
    "4D": 'formats: ANY("4DX","4DX 3D","MX4D","MX4D 3D") OR sessions.format: ANY("4DX","4DX-3D","MX4D")',
    "MX4D": 'formats: ANY("MX4D","MX4D 3D") OR sessions.format: ANY("MX4D")',
    "IMAX 3D": 'formats: ANY("IMAX 3D") OR sessions.format: ANY("IMAX-3D")',
    "IMAX": 'formats: ANY("IMAX 2D","IMAX 3D") OR sessions.format: ANY("IMAX-2D","IMAX-3D")',
    "DOLBY": 'formats: ANY("DOLBY CINEMA 2D","DOLBY CINEMA 3D") OR sessions.format: ANY("dolby-cinema-2d")',
    "DOLBY ATMOS": 'formats: ANY("DOLBY CINEMA 2D","DOLBY CINEMA 3D") OR sessions.format: ANY("dolby-cinema-2d")',
    "SCREENX": 'formats: ANY("2D SCREEN X","3D SCREEN X") OR sessions.format: ANY("2D-SCREEN-X","3D-SCREEN-X")',
    "SCREEN X": 'formats: ANY("2D SCREEN X","3D SCREEN X") OR sessions.format: ANY("2D-SCREEN-X","3D-SCREEN-X")',
    "3D": 'formats: ANY("3D","IMAX 3D","4DX 3D","MX4D 3D") OR sessions.format: ANY("3D","IMAX-3D","4DX-3D")',
    "2D": 'formats: ANY("2D","IMAX 2D") OR sessions.format: ANY("2D","IMAX-2D")',
}

FORMAT_SPEC_MAP_V3: Dict[str, str] = {
    k: "formats: ANY(" + ", ".join(f'"{v}"' for v in vals) + ")"
    for k, vals in FORMAT_VALUES_V3.items()
}

# Exact distinct `genres` values verified from BigQuery `ent_5stores_master_v3`
VALID_GENRES = {
    "Comedy", "Action", "Horror", "Thriller", "Suspense", "Romantic", "Drama",
    "Sci-Fi", "Animation", "Adventure", "Crime", "Mystery", "Musical", "Family",
    "Sufi", "EDM", "Fantasy", "Devotional", "Biography", "Historical", "Sports",
    "Psychological", "Supernatural", "Mythological", "Anime", "Bollywood", "Indie",
    "Folk", "Rock", "Pop", "Indian pop", "Hip Hop", "Techno", "Classical",
    "Indian Classical", "Ghazal", "Retro", "Tribute", "Acoustic", "Accoustic",
    "Jazz", "Blues", "Fusion", "Documentary", "Period", "War", "Political",
}

# Dual-spelling BQ genre expansions (e.g. BQ has 379 `"Accoustic"` + 314 `"Acoustic"` rows)
GENRE_BQ_EXPANSION: Dict[str, List[str]] = {
    "Acoustic": ["Acoustic", "Accoustic"],
    "Accoustic": ["Acoustic", "Accoustic"],
    "Classical": ["Classical", "Indian Classical"],
    "Pop": ["Pop", "Indian pop"],
}

# Exact distinct `categories` values verified from BigQuery `ent_5stores_master_v3`
VALID_CATEGORIES = {
    "stand-up-comedy", "comedy-shows", "open-mic-comedy", "open-mic-music-shows",
    "improv-comedy", "concerts", "music-shows", "music-festivals", "festivals",
    "Carnivals", "workshops", "arts-and-crafts-workshops", "hobby-classes",
    "photography-workshops", "plays", "theatre-plays", "storytelling",
    "club-gigs", "dj-night", "nightlife", "parties", "ladies-night",
    "bollywood-night", "nye-parties", "ny-parties", "nye-live-performances",
    "navratri", "dandiya", "durga-puja", "trekking", "camping", "adventure",
    "activities", "food-and-drinks", "amusement-parks", "exhibitions", "gaming",
    "kids", "entertainment-kids", "education-kids", "kids-workshop",
    "magic", "magic-workshops", "sports", "cricket", "t20", "odi", "football",
    "running", "marathon", "5k", "screening", "films-screening", "karaoke",
}

VALID_LANGUAGES = {
    "Hindi", "English", "Tamil", "Telugu", "Kannada", "Malayalam",
    "Marathi", "Bengali", "Punjabi", "Gujarati", "Bhojpuri", "Odia", "Assamese",
    "Korean", "Japanese",
}
VALID_TIME_SLOTS = {"morning", "afternoon", "evening", "night"}
ALL_RECORD_TYPES = ("movie", "event", "venue", "artist", "offer")

TYPE_ALIAS_MAP: Dict[str, str] = {
    "movie": "movie",
    "movies": "movie",
    "new movie": "movie",
    "new movies": "movie",
    "popular movie": "movie",
    "trending movie": "movie",
    "trending movies": "movie",
    "trending": "movie",
    "ticket": "movie",
    "tickets": "movie",
    "movie ticket": "movie",
    "movie tickets": "movie",
    "film": "movie",
    "films": "movie",
    "padam": "movie",
    "event": "event",
    "events": "event",
    "show": "event",
    "shows": "event",
    "comedy": "event",
    "comedy show": "event",
    "comedy shows": "event",
    "standup": "event",
    "stand up": "event",
    "stand-up-comedy": "event",
    "open mic": "event",
    "open-mic-comedy": "event",
    "concert": "event",
    "concerts": "event",
    "play": "event",
    "plays": "event",
    "natak": "event",
    "natok": "event",
    "workshop": "event",
    "workshops": "event",
    "garba": "event",
    "dandiya": "event",
    "navratri": "event",
    "festival": "event",
    "fest": "event",
    "pass": "event",
    "passes": "event",
    "couple": "event",
    "sports": "event",
    "cricket": "event",
    "match": "event",
    "venue": "venue",
    "venues": "venue",
    "theatre": "venue",
    "theater": "venue",
    "theatres": "venue",
    "theaters": "venue",
    "cinema": "venue",
    "cinemas": "venue",
    "multiplex": "venue",
    "location": "venue",
    "mall": "venue",
    "artist": "artist",
    "director": "artist",
    "actor": "artist",
    "offer": "offer",
    "offers": "offer",
    "discount": "offer",
    "deal": "offer",
    "coupon": "offer",
    "credit card": "offer",
}

# Canonical parent city mapping for Indian neighborhoods, malls, & cultural venues
SUBLOCALITY_TO_CITY_MAP: Dict[str, Tuple[str, str]] = {
    "bandra": ("Mumbai", "Bandra, Mumbai"),
    "bandra west": ("Mumbai", "Bandra West, Mumbai"),
    "andheri": ("Mumbai", "Andheri, Mumbai"),
    "andheri west": ("Mumbai", "Andheri West, Mumbai"),
    "andheri east": ("Mumbai", "Andheri East, Mumbai"),
    "andheri station": ("Mumbai", "Andheri, Mumbai"),
    "chembur": ("Mumbai", "Chembur, Mumbai"),
    "juhu": ("Mumbai", "Juhu, Mumbai"),
    "prithvi theatre": ("Mumbai", "Juhu, Mumbai"),
    "prithvi": ("Mumbai", "Juhu, Mumbai"),
    "bkc": ("Mumbai", "BKC, Mumbai"),
    "nmacc": ("Mumbai", "BKC, Mumbai"),
    "lower parel": ("Mumbai", "Lower Parel, Mumbai"),
    "phoenix palladium": ("Mumbai", "Lower Parel, Mumbai"),
    "palladium": ("Mumbai", "Lower Parel, Mumbai"),
    "ncpa": ("Mumbai", "Nariman Point, Mumbai"),
    "nariman point": ("Mumbai", "Nariman Point, Mumbai"),
    "colaba": ("Mumbai", "Colaba, Mumbai"),
    "powai": ("Mumbai", "Powai, Mumbai"),
    "malad": ("Mumbai", "Malad, Mumbai"),
    "borivali": ("Mumbai", "Borivali, Mumbai"),
    "goregaon": ("Mumbai", "Goregaon, Mumbai"),
    "worli": ("Mumbai", "Worli, Mumbai"),
    "dadar": ("Mumbai", "Dadar, Mumbai"),
    "matunga": ("Mumbai", "Matunga, Mumbai"),
    "kurla": ("Mumbai", "Kurla, Mumbai"),
    "ghatkopar": ("Mumbai", "Ghatkopar, Mumbai"),
    "mulund": ("Mumbai", "Mulund, Mumbai"),
    "vashi": ("Mumbai", "Vashi, Navi Mumbai"),
    "kharghar": ("Mumbai", "Kharghar, Navi Mumbai"),
    "nerul": ("Mumbai", "Nerul, Navi Mumbai"),
    "koramangala": ("Bengaluru", "Koramangala, Bengaluru"),
    "nexus mall koramangala": ("Bengaluru", "Koramangala, Bengaluru"),
    "indiranagar": ("Bengaluru", "Indiranagar, Bengaluru"),
    "whitefield": ("Bengaluru", "Whitefield, Bengaluru"),
    "phoenix marketcity whitefield": ("Bengaluru", "Whitefield, Bengaluru"),
    "cubbon park": ("Bengaluru", "Cubbon Park, Bengaluru"),
    "mg road": ("Bengaluru", "MG Road, Bengaluru"),
    "church street": ("Bengaluru", "Church Street, Bengaluru"),
    "brigade road": ("Bengaluru", "Brigade Road, Bengaluru"),
    "jayanagar": ("Bengaluru", "Jayanagar, Bengaluru"),
    "hsr layout": ("Bengaluru", "HSR Layout, Bengaluru"),
    "marathahalli": ("Bengaluru", "Marathahalli, Bengaluru"),
    "electronic city": ("Bengaluru", "Electronic City, Bengaluru"),
    "malleshwaram": ("Bengaluru", "Malleshwaram, Bengaluru"),
    "jp nagar": ("Bengaluru", "JP Nagar, Bengaluru"),
    "bellandur": ("Bengaluru", "Bellandur, Bengaluru"),
    "hebbal": ("Bengaluru", "Hebbal, Bengaluru"),
    "yelahanka": ("Bengaluru", "Yelahanka, Bengaluru"),
    "banashankari": ("Bengaluru", "Banashankari, Bengaluru"),
    "rajajinagar": ("Bengaluru", "Rajajinagar, Bengaluru"),
    "anna nagar": ("Chennai", "Anna Nagar, Chennai"),
    "velachery": ("Chennai", "Velachery, Chennai"),
    "t nagar": ("Chennai", "T. Nagar, Chennai"),
    "t. nagar": ("Chennai", "T. Nagar, Chennai"),
    "adyar": ("Chennai", "Adyar, Chennai"),
    "omr": ("Chennai", "OMR, Chennai"),
    "nungambakkam": ("Chennai", "Nungambakkam, Chennai"),
    "mylapore": ("Chennai", "Mylapore, Chennai"),
    "vadapalani": ("Chennai", "Vadapalani, Chennai"),
    "porur": ("Chennai", "Porur, Chennai"),
    "tambaram": ("Chennai", "Tambaram, Chennai"),
    "wakad": ("Pune", "Wakad, Pune"),
    "koregaon park": ("Pune", "Koregaon Park, Pune"),
    "hinjewadi": ("Pune", "Hinjewadi, Pune"),
    "kothrud": ("Pune", "Kothrud, Pune"),
    "viman nagar": ("Pune", "Viman Nagar, Pune"),
    "baner": ("Pune", "Baner, Pune"),
    "hadapsar": ("Pune", "Hadapsar, Pune"),
    "kharadi": ("Pune", "Kharadi, Pune"),
    "aundh": ("Pune", "Aundh, Pune"),
    "hitec city": ("Hyderabad", "Hitec City, Hyderabad"),
    "hitech city": ("Hyderabad", "Hitec City, Hyderabad"),
    "gachibowli": ("Hyderabad", "Gachibowli, Hyderabad"),
    "banjara hills": ("Hyderabad", "Banjara Hills, Hyderabad"),
    "jubilee hills": ("Hyderabad", "Jubilee Hills, Hyderabad"),
    "madhapur": ("Hyderabad", "Madhapur, Hyderabad"),
    "kondapur": ("Hyderabad", "Kondapur, Hyderabad"),
    "kukatpally": ("Hyderabad", "Kukatpally, Hyderabad"),
    "secunderabad": ("Hyderabad", "Secunderabad, Hyderabad"),
    "ameerpet": ("Hyderabad", "Ameerpet, Hyderabad"),
    "begumpet": ("Hyderabad", "Begumpet, Hyderabad"),
    "cyber hub": ("National Capital Region (NCR)", "Cyber Hub, Gurgaon"),
    "cyberhub": ("National Capital Region (NCR)", "Cyber Hub, Gurgaon"),
    "dlf cyberhub": ("National Capital Region (NCR)", "Cyber Hub, Gurgaon"),
    "sector 29": ("National Capital Region (NCR)", "Sector 29, Gurgaon"),
    "connaught place": ("National Capital Region (NCR)", "Connaught Place, New Delhi"),
    "nehru place": ("National Capital Region (NCR)", "Nehru Place, New Delhi"),
    "saket": ("National Capital Region (NCR)", "Saket, New Delhi"),
    "hauz khas": ("National Capital Region (NCR)", "Hauz Khas, New Delhi"),
    "dwarka": ("National Capital Region (NCR)", "Dwarka, New Delhi"),
    "vasant kunj": ("National Capital Region (NCR)", "Vasant Kunj, New Delhi"),
    "karol bagh": ("National Capital Region (NCR)", "Karol Bagh, New Delhi"),
    "rajouri garden": ("National Capital Region (NCR)", "Rajouri Garden, New Delhi"),
    "lajpat nagar": ("National Capital Region (NCR)", "Lajpat Nagar, New Delhi"),
    "salt lake": ("Kolkata", "Salt Lake, Kolkata"),
    "park street": ("Kolkata", "Park Street, Kolkata"),
    "new town": ("Kolkata", "New Town, Kolkata"),
    "south city mall": ("Kolkata", "South City, Kolkata"),
    "quest mall": ("Kolkata", "Park Circus, Kolkata"),
    "ballygunge": ("Kolkata", "Ballygunge, Kolkata"),
    "alipore": ("Kolkata", "Alipore, Kolkata"),
    "sg highway": ("Ahmedabad", "SG Highway, Ahmedabad"),
    "navrangpura": ("Ahmedabad", "Navrangpura, Ahmedabad"),
    "vastrapur": ("Ahmedabad", "Vastrapur, Ahmedabad"),
    "prahlad nagar": ("Ahmedabad", "Prahlad Nagar, Ahmedabad"),
    "edappally": ("Kochi", "Edappally, Kochi"),
    "lulu mall": ("Kochi", "Edappally, Kochi"),
    "fort kochi": ("Kochi", "Fort Kochi, Kochi"),
    "kakkanad": ("Kochi", "Kakkanad, Kochi"),
}

ARTIST_NICKNAME_MAP: Dict[str, str] = {
    "rajini": "Rajinikanth",
    "rajnikanth": "Rajinikanth",
    "rajinikanth": "Rajinikanth",
    "thalaivar": "Rajinikanth",
    "srk": "Shah Rukh Khan",
    "shahrukh": "Shah Rukh Khan",
    "shahrukh khan": "Shah Rukh Khan",
    "big b": "Amitabh Bachchan",
    "amitabh": "Amitabh Bachchan",
    "bhaijaan": "Salman Khan",
    "thalapathy": "Vijay",
    "arjit": "Arijit Singh",
    "arjit singh": "Arijit Singh",
    "arijit": "Arijit Singh",
    "ar rahman": "A. R. Rahman",
    "a r rahman": "A. R. Rahman",
    "bassi": "Anubhav Singh Bassi",
    "anubhav bassi": "Anubhav Singh Bassi",
    "samay": "Samay Raina",
    "samay raina": "Samay Raina",
    "zakir": "Zakir Khan",
    "upmanyu": "Abhishek Upmanyu",
}

NON_LOCALITY_TOKENS = {
    "pvr", "inox", "cinepolis", "moviemax", "miraj", "pvr theatres", "inox cinemas",
    "icici", "hdfc", "sbi", "axis", "credit card", "debit card",
}

FORBIDDEN_LOCALITY_WORDS = (
    {w.lower() for w in VALID_LANGUAGES}
    | {g.lower() for g in VALID_GENRES}
    | {c.lower() for c in VALID_CATEGORIES}
    | VALID_TIME_SLOTS
    | {
        "oriya", "imax", "4dx", "4d", "mx4d", "3d", "2d", "dolby", "atmos",
        "screenx", "screen", "ice", "epiq", "recliner", "insignia", "gold", "laser",
        "this", "next", "current", "coming", "upcoming", "releasing", "release",
        "today", "tonight", "tomorrow", "yesterday", "now", "aaj", "kal", "ivattu", "udya",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "weekend", "week", "month", "year", "october", "november", "december",
        "movie", "movies", "film", "films", "padam", "event", "events", "show", "shows",
        "play", "plays", "natak", "natok", "concert", "concerts", "gig", "gigs", "workshop", "workshops",
        "standup", "stand", "open", "mic", "ticket", "tickets", "tkt", "seat", "seats",
        "trending", "popular", "top", "best", "new", "latest",
        "cheap", "cheapest", "budget", "sasta", "free", "under", "below", "less", "price", "rs", "inr",
        "offer", "offers", "discount", "deal", "coupon", "card", "credit", "debit",
        "pvr", "inox", "cinepolis", "moviemax", "miraj",
        "near", "me", "in", "at", "on", "for", "by", "with", "without", "not", "isnt", "other", "than",
        "directed", "director", "actor", "starring",
    }
)

LANG_LOOKUP: Dict[str, str] = {l.lower(): l for l in VALID_LANGUAGES} | {
    "oriya": "Odia",
    "bollywood": "Hindi",
    "kollywood": "Tamil",
    "tollywood": "Telugu",
    "sandalwood": "Kannada",
    "mollywood": "Malayalam",
    "bangla": "Bengali",
}

LANG_GROUNDING_STEMS: Dict[str, Tuple[str, ...]] = {
    "Odia": ("odia", "oriya", "odisha"),
    "Hindi": ("hindi", "bollywood"),
    "English": ("english", "hollywood"),
    "Tamil": ("tamil", "kollywood", "padam", "eppo"),
    "Telugu": ("telugu", "tollywood", "ekkada", "ee weekend"),
    "Kannada": ("kannada", "sandalwood", "ivattu"),
    "Malayalam": ("malayalam", "mollywood"),
    "Marathi": ("marathi", "udya"),
    "Bengali": ("bengali", "bangla", "natok", "shonibar", "pujo"),
    "Punjabi": ("punjabi",),
    "Gujarati": ("gujarati", "garba", "dandiya"),
    "Bhojpuri": ("bhojpuri",),
    "Assamese": ("assamese",),
    "Korean": ("korean", "k-drama", "kpop", "k-pop", "bts"),
    "Japanese": ("japanese", "anime"),
}

GENRE_LOOKUP: Dict[str, str] = {g.lower(): g for g in VALID_GENRES} | {
    "sci fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "romance": "Romantic",
    "romcom": "Romantic",
    "rom-com": "Romantic",
    "romantic comedy": "Romantic",
    "horor": "Horror",
    "thriler": "Thriller",
    "accoustic": "Acoustic",
    "indian classical": "Indian Classical",
    "indian pop": "Indian pop",
    "hip-hop": "Hip Hop",
    "hiphop": "Hip Hop",
    "biopic": "Biography",
    "sport": "Sports",
    "spiritual": "Devotional",
}

GENRE_GROUNDING_STEMS: Dict[str, Tuple[str, ...]] = {
    "Comedy": ("comedy", "comic", "standup", "stand-up", "stand up", "funny", "laugh", "open mic", "mazaa", "chill", "friends", "timepass", "romcom", "rom-com"),
    "Action": ("action", "paisa vasool", "masala", "friends", "fight"),
    "Horror": ("horror", "horor", "scary", "ghost", "spooky", "haunted"),
    "Thriller": ("thriller", "thriler", "suspense", "edge of seat", "gripping"),
    "Suspense": ("suspense", "thriller", "mystery"),
    "Romantic": ("romantic", "romance", "romcom", "rom-com", "date night", "love", "couple", "valentine", "anniversary"),
    "Drama": ("drama", "play", "natak", "natok", "theatre", "emotional", "parents", "family"),
    "Sci-Fi": ("sci-fi", "scifi", "sci fi", "science fiction", "space"),
    "Animation": ("animation", "animated", "anime", "cartoon", "kids"),
    "Adventure": ("adventure", "trek", "kids"),
    "Crime": ("crime", "detective", "murder"),
    "Mystery": ("mystery", "whodunit"),
    "Musical": ("musical", "music", "concert", "gig", "sufi", "anniversary", "parents", "broadway"),
    "Family": ("family", "kids", "children", "parents"),
    "Sufi": ("sufi", "qawwali", "anniversary", "parents"),
    "EDM": ("edm", "dj", "electronic", "techno", "rave", "sunburn"),
    "Fantasy": ("fantasy", "magic", "hanuman", "mytho"),
    "Devotional": ("devotional", "bhajan", "kirtan", "spiritual", "hanuman", "ramayana"),
    "Biography": ("biography", "biopic", "true story", "cricket pe bani", "based on"),
    "Historical": ("historical", "history", "period"),
    "Sports": ("sport", "sports", "cricket", "football"),
    "Psychological": ("psychological", "mind-bending", "mind bending"),
    "Supernatural": ("supernatural", "paranormal"),
    "Mythological": ("mythological", "mythology", "hanuman", "ramayana", "mahabharata"),
    "Anime": ("anime", "manga"),
    "Bollywood": ("bollywood",),
    "Indie": ("indie", "independent"),
    "Folk": ("folk", "lok"),
    "Rock": ("rock", "metal", "band"),
    "Pop": ("pop",),
    "Indian pop": ("pop", "indie pop"),
    "Hip Hop": ("hip hop", "hip-hop", "rap"),
    "Techno": ("techno", "house", "rave", "electronic"),
    "Classical": ("classical", "carnatic", "hindustani", "anniversary", "parents"),
    "Indian Classical": ("classical", "carnatic", "hindustani"),
    "Ghazal": ("ghazal", "shayari", "anniversary", "parents"),
    "Retro": ("retro", "oldies", "90s", "80s", "tribute"),
    "Tribute": ("tribute",),
    "Acoustic": ("acoustic", "accoustic", "unplugged"),
    "Accoustic": ("acoustic", "accoustic", "unplugged"),
}

CAT_LOOKUP: Dict[str, List[str]] = {c.lower(): [c] for c in VALID_CATEGORIES} | {
    "standup": ["stand-up-comedy", "comedy-shows"],
    "stand up": ["stand-up-comedy", "comedy-shows"],
    "stand-up": ["stand-up-comedy", "comedy-shows"],
    "stand-up-comedy": ["stand-up-comedy", "comedy-shows"],
    "comedy": ["stand-up-comedy", "comedy-shows"],
    "open mic": ["open-mic-comedy", "open-mic-music-shows"],
    "open-mic": ["open-mic-comedy", "open-mic-music-shows"],
    "jamming": ["open-mic-music-shows", "music-shows"],
    "natak": ["plays", "theatre-plays"],
    "natok": ["plays", "theatre-plays"],
    "play": ["plays", "theatre-plays"],
    "theatre": ["plays", "theatre-plays"],
    "theatre play": ["plays", "theatre-plays"],
    "concert": ["concerts", "music-shows"],
    "live music": ["concerts", "music-shows"],
    "gig": ["concerts", "music-shows", "club-gigs"],
    "workshop": ["workshops"],
    "pottery": ["arts-and-crafts-workshops", "workshops", "hobby-classes"],
    "art": ["arts-and-crafts-workshops", "workshops"],
    "craft": ["arts-and-crafts-workshops", "workshops"],
    "arts and crafts": ["arts-and-crafts-workshops", "workshops"],
    "dj": ["dj-night", "club-gigs", "nightlife"],
    "party": ["parties", "nightlife", "club-gigs"],
    "ladies night": ["ladies-night", "parties", "nightlife", "club-gigs"],
    "ladies-night": ["ladies-night", "parties", "nightlife", "club-gigs"],
    "nye": ["nye-parties", "ny-parties", "nye-live-performances", "parties"],
    "new year": ["nye-parties", "ny-parties", "nye-live-performances", "parties"],
    "nye-parties": ["nye-parties", "ny-parties", "nye-live-performances", "parties"],
    "garba": ["navratri"],
    "dandiya": ["navratri"],
    "durga puja": ["durga-puja", "festivals"],
    "durgotsav": ["durga-puja", "festivals"],
    "pujo": ["durga-puja", "festivals"],
    "fest": ["festivals", "music-festivals", "Carnivals"],
    "college fest": ["festivals", "music-festivals", "Carnivals"],
    "festival": ["festivals", "music-festivals", "Carnivals"],
    "festivals": ["festivals", "music-festivals", "Carnivals"],
    "carnival": ["Carnivals", "festivals"],
    "carnivals": ["Carnivals", "festivals"],
    "kids": ["kids", "entertainment-kids", "kids-workshop"],
    "magic": ["magic", "magic-workshops", "entertainment-kids"],
    "cricket": ["cricket", "t20", "sports"],
    "t20": ["cricket", "t20", "sports"],
    "ipl": ["cricket", "t20", "sports"],
    "run": ["running", "marathon", "5k", "sports"],
    "running": ["running", "marathon", "5k", "sports"],
    "marathon": ["running", "marathon", "5k", "sports"],
    "trek": ["trekking", "camping", "adventure"],
    "trekking": ["trekking", "camping", "adventure"],
    "screening": ["screening", "films-screening"],
    "film screening": ["screening", "films-screening"],
}


def _quote_any(values: List[str]) -> str:
    inner = ", ".join(f'"{v}"' for v in values)
    return f"ANY({inner})"


def _is_valid_sublocality_phrase(phrase: str) -> bool:
    """Ensure a candidate sub-locality is a real geographic neighborhood and NEVER contains
    formats (`IMAX`, `4DX`), dates (`this Friday`), languages, genres, or whole-city names.
    """
    p_low = phrase.strip().lower().strip(".,-()")
    if not p_low:
        return False
    if p_low in SUBLOCALITY_TO_CITY_MAP:
        return True
    if p_low in WHOLE_CITY_NAMES or p_low in CANONICAL_CITY_MAP or p_low in NON_LOCALITY_TOKENS:
        return False
    words = [w.strip(".,-()") for w in p_low.split() if w.strip(".,-()")]
    if not words or len(words) > 4:
        return False
    allow_digit = words[0] in ("sector", "phase", "block", "stage")
    for w in words:
        if w in FORBIDDEN_LOCALITY_WORDS or w in WHOLE_CITY_NAMES or w in CANONICAL_CITY_MAP:
            return False
        if not allow_digit and any(ch.isdigit() for ch in w):
            return False
    return True


def _resolve_known_city(raw_str: str) -> str:
    """Resolve a string to a canonical Indian city/region, returning `""` if it is not a city."""
    import difflib

    c_clean = raw_str.strip().replace('"', "")
    c_low = c_clean.lower()
    if not c_low or c_low in NON_LOCALITY_TOKENS or c_low in FORBIDDEN_LOCALITY_WORDS:
        return ""
    if c_low in CANONICAL_CITY_MAP:
        return CANONICAL_CITY_MAP[c_low]
    if c_low in WHOLE_CITY_NAMES:
        return c_clean.title() if c_clean.islower() or c_clean.isupper() else c_clean
    if "," not in c_low and " " not in c_low and len(c_low) >= 4:
        close = difflib.get_close_matches(c_low, [w for w in WHOLE_CITY_NAMES if w != "near me"], n=1, cutoff=0.82)
        if close:
            return CANONICAL_CITY_MAP.get(close[0], close[0].title())
    return ""


def _resolve_sublocality_in_phrase(phrase: str) -> Tuple[str, str]:
    """Detect any known Indian sub-locality or landmark inside a phrase (e.g.
    `"Phoenix Marketcity, Whitefield"` -> `("Bengaluru", "Whitefield, Bengaluru")`).
    """
    p_low = phrase.strip().lower().replace('"', "")
    if not p_low:
        return "", ""
    if p_low in SUBLOCALITY_TO_CITY_MAP:
        return SUBLOCALITY_TO_CITY_MAP[p_low]
    for seg in [s.strip() for s in p_low.split(",") if s.strip()]:
        if seg in SUBLOCALITY_TO_CITY_MAP:
            return SUBLOCALITY_TO_CITY_MAP[seg]
    toks = [t.strip(".,-()") for t in p_low.replace(",", " ").split() if t.strip(".,-()")]
    for n in (3, 2, 1):
        for i in range(len(toks) - n, -1, -1):
            cand = " ".join(toks[i : i + n])
            if cand in SUBLOCALITY_TO_CITY_MAP:
                return SUBLOCALITY_TO_CITY_MAP[cand]
    return "", ""


def _recover_city_from_query_text(q_low: str) -> str:
    """Fallback city recovery when an LLM omits `city` even though the query contains an
    unambiguous Indian city, city abbreviation (`lko`, `blr`, `banglore`), or landmark (`phoenix palladium`).
    """
    sub_city, _ = _resolve_sublocality_in_phrase(q_low)
    if sub_city:
        return sub_city
    toks = [t.strip(".,-?!/()") for t in q_low.replace(",", " ").split() if t.strip(".,-?!/()")]
    for n in (2, 1):
        for i in range(len(toks) - n, -1, -1):
            cand = " ".join(toks[i : i + n])
            if cand in SAFE_QUERY_CITY_TOKENS:
                return SAFE_QUERY_CITY_TOKENS[cand]
    preps = {"in", "at", "near", "around", "from", "lo", "mein"}
    for idx, tok in enumerate(toks[:-1]):
        if tok in preps:
            nxt = toks[idx + 1]
            resolved = _resolve_known_city(nxt)
            if resolved:
                return resolved
    return ""


def _normalize_city_and_locality(
    raw_city: Any, raw_locality: Any, rt: str
) -> Tuple[str, str]:
    """Canonicalize LLM-emitted `city` and `locality` slot values while stripping non-geographic
    phrases like `"Imax this Friday, Pune"` down to `canon_city="Pune", final_locality=""`.
    """
    canon_city = ""
    inferred_locality = ""

    if isinstance(raw_city, str) and raw_city.strip():
        c_clean = raw_city.strip().replace('"', "")
        c_low = c_clean.lower()
        if c_low in SUBLOCALITY_TO_CITY_MAP:
            canon_city, inferred_locality = SUBLOCALITY_TO_CITY_MAP[c_low]
        else:
            direct_city = _resolve_known_city(c_clean)
            if direct_city:
                canon_city = direct_city
            elif "," in c_clean:
                parts = [p.strip() for p in c_clean.split(",") if p.strip()]
                head_raw, tail_raw = parts[0], parts[-1]
                sub_city, sub_loc = _resolve_sublocality_in_phrase(c_clean)
                if sub_city:
                    canon_city, inferred_locality = sub_city, sub_loc
                else:
                    tail_city = _resolve_known_city(tail_raw)
                    if tail_city:
                        canon_city = tail_city
                        if _is_valid_sublocality_phrase(head_raw):
                            inferred_locality = f"{head_raw.title() if head_raw.islower() else head_raw}, {canon_city}"
            else:
                tokens = c_clean.split()
                if len(tokens) >= 2:
                    tail_city = _resolve_known_city(tokens[-1])
                    if tail_city:
                        canon_city = tail_city
                        area_w = " ".join(tokens[:-1])
                        if area_w.lower() in SUBLOCALITY_TO_CITY_MAP:
                            canon_city, inferred_locality = SUBLOCALITY_TO_CITY_MAP[area_w.lower()]
                        elif _is_valid_sublocality_phrase(area_w):
                            inferred_locality = f"{area_w.title() if area_w.islower() else area_w}, {canon_city}"
                    else:
                        sub_city, sub_loc = _resolve_sublocality_in_phrase(c_clean)
                        if sub_city:
                            canon_city, inferred_locality = sub_city, sub_loc

    final_locality = ""
    cand_loc = raw_locality.strip().replace('"', "") if isinstance(raw_locality, str) and raw_locality.strip() else ""
    if cand_loc:
        if not canon_city:
            loc_as_city = _resolve_known_city(cand_loc.split(",")[-1].strip())
            if loc_as_city:
                canon_city = loc_as_city
            else:
                sub_city, sub_loc = _resolve_sublocality_in_phrase(cand_loc)
                if sub_city:
                    canon_city, inferred_locality = sub_city, sub_loc
        if rt not in ("venue", "artist", "offer"):
            loc_head_raw = cand_loc.split(",")[0].strip()
            loc_base = loc_head_raw.lower()
            if loc_base in SUBLOCALITY_TO_CITY_MAP:
                mapped_city, mapped_loc = SUBLOCALITY_TO_CITY_MAP[loc_base]
                if not canon_city:
                    canon_city = mapped_city
                final_locality = mapped_loc
            elif _is_valid_sublocality_phrase(loc_head_raw):
                loc_head_fmt = loc_head_raw.title() if loc_head_raw.islower() else loc_head_raw
                if canon_city:
                    final_locality = f"{loc_head_fmt}, {canon_city}"
                elif "," in cand_loc:
                    tail_c = _resolve_known_city(cand_loc.split(",")[-1].strip())
                    if tail_c:
                        canon_city = tail_c
                        final_locality = f"{loc_head_fmt}, {canon_city}"

    if rt not in ("venue", "artist", "offer") and not final_locality and inferred_locality:
        final_locality = inferred_locality

    return canon_city, final_locality


def _detect_query_format_keys(q_low: str) -> List[str]:
    """Detect explicit movie projection/audio format tokens mentioned directly in `q_low`."""
    import re

    found: List[str] = []
    if re.search(r"\bimax\s*[-]?\s*3d\b", q_low):
        found.append("IMAX 3D")
    elif re.search(r"\bimax\b", q_low):
        found.append("IMAX")
    if re.search(r"\b(4dx|4d|mx4d)\b", q_low):
        found.append("4DX")
    if re.search(r"\bdolby\b", q_low):
        found.append("DOLBY")
    if re.search(r"\bscreen\s*x\b|\bscreenx\b", q_low):
        found.append("SCREENX")
    if re.search(r"\bepiq\b", q_low):
        found.append("EPIQ")
    if re.search(r"\b3d\b", q_low) and "IMAX 3D" not in found and not any(w in q_low for w in ("workshop", "printing", "art")):
        found.append("3D")
    if re.search(r"\b2d\b", q_low):
        found.append("2D")
    return found


def _normalize_format_slot(
    raw_fmt: Any,
    app_key: str = DEFAULT_SEARCH_APP,
    q_low: str = "",
) -> Tuple[Optional[str], List[str]]:
    """Compile single or multi-format slots (e.g. `['IMAX','4DX']` or `'IMAX 3D'`) into a schema-valid
    `formats: ANY(...)` condition and return `(condition_str, canonical_keys)`.
    """
    import re

    raw_items: List[str] = []
    if isinstance(raw_fmt, str) and raw_fmt.strip():
        raw_items = [p.strip() for p in re.split(r"[,/|]|\bor\b|\band\b", raw_fmt, flags=re.IGNORECASE) if p.strip()]
    elif isinstance(raw_fmt, list):
        for x in raw_fmt:
            if isinstance(x, str) and x.strip():
                raw_items.extend(p.strip() for p in re.split(r"[,/|]|\bor\b", x, flags=re.IGNORECASE) if p.strip())

    matched_keys: List[str] = []
    ordered_keys = ("IMAX 3D", "4DX", "MX4D", "4D", "IMAX", "DOLBY ATMOS", "DOLBY", "SCREENX", "SCREEN X", "ICE", "EPIQ", "3D", "2D")
    for item in raw_items:
        up = item.upper().replace("-", " ").strip()
        if up in FORMAT_VALUES_V3 and up not in matched_keys:
            matched_keys.append(up)
            continue
        for key in ordered_keys:
            if key in up.split() or up == key or up.startswith(key + " "):
                if key not in matched_keys:
                    matched_keys.append(key)
                break

    q_fmt_keys = _detect_query_format_keys(q_low) if q_low else []
    # Enforce strict `IMAX 3D` when user explicitly asked for `imax 3d` (never widen to `IMAX 2D`)
    if "IMAX 3D" in q_fmt_keys and "IMAX" in matched_keys:
        matched_keys = ["IMAX 3D" if k == "IMAX" else k for k in matched_keys]
    for qfk in q_fmt_keys:
        if qfk not in matched_keys and not (qfk == "IMAX" and "IMAX 3D" in matched_keys):
            matched_keys.append(qfk)

    if not matched_keys:
        return None, []

    if app_key == "v3":
        merged_vals: List[str] = []
        for k in matched_keys:
            for v in FORMAT_VALUES_V3.get(k, []):
                if v not in merged_vals:
                    merged_vals.append(v)
        if not merged_vals:
            return None, []
        return f"formats: {_quote_any(merged_vals)}", matched_keys

    v2_clauses = [FORMAT_SPEC_MAP_V2[k] for k in matched_keys if k in FORMAT_SPEC_MAP_V2]
    if not v2_clauses:
        return None, []
    return " OR ".join(dict.fromkeys(v2_clauses)), matched_keys


CITY_QUERY_STEMS_BY_CANON: Dict[str, List[str]] = {}
for _alias, _canon in CANONICAL_CITY_MAP.items():
    CITY_QUERY_STEMS_BY_CANON.setdefault(_canon, []).append(_alias)
for _subloc, (_p_city, _) in SUBLOCALITY_TO_CITY_MAP.items():
    CITY_QUERY_STEMS_BY_CANON.setdefault(_p_city, []).append(_subloc)

PRICE_GROUNDING_WORDS = (
    "cheap", "budget", "sasta", "saste", "free", "under", "below", "less", "price", "cost",
    "rs", "inr", "₹", "afford", "within", "max", "upto", "up to", "ke andar",
)

QUERY_FILLER_PHRASES = (
    "this_weekend", "this weekend", "this_friday", "this friday", "this_sunday", "this sunday",
    "this_saturday", "this saturday", "this_week", "this week", "next_weekend", "next weekend",
    "next_week", "next week", "next_month", "next month", "this_month", "this month",
    "ee weekend", "is weekend", "iss weekend", "right now",
)
QUERY_FILLER_TOKENS = {
    "trending", "popular", "today", "tonight", "tomorrow", "upcoming", "aaj", "kal", "ivattu", "udya",
}


def _is_city_grounded(canon_city: str, raw_city: Any, q_low: str) -> bool:
    if not canon_city:
        return False
    if "near me" in q_low and canon_city == "Mumbai":
        return True
    q_words = {w.strip(".,-?!/()") for w in q_low.replace(",", " ").split() if w.strip(".,-?!/()")}
    for alias in CITY_QUERY_STEMS_BY_CANON.get(canon_city, []):
        if len(alias) <= 3:
            if alias in q_words:
                return True
        elif alias in q_low:
            return True
    for candidate in (canon_city.lower(), str(raw_city or "").lower()):
        cleaned = candidate.replace("(", " ").replace(")", " ").replace(",", " ")
        for tok in cleaned.split():
            if len(tok) >= 3 and tok in q_low:
                return True
            if len(tok) >= 6 and (tok[:4] in q_low or tok[-5:] in q_low):
                return True
    return False


def _is_locality_grounded(clean_locality: str, q_low: str) -> bool:
    if not clean_locality:
        return False
    loc_head = clean_locality.split(",")[0].strip()
    if not _is_valid_sublocality_phrase(loc_head):
        return False
    loc_low = loc_head.lower()
    if loc_low in q_low:
        return True
    for tok in loc_low.split():
        if len(tok) >= 4 and tok not in WHOLE_CITY_NAMES and tok not in FORBIDDEN_LOCALITY_WORDS and tok in q_low:
            return True
    return False


def _is_lang_grounded(lang_canon: str, q_low: str) -> bool:
    stems = LANG_GROUNDING_STEMS.get(lang_canon, (lang_canon.lower(),))
    return any(s in q_low for s in stems)


def _is_genre_grounded(genre_canon: str, q_low: str) -> bool:
    stems = GENRE_GROUNDING_STEMS.get(genre_canon, (genre_canon.lower(),))
    return any(s in q_low for s in stems)


def _is_artist_grounded(artist_clean: str, q_low: str) -> bool:
    for nick, full_name in ARTIST_NICKNAME_MAP.items():
        if full_name.lower() == artist_clean.lower() and nick in q_low:
            return True
    tokens = [t.strip(".,-()") for t in artist_clean.lower().split() if t.strip(".,-()")]
    if not tokens or all(t in FORBIDDEN_LOCALITY_WORDS for t in tokens):
        return False
    return any(len(t) >= 3 and t not in FORBIDDEN_LOCALITY_WORDS and t in q_low for t in tokens)


OFFER_GROUNDING_WORDS = (
    "offer", "offers", "discount", "discounts", "coupon", "coupons", "promo", "code",
    "cashback", "bogo", "buy 1 get 1", "buy one get one", "card", "credit", "debit",
    "bank", "hdfc", "icici", "sbi", "axis", "kotak", "amex", "rupay", "paytm", "upi",
    "voucher", "deal", "deals",
)

ANY_TEMPORAL_GROUNDING_WORDS = (
    "today", "tonight", "tonite", "tomorrow", "tmrw", "friday", "saturday", "sunday",
    "monday", "tuesday", "wednesday", "thursday", "weekend", "week", "month",
    "right now", "now", "releasing", "upcoming", "2025", "2026", "2027",
    "oct", "october", "nov", "november", "dec", "december",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
    "aaj", "kal", "ivattu", "udya", "naale", "repu", "shonibar", "robbar", "ravivar", "shanivar",
    "nye", "new year", "gandhi jayanti", "diwali", "christmas",
)

WHEN_TOKEN_GROUNDING: Dict[str, Tuple[str, ...]] = {
    "today": ("today", "now", "right now", "aaj", "ivattu", "eroju", "inniku"),
    "tonight": ("tonight", "tonite", "night", "today", "aaj", "raat", "shaam"),
    "right_now": ("right now", "now", "week"),
    "now": ("now", "right now", "today", "week"),
    "tomorrow": ("tomorrow", "tmrw", "kal", "udya", "naale", "repu", "naalai"),
    "this_friday": ("friday", "fri"),
    "friday": ("friday", "fri"),
    "saturday": ("saturday", "sat", "shonibar", "shanivar", "weekend"),
    "sunday": ("sunday", "sun", "robbar", "ravivar", "weekend"),
    "this_weekend": ("weekend", "saturday", "sunday", "sat", "sun", "shonibar", "robbar", "shanivar", "ravivar"),
    "weekend": ("weekend", "saturday", "sunday", "sat", "sun", "shonibar", "robbar", "shanivar", "ravivar"),
    "this_week": ("week", "right now", "now"),
    "current_week": ("week", "right now", "now"),
    "week": ("week",),
    "next_week": ("week",),
    "this_month": ("month",),
    "month": ("month",),
    "next_month": ("month", "november", "december"),
    "november": ("november", "nov"),
    "december": ("december", "dec"),
    "nye": ("nye", "new year", "31st dec", "dec 31"),
    "new_year": ("nye", "new year", "31st dec", "dec 31"),
    "gandhi_jayanti": ("gandhi jayanti", "2nd oct", "oct 2"),
}

TIME_SLOT_GROUNDING: Dict[str, Tuple[str, ...]] = {
    "morning": ("morning", "am", "early", "subah", "breakfast"),
    "afternoon": ("afternoon", "matinee", "noon", "pm", "dopahar", "lunch"),
    "evening": ("evening", "tonight", "tonite", "pm", "sunset", "shaam", "sanjh"),
    "night": ("night", "tonight", "tonite", "late", "pm", "raat", "midnight"),
}


def _is_offer_grounded(q_low: str) -> bool:
    return any(w in q_low for w in OFFER_GROUNDING_WORDS)


def _is_when_grounded(when_tok: str, q_low: str) -> bool:
    stems = WHEN_TOKEN_GROUNDING.get(when_tok)
    if not stems:
        return any(w in q_low for w in ANY_TEMPORAL_GROUNDING_WORDS)
    q_words = set(q_low.replace(",", " ").split())
    return any((s in q_words if len(s) <= 3 else s in q_low) for s in stems)


def _is_time_slot_grounded(slot_tok: str, q_low: str) -> bool:
    stems = TIME_SLOT_GROUNDING.get(slot_tok, (slot_tok,))
    q_words = set(q_low.replace(",", " ").split())
    return any((s in q_words if len(s) <= 2 else s in q_low) for s in stems)


def _is_price_grounded(q_low: str) -> bool:
    import re

    if any(w in q_low for w in PRICE_GROUNDING_WORDS):
        return True
    return bool(re.search(r"\b(?:between|from)\s+\d+\s+(?:and|to|-)\s+\d+\b|\b(?:above|over)\s+\d+\b", q_low))


def _coerce_price_int(val: Any, q_low: str) -> Optional[int]:
    import re

    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return max(0, int(val))
    if isinstance(val, str) and val.strip():
        s_low = val.strip().lower()
        if "free" in s_low and "free" in q_low:
            return 0
        m = re.search(r"\d+", s_low.replace(",", ""))
        if m:
            return int(m.group(0))
    return None


def _parse_price_bounds(slots: Dict[str, Any], q_low: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse `(min_price, max_price)` INR bounds from LLM slots and query cues (`between M and N`, `under N`, `above M`, `cheapest`, `free`)."""
    import re

    if not _is_price_grounded(q_low):
        return None, None

    # 1) Explicit "between M and N" / "from M to N" range in query (excluding ordinal dates like "10th")
    m_rng = re.search(r"\b(?:between|from)\s+(\d+)\s+(?:and|to|-)\s+(\d+)\b", q_low)
    if m_rng:
        lo, hi = int(m_rng.group(1)), int(m_rng.group(2))
        return (min(lo, hi), max(lo, hi))

    raw_min = slots.get("min_price") if slots.get("min_price") is not None else slots.get("minPrice")
    raw_max = (
        slots.get("max_price")
        if slots.get("max_price") is not None
        else (slots.get("maxPrice") if slots.get("maxPrice") is not None else slots.get("price"))
    )
    if raw_max is None:
        raw_max = slots.get("budget") if slots.get("budget") is not None else slots.get("under")

    # Handle array `[1000, 3000]` or string `"1000-3000"` in `price`
    if isinstance(raw_max, list) and len(raw_max) == 2:
        p0 = _coerce_price_int(raw_max[0], q_low)
        p1 = _coerce_price_int(raw_max[1], q_low)
        if p0 is not None and p1 is not None:
            return min(p0, p1), max(p0, p1)
    if isinstance(raw_max, str):
        m_str_rng = re.search(r"(\d+)\s*(?:-|to|and)\s*(\d+)", raw_max.replace(",", ""), flags=re.IGNORECASE)
        if m_str_rng:
            p0, p1 = int(m_str_rng.group(1)), int(m_str_rng.group(2))
            return min(p0, p1), max(p0, p1)

    min_p = _coerce_price_int(raw_min, q_low)
    max_p = _coerce_price_int(raw_max, q_low)

    has_lower_cue = bool(re.search(r"\b(?:between|from|above|over|more\s+than|starting\s+at)\b", q_low))
    # If a small LLM placed a ceiling ("under 1500") inside `min_price` with no `max_price` and no lower-bound cue, treat it as `max_p`
    if min_p is not None and max_p is None and not has_lower_cue:
        max_p, min_p = min_p, None

    if max_p is None:
        m_q = re.search(r"(?:under|below|within|upto|up\s+to)\s+(\d+)|(\d+)\s+ke\s+andar", q_low)
        if m_q:
            max_p = int(m_q.group(1) or m_q.group(2))
        elif "free" in q_low.split():
            max_p = 0
        elif any(w in q_low for w in ("cheapest", "sasta", "saste")):
            max_p = 250

    if min_p is None and has_lower_cue:
        m_lo = re.search(r"\b(?:above|over|more\s+than|starting\s+at)\s+(\d+)\b", q_low)
        if m_lo:
            min_p = int(m_lo.group(1))

    if min_p is not None and max_p is not None and min_p > max_p:
        min_p, max_p = max_p, min_p

    return min_p, max_p


def _parse_price_slot(slots: Dict[str, Any], q_low: str) -> Optional[int]:
    """Backwards-compatible helper returning ceiling price."""
    min_p, max_p = _parse_price_bounds(slots, q_low)
    return max_p if max_p is not None else min_p


def _resolve_temporal_ranges(
    slots: Dict[str, Any],
    rt: str,
    dctx: Dict[str, str],
    q_low: str,
    has_show_context: bool = False,
) -> Tuple[Optional[List[str]], Optional[List[str]], List[str]]:
    """Resolve explicit `['YYYY-MM-DD','YYYY-MM-DD']` ranges or semantic `when`/`rel` tokens
    into `(rel_range, show_range, extra_time_slots)`, grounded against temporal cues in `q_low`.
    """
    rel_range: Optional[List[str]] = None
    show_range: Optional[List[str]] = None
    extra_slots: List[str] = []

    has_any_temporal = any(w in q_low for w in ANY_TEMPORAL_GROUNDING_WORDS)

    raw_rel_r = slots.get("rel_range")
    if (
        has_any_temporal
        and isinstance(raw_rel_r, list)
        and len(raw_rel_r) == 2
        and all(isinstance(x, str) and len(x) == 10 for x in raw_rel_r)
    ):
        rel_range = [raw_rel_r[0], raw_rel_r[1]]

    raw_show_r = slots.get("show_range")
    if (
        has_any_temporal
        and isinstance(raw_show_r, list)
        and len(raw_show_r) == 2
        and all(isinstance(x, str) and len(x) == 10 for x in raw_show_r)
    ):
        show_range = [raw_show_r[0], raw_show_r[1]]

    raw_rel_tok = slots.get("rel") if slots.get("rel") is not None else slots.get("release_date")
    is_rel_flag = (
        raw_rel_tok is True
        or (isinstance(raw_rel_tok, str) and raw_rel_tok.strip().lower() == "true")
    ) and any(w in q_low for w in ("releas", "upcoming", "new movie", "coming soon"))
    rel_str = (
        raw_rel_tok.strip().lower().replace(" ", "_").replace("-", "_")
        if isinstance(raw_rel_tok, str) and raw_rel_tok.strip().lower() != "true"
        else ""
    )
    when_str = (
        str(slots.get("when") or "").strip().lower().replace(" ", "_").replace("-", "_")
    )
    if not when_str and has_any_temporal:
        if "sunday" in q_low or "robbar" in q_low or "ravivar" in q_low:
            when_str = "sunday"
        elif "saturday" in q_low or "shonibar" in q_low or "shanivar" in q_low:
            when_str = "saturday"
        elif "ee weekend" in q_low or "is weekend" in q_low or "this weekend" in q_low:
            when_str = "this_weekend"
        elif "november" in q_low:
            when_str = "november"
        elif "nye" in q_low.split() or "new year" in q_low:
            when_str = "nye"
        elif "gandhi jayanti" in q_low:
            when_str = "gandhi_jayanti"

    yr = dctx["today_iso"][:4]
    next_yr = str(int(yr) + 1)
    token_to_window: Dict[str, List[str]] = {
        "today": [dctx["today_iso"], dctx["today_iso"]],
        "tonight": [dctx["today_iso"], dctx["today_iso"]],
        "right_now": [dctx["today_iso"], dctx["week_end_iso"]],
        "now": [dctx["today_iso"], dctx["week_end_iso"]],
        "tomorrow": [dctx["tomorrow_iso"], dctx["tomorrow_iso"]],
        "this_friday": [dctx["upcoming_friday_iso"], dctx["weekend_sun_iso"]],
        "friday": [dctx["upcoming_friday_iso"], dctx["weekend_sun_iso"]],
        "saturday": [dctx["weekend_sat_iso"], dctx["weekend_sat_iso"]],
        "sunday": [dctx["weekend_sun_iso"], dctx["weekend_sun_iso"]],
        "this_weekend": [dctx["weekend_sat_iso"], dctx["weekend_sun_iso"]],
        "weekend": [dctx["weekend_sat_iso"], dctx["weekend_sun_iso"]],
        "this_week": [dctx["today_iso"], dctx["week_end_iso"]],
        "current_week": [dctx["today_iso"], dctx["week_end_iso"]],
        "week": [dctx["today_iso"], dctx["week_end_iso"]],
        "next_week": [dctx["next_week_start_iso"], dctx["next_week_end_iso"]],
        "this_month": [dctx["today_iso"], dctx["month_end_iso"]],
        "month": [dctx["today_iso"], dctx["month_end_iso"]],
        "next_month": [dctx["next_month_start_iso"], dctx["next_month_end_iso"]],
        "november": [f"{yr}-11-01", f"{yr}-11-30"],
        "december": [f"{yr}-12-01", f"{yr}-12-31"],
        "nye": [f"{yr}-12-31", f"{next_yr}-01-01"],
        "new_year": [f"{yr}-12-31", f"{next_yr}-01-01"],
        "gandhi_jayanti": [f"{yr}-10-02", f"{yr}-10-02"],
    }

    if rel_str in token_to_window and _is_when_grounded(rel_str, q_low) and rel_range is None:
        rel_range = token_to_window[rel_str]

    if when_str == "tonight" and _is_when_grounded("tonight", q_low):
        extra_slots = ["evening", "night"]
    elif any(p in q_low for p in ("after 9 pm", "after 10 pm", "late night")):
        extra_slots = ["night"]

    if when_str in token_to_window and _is_when_grounded(when_str, q_low):
        win = token_to_window[when_str]
        is_release_only = is_rel_flag or (
            rt == "movie"
            and when_str in ("this_friday", "friday", "next_month")
            and not has_show_context
        )
        if is_release_only:
            if rel_range is None:
                rel_range = win
        else:
            if show_range is None:
                show_range = win
            if rt == "movie" and when_str in ("this_week", "current_week", "week", "right_now", "now") and rel_range is None:
                rel_range = win

    return rel_range, show_range, extra_slots


def _iter_slot_strings(val: Any) -> List[str]:
    if isinstance(val, str) and val.strip():
        return [p.strip() for p in val.split(",") if p.strip()]
    if isinstance(val, list):
        out: List[str] = []
        for item in val:
            if isinstance(item, str) and item.strip():
                out.extend(p.strip() for p in item.split(",") if p.strip())
        return out
    return []


PURE_VENUE_TARGET_WORDS = {
    "cinema", "cinemas", "theatre", "theatres", "theater", "theaters",
    "multiplex", "multiplexes", "pvr", "inox", "cinepolis", "moviemax", "miraj",
}
NON_PURE_VENUE_SIGNAL_WORDS = {
    "movie", "movies", "film", "films", "padam", "show", "shows", "showtime", "showtimes",
    "running", "playing", "releasing", "release", "upcoming", "trending", "popular",
    "top", "rated", "play", "plays", "natak", "natok", "drama", "comedy", "standup",
    "concert", "concerts", "gig", "gigs", "workshop", "workshops", "garba", "dandiya",
    "pass", "passes", "ticket", "tickets", "tkt", "offer", "offers", "discount", "discounts",
    "coupon", "coupons", "card", "credit", "debit", "hdfc", "icici", "sbi", "axis",
    "cheap", "cheaper", "cheapest", "budget", "sasta", "free", "under", "below",
    "imax", "4dx", "4d", "mx4d", "3d", "2d", "dolby", "screenx", "epiq",
    "recliner", "insignia", "luxe", "seat", "seats",
    "today", "tonight", "tomorrow", "friday", "saturday", "sunday",
    "weekend", "week", "month", "now", "ivattu", "aaj", "kal", "udya", "timings",
}


def _has_pure_venue_intent(q_low: str) -> bool:
    """Return True when a query asks for cinema halls / theatres / multiplexes / cinema chains
    with zero movie, event, language, genre, format, seating, price, offer, or temporal constraints.
    """
    q_words = {
        w.strip(".,-?!")
        for w in q_low.replace(",", " ").split()
        if w.strip(".,-?!")
    }
    if not (q_words & PURE_VENUE_TARGET_WORDS):
        return False
    if q_words & NON_PURE_VENUE_SIGNAL_WORDS:
        return False
    if _is_offer_grounded(q_low) or _is_price_grounded(q_low):
        return False
    if any(_is_lang_grounded(lc, q_low) for lc in VALID_LANGUAGES):
        return False
    if any(gc in ("Comedy", "Action", "Horror", "Thriller", "Romantic", "Sci-Fi", "Animation") and _is_genre_grounded(gc, q_low) for gc in VALID_GENRES):
        return False
    return True


def _build_region_boost_and_geo_city(canon_city: str, q_low: str, rt: str, is_v3: bool) -> Tuple[List[Dict[str, Any]], str, str]:
    """Build BQ-exact `regions` conditionBoostSpecs (including sub-city NCR/Mumbai tags like
    `"Gurgaon"`, `"Gurugram (Gurgaon)"`, `"Noida"`, `"Navi Mumbai"`, `"Thane"`) and canonical `geo_city`.
    """
    geo_city = GEO_CITY_NAME_MAP.get(canon_city, canon_city)
    if not is_v3:
        if rt == "venue":
            c_cond = f'regionName: ANY("{canon_city}") OR address: ANY("{canon_city}")'
        else:
            c_cond = f'regionName: ANY("{canon_city}") OR {LEGACY_V2_REGIONS_FIELD}: ANY("{canon_city}")'
        return [{"condition": c_cond, "boost": 0.9}], f"({c_cond})", geo_city

    if canon_city == "National Capital Region (NCR)":
        if any(w in q_low for w in ("gurgaon", "gurugram", "ggn", "cyber hub", "cyberhub", "sector 29")):
            c_cond = 'regions: ANY("Gurgaon", "Gurugram (Gurgaon)", "National Capital Region (NCR)")'
            return [{"condition": c_cond, "boost": 0.95}], c_cond, "Gurgaon"
        if "noida" in q_low:
            c_cond = 'regions: ANY("Noida", "National Capital Region (NCR)")'
            return [{"condition": c_cond, "boost": 0.95}], c_cond, "Noida"
        c_cond = 'regions: ANY("National Capital Region (NCR)", "Delhi", "Delhi-NCR")'
        return [{"condition": c_cond, "boost": 0.92}], c_cond, "New Delhi"

    if canon_city == "Mumbai":
        if any(w in q_low for w in ("navi mumbai", "vashi", "kharghar", "nerul", "belapur")):
            c_cond = 'regions: ANY("Navi Mumbai", "Mumbai")'
            return [{"condition": c_cond, "boost": 0.94}], c_cond, "Navi Mumbai"
        if "thane" in q_low:
            c_cond = 'regions: ANY("Thane", "Mumbai")'
            return [{"condition": c_cond, "boost": 0.94}], c_cond, "Thane"

    c_cond = f'regions: ANY("{canon_city}")'
    return [{"condition": c_cond, "boost": 0.92}], c_cond, geo_city


def _clean_rewritten_query(
    raw_q: str,
    original_query: str,
    rt: str,
    canon_city: str,
    matched_langs: List[str],
    fmt_keys: List[str],
    genres_list: List[str],
    cats_list: List[str],
    resolved_artist: str,
    show_range: Optional[List[str]],
    rel_range: Optional[List[str]],
    no_movie: bool,
) -> str:
    """Clean `rewritten_query` so negated clauses, zero-hit seating jargon, and over-stripped
    generic rewrites never degrade Vertex AI Search hybrid retrieval.
    """
    import re

    q_low = original_query.strip().lower()
    rewritten = (raw_q or original_query).strip() or original_query.strip()

    # Guard against tiny models dropping core entity tokens (e.g. `'samay raina indias got letent show'` -> `'Letent show'`)
    orig_core = [
        w.strip(".,-?!")
        for w in q_low.split()
        if len(w.strip(".,-?!")) >= 4 and w.strip(".,-?!") not in FORBIDDEN_LOCALITY_WORDS and w.strip(".,-?!") not in WHOLE_CITY_NAMES
    ]
    if len(orig_core) >= 2:
        kept_core = [w for w in orig_core if w in rewritten.lower()]
        if len(kept_core) < max(1, len(orig_core) // 2):
            rewritten = original_query.strip()

    # 1) Strip explicit negative clauses so BM25/vector search does not retrieve the negated genre/type
    neg_patterns = (
        r"\bwithout\s+any\s+\w+",
        r"\bwithout\s+\w+",
        r"\bbut\s+not\s+a\s+\w+",
        r"\bnot\s+a\s+(?:movie|movies|cartoon|match)\b",
        r"\bthat\s+isn'?t\s+a\s+movie\b",
        r"\bisn'?t\s+a\s+movie\b",
        r"\bother\s+than\s+movies?\b",
        r",?\s*match\s+nahi\b",
    )
    for pat in neg_patterns:
        rewritten = re.sub(pat, " ", rewritten, flags=re.IGNORECASE)
    for p_pat in (
        r"\b(?:between|from)\s+\d+\s+(?:and|to|-)\s+\d+\b",
        r"\b(?:under|below|within|upto|up\s+to|above|over)\s+(?:rs\.?\s*|inr\s*|₹\s*)?\d+\b",
        r"\b\d+\s+ke\s+andar\b",
    ):
        rewritten = re.sub(p_pat, " ", rewritten, flags=re.IGNORECASE)
    if rt == "event" and "movie" not in q_low and "film" not in q_low:
        rewritten = re.sub(r"\bmovies?\b", " ", rewritten, flags=re.IGNORECASE)
    rewritten = " ".join(rewritten.split())

    # 2) Cinema seating/experience/screen queries (`recliner seats`, `gold class or insignia`, `dolby atmos screens`)
    if "gold class" in q_low and "insignia" in q_low:
        return "PVR INOX movie"
    if "recliner" in q_low and rt == "movie" and rewritten.lower() in ("recliner", "recliner seats", "recliner seat"):
        return "movie cinema"
    if rt == "movie" and fmt_keys and re.search(r"\bscreens?\b", rewritten, flags=re.IGNORECASE):
        rewritten = re.sub(r"\bscreens?\b", "movie", rewritten, flags=re.IGNORECASE)

    # 3) Similarity queries (`comedians like zakir khan`) -> search target category, not person name
    if "comedians like" in q_low or "comedian like" in q_low:
        return "stand up comedy"

    # 4) Occasion/vibe expansions when compact local LLMs echo conversational phrasing verbatim
    if "parents" in q_low and "anniversary" in q_low and rt == "event":
        if any(w in rewritten.lower() for w in ("parents", "anniversary", "date night")):
            return "musical sufi theatre concert"
    if "watch with friends" in q_low and any(w in rewritten.lower() for w in ("friends", "watch")):
        return "comedy action movie stand up"
    if "college fest" in q_low and rewritten.lower() in ("college fest", "college fest near me"):
        return "college festival concert carnival"

    # 5) Romantic movie queries: strip `"for date night"` / `"date night"` so 2010 movie `'Date Night'` doesn't hijack #1
    if rt == "movie" and "Romantic" in genres_list and "date night" in rewritten.lower():
        rewritten = re.sub(r"\b(?:for\s+)?date\s+night\b", " ", rewritten, flags=re.IGNORECASE).strip()
        rewritten = " ".join(rewritten.split()) or "romantic romance movies"

    # 6) Availability queries (`seats still available for paradise tonight`): rescue dropped entity name
    if "available for " in q_low and rewritten.lower() in ("seats still available", "seats available", "tickets available", "available"):
        after_for = original_query.lower().split("available for ", 1)[1]
        for tw in ("tonight", "today", "tomorrow", "this weekend", "this friday"):
            after_for = after_for.replace(tw, " ")
        rescued_ent = " ".join(after_for.split()).strip()
        if rescued_ent:
            rewritten = rescued_ent.title()

    # 7) Friday release discovery (`movies releasing this friday`)
    if rt == "movie" and "releasing" in q_low and "friday" in q_low and rewritten.lower() in ("movies", "movie"):
        return "movies releasing friday"

    # 8) Strip pure temporal filler phrases/tokens on movie/event queries using strict word boundaries
    if rt in ("movie", "event") and (show_range or rel_range or any(tok in rewritten.lower().split() for tok in QUERY_FILLER_TOKENS)):
        cleaned_q = rewritten
        for phrase in QUERY_FILLER_PHRASES:
            cleaned_q = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned_q, flags=re.IGNORECASE)
        if canon_city:
            for c_stem in sorted(CITY_QUERY_STEMS_BY_CANON.get(canon_city, []), key=len, reverse=True):
                if len(c_stem) >= 4:
                    cleaned_q = re.sub(rf"\b(?:in|at|near|around)\s+{re.escape(c_stem)}\b", " ", cleaned_q, flags=re.IGNORECASE)
        kept_words = [w for w in cleaned_q.split() if w.lower() not in QUERY_FILLER_TOKENS]
        if kept_words:
            rewritten = " ".join(kept_words)
        else:
            rewritten = "movies" if rt == "movie" else "events"

    if no_movie and rewritten.lower() in ("", "something", "something after 9 pm"):
        rewritten = "events"

    # 9) Preserve resolved superstar name, explicit language, and explicit format keywords if over-stripped
    if resolved_artist and "like " not in q_low and resolved_artist.lower() not in rewritten.lower():
        first_tok = resolved_artist.split()[0].lower()
        if first_tok not in rewritten.lower():
            rewritten = f"{resolved_artist} {rewritten}".strip()
        elif "rajini" in q_low and "rajinikanth" not in rewritten.lower():
            rewritten = re.sub(r"\brajini\b", "Rajinikanth", rewritten, flags=re.IGNORECASE)
    if "padam" in rewritten.lower():
        rewritten = re.sub(r"\bpadam(?:\s+eppo\s+release)?\b", "movie", rewritten, flags=re.IGNORECASE)

    if matched_langs and rt in ("movie", "event"):
        primary_lang = matched_langs[0]
        if primary_lang.lower() not in rewritten.lower():
            # Append/prepend language when query is generic or code-mixed (`natak`, `hyderabad lo ee`, `Hanuman Ansh`)
            if len(rewritten.split()) <= 3 or " lo " in f" {q_low} ":
                if " lo " in f" {rewritten.lower()} ":
                    rewritten = f"{primary_lang} {'stand up comedy' if 'stand-up-comedy' in cats_list else ('movie' if rt == 'movie' else 'show')}"
                else:
                    rewritten = f"{primary_lang} {rewritten}".strip()

    if fmt_keys and rt == "movie":
        for fk in fmt_keys:
            short_fk = "4DX" if fk in ("4D", "MX4D") else fk
            if short_fk.lower() not in rewritten.lower() and fk.lower() not in rewritten.lower():
                rewritten = f"{rewritten} {short_fk}".strip()

    if "ind vs wi" in rewritten.lower():
        rewritten = "India vs West Indies cricket"

    return " ".join(rewritten.split()) or ("movies" if rt == "movie" else "events")


def serialize_llm_slots_to_boost_specs(
    slots: Dict[str, Any],
    original_query: str,
    app_key: str = DEFAULT_SEARCH_APP,
    trust_semantic_slots: bool = False,
) -> Tuple[str, List[Dict[str, Any]], str, str, str]:
    """Serialize structured JSON slots produced by any LLM (Cloud Gemini or Local Metal GPU MLX)
    into exact BigQuery-verified Vertex AI Search `conditionBoostSpecs`, city-only `geo_filter`
    (`location:GEO_DISTANCE("<City>", 10000)`), and `hard_filter` for `v3` or `v2`.
    """
    is_v3 = app_key == "v3"
    q_low = original_query.strip().lower()
    q_words = {w.strip(".,-?!/()") for w in q_low.replace(",", " ").split() if w.strip(".,-?!/()")}
    dctx = _date_context()
    boost_specs: List[Dict[str, Any]] = []
    hard_clauses: List[str] = []
    geo_filter = ""

    raw_rt = str(slots.get("type") or "").strip().lower()
    rt = TYPE_ALIAS_MAP.get(raw_rt, raw_rt)
    if rt not in ALL_RECORD_TYPES and any(vw in raw_rt for vw in PURE_VENUE_TARGET_WORDS) and _has_pure_venue_intent(q_low):
        rt = "venue"

    no_movie_cues = (
        "not a movie", "not movies", "isnt a movie", "isn't a movie", "no movie", "no movies",
        "without movie", "without movies", "without any movie", "other than movie",
        "besides movie", "except movie", "non movie",
    )
    no_event_cues = ("match nahi", "not a match", "not an event", "no event")
    no_movie = any(p in q_low for p in no_movie_cues) or (
        bool(slots.get("no_movie")) and any(w in q_words for w in ("not", "no", "without", "isnt", "isn't", "except", "other"))
    )
    no_event = any(p in q_low for p in no_event_cues) or (
        bool(slots.get("no_event")) and any(w in q_words for w in ("not", "no", "nahi", "without", "except"))
    )

    if rt == "offer" and not _is_offer_grounded(q_low):
        if any(w in q_low for w in ("movie", "movies", "film", "films", "imax", "4dx", "4d", "3d")):
            rt = "movie"
        elif any(w in q_low for w in ("pass", "passes", "garba", "concert", "comedy", "show", "shows", "play", "natak", "event", "events", "workshop")):
            rt = "event"
        else:
            rt = ""

    if no_movie and rt in ("", "movie"):
        rt = "event"
    if no_event and any(w in q_words for w in ("movie", "movies", "film", "films")):
        rt = "movie"

    # Format & cinema seating/experience intent ALWAYS targets `movie` (`venue` rows have `formats = []` in BQ)
    q_fmt_keys = _detect_query_format_keys(q_low)
    has_cinema_seating = any(
        p in q_low for p in ("recliner", "gold class", "insignia", "director's cut", "couple friendly movie", "couple seat")
    )
    if not no_movie and not _is_offer_grounded(q_low) and (q_fmt_keys or has_cinema_seating):
        rt = "movie"

    # Screenings (`open air movie screening`, `anime screenings`), `folk`, and `show dates` (without `movie`/`film`) are `event` records
    has_explicit_movie_word = bool(q_words & {"movie", "movies", "film", "films", "padam"})
    if not _is_offer_grounded(q_low) and (q_words & {"screening", "screenings"}):
        rt = "event"
    elif (
        not _is_offer_grounded(q_low)
        and not has_explicit_movie_word
        and not q_fmt_keys
        and not has_cinema_seating
        and ("show dates" in q_low or bool(q_words & {"folk", "sufi", "garba", "dandiya", "standup", "natak", "natok"}))
    ):
        rt = "event"

    if (
        rt == "event"
        and not no_movie
        and not _is_offer_grounded(q_low)
        and "screening" not in q_low
        and has_explicit_movie_word
    ):
        rt = "movie"

    if not no_movie and rt != "venue" and _has_pure_venue_intent(q_low):
        rt = "venue"

    # Fallback record_type inference if LLM omitted `type` (e.g. "trending movies this week", "sabse zyada book hone wali movie")
    if rt not in ALL_RECORD_TYPES and not _is_offer_grounded(q_low):
        if not no_movie and "screening" not in q_low and has_explicit_movie_word:
            rt = "movie"
        elif q_words & {"concert", "concerts", "standup", "comedy", "natak", "natok", "garba", "dandiya", "workshop", "workshops", "trek", "trekking", "fest", "festival", "t20", "cricket", "folk"}:
            rt = "event"
        elif any(p in q_low for p in ("parents", "anniversary")):
            rt = "event"

    # Resolve city & locality BEFORE finalizing `artist` vs `event` (`anuv jain kochi november` -> `event`)
    canon_city, _ = _normalize_city_and_locality(
        slots.get("city"), slots.get("locality"), rt
    )
    if canon_city and not _is_city_grounded(canon_city, slots.get("city") or slots.get("locality"), q_low):
        canon_city = ""
    if not canon_city and rt != "offer":
        recovered_city = _recover_city_from_query_text(q_low)
        if recovered_city and _is_city_grounded(recovered_city, recovered_city, q_low):
            canon_city = recovered_city

    # `artist` rows in BQ have no `regions` or `show_dates`; artist + city/date or "comedians like X" is an `event` (or `movie`)
    if rt == "artist":
        if "comedians like" in q_low or "comedian like" in q_low or "artists like" in q_low or "singers like" in q_low:
            rt = "event"
        elif has_explicit_movie_word:
            rt = "movie"
        elif canon_city or any(w in q_low for w in ANY_TEMPORAL_GROUNDING_WORDS):
            rt = "event"

    if rt in ALL_RECORD_TYPES and not (rt == "movie" and no_movie):
        rt_cond = f'record_type: ANY("{rt}")'
        boost_specs.append({"condition": rt_cond, "boost": 0.8})
        others = [t for t in ALL_RECORD_TYPES if t != rt]
        boost_specs.append({"condition": f"record_type: {_quote_any(others)}", "boost": -0.75})
        hard_clauses.append(rt_cond)
        # In BQ `ent_5stores_master_v3`, ONLY the 3,676 true cinema halls have `categories: ANY("movies")` (11,390 event/trek venues do not)
        if is_v3 and rt == "venue" and (q_words & PURE_VENUE_TARGET_WORDS):
            boost_specs.append({"condition": 'categories: ANY("movies")', "boost": 0.55})

    if no_movie:
        boost_specs.append({"condition": 'record_type: ANY("movie")', "boost": -0.9})
        hard_clauses.append('NOT (record_type: ANY("movie"))')
    if no_event:
        boost_specs.append({"condition": 'record_type: ANY("event")', "boost": -0.9})
        hard_clauses.append('NOT (record_type: ANY("event"))')

    geo_city_for_filter = ""
    if canon_city:
        reg_boosts, reg_hard, geo_city_for_filter = _build_region_boost_and_geo_city(canon_city, q_low, rt, is_v3)
        boost_specs.extend(reg_boosts)
        hard_clauses.append(reg_hard)

    raw_fmt_val = slots.get("fmt") if slots.get("fmt") is not None else slots.get("format")
    f_cond, fmt_keys = _normalize_format_slot(raw_fmt_val, app_key=app_key, q_low=q_low)

    known_non_tag_keys = {
        "type", "no_movie", "no_event", "city", "locality", "fmt", "format", "lang", "language",
        "genres", "genre", "cats", "categories", "category", "no_genre",
        "artist", "directed_by", "director", "actor",
        "price", "max_price", "min_price", "maxPrice", "minPrice", "budget", "under",
        "avail", "available", "when", "rel", "release_date", "rel_range", "show_range", "slots", "q",
        "rewritten_query", "geo_filter", "condition_boost_specs", "hard_filter",
    }
    extra_unmapped_strings: List[str] = []
    for k, v in slots.items():
        if k not in known_non_tag_keys:
            extra_unmapped_strings.extend(_iter_slot_strings(v))

    matched_langs: List[str] = []
    extra_genre_candidates: List[str] = []
    lang_inputs = _iter_slot_strings(slots.get("lang") if slots.get("lang") is not None else slots.get("language"))
    for l_item in lang_inputs:
        l_low = l_item.lower()
        if l_low in LANG_LOOKUP:
            lc = LANG_LOOKUP[l_low]
            if _is_lang_grounded(lc, q_low) and lc not in matched_langs:
                matched_langs.append(lc)
        else:
            if not f_cond:
                rescued_fmt, r_keys = _normalize_format_slot(l_item, app_key=app_key, q_low=q_low)
                if rescued_fmt and r_keys:
                    f_cond, fmt_keys = rescued_fmt, r_keys
            for tok in l_low.split():
                if tok in LANG_LOOKUP:
                    lc = LANG_LOOKUP[tok]
                    if _is_lang_grounded(lc, q_low) and lc not in matched_langs:
                        matched_langs.append(lc)
                if tok in GENRE_LOOKUP:
                    extra_genre_candidates.append(GENRE_LOOKUP[tok])

    # Recover explicitly stated language words in `q_low` if omitted by LLM (e.g. `malayalam movies with english subtitles`, `telugu stand up`)
    for lw_key, lw_canon in (
        ("hindi", "Hindi"), ("english", "English"), ("tamil", "Tamil"), ("telugu", "Telugu"),
        ("kannada", "Kannada"), ("malayalam", "Malayalam"), ("marathi", "Marathi"),
        ("bengali", "Bengali"), ("bangla", "Bengali"), ("punjabi", "Punjabi"),
        ("gujarati", "Gujarati"), ("odia", "Odia"), ("oriya", "Odia"), ("korean", "Korean"),
    ):
        if lw_key in q_words and lw_canon not in matched_langs:
            matched_langs.append(lw_canon)

    cat_inputs = (
        _iter_slot_strings(slots.get("cats"))
        + _iter_slot_strings(slots.get("categories"))
        + _iter_slot_strings(slots.get("category"))
    )
    if not f_cond:
        for cand_f in cat_inputs + extra_unmapped_strings:
            rescued_fmt, r_keys = _normalize_format_slot(cand_f, app_key=app_key, q_low=q_low)
            if rescued_fmt and r_keys:
                f_cond, fmt_keys = rescued_fmt, r_keys
                break

    if f_cond:
        boost_specs.append({"condition": f_cond, "boost": 0.92})
        hard_clauses.append(f"({f_cond})" if " OR " in f_cond else f_cond)
    elif is_v3 and rt == "movie" and ("gold class" in q_low or "insignia" in q_low):
        boost_specs.append({"condition": 'formats: ANY("IMAX 2D", "IMAX 3D", "DOLBY CINEMA 2D")', "boost": 0.72})

    genre_inputs = (
        _iter_slot_strings(slots.get("genres"))
        + _iter_slot_strings(slots.get("genre"))
        + extra_genre_candidates
        + extra_unmapped_strings
    )

    genres_list: List[str] = []
    for g_raw in genre_inputs:
        g_low = g_raw.lower()
        if g_low in GENRE_LOOKUP:
            gc = GENRE_LOOKUP[g_low]
            if (trust_semantic_slots or _is_genre_grounded(gc, q_low)) and rt in ("movie", "event", ""):
                for exp_g in GENRE_BQ_EXPANSION.get(gc, [gc]):
                    if exp_g not in genres_list:
                        genres_list.append(exp_g)
        elif g_low in LANG_LOOKUP and not matched_langs:
            lc = LANG_LOOKUP[g_low]
            if _is_lang_grounded(lc, q_low):
                matched_langs.append(lc)

    # Ensure explicit genre/typo words in `q_low` aren't missed when an LLM omits `genres`
    if rt in ("movie", "event", "") and "screening" not in q_low:
        for gw_tokens, gw_canon in (
            (("horor", "horror"), "Horror"),
            (("romantic", "romance", "romcom"), "Romantic"),
            (("thriller",), "Thriller"),
            (("folk",), "Folk"),
            (("sufi",), "Sufi"),
        ):
            if any(t in q_words for t in gw_tokens):
                if f"without {gw_canon.lower()}" not in q_low and f"without any {gw_canon.lower()}" not in q_low:
                    if gw_canon not in genres_list:
                        genres_list.append(gw_canon)
        if ("sci-fi" in q_low or "sci fi" in q_low) and "Sci-Fi" not in genres_list:
            genres_list.append("Sci-Fi")

    cats_list: List[str] = []
    for c_raw in cat_inputs:
        c_low = c_raw.lower()
        if c_low in CAT_LOOKUP and rt in ("event", ""):
            for mapped_c in CAT_LOOKUP[c_low]:
                if mapped_c not in cats_list:
                    cats_list.append(mapped_c)
        elif c_low in GENRE_LOOKUP:
            gc = GENRE_LOOKUP[c_low]
            if (trust_semantic_slots or _is_genre_grounded(gc, q_low)) and rt in ("movie", "event", ""):
                for exp_g in GENRE_BQ_EXPANSION.get(gc, [gc]):
                    if exp_g not in genres_list:
                        genres_list.append(exp_g)
        elif c_low in LANG_LOOKUP and not matched_langs:
            lc = LANG_LOOKUP[c_low]
            if _is_lang_grounded(lc, q_low):
                matched_langs.append(lc)

    if "comedians like" in q_low or "comedian like" in q_low:
        for sc in ("stand-up-comedy", "comedy-shows"):
            if sc not in cats_list:
                cats_list.append(sc)
    if "college fest" in q_low:
        for fc in ("festivals", "music-festivals", "Carnivals"):
            if fc not in cats_list:
                cats_list.append(fc)
    if "parents" in q_low and "anniversary" in q_low and rt == "event":
        for pg in ("Musical", "Sufi", "Classical"):
            if pg not in genres_list:
                genres_list.append(pg)
        for pc in ("plays", "theatre-plays", "music-shows", "concerts"):
            if pc not in cats_list:
                cats_list.append(pc)
    if "watch with friends" in q_low and not genres_list:
        genres_list.extend(["Comedy", "Action"])

    if matched_langs:
        is_conjunctive_dual_lang = (
            len(matched_langs) >= 2
            and ("subtitle" in q_low or " with " in q_low or " and " in q_low)
            and " or " not in q_low
        )
        if is_v3 and is_conjunctive_dual_lang:
            for idx_l, l_val in enumerate(matched_langs[:2]):
                vals_l = ["Odia", "Oriya"] if l_val == "Odia" else [l_val]
                l_c = f"languages: {_quote_any(vals_l)}"
                boost_specs.append({"condition": l_c, "boost": 0.75 if idx_l == 0 else 0.55})
                hard_clauses.append(l_c)
        else:
            expanded_v3_langs: List[str] = []
            v2_lang_clauses: List[str] = []
            for l_val in matched_langs:
                if l_val == "Odia":
                    for ov in ("Odia", "Oriya"):
                        if ov not in expanded_v3_langs:
                            expanded_v3_langs.append(ov)
                    v2_lang_clauses.append(
                        f'{LEGACY_V2_LANGUAGES_FIELD}: ANY("Odia", "Oriya") OR sessions.languages: ANY("Odia", "Oriya", "oriya", "odia")'
                    )
                else:
                    if l_val not in expanded_v3_langs:
                        expanded_v3_langs.append(l_val)
                    v2_lang_clauses.append(
                        f'{LEGACY_V2_LANGUAGES_FIELD}: ANY("{l_val}") OR sessions.languages: ANY("{l_val}")'
                    )
            l_cond = (
                f"languages: {_quote_any(expanded_v3_langs)}"
                if is_v3
                else " OR ".join(v2_lang_clauses)
            )
            boost_specs.append({"condition": l_cond, "boost": 0.75})
            hard_clauses.append(f"({l_cond})" if " OR " in l_cond else l_cond)

    if "comedy" in raw_rt and "Comedy" not in genres_list and _is_genre_grounded("Comedy", q_low):
        genres_list.append("Comedy")
    if "open mic" in raw_rt and "open-mic-comedy" not in cats_list:
        cats_list.extend(["open-mic-comedy", "open-mic-music-shows"])

    neg_genres: List[str] = []
    for ng_raw in _iter_slot_strings(slots.get("no_genre")):
        ng_canon = GENRE_LOOKUP.get(ng_raw.lower())
        if ng_canon and _is_genre_grounded(ng_canon, q_low) and ng_canon not in neg_genres:
            neg_genres.append(ng_canon)
    if ("without any horror" in q_low or "without horror" in q_low) and "Horror" not in neg_genres:
        neg_genres.append("Horror")
    if "not a cartoon" in q_low and "Animation" not in neg_genres:
        neg_genres.append("Animation")
    genres_list = [g for g in genres_list if g not in neg_genres]

    gc_parts: List[str] = []
    if genres_list:
        gc_parts.append(f"genres: {_quote_any(genres_list)}")
    if cats_list:
        gc_parts.append(f"categories: {_quote_any(cats_list)}")
    if gc_parts:
        gc_cond = " OR ".join(gc_parts)
        # Slightly softer genre weight when an explicit projection format (`IMAX`/`3D`) is also requested
        gc_boost = 0.52 if f_cond else 0.65
        boost_specs.append({"condition": gc_cond, "boost": gc_boost})
        hard_clauses.append(f"({gc_cond})")

    if neg_genres:
        ng_cond = f"genres: {_quote_any(neg_genres)}"
        boost_specs.append({"condition": ng_cond, "boost": -0.9})
        hard_clauses.append(f"NOT ({ng_cond})")

    raw_artist = (
        slots.get("artist")
        or slots.get("directed_by")
        or slots.get("director")
        or slots.get("actor")
    )
    resolved_artist = ""
    if "like " not in q_low and "similar to " not in q_low:
        if isinstance(raw_artist, str) and raw_artist.strip():
            cand_a = raw_artist.strip()
            mapped_a = ARTIST_NICKNAME_MAP.get(cand_a.lower())
            if mapped_a:
                cand_a = mapped_a
            elif cand_a.islower():
                cand_a = cand_a.title()
            if _is_artist_grounded(cand_a, q_low):
                resolved_artist = cand_a
        if not resolved_artist:
            for nick_k, full_a in ARTIST_NICKNAME_MAP.items():
                if (nick_k in q_words if " " not in nick_k else nick_k in q_low):
                    resolved_artist = full_a
                    break
        if resolved_artist:
            boost_specs.append({"condition": f'artists: ANY("{resolved_artist}")', "boost": 0.8})

    min_p, max_p = _parse_price_bounds(slots, q_low)
    has_grounded_price = (min_p is not None) or (max_p is not None)
    if min_p is not None and max_p is not None:
        p_cond = (
            f"min_price >= {min_p} AND min_price <= {max_p}"
            if is_v3
            else f"sessions.min_price >= {min_p} AND sessions.max_price <= {max_p}"
        )
        boost_specs.append({"condition": p_cond, "boost": 0.65})
        hard_clauses.append(p_cond)
    elif max_p is not None:
        p_cond = f"min_price <= {max_p}" if is_v3 else f"sessions.min_price <= {max_p}"
        boost_specs.append({"condition": p_cond, "boost": 0.65})
        hard_clauses.append(p_cond)
    elif min_p is not None:
        p_cond = f"min_price >= {min_p}" if is_v3 else f"sessions.min_price >= {min_p}"
        boost_specs.append({"condition": p_cond, "boost": 0.65})
        hard_clauses.append(p_cond)

    has_avail = bool(slots.get("avail") or slots.get("available")) or any(
        p in q_low for p in ("seats still available", "tickets left", "seats left", "not sold out")
    )
    if is_v3 and has_avail:
        boost_specs.append({"condition": 'is_sold_out = "false"', "boost": 0.45})

    raw_slots = slots.get("slots")
    t_slots = [
        s.lower()
        for s in (raw_slots if isinstance(raw_slots, list) else [])
        if isinstance(s, str) and s.lower() in VALID_TIME_SLOTS and _is_time_slot_grounded(s.lower(), q_low)
    ]

    has_show_ctx = bool(canon_city or f_cond or has_grounded_price or t_slots)
    rel_range, show_range, extra_slots = _resolve_temporal_ranges(
        slots, rt, dctx, q_low=q_low, has_show_context=has_show_ctx
    )

    is_catalog_new_release_movie = (
        rt == "movie"
        and not show_range
        and not f_cond
        and not t_slots
        and any(w in q_low for w in ("new ", "latest ", "releasing", "upcoming", "coming soon"))
    )

    # GEO_DISTANCE strictly takes ONLY the city name (never sub-localities, formats, or dates)
    if (
        canon_city
        and geo_city_for_filter
        and "near me" not in q_low
        and rt not in ("venue", "artist", "offer")
        and not (rel_range and not show_range)
        and not is_catalog_new_release_movie
    ):
        geo_filter = f'location:GEO_DISTANCE("{geo_city_for_filter}", 10000)'
        hard_clauses.append(geo_filter)

    if show_range:
        if is_v3:
            s_cond = f'show_dates >= "{show_range[0]}T00:00:00+05:30" AND show_dates <= "{show_range[1]}T23:59:59+05:30"'
            boost_specs.append({"condition": s_cond, "boost": 0.75})
        elif rt == "event":
            s_cond = f'{LEGACY_V2_SHOW_DATES_FIELD} >= "{show_range[0]}T00:00:00Z" AND {LEGACY_V2_SHOW_DATES_FIELD} <= "{show_range[1]}T23:59:59Z"'
            boost_specs.append({"condition": s_cond, "boost": 0.75})
        else:
            s_cond = f'sessions.showtime_start >= "{show_range[0]}T00:00:00Z" AND sessions.showtime_start <= "{show_range[1]}T23:59:59Z"'
            boost_specs.append({"condition": s_cond, "boost": 0.65})

    if rel_range:
        rel_boost_val = 0.65 if show_range else 0.85
        if is_v3:
            r_cond = f'release_date >= "{rel_range[0]}T00:00:00Z" AND release_date <= "{rel_range[1]}T23:59:59Z"'
        else:
            r_cond = f'firstReleaseDate >= "{rel_range[0]}T00:00:00Z" AND firstReleaseDate <= "{rel_range[1]}T23:59:59Z"'
        boost_specs.append({"condition": r_cond, "boost": rel_boost_val})

    for es in extra_slots:
        if es not in t_slots:
            t_slots.append(es)
    if t_slots:
        ts_cond = f"time_slots: {_quote_any(t_slots)}" if is_v3 else f"sessions.time_slot: {_quote_any(t_slots)}"
        boost_specs.append({"condition": ts_cond, "boost": 0.55})

    if rt == "venue":
        has_chain_brand = any(
            b in q_words for b in ("pvr", "inox", "cinepolis", "moviemax", "miraj")
        )
        drop_words = {"in", "near", "at", "around"}
        if has_chain_brand:
            drop_words |= {"theatre", "theatres", "theater", "theaters", "cinema", "cinemas"}
        venue_tokens = [
            w for w in original_query.strip().split()
            if w.lower().strip(".,-?!") not in drop_words
        ]
        rewritten = " ".join(venue_tokens) if venue_tokens else original_query.strip()
        if canon_city and canon_city.lower() not in rewritten.lower():
            rewritten = f"{rewritten} {canon_city}".strip()
    else:
        rewritten = _clean_rewritten_query(
            raw_q=str(slots.get("q") or slots.get("rewritten_query") or ""),
            original_query=original_query,
            rt=rt,
            canon_city=canon_city,
            matched_langs=matched_langs,
            fmt_keys=fmt_keys,
            genres_list=genres_list,
            cats_list=cats_list,
            resolved_artist=resolved_artist,
            show_range=show_range,
            rel_range=rel_range,
            no_movie=no_movie,
        )

    summary_items: List[str] = []
    if geo_filter:
        summary_items.append(f"[GEO] {geo_filter}")
    summary_items.extend(
        f"[{'+' if b['boost'] >= 0 else ''}{b['boost']}] {b['condition']}" for b in boost_specs
    )
    summary = " ; ".join(summary_items) or "(no boost conditions extracted)"
    return rewritten, boost_specs, geo_filter, " AND ".join(hard_clauses), summary


def _sanitize_geo_filter_expr(raw_geo: str, q_low: str) -> str:
    """Validate and clean a `location:GEO_DISTANCE("<City>", 10000)` expression so ONLY the
    canonical city name is ever passed to `GEO_DISTANCE`.
    """
    if not raw_geo or not raw_geo.startswith("location:GEO_DISTANCE("):
        return ""
    q1 = raw_geo.find('"')
    q2 = raw_geo.find('"', q1 + 1) if q1 != -1 else -1
    if q1 == -1 or q2 == -1:
        return ""
    place = raw_geo[q1 + 1 : q2].strip()
    canon_city, _ = _normalize_city_and_locality(place, place, "movie")
    if canon_city and _is_city_grounded(canon_city, place, q_low) and "near me" not in q_low:
        geo_city = GEO_CITY_NAME_MAP.get(canon_city, canon_city)
        return f'location:GEO_DISTANCE("{geo_city}", 10000)'
    return ""


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


def build_gemma_chat_messages(dctx: Dict[str, str]) -> List[Dict[str, str]]:
    """Build a compact (~400-token) multi-turn chat template tailored to Gemma 3 / Gemma 4's
    512-token sliding-window `RotatingKVCache` so 100% of rules + examples stay in-window
    even after appending the user query and generating the JSON output.
    """
    _ = dctx  # Semantic `when`/`rel` tokens are resolved deterministically via `_date_context()`
    rules = (
        "Extract ONLY slots in Query as JSON; NEVER invent city/price/when/lang!\n"
        "type: movie(film/IMAX/4DX/3D/Dolby/recliner) | event(show/concert/comedy/play/garba/sports/passes) | venue(PVR/INOX/cinemas/theatre) | artist | offer(card/coupon).\n"
        "Keys: type, no_movie, city(city ONLY), min_price(int), max_price/price(int, 250 cheapest), lang, genres(Romantic), no_genre, cats, fmt, when, artist, q."
    )
    return [
        {"role": "user", "content": rules + "\nQuery: telugu action movie in 4dx this friday in hyderabad"},
        {"role": "assistant", "content": '{"type":"movie","city":"Hyderabad","lang":"Telugu","genres":["Action"],"fmt":"4DX","when":"this friday","q":"telugu action 4DX movie"}'},
        {"role": "user", "content": "Query: standup comedy in koramangala tonight"},
        {"role": "assistant", "content": '{"type":"event","city":"Bengaluru","genres":["Comedy"],"cats":["stand-up-comedy"],"when":"tonight","q":"standup comedy"}'},
        {"role": "user", "content": "Query: cinemas near phoenix marketcity whitefield"},
        {"role": "assistant", "content": '{"type":"venue","city":"Bengaluru","q":"cinemas phoenix marketcity whitefield"}'},
        {"role": "user", "content": "Query: concerts between 1000 and 3000 in jaipur"},
        {"role": "assistant", "content": '{"type":"event","city":"Jaipur","min_price":1000,"max_price":3000,"cats":["concerts"],"q":"concerts"}'},
        {"role": "user", "content": "Query: funny movie without any horror"},
        {"role": "assistant", "content": '{"type":"movie","genres":["Comedy"],"no_genre":"Horror","q":"funny comedy movie"}'},
        {"role": "user", "content": "Query: PLACEHOLDER_QUERY"},
    ]


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


# ==============================================================================
# 1B. Local Gemma 3 Query Routing Decision Engine (`CALL_LLM_INTENT` vs `DIRECT_AGENT_SEARCH`)
# ==============================================================================


def build_gemma_router_chat_messages() -> List[Dict[str, str]]:
    """Build a compact (~367-token) multi-turn chat template for Gemma 3's 512-token
    sliding-window `RotatingKVCache` to decide whether a query requires LLM Intent/Boost
    extraction (`CALL_LLM_INTENT`) or can go directly to Agent Search (`DIRECT_AGENT_SEARCH`).
    """
    rules = (
        "Entertainment Query Router.\n"
        'Output JSON: {"route":"DIRECT_AGENT_SEARCH"|"CALL_LLM_INTENT","reason":"..."}\n'
        "- DIRECT_AGENT_SEARCH: pure movie/show title, artist name, or bank offer with NO filters.\n"
        "- CALL_LLM_INTENT: has city/locality/near, price/under/between/free, date/time/tonight/weekend/late night, format(IMAX/4DX/3D/Dolby/recliner), language, genre/category, negation(without/not), or occasion."
    )
    return [
        {"role": "user", "content": rules + "\nQuery: Spider-Man: No Way Home"},
        {"role": "assistant", "content": '{"route":"DIRECT_AGENT_SEARCH","reason":"Direct movie title lookup with no facet filters"}'},
        {"role": "user", "content": "Query: Zakir Khan show"},
        {"role": "assistant", "content": '{"route":"DIRECT_AGENT_SEARCH","reason":"Direct artist show lookup with no city or date filter"}'},
        {"role": "user", "content": "Query: IMAX 3D movies in Mumbai"},
        {"role": "assistant", "content": '{"route":"CALL_LLM_INTENT","reason":"Has format (IMAX 3D) and city (Mumbai) filters"}'},
        {"role": "user", "content": "Query: concerts between 1000 and 3000 in delhi"},
        {"role": "assistant", "content": '{"route":"CALL_LLM_INTENT","reason":"Has category (concerts), price range (1000-3000), and city (Delhi)"}'},
        {"role": "user", "content": "Query: funny movie without any horror"},
        {"role": "assistant", "content": '{"route":"CALL_LLM_INTENT","reason":"Has genre (Comedy) and negative genre exclusion (without horror)"}'},
        {"role": "user", "content": "Query: PLACEHOLDER_QUERY"},
    ]


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


# ==============================================================================
# 2. Cloud Vertex AI Gemini (`gemini-4-argon` / `gemini-3.5-flash-lite`) Extractor
#    (Unified BigQuery-Grounded Intent Extractor -> Schema-Verified Boost Compiler)
# ==============================================================================

GEMINI_INTENT_SYSTEM_INSTRUCTION = LOCAL_LLM_SYSTEM_PROMPT


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


# ==============================================================================
# 3. Live Vertex AI Search Execution & Side-by-Side Comparison
# ==============================================================================


def build_curl_command(payload: Dict[str, Any], search_path: str = SEARCH_PATH) -> str:
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
    router_model = str(params.get("router_model") or "local:gemma-3-1b-it-qat-4bit").strip()
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
    router_model = str(params.get("router_model") or "local:gemma-3-1b-it-qat-4bit").strip()

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
    is_v3 = app_cfg["key"] == "v3"
    resolved_model, gemini_region, gemini_host, gemini_path, gen_cfg = resolve_gemini_endpoint_and_config(
        "gemini-3.5-flash-lite", max_tokens_default=420, temperature=0.1
    )

    if is_v3:
        schema_ctx = """Known Entertainment v3 Master Schema Context (`ent_5stores_master_v3` / `ent-bq-5stores-master-v3-app`):
1. Option-2 Flat Session Grain (~390,870 docs): `movie` & `event` rows with sessions are 1 row per Session (`id = <entity_code>_<venue>_<session_id>`), binding exact `(session_id, formats, languages, min_price, show_dates, time_slots)` with zero cross-session leakage; `crowdingSpecs` on `entity_code` collapses duplicates.
2. Consolidated Master fields: `regions` (venue city only on session rows — zero 300-city `active_regions` leakage — plus extracted city on `venue` rows), `location` (populated on `movie`, `event`, AND `venue` rows), `formats` (normalized uppercase spaces like `"IMAX 3D"`), `languages` (Title Case), `show_dates`, `release_date`, `min_price`, `is_sold_out`.
3. Curated schema: 0 dynamic facets, 14 indexable fields, 11 searchable fields."""
    else:
        schema_ctx = """Known Entertainment v2 Unified Schema Context (`ent_5stores_unified` / `ent-bq-5stores-v2-app`):
1. `movie` & `event` docs are at Entity+Venue grain (`crowdingSpecs` on `entity_code` collapses theatre duplicates) with nested `sessions[]` arrays (bag-of-values across sessions at a venue).
2. `venue` docs have `address` with city text, while `location` (`lat/lng`) and `regionName` are NULL on `venue` records in v2.
3. Top-level `formats` uses spaces ("4DX 3D") while nested `sessions.format` uses hyphens ("4DX-3D", "IMAX-2D")."""

    prompt = f"""You are a Senior Search Relevance Architect diagnosing Entertainment Search's Vertex AI Agent Search (`{app_cfg['engine_id']}`: 5 record_types = `movie`, `event`, `venue`, `artist`, `offer`).

{schema_ctx}

Analyze this live search result and return a concise diagnosis in JSON:
- Target App: `{app_cfg['engine_id']}` (`{app_cfg['bq_table']}`)
- User Query: "{query}"
- Extracted Filter / Boosts: `{extracted_filters}`
- Rewritten Query: `{rewritten_query}`
- Returned Rank #{hit.get("rank", 1)} Document:
  - ID: `{hit.get("id")}` | entity_code: `{hit.get("entity_code")}` | session_id: `{hit.get("session_id", "")}` | record_type: `{hit.get("record_type")}`
  - Title: "{hit.get("title")}"
  - Region / Venue: {hit.get("regions")} / {hit.get("venues")}
  - Languages: {hit.get("languages")} | Genres: {hit.get("genres")}
  - Formats: {hit.get("formats")} (top-level: {hit.get("top_level_formats")}, nested sessions: {hit.get("session_formats")})
  - Min Price: {hit.get("min_price")} | Dates / Time Slots: {hit.get("date_display")} / {hit.get("time_slots")}
  - Rule Check Verdict: {"MATCH" if hit.get("pass") else "MISS"} ({hit.get("explanation")})

Return ONLY valid JSON with keys:
- `expected`: 1 short clause of what the user wanted.
- `actual`: 1 short clause of what the App actually returned at this rank.
- `why`: 1 short clause explaining why it matched or failed.
- `fix`: 1 concrete actionable fix in BigQuery or Agent Search boost/filter config.
- `one_liner`: A single crisp one-liner (max 45 words) combining Expected, Actual, Why, and Fix.
"""

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
        search_params = dict(params)
        search_params["nlu_mode"] = "auto_route"
        search_params["_precomputed_route_decision"] = decision
        decision = dict(decision)
        decision["search_result"] = run_live_search(search_params)
    return decision


# ==============================================================================
# 4. Frontend HTML / CSS / JS Workbench (Loaded from `realtime_test_ui.html`)
# ==============================================================================

UI_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_test_ui.html")


def load_ui_html() -> str:
    """Load the frontend UI HTML from `realtime_test_ui.html`."""
    with open(UI_HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()


INDEX_HTML = load_ui_html()


class RealtimeTestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html_str: str) -> None:
        raw = html_str.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
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
        return


def main() -> None:
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
