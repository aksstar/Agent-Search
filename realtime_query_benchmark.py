"""
Real-Time Query Resolution & Latency Benchmark for Vertex AI Search (Entertainment & Events).

Compares two execution modes across all 12 natural-language benchmark queries:
1. Single Query (Direct Search):
   - Direct Vertex AI Search API call using pre-defined query + boostSpec (no LLM call).
2. Real-Time Query Resolution (2-Stage Pipeline + Search):
   - Stage 1 (LLM Extraction): Gemini 3.5 Flash-Lite extracts structured intent (what, when, where, language, audience, tags) into `ExtractedSearchIntent`.
   - Stage 2 (Boost Compilation): Deterministic Python compiler (`compile_boost_specs`) builds `conditionBoostSpecs`.
   - Stage 3 (Boosted Search): Vertex AI Search API call using the dynamically extracted query and compiled `boostSpec`.
"""

import json
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import google.auth
from google import genai
from google.auth.transport.requests import AuthorizedSession
from google.genai import types
from pydantic import BaseModel, Field

from config import GEMINI_MODEL, PROJECT_ID, VERTEX_LOCATION
from query_with_boost import (
    BENCHMARK_QUERIES,
    get_discovery_session,
    search_with_boost,
)

# Reference timestamp used for relative date resolution ("tonight", "this friday", "next week", etc.)
REFERENCE_NOW_ISO = "2026-10-07T07:40:00+05:30"

CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "Mumbai": (18.9946, 72.8245),
    "Delhi NCR": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Pune": (18.5204, 73.8567),
    "Kolkata": (22.5726, 88.3639),
    "Goa": (15.4989, 73.8278),
}


# ==============================================================================
# Stage 1: Structured Intent Schema for Gemini Real-Time Extraction
# ==============================================================================
class TimeWindow(BaseModel):
    start_iso: str = Field(
        description="ISO-8601 start timestamp with +05:30 offset, e.g. 2026-10-07T17:00:00+05:30"
    )
    end_iso: str = Field(
        description="ISO-8601 end timestamp with +05:30 offset, e.g. 2026-10-07T23:59:59+05:30"
    )


class ExtractedSearchIntent(BaseModel):
    search_query: str = Field(
        description=(
            "Clean core search query for Vertex AI Search (e.g. 'open mic', 'stand up comedy', "
            "'movies releasing friday', 'events', 'movie', 'garba night', 'new year eve party', "
            "'upcoming telugu movies', 'play theatre', 'pottery workshop')"
        )
    )
    record_type: Optional[Literal["movie", "event"]] = None
    city: Optional[
        Literal[
            "Mumbai",
            "Delhi NCR",
            "Bengaluru",
            "Hyderabad",
            "Chennai",
            "Pune",
            "Kolkata",
            "Goa",
        ]
    ] = None
    languages: List[str] = Field(
        default_factory=list,
        description="Title-cased languages if mentioned, e.g. ['Telugu', 'Hindi']",
    )
    audience: Optional[Literal["kids", "family", "adults"]] = None
    keywords: List[str] = Field(
        default_factory=list,
        description="Lowercase genre/topic/audience keywords for searchKeywords matching",
    )
    hash_tags: List[
        Literal[
            "MORNING_SHOW",
            "LATE_NIGHT_SHOW",
            "NEW_RELEASE",
            "FRIDAY_RELEASE",
            "UPCOMING_MOVIE",
            "KIDS_EVENT",
            "OPEN_MIC",
            "STANDUP_COMEDY",
            "GARBA_NIGHT",
            "NEW_YEAR_EVE",
            "STAGE_PLAY",
            "POTTERY_WORKSHOP",
            "SATURDAY_OCTOBER",
        ]
    ] = Field(default_factory=list)
    date_field: Literal["showtime_start", "release_date"] = Field(
        default="showtime_start",
        description="Use 'release_date' when query asks for 'releasing' or 'upcoming' movies; otherwise 'showtime_start'",
    )
    time_windows: List[TimeWindow] = Field(default_factory=list)


SYSTEM_INSTRUCTION = f"""You are a real-time search query parser for an entertainment & ticketing platform.
Current reference time: {REFERENCE_NOW_ISO} (Wednesday, October 7, 2026).

Convert the user's natural language query into the structured JSON schema:
1. `search_query`: Clean keyword query for the search engine:
   - If the user asks generally for movies (e.g. "late night shows after 10 pm", "morning shows tomorrow before 11"), set `search_query="movie"` and `record_type="movie"`.
   - If the user asks generally what's happening or kids events (e.g. "kids events this sunday morning", "is weekend delhi mein kya chal raha hai"), set `search_query="events"` and `record_type="event"`.
   - If the user asks for plays (e.g. "plays on gandhi jayanti holiday"), set `search_query="play theatre"` and `record_type="event"`.
   - If the user asks for pottery classes, set `search_query="pottery workshop"` and `record_type="event"`.
2. `city`: Map 'delhi' -> 'Delhi NCR', 'bangalore' -> 'Bengaluru', 'pune' -> 'Pune', 'goa' -> 'Goa', etc.
3. `date_field`: Use 'release_date' ONLY for movie release queries ("movies releasing this friday", "upcoming telugu movies next month"). Otherwise use 'showtime_start'.
4. `time_windows`: Resolve relative dates relative to Wednesday, 2026-10-07:
   - "tonight": 2026-10-07T17:00:00+05:30 to 2026-10-07T23:59:59+05:30
   - "after 10 pm": 2026-10-07T22:00:00+05:30 to 2026-10-08T03:00:00+05:30
   - "tomorrow before 11": 2026-10-08T06:00:00+05:30 to 2026-10-08T11:00:00+05:30
   - "this friday": 2026-10-09T00:00:00+05:30 to 2026-10-09T23:59:59+05:30
   - "this weekend": 2026-10-10T00:00:00+05:30 to 2026-10-11T23:59:59+05:30
   - "this sunday morning": 2026-10-11T06:00:00+05:30 to 2026-10-11T12:00:00+05:30
   - "next week": 2026-10-12T00:00:00+05:30 to 2026-10-18T23:59:59+05:30
   - "10th and 20th october": 2026-10-10T00:00:00+05:30 to 2026-10-20T23:59:59+05:30
   - "next month": 2026-11-01T00:00:00+05:30 to 2026-11-30T23:59:59+05:30
   - "new year eve": 2026-12-31T18:00:00+05:30 to 2027-01-01T06:00:00+05:30
   - "gandhi jayanti holiday": 2026-10-02T00:00:00+05:30 to 2026-10-02T23:59:59+05:30
   - "every saturday in october": include windows for 2026-10-10 and 2026-10-17 (and add 'SATURDAY_OCTOBER' to hash_tags).
"""


# ==============================================================================
# Stage 2: Deterministic Python Boost Compiler
# ==============================================================================
def compile_boost_specs(
    intent: ExtractedSearchIntent,
    user_latlng: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """Deterministically compiles `ExtractedSearchIntent` into Vertex AI Search `conditionBoostSpecs`."""
    specs: List[Dict[str, Any]] = []

    # 1. Record Type + Language Boost
    type_clauses: List[str] = []
    if intent.record_type:
        type_clauses.append(f'record_type: ANY("{intent.record_type}")')
    if intent.languages:
        langs = ", ".join(f'"{l}"' for l in intent.languages)
        type_clauses.append(f"languages: ANY({langs})")
    if type_clauses:
        specs.append({"condition": " AND ".join(type_clauses), "boost": 0.6})

    # 2. Genre / Audience Keywords & Hashtags Boost
    if intent.keywords or intent.hash_tags:
        kw_clauses: List[str] = []
        if intent.keywords:
            kws = ", ".join(f'"{k.lower()}"' for k in intent.keywords)
            kw_clauses.append(f"searchKeywords: ANY({kws})")
        if intent.hash_tags:
            tags = ", ".join(f'"{t}"' for t in intent.hash_tags)
            kw_clauses.append(f"hash_tags: ANY({tags})")
        specs.append({"condition": " OR ".join(kw_clauses), "boost": 0.6})

    # 3. Location Boost (Explicit city in query OR concentric rings around user GPS)
    if intent.city and intent.city in CITY_COORDS:
        lat, lng = CITY_COORDS[intent.city]
        specs.append(
            {
                "condition": f"location_city:GEO_DISTANCE({lat}, {lng}, 50000)",
                "boost": 0.8,
            }
        )
    elif user_latlng:
        lat, lng = user_latlng
        for radius_m, boost_val in [
            (50000, 0.4),
            (300000, 0.3),
            (800000, 0.2),
            (1500000, 0.1),
        ]:
            specs.append(
                {
                    "condition": f"location_city:GEO_DISTANCE({lat}, {lng}, {radius_m})",
                    "boost": boost_val,
                }
            )

    # 4. Date / Time Window Boost
    if intent.time_windows:
        field = intent.date_field
        window_clauses = [
            f'({field} >= "{w.start_iso}" AND {field} <= "{w.end_iso}")'
            for w in intent.time_windows
        ]
        specs.append(
            {
                "condition": " OR ".join(window_clauses),
                "boost": 0.9,
            }
        )

    return specs


# ==============================================================================
# Clients & Timed Execution Helpers
# ==============================================================================
_MTLS_SESSION: Optional[AuthorizedSession] = None
_VERTEX_AI_URL: Optional[str] = None

VERTEX_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "search_query": {"type": "STRING"},
        "record_type": {
            "type": "STRING",
            "nullable": True,
            "enum": ["movie", "event"],
        },
        "city": {
            "type": "STRING",
            "nullable": True,
            "enum": [
                "Mumbai",
                "Delhi NCR",
                "Bengaluru",
                "Hyderabad",
                "Chennai",
                "Pune",
                "Kolkata",
                "Goa",
            ],
        },
        "languages": {"type": "ARRAY", "items": {"type": "STRING"}},
        "audience": {
            "type": "STRING",
            "nullable": True,
            "enum": ["kids", "family", "adults"],
        },
        "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
        "hash_tags": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": [
                    "MORNING_SHOW",
                    "LATE_NIGHT_SHOW",
                    "NEW_RELEASE",
                    "FRIDAY_RELEASE",
                    "UPCOMING_MOVIE",
                    "KIDS_EVENT",
                    "OPEN_MIC",
                    "STANDUP_COMEDY",
                    "GARBA_NIGHT",
                    "NEW_YEAR_EVE",
                    "STAGE_PLAY",
                    "POTTERY_WORKSHOP",
                    "SATURDAY_OCTOBER",
                ],
            },
        },
        "date_field": {
            "type": "STRING",
            "enum": ["showtime_start", "release_date"],
        },
        "time_windows": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start_iso": {"type": "STRING"},
                    "end_iso": {"type": "STRING"},
                },
                "required": ["start_iso", "end_iso"],
            },
        },
    },
    "required": [
        "search_query",
        "languages",
        "keywords",
        "hash_tags",
        "date_field",
        "time_windows",
    ],
}


def get_vertex_session() -> Tuple[AuthorizedSession, str]:
    global _MTLS_SESSION, _VERTEX_AI_URL
    if _MTLS_SESSION is None:
        session, disc_base_url = get_discovery_session()
        _MTLS_SESSION = session
        host = (
            "https://aiplatform.mtls.googleapis.com"
            if ".mtls." in disc_base_url
            else "https://aiplatform.googleapis.com"
        )
        _VERTEX_AI_URL = (
            f"{host}/v1/projects/{PROJECT_ID}/locations/{VERTEX_LOCATION}"
            f"/publishers/google/models/{GEMINI_MODEL}:generateContent"
        )
    return _MTLS_SESSION, _VERTEX_AI_URL  # type: ignore[return-value]


def execute_search_timed(
    query: str,
    condition_boost_specs: Optional[List[Dict[str, Any]]] = None,
    page_size: int = 4,
) -> Tuple[Dict[str, Any], float]:
    """Executes Vertex AI Search and returns (response_json, latency_ms)."""
    t0 = time.perf_counter()
    resp_json = search_with_boost(
        query=query,
        condition_boost_specs=condition_boost_specs,
        page_size=page_size,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return resp_json, elapsed_ms


def extract_intent_timed(user_query: str) -> Tuple[ExtractedSearchIntent, float]:
    """Calls Gemini 3.5 Flash-Lite via Vertex AI REST API to extract structured search intent."""
    session, url = get_vertex_session()
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": user_query}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": VERTEX_RESPONSE_SCHEMA,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    t0 = time.perf_counter()
    resp = session.post(url, json=payload, timeout=30)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    intent = ExtractedSearchIntent.model_validate_json(text)
    return intent, elapsed_ms


def get_top_hit_summary(response: Dict[str, Any]) -> str:
    results = response.get("results", [])
    if not results:
        return "No results"
    sd = results[0].get("document", {}).get("structData", {})
    return f"[{sd.get('id')}] {sd.get('title')} ({sd.get('city')})"


def run_benchmark() -> None:
    print(f"Warming up Vertex AI Search session & Gemini 3.5 Flash-Lite ({GEMINI_MODEL}) connection...")
    execute_search_timed("movie", page_size=1)
    extract_intent_timed("open mic next week")
    print("Warm-up complete.\n")

    rows = []

    print("=" * 135)
    print(
        f"{'ID':<4} | {'User Query':<42} | {'Single Query (ms)':>17} | "
        f"{'LLM Extract (ms)':>16} | {'Compile (ms)':>12} | {'RT Search (ms)':>14} | {'RT Total E2E (ms)':>17}"
    )
    print("-" * 135)

    for bq in BENCHMARK_QUERIES:
        qid = bq["id"]
        raw_query = bq["query"]

        # 1. Single Query Mode (Pre-configured query + boostSpec directly to Vertex AI Search)
        single_resp, single_search_ms = execute_search_timed(
            query=bq["what_query"],
            condition_boost_specs=bq["boost_specs"],
            page_size=4,
        )
        single_top1 = get_top_hit_summary(single_resp)

        # 2. Real-Time Query Resolution Mode (Gemini Extraction -> Python Boost Compiler -> Vertex AI Search)
        intent, llm_ms = extract_intent_timed(raw_query)

        t_comp0 = time.perf_counter()
        compiled_specs = compile_boost_specs(intent)
        compile_ms = (time.perf_counter() - t_comp0) * 1000.0

        rt_resp, rt_search_ms = execute_search_timed(
            query=intent.search_query,
            condition_boost_specs=compiled_specs,
            page_size=4,
        )
        rt_total_ms = llm_ms + compile_ms + rt_search_ms
        rt_top1 = get_top_hit_summary(rt_resp)

        rows.append(
            {
                "id": qid,
                "query": raw_query,
                "single_search_ms": single_search_ms,
                "llm_ms": llm_ms,
                "compile_ms": compile_ms,
                "rt_search_ms": rt_search_ms,
                "rt_total_ms": rt_total_ms,
                "extracted_query": intent.search_query,
                "compiled_specs": compiled_specs,
                "single_top1": single_top1,
                "rt_top1": rt_top1,
            }
        )

        print(
            f"{qid:<4} | {raw_query:<42} | {single_search_ms:>14.1f} ms | "
            f"{llm_ms:>13.1f} ms | {compile_ms:>9.3f} ms | {rt_search_ms:>11.1f} ms | {rt_total_ms:>14.1f} ms"
        )

    print("=" * 135)

    n = len(rows)
    avg_single = sum(r["single_search_ms"] for r in rows) / n
    avg_llm = sum(r["llm_ms"] for r in rows) / n
    avg_compile = sum(r["compile_ms"] for r in rows) / n
    avg_rt_search = sum(r["rt_search_ms"] for r in rows) / n
    avg_rt_total = sum(r["rt_total_ms"] for r in rows) / n

    sorted_single = sorted(r["single_search_ms"] for r in rows)
    sorted_rt_total = sorted(r["rt_total_ms"] for r in rows)
    p50_single = sorted_single[n // 2]
    p95_single = sorted_single[min(n - 1, int(n * 0.95))]
    p50_rt = sorted_rt_total[n // 2]
    p95_rt = sorted_rt_total[min(n - 1, int(n * 0.95))]

    print(
        f"{'AVG':<4} | {'Average across 12 queries':<42} | {avg_single:>14.1f} ms | "
        f"{avg_llm:>13.1f} ms | {avg_compile:>9.3f} ms | {avg_rt_search:>11.1f} ms | {avg_rt_total:>14.1f} ms"
    )
    print(
        f"{'P50':<4} | {'Median (P50) Latency':<42} | {p50_single:>14.1f} ms | "
        f"{'-':>16} | {'-':>12} | {'-':>14} | {p50_rt:>14.1f} ms"
    )
    print(
        f"{'P95':<4} | {'95th Percentile (P95) Latency':<42} | {p95_single:>14.1f} ms | "
        f"{'-':>16} | {'-':>12} | {'-':>14} | {p95_rt:>14.1f} ms"
    )
    print("=" * 135)

    print("\n=== Detailed Real-Time Query Resolution & Top-1 Hit Verification ===")
    for r in rows:
        print(f"\n[{r['id']}] Raw Query: {r['query']!r}")
        print(f"  -> Extracted search_query : {r['extracted_query']!r}")
        print(f"  -> Compiled boostSpecs    : {json.dumps(r['compiled_specs'])}")
        print(f"  -> Single Query Top #1    : {r['single_top1']} ({r['single_search_ms']:.1f} ms)")
        print(f"  -> Real-Time Top #1       : {r['rt_top1']} ({r['rt_total_ms']:.1f} ms total)")


if __name__ == "__main__":
    run_benchmark()

# ==============================================================================
# EXECUTION OUTPUT (`python realtime_query_benchmark.py`)
# ==============================================================================
# Warming up Vertex AI Search session & Gemini 3.5 Flash-Lite (gemini-3.5-flash-lite) connection...
# Warm-up complete.
#
# =======================================================================================================================================
# ID   | User Query                                 | Single Query (ms) | LLM Extract (ms) | Compile (ms) | RT Search (ms) | RT Total E2E (ms)
# ---------------------------------------------------------------------------------------------------------------------------------------
# Q1   | open mic next week                         |          353.7 ms |        1746.2 ms |     0.044 ms |       348.7 ms |         2094.9 ms
# Q2   | stand up comedy tonight in pune            |          348.9 ms |        1679.5 ms |     0.014 ms |       328.9 ms |         2008.4 ms
# Q3   | movies releasing this friday               |          331.4 ms |        1857.2 ms |     0.010 ms |       362.9 ms |         2220.1 ms
# Q4   | kids events this sunday morning            |          313.3 ms |        2254.1 ms |     0.007 ms |       312.5 ms |         2566.6 ms
# Q5   | late night shows after 10 pm               |          260.4 ms |        2283.0 ms |     0.023 ms |       318.9 ms |         2601.9 ms
# Q6   | garba nights between 10th and 20th october |          277.4 ms |        1807.5 ms |     0.017 ms |       272.4 ms |         2079.9 ms
# Q7   | new year eve parties in goa                |          328.6 ms |        2292.9 ms |     0.022 ms |       275.9 ms |         2568.9 ms
# Q8   | is weekend delhi mein kya chal raha hai    |          259.0 ms |        1708.1 ms |     0.050 ms |       286.0 ms |         1994.2 ms
# Q9   | upcoming telugu movies next month          |          289.5 ms |        1633.1 ms |     0.029 ms |       380.9 ms |         2014.0 ms
# Q10  | morning shows tomorrow before 11           |          269.1 ms |        1648.8 ms |     0.010 ms |       745.0 ms |         2393.8 ms
# Q11  | plays on gandhi jayanti holiday            |          270.7 ms |        2274.4 ms |     0.022 ms |       274.9 ms |         2549.3 ms
# Q12  | pottery classes every saturday in october  |          275.3 ms |        2282.3 ms |     0.031 ms |       270.7 ms |         2553.0 ms
# =======================================================================================================================================
# AVG  | Average across 12 queries                  |          298.1 ms |        1955.6 ms |     0.023 ms |       348.1 ms |         2303.7 ms
# P50  | Median (P50) Latency                       |          289.5 ms |                - |            - |              - |         2393.8 ms
# P95  | 95th Percentile (P95) Latency              |          353.7 ms |                - |            - |              - |         2601.9 ms
# =======================================================================================================================================
#
# === Detailed Real-Time Query Resolution & Top-1 Hit Verification ===
#
# [Q1] Raw Query: 'open mic next week'
#   -> Extracted search_query : 'open mic'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"event\")", "boost": 0.6}, {"condition": "searchKeywords: ANY(\"open mic\") OR hash_tags: ANY(\"OPEN_MIC\")", "boost": 0.6}, {"condition": "(showtime_start >= \"2026-10-12T00:00:00+05:30\" AND showtime_start <= \"2026-10-18T23:59:59+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [etm100102z] BLR BREWING ACOUSTIC & STANDUP OPEN MIC - BENGALURU (Bengaluru) (353.7 ms)
#   -> Real-Time Top #1       : [etm100102z] BLR BREWING ACOUSTIC & STANDUP OPEN MIC - BENGALURU (Bengaluru) (2094.9 ms total)
#
# [Q2] Raw Query: 'stand up comedy tonight in pune'
#   -> Extracted search_query : 'standup comedy'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"event\")", "boost": 0.6}, {"condition": "searchKeywords: ANY(\"stand up\", \"comedy\") OR hash_tags: ANY(\"STANDUP_COMEDY\")", "boost": 0.6}, {"condition": "location_city:GEO_DISTANCE(18.5204, 73.8567, 50000)", "boost": 0.8}, {"condition": "(showtime_start >= \"2026-10-07T17:00:00+05:30\" AND showtime_start <= \"2026-10-07T23:59:59+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [etm100104z] BASSI KISI KO BATANA MAT STAND UP COMEDY - PUNE (Pune) (348.9 ms)
#   -> Real-Time Top #1       : [etm100104z] BASSI KISI KO BATANA MAT STAND UP COMEDY - PUNE (Pune) (2008.4 ms total)
#
# [Q3] Raw Query: 'movies releasing this friday'
#   -> Extracted search_query : 'movie'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"movie\")", "boost": 0.6}, {"condition": "searchKeywords: ANY(\"movies\", \"releasing\") OR hash_tags: ANY(\"FRIDAY_RELEASE\", \"NEW_RELEASE\")", "boost": 0.6}, {"condition": "(release_date >= \"2026-10-09T00:00:00+05:30\" AND release_date <= \"2026-10-09T23:59:59+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [MV00103] Vettaiyan: The Hunter (Chennai) (331.4 ms)
#   -> Real-Time Top #1       : [MV00102] Bhool Bhulaiyaa 4 (Delhi NCR) (2220.1 ms total)
#
# [Q4] Raw Query: 'kids events this sunday morning'
#   -> Extracted search_query : 'events'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"event\")", "boost": 0.6}, {"condition": "searchKeywords: ANY(\"kids\", \"events\", \"morning\") OR hash_tags: ANY(\"KIDS_EVENT\")", "boost": 0.6}, {"condition": "(showtime_start >= \"2026-10-11T06:00:00+05:30\" AND showtime_start <= \"2026-10-11T12:00:00+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [etm100106z] JUNIOR ROBOTICS & SCIENCE KIDS WORKSHOP - BENGALURU (Bengaluru) (313.3 ms)
#   -> Real-Time Top #1       : [etm100106z] JUNIOR ROBOTICS & SCIENCE KIDS WORKSHOP - BENGALURU (Bengaluru) (2566.6 ms total)
#
# [Q5] Raw Query: 'late night shows after 10 pm'
#   -> Extracted search_query : 'movie'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"movie\")", "boost": 0.6}, {"condition": "searchKeywords: ANY(\"late night\", \"after 10 pm\") OR hash_tags: ANY(\"LATE_NIGHT_SHOW\")", "boost": 0.6}, {"condition": "(showtime_start >= \"2026-10-07T22:00:00+05:30\" AND showtime_start <= \"2026-10-08T03:00:00+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [MV00106] The Batman Part II (Bengaluru) (260.4 ms)
#   -> Real-Time Top #1       : [MV00106] The Batman Part II (Bengaluru) (2601.9 ms total)
#
# [Q6] Raw Query: 'garba nights between 10th and 20th october'
#   -> Extracted search_query : 'garba night'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"event\")", "boost": 0.6}, {"condition": "searchKeywords: ANY(\"garba\") OR hash_tags: ANY(\"GARBA_NIGHT\")", "boost": 0.6}, {"condition": "(showtime_start >= \"2026-10-10T00:00:00+05:30\" AND showtime_start <= \"2026-10-20T23:59:59+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [etm100108z] FALGUNI PATHAK NAVRATRI GARBA NIGHTS - MUMBAI (Mumbai) (277.4 ms)
#   -> Real-Time Top #1       : [etm100108z] FALGUNI PATHAK NAVRATRI GARBA NIGHTS - MUMBAI (Mumbai) (2079.9 ms total)
#
# [Q7] Raw Query: 'new year eve parties in goa'
#   -> Extracted search_query : 'new year eve party'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"event\")", "boost": 0.6}, {"condition": "searchKeywords: ANY(\"new year eve\", \"party\") OR hash_tags: ANY(\"NEW_YEAR_EVE\")", "boost": 0.6}, {"condition": "location_city:GEO_DISTANCE(15.4989, 73.8278, 50000)", "boost": 0.8}, {"condition": "(showtime_start >= \"2026-12-31T18:00:00+05:30\" AND showtime_start <= \"2027-01-01T06:00:00+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [etm100112z] THALASSA SUNSET TO SUNRISE NEW YEAR EVE PARTY - GOA (Goa) (328.6 ms)
#   -> Real-Time Top #1       : [etm100112z] THALASSA SUNSET TO SUNRISE NEW YEAR EVE PARTY - GOA (Goa) (2568.9 ms total)
#
# [Q8] Raw Query: 'is weekend delhi mein kya chal raha hai'
#   -> Extracted search_query : 'events'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"event\")", "boost": 0.6}, {"condition": "location_city:GEO_DISTANCE(28.6139, 77.209, 50000)", "boost": 0.8}, {"condition": "(showtime_start >= \"2026-10-10T00:00:00+05:30\" AND showtime_start <= \"2026-10-11T23:59:59+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [etm100113z] DELHI SUFI & STREET FOOD FESTIVAL - DELHI NCR (Delhi NCR) (259.0 ms)
#   -> Real-Time Top #1       : [etm100113z] DELHI SUFI & STREET FOOD FESTIVAL - DELHI NCR (Delhi NCR) (1994.2 ms total)
#
# [Q9] Raw Query: 'upcoming telugu movies next month'
#   -> Extracted search_query : 'telugu movie'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"movie\") AND languages: ANY(\"Telugu\")", "boost": 0.6}, {"condition": "searchKeywords: ANY(\"telugu\", \"movies\") OR hash_tags: ANY(\"UPCOMING_MOVIE\")", "boost": 0.6}, {"condition": "(release_date >= \"2026-11-01T00:00:00+05:30\" AND release_date <= \"2026-11-30T23:59:59+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [MV00108] Devara Part 2: Red Sea (Hyderabad) (289.5 ms)
#   -> Real-Time Top #1       : [MV00108] Devara Part 2: Red Sea (Hyderabad) (2014.0 ms total)
#
# [Q10] Raw Query: 'morning shows tomorrow before 11'
#   -> Extracted search_query : 'movie'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"movie\")", "boost": 0.6}, {"condition": "hash_tags: ANY(\"MORNING_SHOW\")", "boost": 0.6}, {"condition": "(showtime_start >= \"2026-10-08T06:00:00+05:30\" AND showtime_start <= \"2026-10-08T11:00:00+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [MV00112] Chhaava: The Great Warrior (Mumbai) (269.1 ms)
#   -> Real-Time Top #1       : [MV00112] Chhaava: The Great Warrior (Mumbai) (2393.8 ms total)
#
# [Q11] Raw Query: 'plays on gandhi jayanti holiday'
#   -> Extracted search_query : 'play theatre'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"event\")", "boost": 0.6}, {"condition": "searchKeywords: ANY(\"plays\") OR hash_tags: ANY(\"STAGE_PLAY\")", "boost": 0.6}, {"condition": "(showtime_start >= \"2026-10-02T00:00:00+05:30\" AND showtime_start <= \"2026-10-02T23:59:59+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [etm100117z] TUGHLAQ: THEATRE PLAY - MUMBAI (Mumbai) (270.7 ms)
#   -> Real-Time Top #1       : [etm100117z] TUGHLAQ: THEATRE PLAY - MUMBAI (Mumbai) (2549.3 ms total)
#
# [Q12] Raw Query: 'pottery classes every saturday in october'
#   -> Extracted search_query : 'pottery workshop'
#   -> Compiled boostSpecs    : [{"condition": "record_type: ANY(\"event\")", "boost": 0.6}, {"condition": "searchKeywords: ANY(\"pottery\", \"classes\", \"workshop\") OR hash_tags: ANY(\"POTTERY_WORKSHOP\", \"SATURDAY_OCTOBER\")", "boost": 0.6}, {"condition": "(showtime_start >= \"2026-10-10T00:00:00+05:30\" AND showtime_start <= \"2026-10-10T23:59:59+05:30\") OR (showtime_start >= \"2026-10-17T00:00:00+05:30\" AND showtime_start <= \"2026-10-17T23:59:59+05:30\") OR (showtime_start >= \"2026-10-24T00:00:00+05:30\" AND showtime_start <= \"2026-10-24T23:59:59+05:30\") OR (showtime_start >= \"2026-10-31T00:00:00+05:30\" AND showtime_start <= \"2026-10-31T23:59:59+05:30\")", "boost": 0.9}]
#   -> Single Query Top #1    : [etm100119z] CLAYSTATION WHEEL POTTERY CLASSES & CERAMICS WORKSHOP - BENGALURU (Bengaluru) (275.3 ms)
#   -> Real-Time Top #1       : [etm100119z] CLAYSTATION WHEEL POTTERY CLASSES & CERAMICS WORKSHOP - BENGALURU (Bengaluru) (2553.0 ms total)
