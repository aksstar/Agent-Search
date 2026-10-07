"""Query the Vertex AI Search App using the REST API with boostSpec (conditionBoostSpecs).

Reference:
https://cloud.google.com/generative-ai-app-builder/docs/boost-search-results
"""

import argparse
import json
import math
from config import DATASTORE_ID, ENGINE_ID, LOCATION, PROJECT_ID, PROJECT_NUMBER
from setup_gcs_datastore import get_discovery_session


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in kilometers between two (lat, lng) points."""
    radius_earth_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius_earth_km * math.asin(math.sqrt(a))


_CACHED_SESSION = None


def search_with_boost(
    query: str,
    condition_boost_specs: list[dict] | None = None,
    filter_expr: str = "",
    page_size: int = 10,
) -> dict:
    """Execute a REST API :search call against the configured Vertex AI Search Engine with boostSpec."""
    global _CACHED_SESSION
    if _CACHED_SESSION is None:
        _CACHED_SESSION = get_discovery_session()
    session, base_url = _CACHED_SESSION
    serving_config_url = (
        f"{base_url}/projects/{PROJECT_NUMBER}/locations/{LOCATION}"
        f"/collections/default_collection/engines/{ENGINE_ID}"
        f"/servingConfigs/default_search:search"
    )

    payload: dict = {
        "query": query,
        "pageSize": page_size,
        "queryExpansionSpec": {"condition": "AUTO"},
        "spellCorrectionSpec": {"mode": "AUTO"},
        "languageCode": "en-US",
        "userInfo": {"timeZone": "Asia/Calcutta"},
    }

    if filter_expr:
        payload["filter"] = filter_expr

    if condition_boost_specs:
        payload["boostSpec"] = {
            "conditionBoostSpecs": condition_boost_specs,
        }

    resp = session.post(serving_config_url, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Search failed (HTTP {resp.status_code}): {resp.text}")
    return resp.json()


def print_results(
    title: str,
    response: dict,
    origin_latlng: tuple[float, float] | None = None,
) -> None:
    """Display ranked search results to compare ordering with and without boostSpec."""
    print(f"\n=== {title} ===")
    results = response.get("results", [])
    print(f"Total Matches: {response.get('totalSize', len(results))} (Showing top {len(results)})")
    for rank, item in enumerate(results, start=1):
        doc = item.get("document", {}).get("structData", {})
        rec_id = doc.get("id") or doc.get("_id")
        rec_type = doc.get("record_type", "")
        rec_title = doc.get("title", "")
        city = doc.get("city", "")
        latlng = doc.get("location_latlng", {})
        showtime = doc.get("showtime_start", "")
        release_date = doc.get("release_date", "")
        languages = doc.get("languages", [])
        formats = doc.get("formats", [])
        hash_tags = doc.get("hash_tags", [])
        duration = doc.get("duration")
        age_limit = doc.get("ageLimit")
        extra = []
        if showtime:
            extra.append(f"showtime={showtime[:16]}")
        if release_date:
            extra.append(f"release={release_date[:10]}")
        if languages:
            extra.append(f"lang={languages}")
        if latlng and "lat" in latlng and "lng" in latlng:
            extra.append(f"latlng=({latlng['lat']}, {latlng['lng']})")
            if origin_latlng is not None:
                dist_km = haversine_km(
                    origin_latlng[0], origin_latlng[1], latlng["lat"], latlng["lng"]
                )
                extra.append(f"dist={dist_km:.1f}km")
        if formats:
            extra.append(f"formats={formats}")
        if hash_tags:
            extra.append(f"tags={hash_tags}")
        if duration is not None:
            extra.append(f"duration={duration}m")
        if age_limit is not None:
            extra.append(f"ageLimit={age_limit}+")
        extra_str = " | " + ", ".join(extra) if extra else ""
        print(f"  #{rank}: [{rec_id}] ({rec_type}) {rec_title} — {city}{extra_str}")


# ==============================================================================
# 12 NATURAL LANGUAGE BENCHMARK QUERIES & EXTRACTED-FIELD BOOST SPECS
# Reference "Today": Wednesday, Oct 7, 2026 (2026-10-07+05:30)
# ==============================================================================
BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "query": "open mic next week",
        "what_query": "open mic",
        "difficulty": "easy",
        "tags": "time, genre mood",
        "extracted": "what: open mic, when: next week (2026-10-12 to 2026-10-18)",
        "origin_latlng": None,
        "boost_specs": [
            {
                "condition": 'searchKeywords: ANY("open mic") OR hash_tags: ANY("OPEN_MIC")',
                "boost": 0.6,
            },
            {
                "condition": (
                    'showtime_start >= "2026-10-12T00:00:00+05:30" '
                    'AND showtime_start <= "2026-10-18T23:59:59+05:30"'
                ),
                "boost": 0.9,
            },
        ],
    },
    {
        "id": "Q2",
        "query": "stand up comedy tonight in pune",
        "what_query": "stand up comedy",
        "difficulty": "easy",
        "tags": "time, location, genre mood",
        "extracted": "what: stand up comedy, when: tonight (2026-10-07), where: pune",
        "origin_latlng": (18.5204, 73.8567),  # Pune city center
        "boost_specs": [
            {
                "condition": "location_city:GEO_DISTANCE(18.5204, 73.8567, 50000)",
                "boost": 0.8,
            },
            {
                "condition": (
                    'showtime_start >= "2026-10-07T17:00:00+05:30" '
                    'AND showtime_start <= "2026-10-07T23:59:59+05:30"'
                ),
                "boost": 0.9,
            },
            {
                "condition": 'searchKeywords: ANY("standup comedy", "stand up comedy")',
                "boost": 0.5,
            },
        ],
    },
    {
        "id": "Q3",
        "query": "movies releasing this friday",
        "what_query": "movies releasing friday",
        "difficulty": "easy",
        "tags": "time",
        "extracted": "what: movie, sort: latest, when: this friday (2026-10-09)",
        "origin_latlng": None,
        "boost_specs": [
            {
                "condition": 'record_type: ANY("movie")',
                "boost": 0.5,
            },
            {
                "condition": (
                    'release_date >= "2026-10-09T00:00:00+05:30" '
                    'AND release_date <= "2026-10-09T23:59:59+05:30"'
                ),
                "boost": 0.9,
            },
            {
                "condition": 'hash_tags: ANY("FRIDAY_RELEASE", "NEW_RELEASE")',
                "boost": 0.6,
            },
        ],
    },
    {
        "id": "Q4",
        "query": "kids events this sunday morning",
        "what_query": "events",
        "difficulty": "easy",
        "tags": "time, audience",
        "extracted": "what: events, audience: kids, when: sunday morning (2026-10-11 06:00-12:00)",
        "origin_latlng": None,
        "boost_specs": [
            {
                "condition": 'record_type: ANY("event") AND searchKeywords: ANY("kids", "kids events", "children")',
                "boost": 0.6,
            },
            {
                "condition": (
                    'showtime_start >= "2026-10-11T06:00:00+05:30" '
                    'AND showtime_start <= "2026-10-11T12:00:00+05:30"'
                ),
                "boost": 0.9,
            },
        ],
    },
    {
        "id": "Q5",
        "query": "late night shows after 10 pm",
        "what_query": "movie",
        "difficulty": "medium",
        "tags": "time",
        "extracted": "what: movie, when: after 10 pm (22:00+)",
        "origin_latlng": None,
        "boost_specs": [
            {
                "condition": 'record_type: ANY("movie") AND hash_tags: ANY("LATE_NIGHT_SHOW")',
                "boost": 0.8,
            },
            {
                "condition": (
                    'showtime_start >= "2026-10-07T22:00:00+05:30" '
                    'AND showtime_start <= "2026-10-08T03:00:00+05:30"'
                ),
                "boost": 0.9,
            },
        ],
    },
    {
        "id": "Q6",
        "query": "garba nights between 10th and 20th october",
        "what_query": "garba night",
        "difficulty": "medium",
        "tags": "time, genre mood",
        "extracted": "what: garba night, when: 10th to 20th october (2026-10-10..2026-10-20)",
        "origin_latlng": None,
        "boost_specs": [
            {
                "condition": 'searchKeywords: ANY("garba", "garba night", "garba nights", "dandiya")',
                "boost": 0.6,
            },
            {
                "condition": (
                    'showtime_start >= "2026-10-10T00:00:00+05:30" '
                    'AND showtime_start <= "2026-10-20T23:59:59+05:30"'
                ),
                "boost": 0.9,
            },
        ],
    },
    {
        "id": "Q7",
        "query": "new year eve parties in goa",
        "what_query": "new year eve party",
        "difficulty": "medium",
        "tags": "time, location, genre mood",
        "extracted": "what: party, when: new year eve (2026-12-31), where: goa",
        "origin_latlng": (15.4989, 73.8278),  # Goa
        "boost_specs": [
            {
                "condition": "location_city:GEO_DISTANCE(15.4989, 73.8278, 50000)",
                "boost": 0.8,
            },
            {
                "condition": (
                    'showtime_start >= "2026-12-31T18:00:00+05:30" '
                    'AND showtime_start <= "2027-01-01T06:00:00+05:30"'
                ),
                "boost": 0.9,
            },
            {
                "condition": 'searchKeywords: ANY("party", "new year eve", "nye")',
                "boost": 0.5,
            },
        ],
    },
    {
        "id": "Q8",
        "query": "is weekend delhi mein kya chal raha hai",
        "what_query": "events",
        "difficulty": "hard",
        "tags": "time, location, vague, code mixed",
        "extracted": "what: events, when: this weekend (2026-10-10..2026-10-11), where: delhi",
        "origin_latlng": (28.6139, 77.2090),  # Delhi NCR
        "boost_specs": [
            {
                "condition": 'record_type: ANY("event") AND location_city:GEO_DISTANCE(28.6139, 77.2090, 50000)',
                "boost": 0.8,
            },
            {
                "condition": (
                    'showtime_start >= "2026-10-10T00:00:00+05:30" '
                    'AND showtime_start <= "2026-10-11T23:59:59+05:30"'
                ),
                "boost": 0.9,
            },
        ],
    },
    {
        "id": "Q9",
        "query": "upcoming telugu movies next month",
        "what_query": "upcoming telugu movies",
        "difficulty": "easy",
        "tags": "time, language",
        "extracted": "what: movie, language: telugu, when: next month (Nov 2026)",
        "origin_latlng": None,
        "boost_specs": [
            {
                "condition": 'record_type: ANY("movie") AND languages: ANY("Telugu")',
                "boost": 0.7,
            },
            {
                "condition": (
                    'release_date >= "2026-11-01T00:00:00+05:30" '
                    'AND release_date <= "2026-11-30T23:59:59+05:30"'
                ),
                "boost": 0.9,
            },
        ],
    },
    {
        "id": "Q10",
        "query": "morning shows tomorrow before 11",
        "what_query": "movie",
        "difficulty": "medium",
        "tags": "time",
        "extracted": "what: movie, when: tomorrow before 11 (2026-10-08 06:00-11:00)",
        "origin_latlng": None,
        "boost_specs": [
            {
                "condition": 'record_type: ANY("movie") AND hash_tags: ANY("MORNING_SHOW")',
                "boost": 0.6,
            },
            {
                "condition": (
                    'showtime_start >= "2026-10-08T06:00:00+05:30" '
                    'AND showtime_start < "2026-10-08T11:00:00+05:30"'
                ),
                "boost": 0.9,
            },
        ],
    },
    {
        "id": "Q11",
        "query": "plays on gandhi jayanti holiday",
        "what_query": "play theatre",
        "difficulty": "hard",
        "tags": "time, genre mood, ambiguity",
        "extracted": "what: play, when: 2nd october (2026-10-02)",
        "origin_latlng": None,
        "boost_specs": [
            {
                "condition": 'searchKeywords: ANY("play", "plays", "theatre", "drama")',
                "boost": 0.6,
            },
            {
                "condition": (
                    'showtime_start >= "2026-10-02T00:00:00+05:30" '
                    'AND showtime_start <= "2026-10-02T23:59:59+05:30"'
                ),
                "boost": 0.9,
            },
        ],
    },
    {
        "id": "Q12",
        "query": "pottery classes every saturday in october",
        "what_query": "pottery workshop",
        "difficulty": "hard",
        "tags": "time, genre mood",
        "extracted": "what: pottery workshop, when: every saturday in october (Oct 3, 10, 17, 24, 31)",
        "origin_latlng": None,
        "boost_specs": [
            {
                "condition": 'searchKeywords: ANY("pottery", "pottery classes", "pottery workshop")',
                "boost": 0.6,
            },
            {
                "condition": (
                    'hash_tags: ANY("SATURDAY_OCTOBER") OR ('
                    '(showtime_start >= "2026-10-10T00:00:00+05:30" AND showtime_start <= "2026-10-10T23:59:59+05:30") OR '
                    '(showtime_start >= "2026-10-17T00:00:00+05:30" AND showtime_start <= "2026-10-17T23:59:59+05:30")'
                    ')'
                ),
                "boost": 0.9,
            },
        ],
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query Vertex AI Search App via REST API with boostSpec"
    )
    parser.add_argument("-q", "--query", type=str, default="", help="Custom search query")
    parser.add_argument(
        "-c",
        "--condition",
        type=str,
        default="",
        help='Boost filter condition, e.g. \'location_city:GEO_DISTANCE(13.0827, 80.2707, 50000)\'',
    )
    parser.add_argument(
        "-b",
        "--boost",
        type=float,
        default=0.8,
        help="Boost value in [-1.0, 1.0]",
    )
    args = parser.parse_args()

    if args.condition or args.query:
        q = args.query or "Drishyam"
        specs = [{"condition": args.condition, "boost": args.boost}] if args.condition else None
        res = search_with_boost(query=q, condition_boost_specs=specs)
        print_results(
            f"Query: {q!r} | Boost: condition={args.condition!r}, boost={args.boost}",
            res,
        )
    else:
        # Part 1: Geo-Distance & Core Boost Verification
        q1 = "Welcome to the Jungle"
        chennai_origin = (13.0827, 80.2707)
        boost_specs_concentric_geo = [
            {"condition": "location_city:GEO_DISTANCE(13.0827, 80.2707, 100000)", "boost": 0.4},
            {"condition": "location_city:GEO_DISTANCE(13.0827, 80.2707, 800000)", "boost": 0.3},
            {"condition": "location_city:GEO_DISTANCE(13.0827, 80.2707, 1150000)", "boost": 0.2},
            {"condition": "location_city:GEO_DISTANCE(13.0827, 80.2707, 1500000)", "boost": 0.1},
        ]
        print_results(
            f"GEO. Concentric GEO_DISTANCE Rings (Strict Proximity Order from Chennai) — Query: {q1!r}",
            search_with_boost(
                query=q1, condition_boost_specs=boost_specs_concentric_geo, page_size=5
            ),
            origin_latlng=chennai_origin,
        )

        # Part 2: Run all 12 Benchmark Queries (Baseline vs. Boosted)
        for bq in BENCHMARK_QUERIES:
            print("\n" + "=" * 100)
            print(
                f"[{bq['id']}] User Query: {bq['query']!r} | Difficulty: {bq['difficulty']} "
                f"| Tags: {bq['tags']} | Extracted: {bq['extracted']}"
            )
            print("=" * 100)
            search_q = bq["what_query"]
            print_results(
                f"{bq['id']}-A. Baseline (No Boost) — query={search_q!r}",
                search_with_boost(query=search_q, page_size=4),
                origin_latlng=bq["origin_latlng"],
            )
            print_results(
                f"{bq['id']}-B. With Boost ({json.dumps(bq['boost_specs'])}) — query={search_q!r}",
                search_with_boost(
                    query=search_q,
                    condition_boost_specs=bq["boost_specs"],
                    page_size=4,
                ),
                origin_latlng=bq["origin_latlng"],
            )


# ==============================================================================
# EXECUTION OUTPUT (`python query_with_boost.py`)
# ==============================================================================
#
# === GEO. Concentric GEO_DISTANCE Rings (Strict Proximity Order from Chennai) — Query: 'Welcome to the Jungle' ===
# Total Matches: 5 (Showing top 5)
#   #1: [MV00100] (movie) Welcome to the Jungle — Chennai | showtime=2026-10-09T16:00, release=2026-09-14, lang=['Hindi'], latlng=(13.0556, 80.258), dist=3.3km, formats=['3D'], tags=['BLOCKBUSTER']
#   #2: [MV00098] (movie) Welcome to the Jungle — Goa | showtime=2026-10-23T16:00, release=2026-09-14, lang=['Hindi'], latlng=(15.4989, 73.8278), dist=744.3km, formats=['3D'], tags=['FAMILY_ENTERTAINER']
#   #3: [MV00096] (movie) Welcome to the Jungle — Mumbai | showtime=2026-10-18T09:30, release=2026-09-14, lang=['Hindi'], latlng=(18.9946, 72.8245), dist=1031.8km, formats=['3D'], tags=['MUST_WATCH', 'MORNING_SHOW']
#   #4: [MV00097] (movie) Welcome to the Jungle — Kolkata | showtime=2026-10-20T19:15, release=2026-09-14, lang=['Hindi'], latlng=(22.539, 88.3654), dist=1355.6km, formats=['3D'], tags=['CRITICALLY_ACCLAIMED']
#   #5: [MV00099] (movie) Welcome to the Jungle — Delhi NCR | showtime=2026-10-25T22:30, release=2026-09-14, lang=['Hindi'], latlng=(28.5286, 77.2193), dist=1746.3km, formats=['3D'], tags=['MUST_WATCH', 'LATE_NIGHT_SHOW']
#
# ====================================================================================================
# [Q1] User Query: 'open mic next week' | Difficulty: easy | Tags: time, genre mood | Extracted: what: open mic, when: next week (2026-10-12 to 2026-10-18)
# ====================================================================================================
#
# === Q1-A. Baseline (No Boost) — query='open mic' ===
# Total Matches: 3 (Showing top 3)
#   #1: [etm100100z] (event) UNSCRIPTED OPEN MIC JAM - DELHI NCR — Delhi NCR | showtime=2026-11-22T19:00, release=2026-09-26, lang=['Hindi', 'Urdu'], latlng=(28.5829, 77.2344), tags=['OPEN_MIC'], duration=120m, ageLimit=16+
#   #2: [etm100102z] (event) BLR BREWING ACOUSTIC & STANDUP OPEN MIC - BENGALURU — Bengaluru | showtime=2026-10-16T19:30, release=2026-09-26, lang=['English', 'Hindi'], latlng=(12.9975, 77.6964), tags=['OPEN_MIC', 'NEXT_WEEK'], duration=120m, ageLimit=16+
#   #3: [etm100101z] (event) THE HABITAT COMEDY & POETRY OPEN MIC - MUMBAI — Mumbai | showtime=2026-10-14T20:00, release=2026-09-26, lang=['Hindi', 'English'], latlng=(19.009, 72.8175), tags=['OPEN_MIC', 'NEXT_WEEK'], duration=120m, ageLimit=16+
#
# === Q1-B. With Boost ([{"condition": "searchKeywords: ANY(\"open mic\") OR hash_tags: ANY(\"OPEN_MIC\")", "boost": 0.6}, {"condition": "showtime_start >= \"2026-10-12T00:00:00+05:30\" AND showtime_start <= \"2026-10-18T23:59:59+05:30\"", "boost": 0.9}]) — query='open mic' ===
# Total Matches: 3 (Showing top 3)
#   #1: [etm100102z] (event) BLR BREWING ACOUSTIC & STANDUP OPEN MIC - BENGALURU — Bengaluru | showtime=2026-10-16T19:30, release=2026-09-26, lang=['English', 'Hindi'], latlng=(12.9975, 77.6964), tags=['OPEN_MIC', 'NEXT_WEEK'], duration=120m, ageLimit=16+
#   #2: [etm100101z] (event) THE HABITAT COMEDY & POETRY OPEN MIC - MUMBAI — Mumbai | showtime=2026-10-14T20:00, release=2026-09-26, lang=['Hindi', 'English'], latlng=(19.009, 72.8175), tags=['OPEN_MIC', 'NEXT_WEEK'], duration=120m, ageLimit=16+
#   #3: [etm100100z] (event) UNSCRIPTED OPEN MIC JAM - DELHI NCR — Delhi NCR | showtime=2026-11-22T19:00, release=2026-09-26, lang=['Hindi', 'Urdu'], latlng=(28.5829, 77.2344), tags=['OPEN_MIC'], duration=120m, ageLimit=16+
#
# ====================================================================================================
# [Q2] User Query: 'stand up comedy tonight in pune' | Difficulty: easy | Tags: time, location, genre mood | Extracted: what: stand up comedy, when: tonight (2026-10-07), where: pune
# ====================================================================================================
#
# === Q2-A. Baseline (No Boost) — query='stand up comedy' ===
# Total Matches: 19 (Showing top 4)
#   #1: [etm100104z] (event) BASSI KISI KO BATANA MAT STAND UP COMEDY - PUNE — Pune | showtime=2026-10-07T21:00, release=2026-09-26, lang=['Hindi'], latlng=(18.4892, 73.8205), dist=5.2km, tags=['TONIGHT', 'STANDUP_COMEDY'], duration=100m, ageLimit=16+
#   #2: [etm100103z] (event) AAKASH GUPTA STAND UP COMEDY SPECIAL - PUNE — Pune | showtime=2026-10-07T20:30, release=2026-09-26, lang=['Hindi', 'English'], latlng=(18.4892, 73.8205), dist=5.2km, tags=['TONIGHT', 'STANDUP_COMEDY'], duration=90m, ageLimit=18+
#   #3: [etm100042z] (event) ABHISHEK UPMANYU LIVE - MUMBAI — Mumbai | showtime=2026-10-10T21:00, release=2026-08-14, lang=['Hindi', 'English'], latlng=(19.009, 72.8175), dist=122.2km, duration=90m, ageLimit=16+
#   #4: [etm100012z] (event) ZAKIR KHAN LIVE - BENGALURU — Bengaluru | showtime=2026-10-11T16:00, release=2026-08-17, lang=['Hindi'], latlng=(12.9975, 77.6964), dist=738.8km, duration=90m, ageLimit=16+
#
# === Q2-B. With Boost ([{"condition": "location_city:GEO_DISTANCE(18.5204, 73.8567, 50000)", "boost": 0.8}, {"condition": "showtime_start >= \"2026-10-07T17:00:00+05:30\" AND showtime_start <= \"2026-10-07T23:59:59+05:30\"", "boost": 0.9}, {"condition": "searchKeywords: ANY(\"standup comedy\", \"stand up comedy\")", "boost": 0.5}]) — query='stand up comedy' ===
# Total Matches: 19 (Showing top 4)
#   #1: [etm100104z] (event) BASSI KISI KO BATANA MAT STAND UP COMEDY - PUNE — Pune | showtime=2026-10-07T21:00, release=2026-09-26, lang=['Hindi'], latlng=(18.4892, 73.8205), dist=5.2km, tags=['TONIGHT', 'STANDUP_COMEDY'], duration=100m, ageLimit=16+
#   #2: [etm100103z] (event) AAKASH GUPTA STAND UP COMEDY SPECIAL - PUNE — Pune | showtime=2026-10-07T20:30, release=2026-09-26, lang=['Hindi', 'English'], latlng=(18.4892, 73.8205), dist=5.2km, tags=['TONIGHT', 'STANDUP_COMEDY'], duration=90m, ageLimit=18+
#   #3: [etm100032z] (event) ANUBHAV SINGH BASSI LIVE - PUNE — Pune | showtime=2026-10-25T19:30, release=2026-10-05, lang=['Hindi'], latlng=(18.4892, 73.8205), dist=5.2km, duration=90m, ageLimit=16+
#   #4: [etm100042z] (event) ABHISHEK UPMANYU LIVE - MUMBAI — Mumbai | showtime=2026-10-10T21:00, release=2026-08-14, lang=['Hindi', 'English'], latlng=(19.009, 72.8175), dist=122.2km, duration=90m, ageLimit=16+
#
# ====================================================================================================
# [Q3] User Query: 'movies releasing this friday' | Difficulty: easy | Tags: time | Extracted: what: movie, sort: latest, when: this friday (2026-10-09)
# ====================================================================================================
#
# === Q3-A. Baseline (No Boost) — query='movies releasing friday' ===
# Total Matches: 3 (Showing top 3)
#   #1: [MV00101] (movie) Singham Again — Mumbai | showtime=2026-10-09T19:30, release=2026-09-11, lang=['Hindi'], latlng=(18.9946, 72.8245), formats=['IMAX 2D'], tags=['BLOCKBUSTER']
#   #2: [MV00102] (movie) Bhool Bhulaiyaa 4 — Delhi NCR | showtime=2026-10-09T20:00, release=2026-10-09, lang=['Hindi'], latlng=(28.5286, 77.2193), formats=['2D', '4DX'], tags=['NEW_RELEASE', 'FRIDAY_RELEASE', 'FAMILY_ENTERTAINER']
#   #3: [MV00103] (movie) Vettaiyan: The Hunter — Chennai | showtime=2026-10-09T18:30, release=2026-10-09, lang=['Tamil', 'Telugu', 'Hindi'], latlng=(13.0556, 80.258), formats=['IMAX 2D', 'EPIQ'], tags=['NEW_RELEASE', 'FRIDAY_RELEASE', 'MUST_WATCH']
#
# === Q3-B. With Boost ([{"condition": "record_type: ANY(\"movie\")", "boost": 0.5}, {"condition": "release_date >= \"2026-10-09T00:00:00+05:30\" AND release_date <= \"2026-10-09T23:59:59+05:30\"", "boost": 0.9}, {"condition": "hash_tags: ANY(\"FRIDAY_RELEASE\", \"NEW_RELEASE\")", "boost": 0.6}]) — query='movies releasing friday' ===
# Total Matches: 3 (Showing top 3)
#   #1: [MV00102] (movie) Bhool Bhulaiyaa 4 — Delhi NCR | showtime=2026-10-09T20:00, release=2026-10-09, lang=['Hindi'], latlng=(28.5286, 77.2193), formats=['2D', '4DX'], tags=['NEW_RELEASE', 'FRIDAY_RELEASE', 'FAMILY_ENTERTAINER']
#   #2: [MV00103] (movie) Vettaiyan: The Hunter — Chennai | showtime=2026-10-09T18:30, release=2026-10-09, lang=['Tamil', 'Telugu', 'Hindi'], latlng=(13.0556, 80.258), formats=['IMAX 2D', 'EPIQ'], tags=['NEW_RELEASE', 'FRIDAY_RELEASE', 'MUST_WATCH']
#   #3: [MV00101] (movie) Singham Again — Mumbai | showtime=2026-10-09T19:30, release=2026-09-11, lang=['Hindi'], latlng=(18.9946, 72.8245), formats=['IMAX 2D'], tags=['BLOCKBUSTER']
#
# ====================================================================================================
# [Q4] User Query: 'kids events this sunday morning' | Difficulty: easy | Tags: time, audience | Extracted: what: events, audience: kids, when: sunday morning (2026-10-11 06:00-12:00)
# ====================================================================================================
#
# === Q4-A. Baseline (No Boost) — query='events' ===
# Total Matches: 12 (Showing top 4)
#   #1: [etm100113z] (event) DELHI SUFI & STREET FOOD FESTIVAL - DELHI NCR — Delhi NCR | showtime=2026-10-10T18:00, release=2026-09-26, lang=['Hindi', 'Urdu', 'Punjabi'], latlng=(28.5829, 77.2344), tags=['THIS_WEEKEND', 'DELHI_EVENTS'], duration=300m
#   #2: [etm100114z] (event) JASHN-E-DILLI CARNIVAL & COMEDY NIGHT - DELHI NCR — Delhi NCR | showtime=2026-10-11T17:30, release=2026-09-26, lang=['Hindi', 'English'], latlng=(28.5829, 77.2344), tags=['THIS_WEEKEND', 'DELHI_EVENTS'], duration=240m, ageLimit=12+
#   #3: [etm100106z] (event) JUNIOR ROBOTICS & SCIENCE KIDS WORKSHOP - BENGALURU — Bengaluru | showtime=2026-10-11T10:00, release=2026-09-26, lang=['English'], latlng=(12.9975, 77.6964), tags=['KIDS_EVENT', 'MORNING_SHOW', 'SUNDAY_MORNING'], duration=150m, ageLimit=5+
#   #4: [etm100095z] (event) KIDS SCIENCE CARNIVAL - DELHI NCR — Delhi NCR | showtime=2026-10-23T21:00, release=2026-08-13, lang=['English'], latlng=(28.5829, 77.2344), duration=240m
#
# === Q4-B. With Boost ([{"condition": "record_type: ANY(\"event\") AND searchKeywords: ANY(\"kids\", \"kids events\", \"children\")", "boost": 0.6}, {"condition": "showtime_start >= \"2026-10-11T06:00:00+05:30\" AND showtime_start <= \"2026-10-11T12:00:00+05:30\"", "boost": 0.9}]) — query='events' ===
# Total Matches: 12 (Showing top 4)
#   #1: [etm100106z] (event) JUNIOR ROBOTICS & SCIENCE KIDS WORKSHOP - BENGALURU — Bengaluru | showtime=2026-10-11T10:00, release=2026-09-26, lang=['English'], latlng=(12.9975, 77.6964), tags=['KIDS_EVENT', 'MORNING_SHOW', 'SUNDAY_MORNING'], duration=150m, ageLimit=5+
#   #2: [etm100105z] (event) LITTLE EINSTEINS KIDS MAGIC & PUPPET CARNIVAL - MUMBAI — Mumbai | showtime=2026-10-11T09:30, release=2026-09-26, lang=['English', 'Hindi'], latlng=(19.009, 72.8175), tags=['KIDS_EVENT', 'MORNING_SHOW', 'SUNDAY_MORNING'], duration=120m, ageLimit=3+
#   #3: [etm100095z] (event) KIDS SCIENCE CARNIVAL - DELHI NCR — Delhi NCR | showtime=2026-10-23T21:00, release=2026-08-13, lang=['English'], latlng=(28.5829, 77.2344), duration=240m
#   #4: [etm100096z] (event) KIDS SCIENCE CARNIVAL - BENGALURU — Bengaluru | showtime=2026-10-18T18:30, release=2026-08-13, lang=['English'], latlng=(12.9975, 77.6964), duration=240m
#
# ====================================================================================================
# [Q5] User Query: 'late night shows after 10 pm' | Difficulty: medium | Tags: time | Extracted: what: movie, when: after 10 pm (22:00+)
# ====================================================================================================
#
# === Q5-A. Baseline (No Boost) — query='movie' ===
# Total Matches: 108 (Showing top 4)
#   #1: [MV00017] (movie) Bramayugam 2 — Chennai | showtime=2026-10-09T09:30, release=2026-09-28, lang=['Malayalam'], latlng=(13.0556, 80.258), formats=['4DX'], tags=['CRITICALLY_ACCLAIMED', 'MORNING_SHOW']
#   #2: [MV00018] (movie) Bramayugam 2 — Hyderabad | showtime=2026-10-08T22:30, release=2026-09-28, lang=['Malayalam'], latlng=(17.4126, 78.4664), formats=['4DX'], tags=['FAMILY_ENTERTAINER', 'CRITICALLY_ACCLAIMED', 'LATE_NIGHT_SHOW']
#   #3: [MV00077] (movie) Metro In Dino 2 — Mumbai | showtime=2026-10-24T19:15, release=2026-09-17, lang=['Hindi'], latlng=(18.9946, 72.8245), formats=['IMAX 3D'], tags=['CRITICALLY_ACCLAIMED']
#   #4: [MV00084] (movie) Kantara: Chapter 2 — Chennai | showtime=2026-10-09T19:15, release=2026-09-07, lang=['Kannada', 'Telugu', 'Hindi'], latlng=(13.0556, 80.258), formats=['ICE'], tags=['MUST_WATCH', 'BLOCKBUSTER']
#
# === Q5-B. With Boost ([{"condition": "record_type: ANY(\"movie\") AND hash_tags: ANY(\"LATE_NIGHT_SHOW\")", "boost": 0.8}, {"condition": "showtime_start >= \"2026-10-07T22:00:00+05:30\" AND showtime_start <= \"2026-10-08T03:00:00+05:30\"", "boost": 0.9}]) — query='movie' ===
# Total Matches: 108 (Showing top 4)
#   #1: [MV00106] (movie) The Batman Part II — Bengaluru | showtime=2026-10-07T23:15, release=2026-10-02, lang=['English', 'Hindi'], latlng=(13.011, 77.555), formats=['IMAX 2D', 'DOLBY CINEMA 2D'], tags=['LATE_NIGHT_SHOW', 'BLOCKBUSTER']
#   #2: [MV00105] (movie) John Wick: Chapter 5 — Mumbai | showtime=2026-10-07T22:45, release=2026-10-02, lang=['English', 'Hindi'], latlng=(18.9946, 72.8245), formats=['IMAX 2D', 'DOLBY CINEMA 2D'], tags=['LATE_NIGHT_SHOW', 'MUST_WATCH']
#   #3: [MV00045] (movie) Coolie — Chennai | showtime=2026-10-07T22:30, release=2026-09-18, lang=['Tamil', 'Telugu'], latlng=(13.0556, 80.258), formats=['2D'], tags=['CRITICALLY_ACCLAIMED', 'LATE_NIGHT_SHOW']
#   #4: [MV00018] (movie) Bramayugam 2 — Hyderabad | showtime=2026-10-08T22:30, release=2026-09-28, lang=['Malayalam'], latlng=(17.4126, 78.4664), formats=['4DX'], tags=['FAMILY_ENTERTAINER', 'CRITICALLY_ACCLAIMED', 'LATE_NIGHT_SHOW']
#
# ====================================================================================================
# [Q6] User Query: 'garba nights between 10th and 20th october' | Difficulty: medium | Tags: time, genre mood | Extracted: what: garba night, when: 10th to 20th october (2026-10-10..2026-10-20)
# ====================================================================================================
#
# === Q6-A. Baseline (No Boost) — query='garba night' ===
# Total Matches: 3 (Showing top 3)
#   #1: [etm100108z] (event) FALGUNI PATHAK NAVRATRI GARBA NIGHTS - MUMBAI — Mumbai | showtime=2026-10-12T20:00, release=2026-09-26, lang=['Gujarati', 'Hindi'], latlng=(19.009, 72.8175), tags=['GARBA_NIGHT', 'NAVRATRI'], duration=240m, ageLimit=5+
#   #2: [etm100109z] (event) ROYAL RAAS GARBA & DANDIYA NIGHTS - PUNE — Pune | showtime=2026-10-17T19:30, release=2026-09-26, lang=['Gujarati', 'Hindi'], latlng=(18.4892, 73.8205), tags=['GARBA_NIGHT', 'NAVRATRI'], duration=240m, ageLimit=5+
#   #3: [etm100107z] (event) SHARAD POORNIMA GARBA NIGHTS FINALE - DELHI NCR — Delhi NCR | showtime=2026-10-28T20:00, release=2026-09-26, lang=['Gujarati', 'Hindi'], latlng=(28.5829, 77.2344), tags=['GARBA_NIGHT'], duration=210m, ageLimit=5+
#
# === Q6-B. With Boost ([{"condition": "searchKeywords: ANY(\"garba\", \"garba night\", \"garba nights\", \"dandiya\")", "boost": 0.6}, {"condition": "showtime_start >= \"2026-10-10T00:00:00+05:30\" AND showtime_start <= \"2026-10-20T23:59:59+05:30\"", "boost": 0.9}]) — query='garba night' ===
# Total Matches: 3 (Showing top 3)
#   #1: [etm100108z] (event) FALGUNI PATHAK NAVRATRI GARBA NIGHTS - MUMBAI — Mumbai | showtime=2026-10-12T20:00, release=2026-09-26, lang=['Gujarati', 'Hindi'], latlng=(19.009, 72.8175), tags=['GARBA_NIGHT', 'NAVRATRI'], duration=240m, ageLimit=5+
#   #2: [etm100109z] (event) ROYAL RAAS GARBA & DANDIYA NIGHTS - PUNE — Pune | showtime=2026-10-17T19:30, release=2026-09-26, lang=['Gujarati', 'Hindi'], latlng=(18.4892, 73.8205), tags=['GARBA_NIGHT', 'NAVRATRI'], duration=240m, ageLimit=5+
#   #3: [etm100107z] (event) SHARAD POORNIMA GARBA NIGHTS FINALE - DELHI NCR — Delhi NCR | showtime=2026-10-28T20:00, release=2026-09-26, lang=['Gujarati', 'Hindi'], latlng=(28.5829, 77.2344), tags=['GARBA_NIGHT'], duration=210m, ageLimit=5+
#
# ====================================================================================================
# [Q7] User Query: 'new year eve parties in goa' | Difficulty: medium | Tags: time, location, genre mood | Extracted: what: party, when: new year eve (2026-12-31), where: goa
# ====================================================================================================
#
# === Q7-A. Baseline (No Boost) — query='new year eve party' ===
# Total Matches: 3 (Showing top 3)
#   #1: [etm100110z] (event) MARINE DRIVE ROOFTOP NEW YEAR EVE PARTY - MUMBAI — Mumbai | showtime=2026-12-31T21:00, release=2026-09-26, lang=['Hindi', 'English'], latlng=(19.009, 72.8175), dist=404.8km, tags=['NEW_YEAR_EVE', 'PARTY'], duration=360m, ageLimit=21+
#   #2: [etm100112z] (event) THALASSA SUNSET TO SUNRISE NEW YEAR EVE PARTY - GOA — Goa | showtime=2026-12-31T19:00, release=2026-09-26, lang=['English'], latlng=(15.6029, 73.7339), dist=15.3km, tags=['NEW_YEAR_EVE', 'PARTY'], duration=480m, ageLimit=21+
#   #3: [etm100111z] (event) VAGATOR BEACH NEW YEAR EVE MEGA PARTY - GOA — Goa | showtime=2026-12-31T20:00, release=2026-09-26, lang=['English', 'Hindi'], latlng=(15.6029, 73.7339), dist=15.3km, tags=['NEW_YEAR_EVE', 'PARTY'], duration=420m, ageLimit=18+
#
# === Q7-B. With Boost ([{"condition": "location_city:GEO_DISTANCE(15.4989, 73.8278, 50000)", "boost": 0.8}, {"condition": "showtime_start >= \"2026-12-31T18:00:00+05:30\" AND showtime_start <= \"2027-01-01T06:00:00+05:30\"", "boost": 0.9}, {"condition": "searchKeywords: ANY(\"party\", \"new year eve\", \"nye\")", "boost": 0.5}]) — query='new year eve party' ===
# Total Matches: 3 (Showing top 3)
#   #1: [etm100112z] (event) THALASSA SUNSET TO SUNRISE NEW YEAR EVE PARTY - GOA — Goa | showtime=2026-12-31T19:00, release=2026-09-26, lang=['English'], latlng=(15.6029, 73.7339), dist=15.3km, tags=['NEW_YEAR_EVE', 'PARTY'], duration=480m, ageLimit=21+
#   #2: [etm100111z] (event) VAGATOR BEACH NEW YEAR EVE MEGA PARTY - GOA — Goa | showtime=2026-12-31T20:00, release=2026-09-26, lang=['English', 'Hindi'], latlng=(15.6029, 73.7339), dist=15.3km, tags=['NEW_YEAR_EVE', 'PARTY'], duration=420m, ageLimit=18+
#   #3: [etm100110z] (event) MARINE DRIVE ROOFTOP NEW YEAR EVE PARTY - MUMBAI — Mumbai | showtime=2026-12-31T21:00, release=2026-09-26, lang=['Hindi', 'English'], latlng=(19.009, 72.8175), dist=404.8km, tags=['NEW_YEAR_EVE', 'PARTY'], duration=360m, ageLimit=21+
#
# ====================================================================================================
# [Q8] User Query: 'is weekend delhi mein kya chal raha hai' | Difficulty: hard | Tags: time, location, vague, code mixed | Extracted: what: events, when: this weekend (2026-10-10..2026-10-11), where: delhi
# ====================================================================================================
#
# === Q8-A. Baseline (No Boost) — query='events' ===
# Total Matches: 12 (Showing top 4)
#   #1: [etm100113z] (event) DELHI SUFI & STREET FOOD FESTIVAL - DELHI NCR — Delhi NCR | showtime=2026-10-10T18:00, release=2026-09-26, lang=['Hindi', 'Urdu', 'Punjabi'], latlng=(28.5829, 77.2344), dist=4.2km, tags=['THIS_WEEKEND', 'DELHI_EVENTS'], duration=300m
#   #2: [etm100114z] (event) JASHN-E-DILLI CARNIVAL & COMEDY NIGHT - DELHI NCR — Delhi NCR | showtime=2026-10-11T17:30, release=2026-09-26, lang=['Hindi', 'English'], latlng=(28.5829, 77.2344), dist=4.2km, tags=['THIS_WEEKEND', 'DELHI_EVENTS'], duration=240m, ageLimit=12+
#   #3: [etm100106z] (event) JUNIOR ROBOTICS & SCIENCE KIDS WORKSHOP - BENGALURU — Bengaluru | showtime=2026-10-11T10:00, release=2026-09-26, lang=['English'], latlng=(12.9975, 77.6964), dist=1737.2km, tags=['KIDS_EVENT', 'MORNING_SHOW', 'SUNDAY_MORNING'], duration=150m, ageLimit=5+
#   #4: [etm100095z] (event) KIDS SCIENCE CARNIVAL - DELHI NCR — Delhi NCR | showtime=2026-10-23T21:00, release=2026-08-13, lang=['English'], latlng=(28.5829, 77.2344), dist=4.2km, duration=240m
#
# === Q8-B. With Boost ([{"condition": "record_type: ANY(\"event\") AND location_city:GEO_DISTANCE(28.6139, 77.2090, 50000)", "boost": 0.8}, {"condition": "showtime_start >= \"2026-10-10T00:00:00+05:30\" AND showtime_start <= \"2026-10-11T23:59:59+05:30\"", "boost": 0.9}]) — query='events' ===
# Total Matches: 12 (Showing top 4)
#   #1: [etm100113z] (event) DELHI SUFI & STREET FOOD FESTIVAL - DELHI NCR — Delhi NCR | showtime=2026-10-10T18:00, release=2026-09-26, lang=['Hindi', 'Urdu', 'Punjabi'], latlng=(28.5829, 77.2344), dist=4.2km, tags=['THIS_WEEKEND', 'DELHI_EVENTS'], duration=300m
#   #2: [etm100114z] (event) JASHN-E-DILLI CARNIVAL & COMEDY NIGHT - DELHI NCR — Delhi NCR | showtime=2026-10-11T17:30, release=2026-09-26, lang=['Hindi', 'English'], latlng=(28.5829, 77.2344), dist=4.2km, tags=['THIS_WEEKEND', 'DELHI_EVENTS'], duration=240m, ageLimit=12+
#   #3: [etm100106z] (event) JUNIOR ROBOTICS & SCIENCE KIDS WORKSHOP - BENGALURU — Bengaluru | showtime=2026-10-11T10:00, release=2026-09-26, lang=['English'], latlng=(12.9975, 77.6964), dist=1737.2km, tags=['KIDS_EVENT', 'MORNING_SHOW', 'SUNDAY_MORNING'], duration=150m, ageLimit=5+
#   #4: [etm100061z] (event) KAILASH KHER SUFI NIGHT - BENGALURU — Bengaluru | showtime=2026-10-10T16:00, release=2026-10-05, lang=['Hindi'], latlng=(12.9975, 77.6964), dist=1737.2km, duration=150m
#
# ====================================================================================================
# [Q9] User Query: 'upcoming telugu movies next month' | Difficulty: easy | Tags: time, language | Extracted: what: movie, language: telugu, when: next month (Nov 2026)
# ====================================================================================================
#
# === Q9-A. Baseline (No Boost) — query='upcoming telugu movies' ===
# Total Matches: 8 (Showing top 4)
#   #1: [MV00108] (movie) Devara Part 2: Red Sea — Hyderabad | showtime=2026-11-06T19:00, release=2026-11-06, lang=['Telugu'], latlng=(17.4126, 78.4664), formats=['IMAX 2D', 'DOLBY CINEMA 2D'], tags=['UPCOMING_MOVIE', 'MUST_WATCH', 'BLOCKBUSTER']
#   #2: [MV00107] (movie) OG: They Call Him OG — Hyderabad | showtime=2026-10-08T19:30, release=2026-10-01, lang=['Telugu'], latlng=(17.4126, 78.4664), formats=['2D'], tags=['LIVE_IN_CINEMAS']
#   #3: [MV00110] (movie) SSMB29: Globetrotter — Bengaluru | showtime=2026-11-20T19:30, release=2026-11-20, lang=['Telugu'], latlng=(13.011, 77.555), formats=['IMAX 3D', 'DOLBY CINEMA 2D'], tags=['UPCOMING_MOVIE', 'MUST_WATCH']
#   #4: [MV00109] (movie) Spirit: The Cop Story — Hyderabad | showtime=2026-11-14T18:30, release=2026-11-14, lang=['Telugu'], latlng=(17.4126, 78.4664), formats=['IMAX 2D', 'EPIQ'], tags=['UPCOMING_MOVIE', 'BLOCKBUSTER']
#
# === Q9-B. With Boost ([{"condition": "record_type: ANY(\"movie\") AND languages: ANY(\"Telugu\")", "boost": 0.7}, {"condition": "release_date >= \"2026-11-01T00:00:00+05:30\" AND release_date <= \"2026-11-30T23:59:59+05:30\"", "boost": 0.9}]) — query='upcoming telugu movies' ===
# Total Matches: 8 (Showing top 4)
#   #1: [MV00108] (movie) Devara Part 2: Red Sea — Hyderabad | showtime=2026-11-06T19:00, release=2026-11-06, lang=['Telugu'], latlng=(17.4126, 78.4664), formats=['IMAX 2D', 'DOLBY CINEMA 2D'], tags=['UPCOMING_MOVIE', 'MUST_WATCH', 'BLOCKBUSTER']
#   #2: [MV00110] (movie) SSMB29: Globetrotter — Bengaluru | showtime=2026-11-20T19:30, release=2026-11-20, lang=['Telugu'], latlng=(13.011, 77.555), formats=['IMAX 3D', 'DOLBY CINEMA 2D'], tags=['UPCOMING_MOVIE', 'MUST_WATCH']
#   #3: [MV00109] (movie) Spirit: The Cop Story — Hyderabad | showtime=2026-11-14T18:30, release=2026-11-14, lang=['Telugu'], latlng=(17.4126, 78.4664), formats=['IMAX 2D', 'EPIQ'], tags=['UPCOMING_MOVIE', 'BLOCKBUSTER']
#   #4: [MV00107] (movie) OG: They Call Him OG — Hyderabad | showtime=2026-10-08T19:30, release=2026-10-01, lang=['Telugu'], latlng=(17.4126, 78.4664), formats=['2D'], tags=['LIVE_IN_CINEMAS']
#
# ====================================================================================================
# [Q10] User Query: 'morning shows tomorrow before 11' | Difficulty: medium | Tags: time | Extracted: what: movie, when: tomorrow before 11 (2026-10-08 06:00-11:00)
# ====================================================================================================
#
# === Q10-A. Baseline (No Boost) — query='movie' ===
# Total Matches: 108 (Showing top 4)
#   #1: [MV00017] (movie) Bramayugam 2 — Chennai | showtime=2026-10-09T09:30, release=2026-09-28, lang=['Malayalam'], latlng=(13.0556, 80.258), formats=['4DX'], tags=['CRITICALLY_ACCLAIMED', 'MORNING_SHOW']
#   #2: [MV00018] (movie) Bramayugam 2 — Hyderabad | showtime=2026-10-08T22:30, release=2026-09-28, lang=['Malayalam'], latlng=(17.4126, 78.4664), formats=['4DX'], tags=['FAMILY_ENTERTAINER', 'CRITICALLY_ACCLAIMED', 'LATE_NIGHT_SHOW']
#   #3: [MV00077] (movie) Metro In Dino 2 — Mumbai | showtime=2026-10-24T19:15, release=2026-09-17, lang=['Hindi'], latlng=(18.9946, 72.8245), formats=['IMAX 3D'], tags=['CRITICALLY_ACCLAIMED']
#   #4: [MV00084] (movie) Kantara: Chapter 2 — Chennai | showtime=2026-10-09T19:15, release=2026-09-07, lang=['Kannada', 'Telugu', 'Hindi'], latlng=(13.0556, 80.258), formats=['ICE'], tags=['MUST_WATCH', 'BLOCKBUSTER']
#
# === Q10-B. With Boost ([{"condition": "record_type: ANY(\"movie\") AND hash_tags: ANY(\"MORNING_SHOW\")", "boost": 0.6}, {"condition": "showtime_start >= \"2026-10-08T06:00:00+05:30\" AND showtime_start < \"2026-10-08T11:00:00+05:30\"", "boost": 0.9}]) — query='movie' ===
# Total Matches: 108 (Showing top 4)
#   #1: [MV00112] (movie) Chhaava: The Great Warrior — Mumbai | showtime=2026-10-08T09:00, release=2026-10-02, lang=['Hindi'], latlng=(18.9946, 72.8245), formats=['IMAX 2D'], tags=['MORNING_SHOW', 'MUST_WATCH']
#   #2: [MV00113] (movie) Alpha: Spy Universe — Delhi NCR | showtime=2026-10-08T10:15, release=2026-10-02, lang=['Hindi'], latlng=(28.5286, 77.2193), formats=['2D', 'DOLBY CINEMA 2D'], tags=['MORNING_SHOW', 'BLOCKBUSTER']
#   #3: [MV00017] (movie) Bramayugam 2 — Chennai | showtime=2026-10-09T09:30, release=2026-09-28, lang=['Malayalam'], latlng=(13.0556, 80.258), formats=['4DX'], tags=['CRITICALLY_ACCLAIMED', 'MORNING_SHOW']
#   #4: [MV00079] (movie) Metro In Dino 2 — Delhi NCR | showtime=2026-10-23T09:30, release=2026-09-17, lang=['Hindi'], latlng=(28.5286, 77.2193), formats=['IMAX 3D'], tags=['MUST_WATCH', 'MORNING_SHOW']
#
# ====================================================================================================
# [Q11] User Query: 'plays on gandhi jayanti holiday' | Difficulty: hard | Tags: time, genre mood, ambiguity | Extracted: what: play, when: 2nd october (2026-10-02)
# ====================================================================================================
#
# === Q11-A. Baseline (No Boost) — query='play theatre' ===
# Total Matches: 3 (Showing top 3)
#   #1: [etm100117z] (event) TUGHLAQ: THEATRE PLAY - MUMBAI — Mumbai | showtime=2026-10-02T15:30, release=2026-09-26, lang=['Hindi'], latlng=(19.009, 72.8175), tags=['STAGE_PLAY', 'HOLIDAY_SPECIAL'], duration=150m, ageLimit=12+
#   #2: [etm100115z] (event) ANDHA YUG: CLASSIC HINDI STAGE PLAY - MUMBAI — Mumbai | showtime=2026-10-25T19:00, release=2026-09-26, lang=['Hindi'], latlng=(19.009, 72.8175), tags=['STAGE_PLAY'], duration=140m, ageLimit=12+
#   #3: [etm100116z] (event) MAHATMA: THE STAGE PLAY - DELHI NCR — Delhi NCR | showtime=2026-10-02T18:00, release=2026-09-26, lang=['Hindi', 'English'], latlng=(28.5829, 77.2344), tags=['STAGE_PLAY', 'HOLIDAY_SPECIAL'], duration=135m, ageLimit=5+
#
# === Q11-B. With Boost ([{"condition": "searchKeywords: ANY(\"play\", \"plays\", \"theatre\", \"drama\")", "boost": 0.6}, {"condition": "showtime_start >= \"2026-10-02T00:00:00+05:30\" AND showtime_start <= \"2026-10-02T23:59:59+05:30\"", "boost": 0.9}]) — query='play theatre' ===
# Total Matches: 3 (Showing top 3)
#   #1: [etm100117z] (event) TUGHLAQ: THEATRE PLAY - MUMBAI — Mumbai | showtime=2026-10-02T15:30, release=2026-09-26, lang=['Hindi'], latlng=(19.009, 72.8175), tags=['STAGE_PLAY', 'HOLIDAY_SPECIAL'], duration=150m, ageLimit=12+
#   #2: [etm100116z] (event) MAHATMA: THE STAGE PLAY - DELHI NCR — Delhi NCR | showtime=2026-10-02T18:00, release=2026-09-26, lang=['Hindi', 'English'], latlng=(28.5829, 77.2344), tags=['STAGE_PLAY', 'HOLIDAY_SPECIAL'], duration=135m, ageLimit=5+
#   #3: [etm100115z] (event) ANDHA YUG: CLASSIC HINDI STAGE PLAY - MUMBAI — Mumbai | showtime=2026-10-25T19:00, release=2026-09-26, lang=['Hindi'], latlng=(19.009, 72.8175), tags=['STAGE_PLAY'], duration=140m, ageLimit=12+
#
# ====================================================================================================
# [Q12] User Query: 'pottery classes every saturday in october' | Difficulty: hard | Tags: time, genre mood | Extracted: what: pottery workshop, when: every saturday in october (Oct 3, 10, 17, 24, 31)
# ====================================================================================================
#
# === Q12-A. Baseline (No Boost) — query='pottery workshop' ===
# Total Matches: 3 (Showing top 3)
#   #1: [etm100119z] (event) CLAYSTATION WHEEL POTTERY CLASSES & CERAMICS WORKSHOP - BENGALURU — Bengaluru | showtime=2026-10-10T11:00, release=2026-09-26, lang=['English', 'Hindi'], latlng=(12.9975, 77.6964), tags=['POTTERY_WORKSHOP', 'SATURDAY_OCTOBER'], duration=150m, ageLimit=8+
#   #2: [etm100120z] (event) BANDRA STUDIO HAND-BUILDING POTTERY CLASSES - MUMBAI — Mumbai | showtime=2026-10-17T15:00, release=2026-09-26, lang=['English', 'Hindi'], latlng=(19.009, 72.8175), tags=['POTTERY_WORKSHOP', 'SATURDAY_OCTOBER'], duration=120m, ageLimit=8+
#   #3: [etm100118z] (event) PUNE CLAY COLLECTIVE POTTERY CLASSES - PUNE — Pune | showtime=2026-11-11T16:00, release=2026-09-26, lang=['English', 'Marathi'], latlng=(18.4892, 73.8205), tags=['POTTERY_WORKSHOP'], duration=120m, ageLimit=8+
#
# === Q12-B. With Boost ([{"condition": "searchKeywords: ANY(\"pottery\", \"pottery classes\", \"pottery workshop\")", "boost": 0.6}, {"condition": "hash_tags: ANY(\"SATURDAY_OCTOBER\") OR ((showtime_start >= \"2026-10-10T00:00:00+05:30\" AND showtime_start <= \"2026-10-10T23:59:59+05:30\") OR (showtime_start >= \"2026-10-17T00:00:00+05:30\" AND showtime_start <= \"2026-10-17T23:59:59+05:30\"))", "boost": 0.9}]) — query='pottery workshop' ===
# Total Matches: 3 (Showing top 3)
#   #1: [etm100119z] (event) CLAYSTATION WHEEL POTTERY CLASSES & CERAMICS WORKSHOP - BENGALURU — Bengaluru | showtime=2026-10-10T11:00, release=2026-09-26, lang=['English', 'Hindi'], latlng=(12.9975, 77.6964), tags=['POTTERY_WORKSHOP', 'SATURDAY_OCTOBER'], duration=150m, ageLimit=8+
#   #2: [etm100120z] (event) BANDRA STUDIO HAND-BUILDING POTTERY CLASSES - MUMBAI — Mumbai | showtime=2026-10-17T15:00, release=2026-09-26, lang=['English', 'Hindi'], latlng=(19.009, 72.8175), tags=['POTTERY_WORKSHOP', 'SATURDAY_OCTOBER'], duration=120m, ageLimit=8+
#   #3: [etm100118z] (event) PUNE CLAY COLLECTIVE POTTERY CLASSES - PUNE — Pune | showtime=2026-11-11T16:00, release=2026-09-26, lang=['English', 'Marathi'], latlng=(18.4892, 73.8205), tags=['POTTERY_WORKSHOP'], duration=120m, ageLimit=8+
