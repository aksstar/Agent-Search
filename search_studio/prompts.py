"""LLM System Prompts & Multi-Turn Few-Shot Chat Templates for Search Studio.

Contains:
1. `LOCAL_LLM_SYSTEM_PROMPT` / `GEMINI_INTENT_SYSTEM_INSTRUCTION`:
   Full schema-grounded 8-key JSON slot extraction prompt for Qwen 2.5 / Cloud Gemini.
2. `build_gemma_chat_messages()`:
   Compact (~404-token) multi-turn chat template for Local Gemma 3 1B / 270M Intent Extraction.
3. `build_gemma_router_chat_messages()`:
   Compact (~367-token) multi-turn chat template for Local Gemma 3 Query Route Decision
   (`DIRECT_AGENT_SEARCH` vs `CALL_LLM_INTENT`).
4. `build_explain_hit_prompt()`:
   Diagnostic prompt for `gemini-3.5-flash-lite` live hit explanation.
"""

from __future__ import annotations

from typing import Any, Dict, List

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

GEMINI_INTENT_SYSTEM_INSTRUCTION = LOCAL_LLM_SYSTEM_PROMPT


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



def build_gemma_router_chat_messages() -> List[Dict[str, str]]:
    """Build a compact (~405-token) multi-turn chat template for Gemma 3's 512-token
    sliding-window `RotatingKVCache` to decide whether a query requires LLM Intent/Boost
    extraction (`CALL_LLM_INTENT`) or can go directly to Agent Search (`DIRECT_AGENT_SEARCH`).
    """
    rules = (
        "Entertainment Query Router.\n"
        'Output JSON: {"route":"DIRECT_AGENT_SEARCH"|"CALL_LLM_INTENT","reason":"..."}\n'
        "- DIRECT_AGENT_SEARCH: pure movie/show title, artist name, venue/theatre name, or bank offer with NO filters.\n"
        "- CALL_LLM_INTENT: has city/locality/near, price/under/between/free, date/time/tonight/tomorrow/weekend/late night, format(IMAX/4DX/3D/Dolby/recliner), language, genre/category, or negation(without/not)."
    )
    return [
        {"role": "user", "content": rules + "\nQuery: Spider-Man: No Way Home"},
        {"role": "assistant", "content": '{"route":"DIRECT_AGENT_SEARCH","reason":"Direct movie title lookup with no facet filters"}'},
        {"role": "user", "content": "Query: Zakir Khan show"},
        {"role": "assistant", "content": '{"route":"DIRECT_AGENT_SEARCH","reason":"Direct artist show lookup with no city or date filter"}'},
        {"role": "user", "content": "Query: HDFC bank Buy 1 Get 1 offer"},
        {"role": "assistant", "content": '{"route":"DIRECT_AGENT_SEARCH","reason":"Direct bank offer lookup with no city or date filter"}'},
        {"role": "user", "content": "Query: IMAX 3D movies in Mumbai"},
        {"role": "assistant", "content": '{"route":"CALL_LLM_INTENT","reason":"Has format (IMAX 3D) and city (Mumbai) filters"}'},
        {"role": "user", "content": "Query: concerts between 1000 and 3000 in delhi"},
        {"role": "assistant", "content": '{"route":"CALL_LLM_INTENT","reason":"Has category (concerts), price range (1000-3000), and city (Delhi)"}'},
        {"role": "user", "content": "Query: funny movie without any horror"},
        {"role": "assistant", "content": '{"route":"CALL_LLM_INTENT","reason":"Has genre (Comedy) and negative genre exclusion (without horror)"}'},
        {"role": "user", "content": "Query: PLACEHOLDER_QUERY"},
    ]



def build_explain_hit_prompt(
    query: str,
    extracted_filters: str,
    rewritten_query: str,
    hit: Dict[str, Any],
    app_cfg: Dict[str, Any],
) -> str:
    """Build the diagnostic prompt for `gemini-3.5-flash-lite` to explain a single ranked hit."""
    is_v3 = app_cfg["key"] == "v3"
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

    return f"""You are a Senior Search Relevance Architect diagnosing Entertainment Search's Vertex AI Agent Search (`{app_cfg['engine_id']}`: 5 record_types = `movie`, `event`, `venue`, `artist`, `offer`).

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
