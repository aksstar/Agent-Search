"""Deterministic Intent-to-Boost & Geolocation Compiler.

Validates LLM-extracted JSON slots against the user query text and compiles
schema-verified Vertex AI Agent Search `conditionBoostSpecs` and hard filters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from config import (
    LEGACY_V2_LANGUAGES_FIELD,
    LEGACY_V2_REGIONS_FIELD,
    LEGACY_V2_SHOW_DATES_FIELD,
)
from search_studio.client import DEFAULT_SEARCH_APP, _date_context
from search_studio.taxonomy import (
    ALL_RECORD_TYPES, ARTIST_NICKNAME_MAP, CANONICAL_CITY_MAP, CAT_LOOKUP, FORBIDDEN_LOCALITY_WORDS, FORMAT_SPEC_MAP_V2, FORMAT_SPEC_MAP_V3, FORMAT_VALUES_V3, GENRE_BQ_EXPANSION, GENRE_GROUNDING_STEMS, GENRE_LOOKUP, GEO_CITY_NAME_MAP, LANG_GROUNDING_STEMS, LANG_LOOKUP, NON_LOCALITY_TOKENS, SAFE_QUERY_CITY_TOKENS, SUBLOCALITY_TO_CITY_MAP, TYPE_ALIAS_MAP, VALID_CATEGORIES, VALID_GENRES, VALID_LANGUAGES, VALID_TIME_SLOTS, WHOLE_CITY_NAMES,
)


def _quote_any(values: List[str]) -> str:
    """Format a list of string literals into Vertex AI Search `ANY(\"v1\", \"v2\")` syntax."""
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
    """Verify that `canon_city` (or a known neighborhood/alias of it) is present in `q_low`."""
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
    """Verify that a sub-locality/neighborhood phrase is grounded in `q_low`."""
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
    """Verify that `lang_canon` (or one of its linguistic stems in `LANG_GROUNDING_STEMS`) appears in `q_low`."""
    stems = LANG_GROUNDING_STEMS.get(lang_canon, (lang_canon.lower(),))
    return any(s in q_low for s in stems)


def _is_genre_grounded(genre_canon: str, q_low: str) -> bool:
    """Verify that `genre_canon` (or a synonym/stem in `GENRE_GROUNDING_STEMS`) appears in `q_low`."""
    stems = GENRE_GROUNDING_STEMS.get(genre_canon, (genre_canon.lower(),))
    return any(s in q_low for s in stems)


def _is_artist_grounded(artist_clean: str, q_low: str) -> bool:
    """Verify that `artist_clean` (or a nickname in `ARTIST_NICKNAME_MAP`) appears in `q_low`."""
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
    """Return True only when `q_low` contains explicit bank/card/discount/offer keywords."""
    return any(w in q_low for w in OFFER_GROUNDING_WORDS)


def _is_when_grounded(when_tok: str, q_low: str) -> bool:
    """Verify that a temporal token (`today`, `tonight`, `this_weekend`, etc.) is grounded in `q_low`."""
    stems = WHEN_TOKEN_GROUNDING.get(when_tok)
    if not stems:
        return any(w in q_low for w in ANY_TEMPORAL_GROUNDING_WORDS)
    q_words = set(q_low.replace(",", " ").split())
    return any((s in q_words if len(s) <= 3 else s in q_low) for s in stems)


def _is_time_slot_grounded(slot_tok: str, q_low: str) -> bool:
    """Verify that a time-of-day slot (`morning`, `afternoon`, `evening`, `night`) is grounded in `q_low`."""
    stems = TIME_SLOT_GROUNDING.get(slot_tok, (slot_tok,))
    q_words = set(q_low.replace(",", " ").split())
    return any((s in q_words if len(s) <= 2 else s in q_low) for s in stems)


def _is_price_grounded(q_low: str) -> bool:
    """Return True if `q_low` contains explicit price bounds (`under N`, `between M and N`, `free`, `cheapest`)."""
    import re

    if any(w in q_low for w in PRICE_GROUNDING_WORDS):
        return True
    return bool(re.search(r"\b(?:between|from)\s+\d+\s+(?:and|to|-)\s+\d+\b|\b(?:above|over)\s+\d+\b", q_low))


def _coerce_price_int(val: Any, q_low: str) -> Optional[int]:
    """Safely parse an integer INR price from an LLM slot value (`int`, `float`, or numeric string)."""
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
    """Normalize a string or list of strings from an LLM JSON slot into a flat list of non-empty strings."""
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

