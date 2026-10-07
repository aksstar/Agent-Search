"""Create, Insert (Batch & Single), Modify (Patch), and Delete documents in Vertex AI Search DataStore.

Demonstrates real-time Document CRUD operations against the Discovery Engine REST API:
1. Insert 10 new records (5 movies + 5 live events) using:
   - Single-document creation (`POST .../documents?documentId={id}`)
   - Batch inline import (`POST .../documents:import` with `inlineSource`)
2. Modify existing records (`GET` + `PATCH .../documents/{id}?allowMissing=true`)
3. Delete records (`DELETE .../documents/{id}`) and verify `404 NOT_FOUND`

Reference Documentation:
- Create Document: https://cloud.google.com/generative-ai-app-builder/docs/reference/rest/v1/projects.locations.collections.dataStores.branches.documents/create
- Import Documents (Inline / GCS): https://cloud.google.com/generative-ai-app-builder/docs/reference/rest/v1/projects.locations.collections.dataStores.branches.documents/import
- Patch / Update Document: https://cloud.google.com/generative-ai-app-builder/docs/reference/rest/v1/projects.locations.collections.dataStores.branches.documents/patch
- Delete Document: https://cloud.google.com/generative-ai-app-builder/docs/reference/rest/v1/projects.locations.collections.dataStores.branches.documents/delete
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from google.auth.transport.requests import AuthorizedSession

from config import DATASTORE_ID, LOCATION, PROJECT_ID
from setup_gcs_datastore import get_discovery_session


def get_documents_base_url(base_url: str) -> str:
    """Return the REST base URL for the default_branch documents collection."""
    return (
        f"{base_url}/projects/{PROJECT_ID}/locations/{LOCATION}"
        f"/collections/default_collection/dataStores/{DATASTORE_ID}"
        f"/branches/default_branch/documents"
    )


# ==============================================================================
# Core Document CRUD Functions (Vertex AI Search / Discovery Engine REST API)
# ==============================================================================
def get_document(
    session: AuthorizedSession, base_url: str, doc_id: str
) -> Optional[Dict[str, Any]]:
    """Fetch a single document by ID (`GET .../documents/{doc_id}`). Returns None if 404."""
    url = f"{get_documents_base_url(base_url)}/{doc_id}"
    resp = session.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def create_document(
    session: AuthorizedSession,
    base_url: str,
    doc_id: str,
    struct_data: Dict[str, Any],
    upsert_if_exists: bool = True,
) -> Dict[str, Any]:
    """Create a single document (`POST .../documents?documentId={doc_id}`).

    If the document ID already exists (HTTP 409) and `upsert_if_exists=True`,
    falls back to `PATCH .../documents/{doc_id}` so repeated runs are idempotent.
    """
    docs_url = get_documents_base_url(base_url)
    payload = {
        "id": doc_id,
        "schemaId": "default_schema",
        "structData": struct_data,
    }
    resp = session.post(f"{docs_url}?documentId={doc_id}", json=payload)
    if resp.status_code == 409 and upsert_if_exists:
        resp = session.patch(f"{docs_url}/{doc_id}?allowMissing=true", json=payload)
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"create_document({doc_id}) failed (HTTP {resp.status_code}): {resp.text}"
        )
    return resp.json()


def batch_insert_documents(
    session: AuthorizedSession,
    base_url: str,
    records: List[Dict[str, Any]],
    reconciliation_mode: str = "INCREMENTAL",
) -> Dict[str, Any]:
    """Batch-insert or upsert up to 100 documents inline (`POST .../documents:import`).

    Args:
        session: Authenticated Discovery Engine session.
        base_url: Discovery Engine REST endpoint base URL.
        records: List of raw record dicts (each must contain `id` or `_id`).
        reconciliation_mode: `INCREMENTAL` (upsert into existing index) or `FULL`.
    """
    docs_url = get_documents_base_url(base_url)
    inline_docs = []
    for rec in records:
        doc_id = rec.get("id") or rec.get("_id")
        if not doc_id:
            raise ValueError(f"Record missing 'id' or '_id': {rec}")
        normalized = dict(rec)
        normalized["id"] = doc_id
        normalized["_id"] = doc_id
        inline_docs.append(
            {
                "id": doc_id,
                "schemaId": "default_schema",
                "structData": normalized,
            }
        )

    payload = {
        "inlineSource": {"documents": inline_docs},
        "reconciliationMode": reconciliation_mode,
    }
    resp = session.post(f"{docs_url}:import", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(
            f"batch_insert_documents failed (HTTP {resp.status_code}): {resp.text}"
        )
    lro = resp.json()
    lro_name = lro.get("name", "")

    # Poll long-running import operation until done
    if lro_name and not lro.get("done"):
        for _ in range(20):
            op_resp = session.get(f"{base_url}/{lro_name}")
            if op_resp.status_code == 200:
                op_data = op_resp.json()
                if op_data.get("done"):
                    return op_data
            time.sleep(1.5)
    return lro


def modify_document(
    session: AuthorizedSession,
    base_url: str,
    doc_id: str,
    updates: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Modify specific fields of an existing document (`GET` + `PATCH .../documents/{doc_id}`).

    Note: In Vertex AI Search, `PATCH` replaces the full `structData` object.
    To perform a partial field update safely, we first `GET` the existing document,
    merge `updates` into its `structData`, and then `PATCH` the merged document.

    Returns:
        Tuple of `(before_struct_data, updated_document_response)`.
    """
    existing = get_document(session, base_url, doc_id)
    if existing is None:
        raise RuntimeError(f"Cannot modify '{doc_id}': document does not exist (404).")

    before_struct = dict(existing.get("structData", {}))
    merged_struct = dict(before_struct)
    merged_struct.update(updates)

    docs_url = get_documents_base_url(base_url)
    payload = {
        "id": doc_id,
        "schemaId": existing.get("schemaId", "default_schema"),
        "structData": merged_struct,
    }
    resp = session.patch(f"{docs_url}/{doc_id}?allowMissing=false", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(
            f"modify_document({doc_id}) failed (HTTP {resp.status_code}): {resp.text}"
        )
    return before_struct, resp.json()


def delete_document(
    session: AuthorizedSession, base_url: str, doc_id: str
) -> bool:
    """Delete a single document by ID (`DELETE .../documents/{doc_id}`).

    Returns True if deleted, False if already absent (404).
    """
    url = f"{get_documents_base_url(base_url)}/{doc_id}"
    resp = session.delete(url)
    if resp.status_code == 404:
        return False
    if resp.status_code != 200:
        raise RuntimeError(
            f"delete_document({doc_id}) failed (HTTP {resp.status_code}): {resp.text}"
        )
    return True


# ==============================================================================
# 10 New Sample Records (5 Movies + 5 Live Events)
# ==============================================================================
NEW_RECORDS_10: List[Dict[str, Any]] = [
    # --- 5 New Movies (MV00201 - MV00205) ---
    {
        "id": "MV00201",
        "_id": "MV00201",
        "record_type": "movie",
        "media_type": "movie",
        "defaultVariantCode": "ET00500201",
        "title": "Dhurandhar: The Spy Chronicles",
        "categories": ["movies"],
        "slug": "dhurandhar-the-spy-chronicles",
        "available_time": "2026-10-09T00:00:00+05:30",
        "release_date": "2026-10-09T00:00:00+05:30",
        "showtime_start": "2026-10-09T20:00:00+05:30",
        "showtime_end": "2026-10-09T22:45:00+05:30",
        "hash_tags": ["NEW_RELEASE", "FRIDAY_RELEASE", "BLOCKBUSTER"],
        "genres": ["Action", "Spy", "Thriller"],
        "languages": ["Hindi"],
        "searchKeywords": ["movie", "cinema", "action", "spy", "thriller", "hindi", "dhurandhar"],
        "formats": ["2D", "IMAX 2D"],
        "artists": ["Ranveer Singh", "R. Madhavan", "Akshaye Khanna"],
        "city": "Mumbai",
        "venue": "PVR ICON Phoenix Palladium, Lower Parel",
        "location_city": {"address": "Mumbai, Maharashtra, India"},
        "location_latlng": {"lat": 18.9946, "lng": 72.8245},
        "uri": "https://events.example.com/mumbai/movies/dhurandhar-the-spy-chronicles/ET00500201",
    },
    {
        "id": "MV00202",
        "_id": "MV00202",
        "record_type": "movie",
        "media_type": "movie",
        "defaultVariantCode": "ET00500202",
        "title": "Kantara: A Legend Part 3",
        "categories": ["movies"],
        "slug": "kantara-a-legend-part-3",
        "available_time": "2026-10-16T00:00:00+05:30",
        "release_date": "2026-10-16T00:00:00+05:30",
        "showtime_start": "2026-10-16T19:15:00+05:30",
        "showtime_end": "2026-10-16T22:05:00+05:30",
        "hash_tags": ["MUST_WATCH", "BLOCKBUSTER", "CRITICALLY_ACCLAIMED"],
        "genres": ["Action", "Folklore", "Drama"],
        "languages": ["Kannada", "Hindi", "Telugu"],
        "searchKeywords": ["movie", "cinema", "kantara", "folklore", "kannada", "hindi"],
        "formats": ["2D", "4DX", "IMAX 2D"],
        "artists": ["Rishab Shetty", "Sapthami Gowda"],
        "city": "Bengaluru",
        "venue": "PVR Orion Mall, Rajajinagar",
        "location_city": {"address": "Bengaluru, Karnataka, India"},
        "location_latlng": {"lat": 13.0110, "lng": 77.5550},
        "uri": "https://events.example.com/bengaluru/movies/kantara-a-legend-part-3/ET00500202",
    },
    {
        "id": "MV00203",
        "_id": "MV00203",
        "record_type": "movie",
        "media_type": "movie",
        "defaultVariantCode": "ET00500203",
        "title": "Jana Nayagan: The Leader",
        "categories": ["movies"],
        "slug": "jana-nayagan-the-leader",
        "available_time": "2026-10-23T00:00:00+05:30",
        "release_date": "2026-10-23T00:00:00+05:30",
        "showtime_start": "2026-10-23T09:30:00+05:30",
        "showtime_end": "2026-10-23T12:30:00+05:30",
        "hash_tags": ["MORNING_SHOW", "BLOCKBUSTER"],
        "genres": ["Action", "Political", "Drama"],
        "languages": ["Tamil", "Telugu"],
        "searchKeywords": ["movie", "cinema", "tamil", "action", "morning show"],
        "formats": ["2D", "EPIQ"],
        "artists": ["Vijay", "Pooja Hegde", "Bobby Deol"],
        "city": "Chennai",
        "venue": "Sathyam Cinemas, Royapettah",
        "location_city": {"address": "Chennai, Tamil Nadu, India"},
        "location_latlng": {"lat": 13.0556, "lng": 80.2580},
        "uri": "https://events.example.com/chennai/movies/jana-nayagan-the-leader/ET00500203",
    },
    {
        "id": "MV00204",
        "_id": "MV00204",
        "record_type": "movie",
        "media_type": "movie",
        "defaultVariantCode": "ET00500204",
        "title": "Don 3: The Final Heist",
        "categories": ["movies"],
        "slug": "don-3-the-final-heist",
        "available_time": "2026-11-06T00:00:00+05:30",
        "release_date": "2026-11-06T00:00:00+05:30",
        "showtime_start": "2026-11-06T22:45:00+05:30",
        "showtime_end": "2026-11-07T01:25:00+05:30",
        "hash_tags": ["UPCOMING_MOVIE", "LATE_NIGHT_SHOW"],
        "genres": ["Action", "Crime", "Thriller"],
        "languages": ["Hindi"],
        "searchKeywords": ["movie", "cinema", "don 3", "crime", "late night"],
        "formats": ["2D", "DOLBY CINEMA 2D"],
        "artists": ["Ranveer Singh", "Kiara Advani"],
        "city": "Delhi NCR",
        "venue": "PVR Select Citywalk, Saket",
        "location_city": {"address": "Delhi NCR, Delhi, India"},
        "location_latlng": {"lat": 28.5286, "lng": 77.2193},
        "uri": "https://events.example.com/delhi-ncr/movies/don-3-the-final-heist/ET00500204",
    },
    {
        "id": "MV00205",
        "_id": "MV00205",
        "record_type": "movie",
        "media_type": "movie",
        "defaultVariantCode": "ET00500205",
        "title": "SSMB29: Garuda Rising",
        "categories": ["movies"],
        "slug": "ssmb29-garuda-rising",
        "available_time": "2026-11-20T00:00:00+05:30",
        "release_date": "2026-11-20T00:00:00+05:30",
        "showtime_start": "2026-11-20T18:30:00+05:30",
        "showtime_end": "2026-11-20T21:40:00+05:30",
        "hash_tags": ["UPCOMING_MOVIE", "BLOCKBUSTER"],
        "genres": ["Adventure", "Action"],
        "languages": ["Telugu", "Hindi", "English"],
        "searchKeywords": ["movie", "upcoming", "telugu", "adventure", "ssmb29"],
        "formats": ["IMAX 3D", "3D"],
        "artists": ["Mahesh Babu", "Priyanka Chopra"],
        "city": "Hyderabad",
        "venue": "Prasads Multiplex, Khairatabad",
        "location_city": {"address": "Hyderabad, Telangana, India"},
        "location_latlng": {"lat": 17.4126, "lng": 78.4664},
        "uri": "https://events.example.com/hyderabad/movies/ssmb29-garuda-rising/ET00500205",
    },
    # --- 5 New Live Events (etm100201z - etm100205z) ---
    {
        "id": "etm100201z",
        "_id": "etm100201z",
        "record_type": "event",
        "eventGroupCode": "EG50000201",
        "title": "COLDPLAY MUSIC OF THE SPHERES WORLD TOUR - MUMBAI",
        "slug": "coldplay-music-of-the-spheres-world-tour-mumbai",
        "persons": ["Chris Martin", "Coldplay"],
        "languages": ["English"],
        "duration": 180,
        "ageLimit": 5,
        "ageLimitDisplay": "5yrs +",
        "isActive": True,
        "isSearchable": True,
        "createdAt": "2026-10-01T10:00:00+05:30",
        "release_date": "2026-10-01T10:00:00+05:30",
        "showtime_start": "2026-10-18T19:00:00+05:30",
        "showtime_end": "2026-10-18T22:00:00+05:30",
        "hash_tags": ["NEXT_WEEK", "MUST_WATCH"],
        "searchKeywords": ["music", "live concert", "coldplay", "rock", "pop"],
        "city": "Mumbai",
        "venue": "NSCI Dome, Worli",
        "location_city": {"address": "Mumbai, Maharashtra, India"},
        "location_latlng": {"lat": 19.0090, "lng": 72.8175},
    },
    {
        "id": "etm100202z",
        "_id": "etm100202z",
        "record_type": "event",
        "eventGroupCode": "EG50000202",
        "title": "ED SHEERAN MATHEMATICS TOUR - BENGALURU",
        "slug": "ed-sheeran-mathematics-tour-bengaluru",
        "persons": ["Ed Sheeran"],
        "languages": ["English"],
        "duration": 150,
        "ageLimit": 5,
        "ageLimitDisplay": "5yrs +",
        "isActive": True,
        "isSearchable": True,
        "createdAt": "2026-10-02T12:00:00+05:30",
        "release_date": "2026-10-02T12:00:00+05:30",
        "showtime_start": "2026-10-17T18:30:00+05:30",
        "showtime_end": "2026-10-17T21:00:00+05:30",
        "hash_tags": ["NEXT_WEEK", "SATURDAY_OCTOBER"],
        "searchKeywords": ["music", "concert", "ed sheeran", "acoustic", "pop"],
        "city": "Bengaluru",
        "venue": "Phoenix Marketcity, Whitefield",
        "location_city": {"address": "Bengaluru, Karnataka, India"},
        "location_latlng": {"lat": 12.9975, "lng": 77.6964},
    },
    {
        "id": "etm100203z",
        "_id": "etm100203z",
        "record_type": "event",
        "eventGroupCode": "EG50000203",
        "title": "VIR DAS MIND FOOL INDIA TOUR - DELHI NCR",
        "slug": "vir-das-mind-fool-india-tour-delhi-ncr",
        "persons": ["Vir Das"],
        "languages": ["English", "Hindi"],
        "duration": 95,
        "ageLimit": 16,
        "ageLimitDisplay": "16yrs +",
        "isActive": True,
        "isSearchable": True,
        "createdAt": "2026-10-03T09:00:00+05:30",
        "release_date": "2026-10-03T09:00:00+05:30",
        "showtime_start": "2026-10-10T20:00:00+05:30",
        "showtime_end": "2026-10-10T21:35:00+05:30",
        "hash_tags": ["STANDUP_COMEDY", "SATURDAY_OCTOBER"],
        "searchKeywords": ["stand up comedy", "comedy", "vir das", "live"],
        "city": "Delhi NCR",
        "venue": "Jawaharlal Nehru Stadium",
        "location_city": {"address": "Delhi NCR, Delhi, India"},
        "location_latlng": {"lat": 28.5829, "lng": 77.2344},
    },
    {
        "id": "etm100204z",
        "_id": "etm100204z",
        "record_type": "event",
        "eventGroupCode": "EG50000204",
        "title": "ZOMALAND FOOD & MUSIC CARNIVAL - PUNE",
        "slug": "zomaland-food-and-music-carnival-pune",
        "persons": ["DIVINE", "Ritviz", "Celebrity Chefs"],
        "languages": ["Hindi", "English"],
        "duration": 360,
        "ageLimit": None,
        "ageLimitDisplay": "All Ages",
        "isActive": True,
        "isSearchable": True,
        "createdAt": "2026-10-04T11:00:00+05:30",
        "release_date": "2026-10-04T11:00:00+05:30",
        "showtime_start": "2026-10-11T14:00:00+05:30",
        "showtime_end": "2026-10-11T20:00:00+05:30",
        "hash_tags": ["FAMILY_ENTERTAINER"],
        "searchKeywords": ["food festival", "music", "carnival", "pune", "events"],
        "city": "Pune",
        "venue": "Mahalaxmi Lawns, Karve Nagar",
        "location_city": {"address": "Pune, Maharashtra, India"},
        "location_latlng": {"lat": 18.4892, "lng": 73.8205},
    },
    {
        "id": "etm100205z",
        "_id": "etm100205z",
        "record_type": "event",
        "eventGroupCode": "EG50000205",
        "title": "SUNBURN ARENA FT. ALAN WALKER - GOA",
        "slug": "sunburn-arena-ft-alan-walker-goa",
        "persons": ["Alan Walker"],
        "languages": ["English"],
        "duration": 240,
        "ageLimit": 18,
        "ageLimitDisplay": "18yrs +",
        "isActive": True,
        "isSearchable": True,
        "createdAt": "2026-10-05T15:00:00+05:30",
        "release_date": "2026-10-05T15:00:00+05:30",
        "showtime_start": "2026-12-30T18:00:00+05:30",
        "showtime_end": "2026-12-30T22:00:00+05:30",
        "hash_tags": ["EDM_NIGHT"],
        "searchKeywords": ["edm", "party", "alan walker", "sunburn", "goa"],
        "city": "Goa",
        "venue": "Vagator Beach Grounds",
        "location_city": {"address": "Goa, Goa, India"},
        "location_latlng": {"lat": 15.6029, "lng": 73.7339},
    },
]


# ==============================================================================
# End-to-End Demo Workflow (Insert 10 -> Modify Existing -> Delete Few)
# ==============================================================================
def run_crud_demo() -> None:
    session, base_url = get_discovery_session()
    print(f"Connected to Discovery Engine DataStore: {DATASTORE_ID} (Project: {PROJECT_ID})\n")

    # ------------------------------------------------------------------
    # STEP 1: Insert 10 New Records (2 via Single Create + 8 via Batch Inline Import)
    # ------------------------------------------------------------------
    print("=" * 95)
    print("STEP 1: Inserting 10 New Records (5 Movies MV00201..MV00205 + 5 Events etm100201z..etm100205z)")
    print("=" * 95)

    single_records = NEW_RECORDS_10[:2]
    batch_records = NEW_RECORDS_10[2:]

    for rec in single_records:
        doc_id = rec["id"]
        created = create_document(session, base_url, doc_id, rec)
        sd = created.get("structData", {})
        print(
            f"  [Single Create POST] {doc_id:<12} | ({sd.get('record_type')}) "
            f"{sd.get('title')} — {sd.get('city')}"
        )

    lro_result = batch_insert_documents(
        session, base_url, batch_records, reconciliation_mode="INCREMENTAL"
    )
    meta = lro_result.get("metadata", {})
    print(
        f"  [Batch Import POST]  Imported {len(batch_records)} documents via inlineSource "
        f"(successCount={meta.get('successCount', len(batch_records))}, "
        f"failureCount={meta.get('failureCount', 0)})"
    )
    for rec in batch_records:
        print(
            f"    -> {rec['id']:<12} | ({rec['record_type']}) {rec['title']} — {rec['city']}"
        )

    # ------------------------------------------------------------------
    # STEP 2: Modify Existing Records (`GET` + `PATCH .../documents/{id}`)
    # ------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("STEP 2: Modifying Existing Records in DataStore (GET + PATCH)")
    print("=" * 95)

    modifications: List[Tuple[str, Dict[str, Any]]] = [
        (
            "MV00001",
            {
                "formats": ["2D", "IMAX 2D", "DOLBY ATMOS"],
                "hash_tags": [
                    "LIVE_IN_CINEMAS",
                    "FAMILY_ENTERTAINER",
                    "LATE_NIGHT_SHOW",
                    "IMAX_SPECIAL",
                ],
            },
        ),
        (
            "MV00201",
            {
                "languages": ["Hindi", "English", "Telugu"],
                "hash_tags": [
                    "NEW_RELEASE",
                    "FRIDAY_RELEASE",
                    "BLOCKBUSTER",
                    "SELLING_FAST",
                ],
                "formats": ["2D", "IMAX 2D", "4DX"],
            },
        ),
        (
            "etm100201z",
            {
                "duration": 210,
                "hash_tags": ["NEXT_WEEK", "MUST_WATCH", "EXTRA_SHOW_ADDED"],
            },
        ),
    ]

    for doc_id, field_updates in modifications:
        before_sd, updated_doc = modify_document(session, base_url, doc_id, field_updates)
        after_sd = updated_doc.get("structData", {})
        print(f"\n  [PATCH] Document '{doc_id}' ({after_sd.get('title')}):")
        for k in field_updates:
            print(f"    - {k}: {before_sd.get(k)!r}  -->  {after_sd.get(k)!r}")

    # ------------------------------------------------------------------
    # STEP 3: Delete a Few Records (`DELETE .../documents/{id}`) & Verify 404
    # ------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("STEP 3: Deleting Records from DataStore (DELETE) & Verifying Removal")
    print("=" * 95)

    ids_to_delete = ["MV00204", "MV00205", "etm100205z"]
    for doc_id in ids_to_delete:
        deleted = delete_document(session, base_url, doc_id)
        verify_after = get_document(session, base_url, doc_id)
        status_str = "404 NOT_FOUND (Verified Deleted)" if verify_after is None else "Still Present"
        print(f"  [DELETE] {doc_id:<12} | deleted={deleted} | post-delete GET: {status_str}")

    print("\n" + "=" * 95)
    print("Summary: 10 inserted, 3 modified, 3 deleted (7 net new records active: MV00201..MV00203, etm100201z..etm100204z)")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Insert, Modify, and Delete documents in Vertex AI Search DataStore."
    )
    parser.add_argument(
        "--get",
        metavar="DOC_ID",
        help="Fetch and print a single document by ID from the DataStore.",
    )
    parser.add_argument(
        "--delete",
        nargs="+",
        metavar="DOC_ID",
        help="Delete one or more document IDs from the DataStore.",
    )
    args = parser.parse_args()

    if args.get:
        sess, api_base = get_discovery_session()
        doc = get_document(sess, api_base, args.get)
        print(json.dumps(doc, indent=2) if doc else f"Document '{args.get}' not found (404).")
    elif args.delete:
        sess, api_base = get_discovery_session()
        for did in args.delete:
            ok = delete_document(sess, api_base, did)
            print(f"Deleted '{did}': {ok}")
    else:
        run_crud_demo()

# ==============================================================================
# EXECUTION OUTPUT (`python manage_datastore_documents.py`)
# ==============================================================================
# Connected to Discovery Engine DataStore: sample-data-connector_gcs_store (Project: your-gcp-project-id)
#
# ===============================================================================================
# STEP 1: Inserting 10 New Records (5 Movies MV00201..MV00205 + 5 Events etm100201z..etm100205z)
# ===============================================================================================
#   [Single Create POST] MV00201      | (movie) Dhurandhar: The Spy Chronicles — Mumbai
#   [Single Create POST] MV00202      | (movie) Kantara: A Legend Part 3 — Bengaluru
#   [Batch Import POST]  Imported 8 documents via inlineSource (successCount=8, failureCount=0)
#     -> MV00203      | (movie) Jana Nayagan: The Leader — Chennai
#     -> MV00204      | (movie) Don 3: The Final Heist — Delhi NCR
#     -> MV00205      | (movie) SSMB29: Garuda Rising — Hyderabad
#     -> etm100201z   | (event) COLDPLAY MUSIC OF THE SPHERES WORLD TOUR - MUMBAI — Mumbai
#     -> etm100202z   | (event) ED SHEERAN MATHEMATICS TOUR - BENGALURU — Bengaluru
#     -> etm100203z   | (event) VIR DAS MIND FOOL INDIA TOUR - DELHI NCR — Delhi NCR
#     -> etm100204z   | (event) ZOMALAND FOOD & MUSIC CARNIVAL - PUNE — Pune
#     -> etm100205z   | (event) SUNBURN ARENA FT. ALAN WALKER - GOA — Goa
#
# ===============================================================================================
# STEP 2: Modifying Existing Records in DataStore (GET + PATCH)
# ===============================================================================================
#
#   [PATCH] Document 'MV00001' (Drishyam: The Conclusion):
#     - formats: ['2D']  -->  ['2D', 'IMAX 2D', 'DOLBY ATMOS']
#     - hash_tags: ['LIVE_IN_CINEMAS', 'FAMILY_ENTERTAINER', 'LATE_NIGHT_SHOW']  -->  ['LIVE_IN_CINEMAS', 'FAMILY_ENTERTAINER', 'LATE_NIGHT_SHOW', 'IMAX_SPECIAL']
#
#   [PATCH] Document 'MV00201' (Dhurandhar: The Spy Chronicles):
#     - languages: ['Hindi']  -->  ['Hindi', 'English', 'Telugu']
#     - hash_tags: ['NEW_RELEASE', 'FRIDAY_RELEASE', 'BLOCKBUSTER']  -->  ['NEW_RELEASE', 'FRIDAY_RELEASE', 'BLOCKBUSTER', 'SELLING_FAST']
#     - formats: ['2D', 'IMAX 2D']  -->  ['2D', 'IMAX 2D', '4DX']
#
#   [PATCH] Document 'etm100201z' (COLDPLAY MUSIC OF THE SPHERES WORLD TOUR - MUMBAI):
#     - duration: 180  -->  210
#     - hash_tags: ['NEXT_WEEK', 'MUST_WATCH']  -->  ['NEXT_WEEK', 'MUST_WATCH', 'EXTRA_SHOW_ADDED']
#
# ===============================================================================================
# STEP 3: Deleting Records from DataStore (DELETE) & Verifying Removal
# ===============================================================================================
#   [DELETE] MV00204      | deleted=True | post-delete GET: 404 NOT_FOUND (Verified Deleted)
#   [DELETE] MV00205      | deleted=True | post-delete GET: 404 NOT_FOUND (Verified Deleted)
#   [DELETE] etm100205z   | deleted=True | post-delete GET: 404 NOT_FOUND (Verified Deleted)
#
# ===============================================================================================
# Summary: 10 inserted, 3 modified, 3 deleted (7 net new records active: MV00201..MV00203, etm100201z..etm100204z)
# ===============================================================================================
