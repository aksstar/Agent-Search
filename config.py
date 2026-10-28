"""Centralized configuration loader for Agent Search & Gemini scripts.

Reads settings from environment variables or a local gitignored `config.local.json` file,
falling back to safe generic placeholders so no project IDs, project numbers, app names,
or credentials are hardcoded in version-controlled code.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

_CONFIG_DIR = Path(__file__).resolve().parent
_LOCAL_CONFIG_PATH = _CONFIG_DIR / "config.local.json"


def _load_local_config() -> Dict[str, Any]:
    if _LOCAL_CONFIG_PATH.exists():
        try:
            with open(_LOCAL_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


_CFG = _load_local_config()


def _get(env_key: str, json_key: str, default: str) -> str:
    val = os.getenv(env_key)
    if val is not None and val != "":
        return val
    cfg_val = _CFG.get(json_key)
    if cfg_val is not None and str(cfg_val) != "":
        return str(cfg_val)
    return default


PROJECT_ID: str = _get("GCP_PROJECT_ID", "project_id", "your-gcp-project-id")
PROJECT_NUMBER: str = _get("GCP_PROJECT_NUMBER", "project_number", "000000000000")
LOCATION: str = _get("DISCOVERY_ENGINE_LOCATION", "location", "global")
VERTEX_AI_LOCATION: str = _get("VERTEX_AI_LOCATION", "vertex_ai_location", "us-central1")
VERTEX_LOCATION: str = VERTEX_AI_LOCATION

GCS_BUCKET: str = _get("GCS_BUCKET", "gcs_bucket", "your-gcs-bucket-name")
GCS_FOLDER: str = _get("GCS_FOLDER", "gcs_folder", "sample_search_data")
LOCAL_JSON_FILE: str = _get("LOCAL_JSON_FILE", "local_json_file", "sample_metadata_200.json")
LOCAL_JSONL_FILE: str = _get("LOCAL_JSONL_FILE", "local_jsonl_file", "sample_metadata_200.jsonl")

COLLECTION_ID: str = _get("COLLECTION_ID", "collection_id", "sample-data-connector")
COLLECTION_DISPLAY_NAME: str = _get(
    "COLLECTION_DISPLAY_NAME", "collection_display_name", "Entertainment Sample Data"
)
DATASTORE_ID: str = _get(
    "DATASTORE_ID", "datastore_id", "sample-data-connector_gcs_store"
)
ENGINE_ID: str = _get("SEARCH_ENGINE_ID", "engine_id", "your-search-engine-id")
ENGINE_DISPLAY_NAME: str = _get(
    "ENGINE_DISPLAY_NAME", "engine_display_name", "Entertainment Search App"
)
REFRESH_INTERVAL_SECONDS: str = _get(
    "REFRESH_INTERVAL_SECONDS", "refresh_interval_seconds", "14400s"
)

GCLOUD_ACCOUNT: str = _get("GCLOUD_ACCOUNT", "gcloud_account", "")
GEMINI_MODEL: str = _get("GEMINI_MODEL", "gemini_model", "gemini-3.5-flash-lite")
GEMINI_PROJECT_ID: str = _get("GEMINI_PROJECT_ID", "gemini_project_id", PROJECT_ID)
ECP_HTTP_PROXY_BIN: str = _get("ECP_HTTP_PROXY_BIN", "ecp_http_proxy_bin", "ecp_http_proxy")

# 5-Store Unified Entertainment Search Identifiers (loaded from config.local.json or env vars)
SEARCH_5STORES_PROJECT_ID: str = _get(
    "SEARCH_5STORES_PROJECT_ID", "search_5stores_project_id", PROJECT_ID
)
SEARCH_5STORES_PROJECT_NUMBER: str = _get(
    "SEARCH_5STORES_PROJECT_NUMBER", "search_5stores_project_number", PROJECT_NUMBER
)
SEARCH_5STORES_DISPLAY_PROJECT_ID: str = _get(
    "SEARCH_5STORES_DISPLAY_PROJECT_ID", "search_5stores_display_project_id", "your-gcp-project-id"
)
GCS_5STORES_BUCKET: str = _get(
    "GCS_5STORES_BUCKET", "gcs_5stores_bucket", "your-gcs-import-bucket"
)

BQ_5STORES_DATASET: str = _get("BQ_5STORES_DATASET", "bq_5stores_dataset", "ent_search_bq")
BQ_TABLE_V3_RUNTIME: str = _get("BQ_TABLE_V3", "bq_table_v3", "ent_5stores_master_v3")
BQ_TABLE_V2_RUNTIME: str = _get("BQ_TABLE_V2", "bq_table_v2", "ent_5stores_unified")
BQ_TABLE_V3_DISPLAY: str = _get(
    "BQ_TABLE_V3_DISPLAY", "bq_table_v3_display", "ent_5stores_master_v3"
)
BQ_TABLE_V2_DISPLAY: str = _get(
    "BQ_TABLE_V2_DISPLAY", "bq_table_v2_display", "ent_5stores_unified"
)

ENGINE_ID_V3_RUNTIME: str = _get(
    "ENGINE_ID_V3", "engine_id_v3", "ent-bq-5stores-master-v3-app"
)
DATASTORE_ID_V3_RUNTIME: str = _get(
    "DATASTORE_ID_V3", "datastore_id_v3", "ent-bq-5stores-master-v3-ds"
)
ENGINE_ID_V3_DISPLAY: str = _get(
    "ENGINE_ID_V3_DISPLAY", "engine_id_v3_display", "ent-bq-5stores-master-v3-app"
)
DATASTORE_ID_V3_DISPLAY: str = _get(
    "DATASTORE_ID_V3_DISPLAY", "datastore_id_v3_display", "ent-bq-5stores-master-v3-ds"
)

ENGINE_ID_V2_RUNTIME: str = _get(
    "ENGINE_ID_V2", "engine_id_v2", "ent-bq-5stores-v2-app"
)
DATASTORE_ID_V2_RUNTIME: str = _get(
    "DATASTORE_ID_V2", "datastore_id_v2", "ent-bq-5stores-v2-ds"
)
ENGINE_ID_V2_DISPLAY: str = _get(
    "ENGINE_ID_V2_DISPLAY", "engine_id_v2_display", "ent-bq-5stores-v2-app"
)
DATASTORE_ID_V2_DISPLAY: str = _get(
    "DATASTORE_ID_V2_DISPLAY", "datastore_id_v2_display", "ent-bq-5stores-v2-ds"
)

LEGACY_V2_LANGUAGES_FIELD: str = _get(
    "LEGACY_V2_LANGUAGES_FIELD", "legacy_v2_languages_field", "languages"
)
LEGACY_V2_SOLD_OUT_FIELD: str = _get(
    "LEGACY_V2_SOLD_OUT_FIELD", "legacy_v2_sold_out_field", "is_sold_out"
)

CROWDING_FIELD: str = _get("CROWDING_FIELD", "crowding_field", "entity_code")
LEGACY_V2_REGIONS_FIELD: str = _get(
    "LEGACY_V2_REGIONS_FIELD", "legacy_v2_regions_field", "active_regions"
)
LEGACY_V2_SHOW_DATES_FIELD: str = _get(
    "LEGACY_V2_SHOW_DATES_FIELD", "legacy_v2_show_dates_field", "event_dates"
)
