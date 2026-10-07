"""Centralized configuration loader for Agent Search & Gemini scripts.

Reads settings from environment variables or a local gitignored `config.local.json` file,
falling back to safe generic placeholders so no project IDs, app names, or credentials
are hardcoded in version-controlled code.
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
    return os.getenv(env_key) or str(_CFG.get(json_key, default))


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
