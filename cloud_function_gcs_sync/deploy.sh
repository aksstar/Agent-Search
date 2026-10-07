#!/usr/bin/env bash
# ==============================================================================
# Deploy 2nd-Gen Cloud Run Function with Eventarc GCS Trigger
# Automatically triggers Agent Search DataStore sync on any new .jsonl upload
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load settings from ../config.local.json (or environment variables) via config.py
eval "$(python3 -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
import config
print(f'export GCP_PROJECT_ID=\"{config.PROJECT_ID}\"')
print(f'export GCP_PROJECT_NUMBER=\"{config.PROJECT_NUMBER}\"')
print(f'export DISCOVERY_ENGINE_LOCATION=\"{config.LOCATION}\"')
print(f'export REGION=\"{config.VERTEX_AI_LOCATION}\"')
print(f'export GCS_BUCKET=\"{config.GCS_BUCKET}\"')
print(f'export GCS_FOLDER=\"{config.GCS_FOLDER}\"')
print(f'export COLLECTION_ID=\"{config.COLLECTION_ID}\"')
print(f'export DATASTORE_ID=\"{config.DATASTORE_ID}\"')
")"

FUNCTION_NAME="${FUNCTION_NAME:-gcs-to-vertex-search-sync}"
BUCKET_LOCATION="$(gcloud storage buckets describe "gs://${GCS_BUCKET}" --project="${GCP_PROJECT_ID}" --format="value(location)" | tr '[:upper:]' '[:lower:]')"
TRIGGER_LOCATION="${BUCKET_LOCATION:-${REGION}}"

echo "=================================================================="
echo "Deploying 2nd-Gen Cloud Function: ${FUNCTION_NAME}"
echo "  Project          : ${GCP_PROJECT_ID}"
echo "  Function Region  : ${REGION}"
echo "  Bucket / Trigger : gs://${GCS_BUCKET}/${GCS_FOLDER}/*.jsonl (location: ${TRIGGER_LOCATION})"
echo "  Target DataStore : ${DATASTORE_ID} (collection: ${COLLECTION_ID})"
echo "=================================================================="

# 1. Ensure the GCS service account has pubsub.publisher permission for Eventarc GCS triggers
GCS_SERVICE_ACCOUNT="$(gcloud storage service-agent --project="${GCP_PROJECT_ID}")"
echo "Granting roles/pubsub.publisher to GCS service agent (${GCS_SERVICE_ACCOUNT})..."
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${GCS_SERVICE_ACCOUNT}" \
  --role="roles/pubsub.publisher" \
  --condition=None \
  --quiet >/dev/null

# 2. Deploy the 2nd-Gen Cloud Function triggered by google.cloud.storage.object.v1.finalized
gcloud functions deploy "${FUNCTION_NAME}" \
  --gen2 \
  --project="${GCP_PROJECT_ID}" \
  --region="${REGION}" \
  --runtime=python312 \
  --source="${SCRIPT_DIR}" \
  --entry-point=sync_gcs_to_datastore \
  --trigger-location="${TRIGGER_LOCATION}" \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=${GCS_BUCKET}" \
  --set-env-vars="GCP_PROJECT_ID=${GCP_PROJECT_ID},DISCOVERY_ENGINE_LOCATION=${DISCOVERY_ENGINE_LOCATION},GCS_BUCKET=${GCS_BUCKET},GCS_FOLDER=${GCS_FOLDER},COLLECTION_ID=${COLLECTION_ID},DATASTORE_ID=${DATASTORE_ID},RECONCILIATION_MODE=INCREMENTAL" \
  --timeout=120s \
  --memory=512Mi

echo "Deployment complete. Uploading any .jsonl file to gs://${GCS_BUCKET}/${GCS_FOLDER}/ will now automatically trigger ${FUNCTION_NAME}."
