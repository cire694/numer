#!/bin/bash
# submit_fat_boi_final.sh — build a Model-Upload-ready pickle for the final model.
#
# Usage: ./submit_fat_boi_final.sh models/dynamic_ensemble_20260728_XXXXXX
# (path WITHOUT .pkl extension)

set -euo pipefail

TRAIN_ENV="numer"
UPLOAD_ENV="numer_upload"
NUMERAI_MODEL_NAME="fat_boi"

[ $# -lt 1 ] && { echo "Usage: $0 <model_base_path (no .pkl)>"; exit 1; }
MODEL_BASE="$1"
PORTABLE_DIR="${MODEL_BASE}_portable"
DOWNLOAD_NAME="predict_${NUMERAI_MODEL_NAME}_final"

module load miniforge

echo "=== Step 1/2: exporting portable artifacts (env=$TRAIN_ENV) ==="
conda activate "$TRAIN_ENV"
python export_portable.py --model-path "${MODEL_BASE}.pkl" --out-dir "$PORTABLE_DIR"

echo "=== Step 2/2: building upload pickle (env=$UPLOAD_ENV) ==="
conda activate "$UPLOAD_ENV"
python predict_upload.py --portable-dir "$PORTABLE_DIR" --download-name "$DOWNLOAD_NAME"

echo ""
echo "Built: ${DOWNLOAD_NAME}.pkl — upload this at numer.ai/submissions for model '$NUMERAI_MODEL_NAME'"