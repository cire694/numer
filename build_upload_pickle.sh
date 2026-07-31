#!/bin/bash
# build_upload_pickle.sh
#
# Full pipeline for producing a Numerai Model-Upload-ready pickle from a
# model trained in the main (3.14) environment:
#
#   1. Export the trained ensemble to version-agnostic portable files
#      (LightGBM .txt boosters + torch .pt state dict + JSON metadata),
#      run under the 3.14 env where the original .pkl actually loads.
#   2. Reconstruct the ensemble from those portable files and cloudpickle
#      the predict() wrapper, run under 3.12 to match Numerai's sandbox.
#
# Usage:
#   ./build_upload_pickle.sh models/dynamic_ensemble_20260728_225421
#
# (pass the model path WITHOUT the .pkl extension — the script derives
#  both the .pkl and the _portable/ dir from that base name)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <model_base_path (no .pkl extension)>"
    echo "Example: $0 models/dynamic_ensemble_20260728_225421"
    exit 1
fi

MODEL_BASE="$1"
MODEL_PKL="${MODEL_BASE}.pkl"
PORTABLE_DIR="${MODEL_BASE}_portable"
DOWNLOAD_NAME="predict_dynamic_ensemble"

if [ ! -f "$MODEL_PKL" ]; then
    echo "ERROR: $MODEL_PKL not found."
    exit 1
fi

echo "=== Step 1/2: exporting portable artifacts (3.14 env: numer) ==="
echo "Model: $MODEL_PKL"
echo "Output dir: $PORTABLE_DIR"

source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate numer

python "$SCRIPT_DIR/export_portable.py" --model-path "$MODEL_PKL" --out-dir "$PORTABLE_DIR"

echo ""
echo "=== Step 2/2: building upload pickle (3.12 env: numer_upload) ==="

conda activate numer_upload

python "$SCRIPT_DIR/predict_upload.py" --portable-dir "$PORTABLE_DIR" --download-name "$DOWNLOAD_NAME"

echo ""
echo "=== Done ==="
echo "Upload pickle: ${DOWNLOAD_NAME}.pkl"
echo ""
echo "Next: test it in the actual sandbox before uploading to numer.ai:"
echo "  docker run -i --rm -v \"\$PWD:\$PWD\" ghcr.io/numerai/numerai_predict_py_3_12:stable --debug --model \$PWD/${DOWNLOAD_NAME}.pkl"