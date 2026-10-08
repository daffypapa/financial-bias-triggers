#!/usr/bin/env bash
# RQ2: bias removal by rewriting, using gold and detected bias labels.
# Requires: rq1_detection.sh (uses each model's own detected labels).
# Output: outputs/bias_rewriting/<model_name>/
set -euo pipefail
cd "$(dirname "$0")/../src"
source ../bash_scripts/models.sh

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_TYPE MODEL_PATH MODEL_NAME <<< "$entry"
  python -m experiments.bias_rewriting \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME"
done
