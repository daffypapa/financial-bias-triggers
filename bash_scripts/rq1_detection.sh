#!/usr/bin/env bash
# RQ1: zero-shot bias trigger detection on data/triggerbank.csv.
# Output: outputs/bias_detection/<model_name>/
set -euo pipefail
cd "$(dirname "$0")/../src"
source ../bash_scripts/models.sh

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_TYPE MODEL_PATH MODEL_NAME <<< "$entry"
  python -m experiments.bias_detection \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME"
done
