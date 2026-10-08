#!/usr/bin/env bash
# RQ3: sentiment classification under the defense conditions.
# Requires: rq1_detection.sh and rq2_rewriting.sh for the same model.
# Output: outputs/bias_defense/<model_name>/
#
# Runs all conditions by default. Use --conditions to run a subset; existing
# predictions are checkpointed and resumed unless --override is passed.
set -euo pipefail
cd "$(dirname "$0")/../src"
source ../bash_scripts/models.sh

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_TYPE MODEL_PATH MODEL_NAME <<< "$entry"
  python -m experiments.bias_defense \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME"
done
