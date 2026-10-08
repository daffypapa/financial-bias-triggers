#!/usr/bin/env bash
# RQ2: LLM-as-judge evaluation of all rewrites in outputs/bias_rewriting/.
# Requires: rq2_rewriting.sh
# Output: outputs/bias_judge/gemma-4-31b/
set -euo pipefail
cd "$(dirname "$0")/../src"

python -m experiments.bias_judge \
  --model_type vlm \
  --model_path unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --model_name gemma-4-31b

# To tell the judge the gold bias labels instead of the detected ones, add:
#   --use_gold_labels
