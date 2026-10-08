#!/usr/bin/env bash
# Injects bias triggers into the 150 base sentences with GPT-5.4 (temperature 1.0).
# Output: outputs/bias_injection/biased_sentences_gpt-5.4_injection_rewrite.csv
#
# Generation is sampled, so a rerun will not reproduce data/triggerbank.csv exactly.
# The RQ experiments use the released dataset. After running this script, see
# notebooks/01_dataset_postprocessing.ipynb for the postprocessing step.
set -euo pipefail
cd "$(dirname "$0")/../src"

python -m experiments.bias_injection \
  --model_type openai \
  --model_path gpt-5.4-2026-03-05 \
  --model_name gpt-5.4 \
  --prompt_module prompts.injection_rewrite \
  --temperature 1.0
