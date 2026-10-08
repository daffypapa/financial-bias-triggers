#!/usr/bin/env bash
# Prompt sensitivity of bias injection: detection on datasets produced with two
# alternative injection prompts (data/prompt_sensitivity/biased_sample_alt_{1,2}.csv),
# built from a stratified sample of 30 base sentences (data/prompt_sensitivity/sample_30.csv).
# Output: outputs/prompt_sensitivity/bias_detection/<model_name>_alt{1,2}/
# Analysis: notebooks/rq1_prompt_sensitivity.ipynb (also needs rq1_detection.sh outputs).
set -euo pipefail
cd "$(dirname "$0")/../src"

DATA_DIR="../data/prompt_sensitivity"
OUTPUT_DIR="../outputs/prompt_sensitivity"

# To regenerate the alternative datasets (sampled, so not exactly reproducible), uncomment:
# for ALT in 1 2; do
#   python -m experiments.bias_injection \
#     --dataset_path "$DATA_DIR/sample_30.csv" \
#     --output_path "$OUTPUT_DIR/biased_sample.csv" \
#     --prompt_module "prompts.injection_rewrite_prompt_sensitivity_alt_${ALT}"
# done

MODELS=(
  "openai|gpt-5.4-mini-2026-03-17|gpt-5.4-mini"
  "transformers|unsloth/Llama-3.1-8B-Instruct|llama-3.1-8b"
  "transformers|unsloth/Qwen3-8B|qwen3-8b"
  "vlm|unsloth/gemma-4-31B-it-unsloth-bnb-4bit|gemma-4-31b"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_TYPE MODEL_PATH MODEL_NAME <<< "$entry"
  for ALT in 1 2; do
    python -m experiments.bias_detection \
      --model_type "$MODEL_TYPE" \
      --model_path "$MODEL_PATH" \
      --model_name "${MODEL_NAME}_alt${ALT}" \
      --dataset_path "$DATA_DIR/biased_sample_alt_${ALT}.csv" \
      --output_dir "$OUTPUT_DIR"
  done
done
