# Models evaluated in the paper, as "model_type|model_path|model_name".
# Sourced by the RQ scripts. Comment out entries to run a subset.
#
# API models read their keys from the .env file in the repository root.
# Open models are loaded from HuggingFace (4-bit unsloth checkpoints where noted).

MODELS=(
  # API models
  "openai|gpt-5.4-mini-2026-03-17|gpt-5.4-mini"
  "anthropic|claude-haiku-4-5-20251001|claude-haiku-4.5"
  "gemini|gemini-3-flash-preview|gemini-3-flash"

  # Open models
  "transformers|unsloth/Llama-3.1-8B-Instruct|llama-3.1-8b"
  "transformers|unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit|llama-3.1-70b"
  "transformers|unsloth/Llama-3.3-70B-Instruct-bnb-4bit|llama-3.3-70b"
  "transformers|unsloth/Qwen3-8B|qwen3-8b"
  "transformers|unsloth/Qwen3.5-9B|qwen3.5-9b"
  "transformers|unsloth/Qwen3-32B-bnb-4bit|qwen3-32b"
  "vlm|unsloth/gemma-4-12b-it|gemma-4-12b"
  "vlm|unsloth/gemma-4-31B-it-unsloth-bnb-4bit|gemma-4-31b"
)
