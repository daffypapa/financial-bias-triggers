"""
LLM-as-judge evaluation of bias rewriting (RQ2).
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path

from core.inference import load_model, run_inference, unload_model, get_api_key, OUTPUTS_DIR
from prompts.llm_as_judge import SYSTEM_PROMPT_JUDGE, USER_PROMPT_JUDGE


def parse_judge_response(text: str) -> dict:
    text = str(text).strip()

    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
        return {
            "trigger_removed": int(result.get("trigger_removed", 0)),
            "facts_preserved": int(result.get("facts_preserved", 0)),
            "natural_language": int(result.get("natural_language", 0)),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "trigger_removed": -1,
            "facts_preserved": -1,
            "natural_language": -1,
        }


def judge_dataframe(model, df: pd.DataFrame, bias_column: str = "predicted_biases", desc: str = "Judging") -> pd.DataFrame:
    df = df.copy()

    # Filter out unbiased rows if column exists
    if "biases" in df.columns:
        df = df[df["biases"] != "none"].copy()

    df["judge_prompt"] = df.apply(
        lambda row: USER_PROMPT_JUDGE.format(
            original_sentence=row["original_sentence"],
            biased_sentence=row["biased_sentence"],
            debiased_sentence=row["detected_bias_rewrite"],
            predicted_biases=row[bias_column]
        ),
        axis=1
    )

    results_df = run_inference(
        model=model,
        df=df,
        system_prompt=SYSTEM_PROMPT_JUDGE,
        user_prompt_template="{sentence}",
        input_column="judge_prompt",
        output_column="judge_response",
        desc=desc,
    )

    parsed = results_df["judge_response"].apply(parse_judge_response)
    results_df["trigger_removed"] = parsed.apply(lambda x: x["trigger_removed"])
    results_df["facts_preserved"] = parsed.apply(lambda x: x["facts_preserved"])
    results_df["natural_language"] = parsed.apply(lambda x: x["natural_language"])

    results_df = results_df.drop(columns=["judge_prompt"])

    return results_df


def compute_summary(results_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    valid_df = results_df[results_df["trigger_removed"] != -1]

    summary = {
        "total_evaluated": len(results_df),
        "valid_responses": len(valid_df),
        "parse_failures": len(results_df) - len(valid_df),
        "trigger_removed_rate": valid_df["trigger_removed"].mean() if len(valid_df) > 0 else 0,
        "facts_preserved_rate": valid_df["facts_preserved"].mean() if len(valid_df) > 0 else 0,
        "natural_language_rate": valid_df["natural_language"].mean() if len(valid_df) > 0 else 0,
    }

    if "biases" in valid_df.columns:
        per_bias = valid_df.groupby("biases").agg({
            "trigger_removed": "mean",
            "facts_preserved": "mean",
            "natural_language": "mean",
        }).round(3)
    else:
        per_bias = pd.DataFrame()

    return summary, per_bias


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluation for bias rewriting")

    parser.add_argument("--model_type", type=str, required=True,
                        choices=["transformers", "vlm", "openai", "anthropic", "gemini"])
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace repo ID, local path, or API model name")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pseudoname for the judge model")

    parser.add_argument("--rewriting_dir", type=str,
                        default=str(OUTPUTS_DIR / "bias_rewriting"),
                        help="Directory containing model rewriting outputs (one subfolder per model)")
    parser.add_argument("--skip", type=str, nargs="+", default=[],
                        help="List of model names to skip")

    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR))

    parser.add_argument("--use_gold_labels", action="store_true",
                        help="Use gold bias labels instead of predicted biases")

    parser.add_argument("--max_out_tokens", type=int, default=128)
    parser.add_argument("--max_in_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--enable_thinking", action="store_true")

    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)

    args = parser.parse_args()

    # Determine which bias column to use
    bias_column = "biases" if args.use_gold_labels else "predicted_biases"

    # Determine judge folder name
    judge_folder = f"{args.model_name}_gold_labels" if args.use_gold_labels else args.model_name

    judge_out_dir = os.path.join(args.output_dir, "bias_judge", judge_folder)
    os.makedirs(judge_out_dir, exist_ok=True)

    if args.model_type in ["transformers", "vlm"]:
        model_kwargs = {
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
            "max_out_tokens": args.max_out_tokens,
            "max_in_tokens": args.max_in_tokens,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
        }

    else:
        api_key = get_api_key(args.model_type, args.api_key)
        model_kwargs = {
            "api_key": api_key,
            "max_out_tokens": args.max_out_tokens,
            "temperature": args.temperature,
        }
        if args.model_type == "openai":
            model_kwargs["base_url"] = args.base_url

    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        name=args.model_name,
        **model_kwargs,
    )

    all_summaries = []

    for judged_model_dir in sorted(Path(args.rewriting_dir).iterdir()):
        if not judged_model_dir.is_dir():
            continue

        judged_model_name = judged_model_dir.name

        if judged_model_name in args.skip:
            print(f"Skipping {judged_model_name}")
            continue

        rewrite_file = None
        for f in judged_model_dir.iterdir():
            if f.name.startswith("debiased_sentences_") and f.suffix == ".csv":
                rewrite_file = f
                break

        if rewrite_file is None:
            print(f"No rewriting file found for {judged_model_name}, skipping")
            continue

        print(f"\nJudging {judged_model_name}...")
        df = pd.read_csv(rewrite_file)

        results_df = judge_dataframe(model, df, bias_column=bias_column, desc=f"Judging [{judged_model_name}]")
        summary, per_bias = compute_summary(results_df)
        summary["judged_model"] = judged_model_name
        all_summaries.append(summary)

        judged_out_dir = os.path.join(judge_out_dir, judged_model_name)
        os.makedirs(judged_out_dir, exist_ok=True)

        results_path = os.path.join(judged_out_dir, "judge_results.csv")
        per_bias_path = os.path.join(judged_out_dir, "judge_per_bias.csv")

        results_df.to_csv(results_path, index=False)
        per_bias.to_csv(per_bias_path, index=True)

        print(f"  Valid: {summary['valid_responses']}/{summary['total_evaluated']}")
        print(f"  Trigger removed: {summary['trigger_removed_rate']:.1%}")
        print(f"  Facts preserved: {summary['facts_preserved_rate']:.1%}")
        print(f"  Natural language: {summary['natural_language_rate']:.1%}")

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(judge_out_dir, "all_models_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved overall summary to: {summary_path}")

    unload_model(model)


if __name__ == "__main__":
    main()
