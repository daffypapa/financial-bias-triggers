"""
Bias defense experiment (RQ3): Sentiment prediction under various defense conditions.
"""

import os
import argparse
import pandas as pd
from sklearn.metrics import classification_report

from core.inference import load_model, run_inference, unload_model, get_api_key, DATA_DIR, OUTPUTS_DIR
from prompts.defense import (
    SYSTEM_PROMPT_SENTIMENT,
    SYSTEM_PROMPT_COT,
    USER_PROMPT_BASELINE,
    USER_PROMPT_GENERIC_DEFENSE,
    USER_PROMPT_COT,
    USER_PROMPT_TARGETED,
)


ALL_CONDITIONS = [
    "1_clean_baseline",
    "2_no_defense", 
    "3_generic_defense",
    "4_cot",
    "4_cot_clean",
    "5_targeted_detected",
    "6_targeted_gold",
    "7_rewrite_detected",
    "8_rewrite_gold",
]

BIASED_CONDITIONS = [
    "2_no_defense", 
    "3_generic_defense",
    "4_cot",
    "5_targeted_detected",
    "6_targeted_gold",
    "7_rewrite_detected",
    "8_rewrite_gold",
]

CLEAN_CONDITIONS = [
    "1_clean_baseline",
    "4_cot_clean",
]


def run_condition_with_checkpointing(
    model, 
    df, 
    condition_name, 
    system_prompt, 
    user_prompt_template, 
    input_column,
    existing_df,
    predictions_path,
    checkpoint_every=1000
):
    """Run a single defense condition with intra-condition checkpointing."""
    
    # Determine the key column based on condition type
    if condition_name in CLEAN_CONDITIONS:
        key_column = "original_sentence"
    else:
        key_column = "biased_sentence"
    
    # Check which rows are already done for this condition
    if existing_df is not None and len(existing_df) > 0:
        existing_cond = existing_df[existing_df["condition"] == condition_name]
        if key_column in existing_cond.columns:
            done_sentences = set(existing_cond[key_column].tolist())
        else:
            done_sentences = set()
    else:
        existing_cond = pd.DataFrame()
        done_sentences = set()
    
    # Filter to remaining rows
    if key_column in df.columns:
        df_remaining = df[~df[key_column].isin(done_sentences)].copy()
    else:
        df_remaining = df.copy()
    
    if len(df_remaining) == 0:
        print(f"Condition {condition_name}: Already complete ({len(done_sentences)} rows)")
        return existing_cond
    
    print(f"Condition {condition_name}: {len(done_sentences)} done, {len(df_remaining)} remaining")
    
    # Collect results incrementally
    all_new_results = []
    
    for i in range(0, len(df_remaining), checkpoint_every):
        batch_df = df_remaining.iloc[i:i+checkpoint_every].copy()
        
        batch_results = run_inference(
            model=model,
            df=batch_df,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            input_column=input_column,
            output_column="predicted_sentiment",
            desc=f"Defense [{condition_name}] batch {i//checkpoint_every + 1}",
        )
        batch_results["condition"] = condition_name
        all_new_results.append(batch_results)
        
        # Save checkpoint after each batch
        combined_new = pd.concat(all_new_results, ignore_index=True)
        if len(existing_cond) > 0:
            checkpoint_df = pd.concat([existing_cond, combined_new], ignore_index=True)
        else:
            checkpoint_df = combined_new
        
        # Load full existing predictions and update
        if os.path.exists(predictions_path):
            full_existing = pd.read_csv(predictions_path)
            # Remove old entries for this condition
            full_existing = full_existing[full_existing["condition"] != condition_name]
            full_existing = pd.concat([full_existing, checkpoint_df], ignore_index=True)
        else:
            full_existing = checkpoint_df
        
        full_existing.to_csv(predictions_path, index=False)
        print(f"  Checkpoint saved: {len(checkpoint_df)} rows for {condition_name}")
    
    # Return all results for this condition
    if len(existing_cond) > 0:
        return pd.concat([existing_cond, pd.concat(all_new_results, ignore_index=True)], ignore_index=True)
    else:
        return pd.concat(all_new_results, ignore_index=True)


def parse_sentiment(text: str) -> str:
    """Parse model output to sentiment label."""
    text = str(text).lower().strip()
    if "positive" in text:
        return "positive"
    elif "negative" in text:
        return "negative"
    elif "neutral" in text:
        return "neutral"
    return "neutral"


def parse_sentiment_cot(text: str) -> str:
    """Parse COT output — prioritize FINAL ANSWER if present."""
    text = str(text).lower().strip()

    if "final answer:" in text:
        answer_part = text.split("final answer:")[-1].strip()
        if "positive" in answer_part:
            return "positive"
        elif "negative" in answer_part:
            return "negative"
        elif "neutral" in answer_part:
            return "neutral"

    return parse_sentiment(text)


def evaluate_condition(df: pd.DataFrame, true_col: str, pred_col: str, condition_name: str) -> pd.DataFrame:
    """Generate classification report for a condition."""
    report = classification_report(
        df[true_col],
        df[pred_col],
        labels=["positive", "neutral", "negative"],
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    report_df["condition"] = condition_name
    return report_df


def main():
    parser = argparse.ArgumentParser(description="Run bias defense experiments (RQ3)")

    parser.add_argument("--model_type", type=str, required=True,
                        choices=["transformers", "vlm", "openai", "anthropic", "gemini"])
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace repo ID, local path, or API model name")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pseudoname used for the output files and folders")

    parser.add_argument("--dataset_path", type=str, default=str(DATA_DIR / "triggerbank.csv"),
                        help="Path to dataset")
    parser.add_argument("--detection_results_path", type=str, default=None,
                        help="Path to detection results. Must contain predicted_biases column. "
                             "Defaults to the same model's RQ1 output in --output_dir.")
    parser.add_argument("--rewriting_results_path", type=str, default=None,
                        help="Path to rewriting results. Must contain detected_bias_rewrite and true_bias_rewrite columns. "
                             "Defaults to the same model's RQ2 output in --output_dir.")

    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--label_column", type=str, default="label")

    parser.add_argument("--max_out_tokens", type=int, default=1024)
    parser.add_argument("--max_in_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--enable_thinking", action="store_true")

    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)

    parser.add_argument("--checkpoint_every", type=int, default=1000,
                        help="Save checkpoint every N rows within a condition")

    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        choices=ALL_CONDITIONS,
                        help="Specific conditions to run. If not provided, runs all conditions.")

    parser.add_argument("--override", action="store_true",
                        help="Run experiment from start, ignoring existing saved results")

    args = parser.parse_args()

    # Determine which conditions to run
    conditions_to_run = args.conditions if args.conditions else ALL_CONDITIONS

    out_dir = os.path.join(args.output_dir, "bias_defense", args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    predictions_path = os.path.join(out_dir, f"predictions_{args.model_name}.csv")
    report_path = os.path.join(out_dir, f"classification_report_{args.model_name}.csv")

    detection_results_path = args.detection_results_path or os.path.join(
        args.output_dir, "bias_detection", args.model_name, f"dataset_predictions_{args.model_name}.csv"
    )
    rewriting_results_path = args.rewriting_results_path or os.path.join(
        args.output_dir, "bias_rewriting", args.model_name, f"debiased_sentences_{args.model_name}.csv"
    )

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

    df = pd.read_csv(args.dataset_path)

    df_biased = df[df["biases"] != "none"].copy()
    df_clean = df[df["biases"] == "none"].copy()

    detection_df = pd.read_csv(detection_results_path)
    df_biased = df_biased.merge(
        detection_df[["biased_sentence", "predicted_biases"]],
        on="biased_sentence",
        how="left"
    )

    rewriting_df = pd.read_csv(rewriting_results_path)
    df_biased = df_biased.merge(
        rewriting_df[["biased_sentence", "detected_bias_rewrite", "true_bias_rewrite"]],
        on="biased_sentence",
        how="left"
    )

    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        name=args.model_name,
        **model_kwargs,
    )

    # Load existing predictions or start fresh
    if not args.override and os.path.exists(predictions_path):
        existing_df = pd.read_csv(predictions_path)
        # Remove rows for conditions we're about to rerun
        existing_df = existing_df[~existing_df["condition"].isin(conditions_to_run)]
        print(f"Loaded existing predictions, will update conditions: {conditions_to_run}")
    else:
        existing_df = pd.DataFrame()
        if args.override and os.path.exists(predictions_path):
            os.remove(predictions_path)

    all_results = [existing_df] if len(existing_df) > 0 else []

    # Condition 1: Clean baseline
    if "1_clean_baseline" in conditions_to_run and len(df_clean) > 0:
        results_c1 = run_condition_with_checkpointing(
            model=model,
            df=df_clean,
            condition_name="1_clean_baseline",
            system_prompt=SYSTEM_PROMPT_SENTIMENT,
            user_prompt_template=USER_PROMPT_BASELINE,
            input_column="original_sentence",
            existing_df=existing_df,
            predictions_path=predictions_path,
            checkpoint_every=args.checkpoint_every,
        )
        results_c1["parsed_sentiment"] = results_c1["predicted_sentiment"].apply(parse_sentiment)
        all_results.append(results_c1)

    # Condition 2: No defense
    if "2_no_defense" in conditions_to_run:
        results_c2 = run_condition_with_checkpointing(
            model=model,
            df=df_biased,
            condition_name="2_no_defense",
            system_prompt=SYSTEM_PROMPT_SENTIMENT,
            user_prompt_template=USER_PROMPT_BASELINE,
            input_column="biased_sentence",
            existing_df=existing_df,
            predictions_path=predictions_path,
            checkpoint_every=args.checkpoint_every,
        )
        results_c2["parsed_sentiment"] = results_c2["predicted_sentiment"].apply(parse_sentiment)
        all_results.append(results_c2)

    # Condition 3: Generic defense
    if "3_generic_defense" in conditions_to_run:
        results_c3 = run_condition_with_checkpointing(
            model=model,
            df=df_biased,
            condition_name="3_generic_defense",
            system_prompt=SYSTEM_PROMPT_SENTIMENT,
            user_prompt_template=USER_PROMPT_GENERIC_DEFENSE,
            input_column="biased_sentence",
            existing_df=existing_df,
            predictions_path=predictions_path,
            checkpoint_every=args.checkpoint_every,
        )
        results_c3["parsed_sentiment"] = results_c3["predicted_sentiment"].apply(parse_sentiment)
        all_results.append(results_c3)

    # Condition 4: Chain-of-thought on biased
    if "4_cot" in conditions_to_run:
        results_c4 = run_condition_with_checkpointing(
            model=model,
            df=df_biased,
            condition_name="4_cot",
            system_prompt=SYSTEM_PROMPT_COT,
            user_prompt_template=USER_PROMPT_COT,
            input_column="biased_sentence",
            existing_df=existing_df,
            predictions_path=predictions_path,
            checkpoint_every=args.checkpoint_every,
        )
        results_c4["parsed_sentiment"] = results_c4["predicted_sentiment"].apply(parse_sentiment_cot)
        all_results.append(results_c4)

    # Condition 4_cot_clean: Chain-of-thought on clean
    if "4_cot_clean" in conditions_to_run and len(df_clean) > 0:
        results_c4_clean = run_condition_with_checkpointing(
            model=model,
            df=df_clean,
            condition_name="4_cot_clean",
            system_prompt=SYSTEM_PROMPT_COT,
            user_prompt_template=USER_PROMPT_COT,
            input_column="original_sentence",
            existing_df=existing_df,
            predictions_path=predictions_path,
            checkpoint_every=args.checkpoint_every,
        )
        results_c4_clean["parsed_sentiment"] = results_c4_clean["predicted_sentiment"].apply(parse_sentiment_cot)
        all_results.append(results_c4_clean)

    # Condition 5: Targeted defense with detected labels
    if "5_targeted_detected" in conditions_to_run:
        df_c5 = df_biased.copy()
        df_c5["targeted_prompt_detected"] = df_c5.apply(
            lambda row: USER_PROMPT_TARGETED.format(
                sentence=row["biased_sentence"],
                bias_type=row["predicted_biases"]
            ),
            axis=1
        )
        results_c5 = run_condition_with_checkpointing(
            model=model,
            df=df_c5,
            condition_name="5_targeted_detected",
            system_prompt=SYSTEM_PROMPT_SENTIMENT,
            user_prompt_template="{sentence}",
            input_column="targeted_prompt_detected",
            existing_df=existing_df,
            predictions_path=predictions_path,
            checkpoint_every=args.checkpoint_every,
        )
        results_c5["parsed_sentiment"] = results_c5["predicted_sentiment"].apply(parse_sentiment)
        all_results.append(results_c5)

    # Condition 6: Targeted defense with gold labels
    if "6_targeted_gold" in conditions_to_run:
        df_c6 = df_biased.copy()
        df_c6["targeted_prompt_gold"] = df_c6.apply(
            lambda row: USER_PROMPT_TARGETED.format(
                sentence=row["biased_sentence"],
                bias_type=row["biases"]
            ),
            axis=1
        )
        results_c6 = run_condition_with_checkpointing(
            model=model,
            df=df_c6,
            condition_name="6_targeted_gold",
            system_prompt=SYSTEM_PROMPT_SENTIMENT,
            user_prompt_template="{sentence}",
            input_column="targeted_prompt_gold",
            existing_df=existing_df,
            predictions_path=predictions_path,
            checkpoint_every=args.checkpoint_every,
        )
        results_c6["parsed_sentiment"] = results_c6["predicted_sentiment"].apply(parse_sentiment)
        all_results.append(results_c6)

    # Condition 7: Rewrite with detected labels then classify
    if "7_rewrite_detected" in conditions_to_run:
        results_c7 = run_condition_with_checkpointing(
            model=model,
            df=df_biased,
            condition_name="7_rewrite_detected",
            system_prompt=SYSTEM_PROMPT_SENTIMENT,
            user_prompt_template=USER_PROMPT_BASELINE,
            input_column="detected_bias_rewrite",
            existing_df=existing_df,
            predictions_path=predictions_path,
            checkpoint_every=args.checkpoint_every,
        )
        results_c7["parsed_sentiment"] = results_c7["predicted_sentiment"].apply(parse_sentiment)
        all_results.append(results_c7)

    # Condition 8: Rewrite with gold labels then classify
    if "8_rewrite_gold" in conditions_to_run:
        results_c8 = run_condition_with_checkpointing(
            model=model,
            df=df_biased,
            condition_name="8_rewrite_gold",
            system_prompt=SYSTEM_PROMPT_SENTIMENT,
            user_prompt_template=USER_PROMPT_BASELINE,
            input_column="true_bias_rewrite",
            existing_df=existing_df,
            predictions_path=predictions_path,
            checkpoint_every=args.checkpoint_every,
        )
        results_c8["parsed_sentiment"] = results_c8["predicted_sentiment"].apply(parse_sentiment)
        all_results.append(results_c8)

    # Combine all results and save final predictions
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Deduplicate biased conditions
    df_biased_results = combined_df[combined_df["condition"].isin(BIASED_CONDITIONS)]
    df_biased_results = df_biased_results.drop_duplicates(subset=["condition", "biased_sentence"], keep="last")
    
    # Deduplicate clean conditions
    df_clean_results = combined_df[combined_df["condition"].isin(CLEAN_CONDITIONS)]
    df_clean_results = df_clean_results.drop_duplicates(subset=["condition", "original_sentence"], keep="last")
    
    combined_df = pd.concat([df_biased_results, df_clean_results], ignore_index=True)
    combined_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")

    # Generate classification reports for all conditions present
    all_reports = []
    for condition_name in ALL_CONDITIONS:
        cond_df = combined_df[combined_df["condition"] == condition_name]
        if len(cond_df) > 0:
            report = evaluate_condition(cond_df, args.label_column, "parsed_sentiment", condition_name)
            all_reports.append(report)

    if all_reports:
        combined_reports = pd.concat(all_reports)
        combined_reports.to_csv(report_path, index=True)
        print(f"Saved classification report to: {report_path}")

    unload_model(model)


if __name__ == "__main__":
    main()