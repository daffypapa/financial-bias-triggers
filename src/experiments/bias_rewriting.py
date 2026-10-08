import os
import argparse
import pandas as pd

from core.inference import load_model, run_inference, unload_model, get_api_key, DATA_DIR, OUTPUTS_DIR
from prompts.rewriting import SYSTEM_PROMPT_REWRITING, USER_PROMPT_REWRITING


def main():
    parser = argparse.ArgumentParser(description="Run bias removal using LLMs")
    
    parser.add_argument("--model_type", type=str, default="transformers",
                        choices=["transformers", "vlm", "openai", "anthropic", "gemini"])
    parser.add_argument("--model_path", type=str, required=True, 
                        help="HuggingFace repo ID, local path, or API model name")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Pseudoname used for the output files and folders")
    parser.add_argument("--dataset_path", type=str, default=str(DATA_DIR / "triggerbank.csv"))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--text_column", type=str, default="biased_sentence")

    parser.add_argument("--detection_results_path", type=str, default=None,
                        help="Path to detection results. Must contain predicted_biases column. "
                             "Defaults to the same model's RQ1 output in --output_dir.")

    parser.add_argument("--max_out_tokens", type=int, default=256)
    parser.add_argument("--max_in_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--enable_thinking", action="store_true")
    
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    
    args = parser.parse_args()

    out_dir = os.path.join(args.output_dir, "bias_rewriting", args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    detection_results_path = args.detection_results_path or os.path.join(
        args.output_dir, "bias_detection", args.model_name, f"dataset_predictions_{args.model_name}.csv"
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

    detection_df = pd.read_csv(detection_results_path)
    df = df.merge(
        detection_df[["biased_sentence", "predicted_biases"]], 
        on="biased_sentence", 
        how="left"
    )

    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        name=args.model_name,
        **model_kwargs,
    )
    
    df["true_bias_prompt"] = df.apply(
        lambda row: USER_PROMPT_REWRITING.format(
            predicted_biases=row["biases"],
            sentence=row[args.text_column]
        ),
        axis=1
    )
    
    results_df = run_inference(
        model=model,
        df=df,
        system_prompt=SYSTEM_PROMPT_REWRITING,
        user_prompt_template="{sentence}",
        input_column="true_bias_prompt",
        output_column="true_bias_rewrite",
        desc=f"Bias Removal [gold labels] [{args.model_name}]",
    )
    
    results_df["detected_bias_prompt"] = results_df.apply(
        lambda row: USER_PROMPT_REWRITING.format(
            predicted_biases=row["predicted_biases"],
            sentence=row[args.text_column]
        ),
        axis=1
    )
    
    results_df = run_inference(
        model=model,
        df=results_df,
        system_prompt=SYSTEM_PROMPT_REWRITING,
        user_prompt_template="{sentence}",
        input_column="detected_bias_prompt",
        output_column="detected_bias_rewrite",
        desc=f"Bias Removal [detected labels] [{args.model_name}]",
    )
    
    results_df = results_df.drop(columns=["true_bias_prompt", "detected_bias_prompt"])
    
    output_path = os.path.join(out_dir, f"debiased_sentences_{args.model_name}.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"Saved debiased sentences to: {output_path}")
    
    unload_model(model)


if __name__ == "__main__":
    main()