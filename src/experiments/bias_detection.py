import os
import argparse
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

from core.inference import load_model, run_inference, unload_model, get_api_key, DATA_DIR, OUTPUTS_DIR
from prompts.detection import SYSTEM_PROMPT_BIAS_DETECTION, USER_PROMPT_DETECTION


def parse_predictions(text: str) -> list[str]:
    valid_biases = ['anchoring', 'authority', 'framing', 'herding', 'recency', 'overconfidence']
    text = str(text).lower()
    
    if "none" in text and not any(b in text for b in valid_biases):
        return ["none"]  
        
    detected_biases = [bias for bias in valid_biases if bias in text]
    
    return detected_biases if detected_biases else ["none"]


def parse_ground_truth(text: str) -> list[str]:
    if pd.isna(text) or str(text).strip() == "" or str(text).strip().lower() == "none":
        return ["none"]
        
    return [b.strip().lower() for b in str(text).split(',') if b.strip()]


def evaluate_detection(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    y_true = df[true_col].apply(parse_ground_truth)
    y_pred = df[pred_col].apply(parse_ground_truth)
    
    classes = ['anchoring', 'authority', 'framing', 'herding', 'recency', 'overconfidence', 'none']
    
    mlb = MultiLabelBinarizer(classes=classes)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    report = classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, output_dict=True)
    return pd.DataFrame(report).transpose()


def main():
    parser = argparse.ArgumentParser(description="Run zero-shot bias detection using LLMs")
    
    parser.add_argument("--model_type", type=str, default="transformers",
                        choices=["transformers", "vlm", "openai", "anthropic", "gemini"])
    parser.add_argument("--model_path", type=str, required=True, 
                        help="HuggingFace repo ID, local path, or API model name")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Pseudoname used for the output files and folders")
    parser.add_argument("--dataset_path", type=str, default=str(DATA_DIR / "triggerbank.csv"))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--text_column", type=str, default="biased_sentence")
    parser.add_argument("--true_label_column", type=str, default="biases")
    
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

    out_dir = os.path.join(args.output_dir, "bias_detection", args.model_name)
    os.makedirs(out_dir, exist_ok=True)
    
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

    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        name=args.model_name,
        **model_kwargs,
    )
    
    results_df = run_inference(
        model=model,
        df=df,
        system_prompt=SYSTEM_PROMPT_BIAS_DETECTION,
        user_prompt_template=USER_PROMPT_DETECTION,
        input_column=args.text_column,
        output_column="raw_model_response",
        desc=f"Zero-Shot Detection [{args.model_name}]",
    )
    
    results_df["predicted_biases"] = results_df["raw_model_response"].apply(
        lambda x: ",".join(parse_predictions(x))
    )
    
    report_df = evaluate_detection(
        df=results_df, 
        true_col=args.true_label_column, 
        pred_col="predicted_biases"
    )
    
    dataset_output_path = os.path.join(out_dir, f"dataset_predictions_{args.model_name}.csv")
    report_output_path = os.path.join(out_dir, f"classification_report_{args.model_name}.csv")

    results_df.to_csv(dataset_output_path, index=False)
    report_df.to_csv(report_output_path, index=True)
    
    unload_model(model)


if __name__ == "__main__":
    main()