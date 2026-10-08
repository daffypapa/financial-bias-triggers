"""
Bias injection script. Injects cognitive bias triggers into financial sentences.
"""

import os
import argparse
import importlib
import pandas as pd
from itertools import combinations

from core.inference import load_model, run_inference, unload_model, get_api_key, DATA_DIR, OUTPUTS_DIR


def load_prompts(prompt_module: str):
    """Load prompts from specified module."""
    module = importlib.import_module(prompt_module)
    
    return {
        "system": module.SYSTEM_PROMPT_BIAS_INJECTION,
        "bias_templates": {
            "anchoring": module.USER_PROMPT_ANCHORING,
            "authority": module.USER_PROMPT_AUTHORITY,
            "herding": module.USER_PROMPT_HERDING,
            "recency": module.USER_PROMPT_RECENCY,
            "overconfidence": module.USER_PROMPT_OVERCONFIDENCE,
        },
        # Framing is applied against the gold sentiment: positive sentences receive a
        # negative spin and vice versa. Neutral sentences are not framed.
        "framing_templates": {
            "positive": module.USER_PROMPT_FRAMING_NEGATIVE,
            "negative": module.USER_PROMPT_FRAMING_POSITIVE,
        },
    }


def inject_single_biases(
    model,
    df: pd.DataFrame,
    prompts: dict,
    text_column: str = "sentence",
    label_column: str = "label",
) -> pd.DataFrame:
    """Inject individual biases."""
    results = []
    
    for bias_name, bias_template in prompts["bias_templates"].items():
        biased_df = run_inference(
            model=model,
            df=df,
            system_prompt=prompts["system"],
            user_prompt_template=bias_template,
            input_column=text_column,
            output_column="biased_sentence",
            desc=f"Injecting {bias_name}",
        )
        biased_df = biased_df.rename(columns={text_column: "original_sentence"})
        biased_df["biases"] = bias_name
        results.append(biased_df)
    
    for sentiment, framing_template in prompts["framing_templates"].items():
        df_subset = df[df[label_column] == sentiment].copy()
        if len(df_subset) == 0:
            continue
        
        biased_df = run_inference(
            model=model,
            df=df_subset,
            system_prompt=prompts["system"],
            user_prompt_template=framing_template,
            input_column=text_column,
            output_column="biased_sentence",
            desc=f"Injecting framing ({sentiment})",
        )
        biased_df = biased_df.rename(columns={text_column: "original_sentence"})
        biased_df["biases"] = "framing"
        results.append(biased_df)
    
    return pd.concat(results, ignore_index=True)


def inject_pairwise_biases(
    model,
    df: pd.DataFrame,
    prompts: dict,
    text_column: str = "sentence",
    label_column: str = "label",
) -> pd.DataFrame:
    """Inject pairwise bias combinations."""
    results = []
    non_framing_pairs = list(combinations(prompts["bias_templates"].keys(), 2))
    
    for bias1, bias2 in non_framing_pairs:
        intermediate_df = run_inference(
            model=model,
            df=df,
            system_prompt=prompts["system"],
            user_prompt_template=prompts["bias_templates"][bias1],
            input_column=text_column,
            output_column="intermediate_sentence",
            desc=f"Injecting {bias1} (pair 1/2)",
        )

        intermediate_df["intermediate_sentence"] = intermediate_df["intermediate_sentence"].astype(str).apply(
            lambda x: "".join(ch for ch in x if ch.isprintable())
        )
        
        biased_df = run_inference(
            model=model,
            df=intermediate_df,
            system_prompt=prompts["system"],
            user_prompt_template=prompts["bias_templates"][bias2],
            input_column="intermediate_sentence",
            output_column="biased_sentence",
            desc=f"Injecting {bias2} (pair 2/2)",
        )
        
        biased_df = biased_df.rename(columns={text_column: "original_sentence"})
        biased_df["biases"] = f"{bias1},{bias2}"
        biased_df = biased_df.drop(columns=["intermediate_sentence"])
        results.append(biased_df)
    
    for sentiment, framing_template in prompts["framing_templates"].items():
        df_subset = df[df[label_column] == sentiment].copy()
        if len(df_subset) == 0:
            continue
        
        framed_df = run_inference(
            model=model,
            df=df_subset,
            system_prompt=prompts["system"],
            user_prompt_template=framing_template,
            input_column=text_column,
            output_column="framed_sentence",
            desc=f"Injecting framing ({sentiment}, pair 1/2)",
        )

        framed_df["framed_sentence"] = framed_df["framed_sentence"].astype(str).apply(
            lambda x: "".join(ch for ch in x if ch.isprintable())
        )

        for bias_name, bias_template in prompts["bias_templates"].items():
            biased_df = run_inference(
                model=model,
                df=framed_df,
                system_prompt=prompts["system"],
                user_prompt_template=bias_template,
                input_column="framed_sentence",
                output_column="biased_sentence",
                desc=f"Injecting framing+{bias_name} (pair 2/2)",
            )
            
            biased_df = biased_df.rename(columns={text_column: "original_sentence"})
            biased_df["biases"] = f"framing,{bias_name}"
            biased_df = biased_df.drop(columns=["framed_sentence"])
            results.append(biased_df)
    
    return pd.concat(results, ignore_index=True)


def inject_all_biases(
    model,
    prompts: dict,
    dataset_path: str,
    text_column: str = "sentence",
    label_column: str = "label",
) -> pd.DataFrame:
    """Run full bias injection pipeline."""
    df = pd.read_csv(dataset_path)
    
    single_df = inject_single_biases(model, df, prompts, text_column, label_column)
    pairwise_df = inject_pairwise_biases(model, df, prompts, text_column, label_column)
    
    return pd.concat([single_df, pairwise_df], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Run bias injection using LLMs")
    
    parser.add_argument("--model_type", type=str, default="openai",
                        choices=["transformers", "vlm", "openai", "anthropic", "gemini"])
    parser.add_argument("--model_path", type=str, default="gpt-5.4-2026-03-05",
                        help="HuggingFace repo ID, local path, or API model name")
    parser.add_argument("--model_name", type=str, default="gpt-5.4",
                        help="Pseudoname used in the output file name")
    parser.add_argument("--dataset_path", type=str,
                        default=str(DATA_DIR / "base_sentences.csv"))
    parser.add_argument("--output_path", type=str,
                        default=str(OUTPUTS_DIR / "bias_injection" / "biased_sentences.csv"),
                        help="Model and prompt module names are appended to the file name")

    parser.add_argument("--prompt_module", type=str, default="prompts.injection_rewrite",
                        help="Module path for prompts (default: prompts.injection_rewrite)")
    
    parser.add_argument("--text_column", type=str, default="sentence")
    parser.add_argument("--label_column", type=str, default="label")
    
    parser.add_argument("--max_out_tokens", type=int, default=256)
    parser.add_argument("--max_in_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--enable_thinking", action="store_true")

    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_module)
    
    prompt_name = args.prompt_module.split(".")[-1]
    if args.output_path.endswith(".csv"):
        output_path = args.output_path.replace(".csv", f"_{args.model_name}_{prompt_name}.csv")
    else:
        output_path = f"{args.output_path}_{args.model_name}_{prompt_name}.csv"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

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
    
    biases_df = inject_all_biases(
        model=model,
        prompts=prompts,
        dataset_path=args.dataset_path,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    
    biases_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    unload_model(model)


if __name__ == "__main__":
    main()