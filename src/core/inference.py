"""
Core inference module. Handles model loading, dataset iteration, and response collection.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import torch
import gc
from core.model import TransformersModel, VLMModel, OpenAIModel, AnthropicModel, GeminiModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"

load_dotenv(REPO_ROOT / ".env")


def get_api_key(model_type: str, api_key_arg: str | None = None) -> str:
    """
    Get API key from argument or environment variable.
    
    Priority: argument > environment variable
    
    Environment variables:
        - openai: OPENAI_API_KEY
        - anthropic: ANTHROPIC_API_KEY
        - gemini: GOOGLE_API_KEY
    """
    if api_key_arg:
        return api_key_arg
    
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }
    
    env_var = env_var_map.get(model_type)
    if env_var:
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key
        raise ValueError(
            f"No API key provided. Either pass --api_key or set {env_var} environment variable."
        )
    
    raise ValueError(f"Unknown model type for API key: {model_type}")

def load_model(model_type: str, model_path: str, name: str, **kwargs):
    """Load a model based on type."""
    if model_type == "transformers":
        return TransformersModel(model_path=model_path, name=name, **kwargs)
    elif model_type == "vlm":
        return VLMModel(model_path=model_path, name=name, **kwargs)
    elif model_type == "openai":
        return OpenAIModel(model_name=model_path, name=name, **kwargs)
    elif model_type == "anthropic":
        return AnthropicModel(model_name=model_path, name=name, **kwargs)
    elif model_type == "gemini":
        return GeminiModel(model_name=model_path, name=name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_inference(
    model,
    df: pd.DataFrame,
    system_prompt: str,
    user_prompt_template: str,
    input_column: str,
    output_column: str = "response",
    desc: str = "Running inference",
) -> pd.DataFrame:
    """
    Run single inference per row.
    
    Args:
        model: Loaded model instance
        df: Input dataframe
        system_prompt: System prompt to use
        user_prompt_template: User prompt template with {sentence} placeholder
        input_column: Column name to read input from
        output_column: Column name to write output to
        desc: Progress bar description
    
    Returns:
        DataFrame with added output column
    """
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        text = row[input_column]
        user_prompt = user_prompt_template.format(sentence=text)
        response = model.prompt(system_prompt, user_prompt)
        results.append(response)
    
    df = df.copy()
    df[output_column] = results
    return df


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def unload_model(model):
    """Unload model and clear memory."""
    if hasattr(model, 'unload'):
        model.unload()
    clear_memory()