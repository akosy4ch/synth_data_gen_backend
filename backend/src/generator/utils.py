import os
import json
import time
import torch
import random
import string
import logging
import numpy as np
import pandas as pd
from typing import Any

# Global cache for models and markov
_MODEL_CACHE = {}
_MARKOV_MODEL = None

def get_device() -> str:
    """
    Return the best available torch device: MPS (Apple), CUDA, or CPU.
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def move_to_device(model):
    """Move model to the appropriate device."""
    return model.to(get_device())

def _warmup_model(model_name, model, tokenizer):
    """Warm up model with dummy input to reduce latency."""
    try:
        if model_name.lower() == "distil-cmlm":
            dummy = "Hello " + (tokenizer.mask_token or "[MASK]")
            inputs = tokenizer(dummy, return_tensors="pt").to(get_device())
            with torch.inference_mode():
                _ = model(**inputs)
        else:
            inputs = tokenizer("Hello", return_tensors="pt")
            input_ids = inputs["input_ids"].to(get_device())
            with torch.inference_mode():
                _ = model.generate(input_ids, max_new_tokens=1)
    except Exception as e:
        logging.warning(f"Warm-up failed for {model_name}: {e}")

def get_generator_status():
    """Get current device, torch version, and loaded models."""
    device = get_device()
    if device.startswith("cuda"):
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        device_info = f"cuda ({device_name})"
    elif device == "mps":
        device_info = "mps (Apple Metal)"
    else:
        device_info = "cpu"
    return {
        "device": device_info,
        "torch_version": torch.__version__,
        "cached_models": list(_MODEL_CACHE.keys())
    }

def run_with_timeout(func, timeout=10, *args, **kwargs):
    result, exc = None, None
    def wrapper():
        nonlocal result, exc
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exc = e
    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"Function exceeded timeout of {timeout}s")
    if exc:
        raise exc
    return result

def load_markov_chain(text_corpus: str):
    global _MARKOV_MODEL
    import markovify
    _MARKOV_MODEL = markovify.Text(text_corpus)
    return _MARKOV_MODEL

def sanitize_filename(filename: str) -> str:
    keepchars = (" ", ".", "_", "-")
    return "".join(c for c in filename if c.isalnum() or c in keepchars).rstrip()

def random_string(length: int = 8) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [json_safe(item) for item in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return json_safe(obj.to_dict(orient="records") if isinstance(obj, pd.DataFrame) else obj.tolist())
    elif isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (str, bool, int, float, type(None))):
        return obj
    else:
        return str(obj)

def safe_json_dumps(data: Any, indent: int = None) -> str:
    try:
        return json.dumps(json_safe(data), indent=indent, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to JSON serialize: {e}")
        return "{}"

def format_bytes(size: int) -> str:
    power = 2**10
    n = 0
    labels = ["B", "KB", "MB", "GB", "TB"]
    while size > power and n < len(labels) - 1:
        size /= power
        n += 1
    return f"{size:.2f} {labels[n]}"

def try_parse_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except:
        return fallback

def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logging.info(f"⏱️ {func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

def validate_model_for_columns(data_type: str, model_type: str, df: pd.DataFrame) -> str:
    model_type = model_type.lower()
    if df.shape[1] == 1:
        if data_type == "numeric" and model_type in {"ctgan", "tvae"}:
            logging.info(f"Model {model_type} is not valid for 1-column numeric data. Using GMM instead.")
            return "gmm"
        if data_type == "text" and model_type not in {"markov", "gpt-j", "flan-t5", "distil-cmlm", "llama-3.2", "deepseek", "openelm"}:
            logging.info(f"Text model {model_type} not supported. Defaulting to MARKOV.")
            return "markov"
    return model_type

def estimate_max_tokens_from_text(text: str, tokenizer=None, min_tokens=10, max_tokens=200, multiplier=1.1) -> int:
    """
    Estimate a suitable max_new_tokens value based on the original text length.
    Uses tokenizer if provided; otherwise falls back to word count estimation.
    """
    if tokenizer:
        try:
            token_count = len(tokenizer.tokenize(text))
        except Exception as e:
            logging.warning(f"Tokenizer failed on text '{text[:30]}...': {e}")
            token_count = len(text.split())
    else:
        token_count = len(text.split())

    estimated = int(token_count * multiplier)
    return min(max(estimated, min_tokens), max_tokens)
