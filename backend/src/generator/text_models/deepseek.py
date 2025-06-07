import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import threading
from utils.generator_utils import estimate_max_tokens_from_text

_MODEL_CACHE = {}

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_deepseek():
    if "deepseek" in _MODEL_CACHE:
        return _MODEL_CACHE["deepseek"]

    model_id = "deepseek-ai/DeepSeek-V2"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model.to(get_device())
    model.eval()

    _MODEL_CACHE["deepseek"] = (model, tokenizer)
    logging.info(f"✅ DeepSeek model loaded on {get_device()}")
    return model, tokenizer

def _generate_with_timeout(fn, args=(), kwargs=None, timeout=30):
    result = None
    exc = None
    if kwargs is None:
        kwargs = {}

    def wrapper():
        nonlocal result, exc
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            exc = e

    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError("⚠️ DeepSeek generation timed out")
    if exc:
        raise exc
    return result

def generate_text(prompt: str, num_return_sequences=1, max_tokens=None, temperature=0.7, timeout=30):
    model, tokenizer = load_deepseek()
    device = get_device()

    # 🔄 Dynamically estimate token count if not provided
    if max_tokens is None:
        max_tokens = estimate_max_tokens_from_text(prompt, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_args = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "num_return_sequences": num_return_sequences,
        "do_sample": True,
    }

    def generate():
        with torch.inference_mode():
            output = model.generate(**inputs, **gen_args)
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        trimmed = [s[len(prompt):].strip() for s in decoded]
        return trimmed

    return _generate_with_timeout(generate, timeout=timeout)

def generate_deepseek_df(df: pd.DataFrame, column: str, samples=1, max_tokens=None, timeout=30):
    results = []
    for prompt in df[column].dropna().astype(str).tolist():
        try:
            outputs = generate_text(prompt, num_return_sequences=samples, max_tokens=max_tokens, timeout=timeout)
            results.extend(outputs)
        except Exception as e:
            logging.warning(f"DeepSeek failed on prompt '{prompt[:30]}...': {e}")
            results.extend([""] * samples)

    results = results[:samples]
    return pd.DataFrame({column: results})
