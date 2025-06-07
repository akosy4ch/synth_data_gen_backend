import torch
import logging
import pandas as pd
import threading
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils.generator_utils import estimate_max_tokens_from_text

_MODEL_CACHE = {}

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_distilbert():
    if "distil-cmlm" in _MODEL_CACHE:
        return _MODEL_CACHE["distil-cmlm"]

    model_id = "distilbert-base-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.to(get_device())
    model.eval()

    _MODEL_CACHE["distil-cmlm"] = (model, tokenizer)
    logging.info(f"✅ Loaded DistilBERT (C-MLM) on {get_device()}")
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
        raise TimeoutError("⚠️ C-MLM generation timed out")
    if exc:
        raise exc
    return result

def generate_with_masked_lm(prompt: str, max_tokens=20) -> str:
    model, tokenizer = load_distilbert()
    device = get_device()
    mask_token = tokenizer.mask_token

    text = prompt.strip()
    for _ in range(max_tokens):
        if not text.endswith(" "):
            text += " "
        text += mask_token

        inputs = tokenizer(text, return_tensors="pt").to(device)
        mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            break

        mask_index = mask_positions[1].item()
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits[0, mask_index]
        predicted_token_id = int(torch.argmax(logits))

        if predicted_token_id in {tokenizer.pad_token_id, tokenizer.eos_token_id}:
            break

        new_token = tokenizer.decode([predicted_token_id])
        text = text.replace(mask_token, new_token, 1)

    return text.strip()

def generate_text(prompt: str, num_samples: int = 1, model_name: str = None, max_tokens: int = None, **kwargs) -> list[str]:
    model, tokenizer = load_distilbert()
    
    # Dynamically estimate max_tokens if not provided
    if max_tokens is None:
        max_tokens = estimate_max_tokens_from_text(prompt, tokenizer)

    def run():
        return [generate_with_masked_lm(prompt, max_tokens) for _ in range(num_samples)]

    return _generate_with_timeout(run, timeout=kwargs.get("timeout", 30))

def generate_cmlm_df(df: pd.DataFrame, column: str, samples=1, max_tokens=None, timeout=30):
    results = []
    for prompt in df[column].dropna().astype(str).tolist():
        try:
            outputs = generate_text(prompt, num_samples=samples, max_tokens=max_tokens, timeout=timeout)
            results.extend(outputs)
        except Exception as e:
            logging.warning(f"Distil-CMLM failed on prompt '{prompt[:30]}...': {e}")
            results.extend([""] * samples)

    results = results[:samples]
    return pd.DataFrame({column: results})
