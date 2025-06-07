import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import threading
import logging

_MODEL_CACHE = {}
MODEL_NAME = "EleutherAI/gpt-j-6B"

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_gpt_j():
    if "gpt-j" in _MODEL_CACHE:
        return _MODEL_CACHE["gpt-j"]

    logging.info(f"📦 Loading GPT-J model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = model.to(get_device())
    model.eval()

    _MODEL_CACHE["gpt-j"] = (model, tokenizer)
    _warmup(model, tokenizer)
    return model, tokenizer

def _warmup(model, tokenizer):
    try:
        inputs = tokenizer("Hello", return_tensors="pt").to(get_device())
        with torch.inference_mode():
            _ = model.generate(inputs["input_ids"], max_new_tokens=1)
        logging.info("🔥 Warm-up complete for GPT-J.")
    except Exception as e:
        logging.warning(f"⚠️ Warm-up failed for GPT-J: {e}")

def _generate(prompt, max_new_tokens=50, num_return_sequences=1, temperature=0.8):
    model, tokenizer = load_gpt_j()

    inputs = tokenizer(prompt, return_tensors="pt").to(get_device())
    input_ids = inputs["input_ids"]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "do_sample": num_return_sequences > 1,
        "temperature": temperature
    }

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)

    outputs = []
    for seq in output_ids:
        full_text = tokenizer.decode(seq, skip_special_tokens=True)
        continuation = tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True).strip()
        outputs.append(continuation or full_text)
    return outputs

def _run_with_timeout(func, args=(), kwargs=None, timeout=120):
    result = None
    exc = None
    if kwargs is None:
        kwargs = {}

    def wrapper():
        nonlocal result, exc
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exc = e

    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"⚠️ GPT-J generation timed out after {timeout}s")
    if exc:
        raise exc
    return result

def generate_text(prompt: str, num_samples: int = 1, model_name: str = None,
                  max_new_tokens: int = 50, timeout: int = 120, **kwargs) -> list[str]:
    return _run_with_timeout(
        _generate,
        args=(prompt,),
        kwargs={
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_samples,
            "temperature": kwargs.get("temperature", 0.8)
        },
        timeout=timeout
    )

def generate_gptj_df(df: pd.DataFrame, column: str, samples=1, max_new_tokens=50, timeout=120):
    results = []
    prompts = df[column].dropna().astype(str).tolist()

    for prompt in prompts:
        try:
            outputs = generate_text(prompt, num_samples=samples,
                                    max_new_tokens=max_new_tokens, timeout=timeout)
            results.extend(outputs)
        except Exception as e:
            logging.warning(f"⚠️ GPT-J failed on prompt '{prompt[:30]}...': {e}")
            results.extend([""] * samples)

    # Ensure length matches requested size
    while len(results) < samples:
        results += results
    results = results[:samples]

    return pd.DataFrame({column: results})
