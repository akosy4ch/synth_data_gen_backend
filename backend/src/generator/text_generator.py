import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
import pandas as pd
import threading, logging, asyncio

# Global cache and model holders
_MODEL_CACHE = {}
_MARKOV_MODEL = None

torch.set_num_threads(torch.get_num_threads())

def get_device():
    """Return 'mps' (Apple Silicon), 'cuda', or 'cpu'."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def move_to_device(model):
    """Move model to the appropriate device."""
    return model.to(get_device())

def load_markov_chain(text_corpus: str):
    """
    Initialize the global Markov model using the provided text corpus.
    Requires markovify library.
    """
    global _MARKOV_MODEL
    try:
        import markovify
    except ImportError:
        raise ImportError("markovify library is required for Markov model. Install it via pip.")
    _MARKOV_MODEL = markovify.Text(text_corpus)
    return _MARKOV_MODEL

def _warmup_model(model_name, model, tokenizer):
    """Internal: warm up the model to reduce first-time latency."""
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

def _get_model_and_tokenizer(model_name: str):
    """Internal: load model & tokenizer or fetch from cache."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    mn = model_name.lower()
    if mn == "flan-t5":
        mid = "google/flan-t5-base"
        model = AutoModelForSeq2SeqLM.from_pretrained(mid)
        tokenizer = AutoTokenizer.from_pretrained(mid)
    elif mn == "gpt-j":
        mid = "EleutherAI/gpt-j-6B"
        model = AutoModelForCausalLM.from_pretrained(mid)
        tokenizer = AutoTokenizer.from_pretrained(mid)
    elif mn == "llama-3.2":
        mid = "meta-llama/Llama-3-2-1b"
        model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    elif mn == "deepseek":
        mid = "deepseek-ai/DeepSeek-V2"
        model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    elif mn == "openelm":
        mid = "apple/OpenELM-3B"
        model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    elif mn == "distil-cmlm":
        mid = "distilbert-base-uncased"
        model = AutoModelForMaskedLM.from_pretrained(mid)
        tokenizer = AutoTokenizer.from_pretrained(mid)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    model = move_to_device(model)
    model.eval()
    device = get_device()
    logging.info(f"✅ Loading model '{model_name}' on device: {device}")
    _MODEL_CACHE[model_name] = (model, tokenizer)
    _warmup_model(model_name, model, tokenizer)
    return (model, tokenizer)

def _generate_with_timeout(func, args=(), kwargs=None, timeout=None):
    """Internal: run func(*args, **kwargs) with a time limit."""
    if kwargs is None:
        kwargs = {}
    result = None
    exc = None
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
        # If still running after timeout, we consider it a failure
        raise TimeoutError(f"Generation exceeded {timeout} seconds")
    if exc:
        raise exc
    return result

def _generate_with_masked_lm(model, tokenizer, prompt: str, max_tokens: int):
    """Internal: generate text iteratively using a masked LM (BERT-like model)."""
    text = prompt
    for _ in range(max_tokens):
        mask_token = tokenizer.mask_token
        if mask_token is None:
            break
        # Ensure proper spacing
        if text and not text.endswith(" "):
            text += " "
        masked_input = text + mask_token
        inputs = tokenizer(masked_input, return_tensors="pt").to(get_device())
        # Find mask position
        mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            break
        mask_idx = mask_positions[1].item()
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits[0, mask_idx]
        new_token_id = int(torch.argmax(logits))
        # Stop if end-of-sequence
        if tokenizer.eos_token_id and new_token_id == tokenizer.eos_token_id:
            break
        if tokenizer.pad_token_id and new_token_id == tokenizer.pad_token_id:
            break
        new_token = tokenizer.decode([new_token_id])
        text += new_token
    return text

def _generate_from_model(model_name: str, prompt: str, max_new_tokens: int = 50, num_return_sequences: int = 1):
    """Internal: generate text from prompt using specified model (handles all model types)."""

    mn = model_name.lower()
    if mn == "markov":
        if _MARKOV_MODEL is None:
            raise ValueError("Markov model is not loaded. Call load_markov_chain first.")
        try:
            sentence = _MARKOV_MODEL.make_sentence_with_start(prompt)
        except Exception:
            sentence = _MARKOV_MODEL.make_sentence()
        if sentence is None:
            sentence = prompt or ""
        return [sentence]

    # Use transformers model for other types
    model, tokenizer = _get_model_and_tokenizer(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(get_device())

    if mn == "distil-cmlm":
        text = _generate_with_masked_lm(model, tokenizer, prompt, max_new_tokens)
        return [text]

    gen_kwargs = {"max_new_tokens": max_new_tokens, "num_return_sequences": num_return_sequences}
    if num_return_sequences > 1:
        gen_kwargs.update({"do_sample": True, "temperature": 0.8})

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)

    outputs = []
    for seq in output_ids:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        if mn in {"gpt-j", "llama-3.2", "deepseek", "openelm"}:
            prompt_len = input_ids.shape[1]
            continuation_ids = seq[prompt_len:]
            text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
        logging.debug(f"🔹 Decoded output: {text!r}")
        outputs.append(text)

    # ✅ Now check for empty outputs AFTER outputs is defined
    if not outputs or all(not str(o).strip() for o in outputs):
        logging.warning(f"🛑 Model {model_name} produced only empty outputs for prompt: {prompt[:50]!r}")

    return outputs

def generate_text(prompt: str, model_name: str, num_samples: int = 1, max_new_tokens: int = 50, timeout: float = None):
    """
    Generate text from prompt using the specified model. Returns a DataFrame with a 'generated' column.
    """
    results = []
    for i in range(num_samples):
        try:
            if timeout is not None:
                out_list = _generate_with_timeout(_generate_from_model, args=(model_name, prompt, max_new_tokens, 1), timeout=timeout)
            else:
                out_list = _generate_from_model(model_name, prompt, max_new_tokens, 1)
            # _generate_from_model returns a list of outputs; take the first element
            generated_text = out_list[0] if isinstance(out_list, list) else out_list
            
            if generated_text is None:
                logging.warning(f"⚠️ Got None output for sample {i+1} (model={model_name})")
                continue
            logging.debug(f"[{model_name}] Generated raw: {generated_text!r}")
            results.append(generated_text.strip())  # keep empty strings for debugging

        except Exception as e:
            logging.error(f"Error in generation (model={model_name}): {e}")
            # Attempt fallback to Markov chain if available and if not already using it
            if model_name.lower() != "markov" and _MARKOV_MODEL is not None:
                try:
                    fb_text = _MARKOV_MODEL.make_sentence() or (prompt or "")
                    logging.warning(f"Falling back to Markov model for {model_name} failure.")
                    results.append(fb_text)
                    continue
                except Exception as fe:
                    logging.error(f"Markov fallback failed: {fe}")
            # If no fallback could be used, append None
            results.append(None)
    # Pad results to ensure length
    if len(results) < num_samples:
        results.extend([""] * (num_samples - len(results)))
    return pd.DataFrame({"generated": results})



async def generate_text_async(df: pd.DataFrame, model_name: str, num_samples: int, column: str, max_new_tokens: int = 50, timeout: float = None) -> pd.DataFrame:
    loop = asyncio.get_running_loop()
    prompt_texts = df[column].dropna().astype(str).tolist()

    def batch_generate():
        results = []
        for prompt in prompt_texts:
            try:
                out_df = generate_text(prompt, model_name, num_samples, max_new_tokens, timeout)
                for text in out_df.get("generated", []):
                    if text is not None and str(text).strip():
                        results.append(text.strip() if text is not None else "")
            except Exception as e:
                logging.warning(f"Generation error for prompt: {prompt[:50]} | {e}")
        
        # Ensure minimum size
        while len(results) < num_samples:
            results += results  # Duplicate to pad
        results = results[:num_samples]  # Trim

        return pd.DataFrame({column: results})

    result_df = await loop.run_in_executor(None, batch_generate)

    if column not in result_df.columns:
        result_df.columns = [column]

    return result_df

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
