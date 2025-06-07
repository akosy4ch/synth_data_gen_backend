import requests
import pandas as pd
import logging
import threading

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "llama3"

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
        raise TimeoutError(f"⚠️ Ollama generation timed out after {timeout}s")
    if exc:
        raise exc
    return result

def _generate_with_ollama(prompt, model, temperature, max_tokens):
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        logging.warning(f"[Ollama] Failed to generate: {e}")
        return ""

def generate_text(prompt: str, model_name: str = None, num_samples: int = 1, **kwargs) -> pd.DataFrame:
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_new_tokens", 100)
    timeout = kwargs.get("timeout", 120)

    results = []
    for _ in range(num_samples):
        try:
            result = _run_with_timeout(
                _generate_with_ollama,
                args=(prompt,),
                kwargs={
                    "model": OLLAMA_MODEL_NAME,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=timeout
            )
            results.append(result)
        except Exception as e:
            logging.warning(f"[Ollama] Timeout/Error: {e}")
            results.append("")

    return pd.DataFrame({"generated": results})
