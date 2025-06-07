import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import threading

class FlanT5Generator:

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
            raise TimeoutError(f"⚠️ Generation timed out after {timeout}s")
        if exc:
            raise exc
        return result

    def __init__(self, model_id="google/flan-t5-base", device=None):
        self.model_id = model_id
        self.device = device or self._get_device()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _get_device(self):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _load_model(self):
        logging.info(f"Loading FLAN-T5 model: {self.model_id} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        logging.info("Model loaded and ready.")

    def generate_batch(self, prompts: list[str], max_tokens: int = 50, temperature: float = 1.0):
        if not prompts:
            raise ValueError("Prompts list is empty.")

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        gen_args = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
        }

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_args)

        return [self.tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]


def generate_text(prompt: str, num_samples: int = 1, model_name: str = None, **kwargs) -> list[str]:
    return _run_with_timeout(
        _generator_instance.generate_batch,
        kwargs={
            "prompts": [prompt] * num_samples,
            "max_tokens": kwargs.get("max_tokens", 50),
            "temperature": kwargs.get("temperature", 1.0),
        },
        timeout=kwargs.get("timeout", 60)
    )
