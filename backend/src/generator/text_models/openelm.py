import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Глобальный кэш для хранения загруженных моделей
_MODEL_CACHE = {}

def get_device():
    """Определяет доступное устройство: MPS (Apple), CUDA или CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_openelm_model(model_id: str = "apple/OpenELM-3B"):
    """
    Загружает модель OpenELM и токенизатор с кэшированием.

    Args:
        model_id (str): Идентификатор модели на HuggingFace.

    Returns:
        Tuple[model, tokenizer]: Загруженные модель и токенизатор.
    """
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    logging.info(f"⏳ Loading OpenELM model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(get_device()).eval()

    _MODEL_CACHE[model_id] = (model, tokenizer)
    logging.info(f"✅ OpenELM model loaded on device: {get_device()}")

    return model, tokenizer

def generate_openelm_text(prompt: str, num_samples: int = 1, max_new_tokens: int = 50, model_id: str = "apple/OpenELM-3B") -> list[str]:
    """
    Генерирует текст с помощью модели OpenELM.

    Args:
        prompt (str): Исходный текст.
        num_samples (int): Количество сэмплов.
        max_new_tokens (int): Макс. число токенов в генерации.
        model_id (str): Идентификатор модели.

    Returns:
        List[str]: Сгенерированные строки.
    """
    model, tokenizer = load_openelm_model(model_id)
    inputs = tokenizer(prompt, return_tensors="pt").to(get_device())

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_samples,
        "do_sample": num_samples > 1,
        "temperature": 0.7 if num_samples > 1 else None
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})

    outputs = []
    for seq in output_ids:
        full_text = tokenizer.decode(seq, skip_special_tokens=True)
        prompt_len = inputs["input_ids"].shape[1]
        continuation_ids = seq[prompt_len:]
        generated_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
        outputs.append(generated_text or full_text)

    return outputs

def generate_openelm_df(df: pd.DataFrame, column: str, num_samples: int = 10, max_new_tokens: int = 50, model_id: str = "apple/OpenELM-3B") -> pd.DataFrame:
    """
    Генерирует DataFrame с синтетическим текстом, используя OpenELM.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        column (str): Название колонки с промптами.
        num_samples (int): Кол-во сгенерированных строк.
        max_new_tokens (int): Макс. токенов на ответ.
        model_id (str): Идентификатор модели.

    Returns:
        pd.DataFrame: Таблица с сгенерированными текстами.
    """
    prompts = df[column].dropna().astype(str).tolist()
    if not prompts:
        return pd.DataFrame({column: [""] * num_samples})

    results = []
    i = 0
    while len(results) < num_samples:
        prompt = prompts[i % len(prompts)]
        try:
            output = generate_openelm_text(prompt, 1, max_new_tokens, model_id)[0]
            results.append(output)
        except Exception as e:
            logging.warning(f"⚠️ OpenELM generation failed for prompt: {prompt[:40]} | {e}")
            results.append("")
        i += 1

    return pd.DataFrame({column: results[:num_samples]})

def generate_text(prompt: str, num_samples: int = 1, model_name: str = "apple/OpenELM-3B", max_new_tokens: int = 50) -> list[str]:
    """
    Универсальная функция генерации текста.

    Args:
        prompt (str): Стартовый текст.
        num_samples (int): Количество вариаций.
        model_name (str): Название модели.
        max_new_tokens (int): Макс. токенов.

    Returns:
        list[str]: Результаты генерации.
    """
    return generate_openelm_text(
        prompt=prompt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        model_id=model_name
    )
