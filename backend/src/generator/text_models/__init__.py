from .flan_t5 import generate_text as generate_flan
from .gpt_j import generate_text as generate_gptj
from .llama import generate_text as generate_llama
from .deepseek import generate_text as generate_deepseek
from .openelm import generate_text as generate_openelm
from .distil_cmlm import generate_text as generate_distil_cmlm
from .markov_gen import generate_text as generate_markov

MODEL_DISPATCH = {
    "flan-t5": generate_flan,
    "gpt-j": generate_gptj,
    "llama-3.2": generate_llama,
    "deepseek": generate_deepseek,
    "openelm": generate_openelm,
    "distil-cmlm": generate_distil_cmlm,
    "markov": generate_markov,
}

def generate_text(model_name, prompt, num_samples=1, **kwargs):
    model_name = model_name.lower()
    if model_name not in MODEL_DISPATCH:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_DISPATCH[model_name](prompt, num_samples, **kwargs)
