import pandas as pd

def validate_model_for_columns(data_type: str, model_type: str, df: pd.DataFrame) -> str:
    if df.shape[1] == 1:
        if data_type == "numeric" and model_type in {"CTGAN", "TVAE"}:
            return "GMM"
        if data_type == "text" and model_type not in {"MARKOV", "GPT-J", "FLAN-T5", "DISTIL-CMLM", "LLAMA-2", "DEEPSEEK"}:
            return "MARKOV"
    return model_type
