import pandas as pd
import logging

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
