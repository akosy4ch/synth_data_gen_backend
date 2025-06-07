import pandas as pd
import markovify
import logging

_MARKOV_MODEL = None

def load_markov_model(df: pd.DataFrame, column: str, state_size: int = 2) -> None:
    """
    Loads a global Markov model from the given DataFrame and column.
    
    Args:
        df (pd.DataFrame): DataFrame containing textual data.
        column (str): Column name with text data.
        state_size (int): Order of the Markov chain (default=2).
    """
    global _MARKOV_MODEL

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    text_data = df[column].dropna().astype(str).tolist()
    full_text = "\n".join(text_data)

    try:
        _MARKOV_MODEL = markovify.Text(full_text, state_size=state_size)
        logging.info(f"✅ Markov model trained on column '{column}' with {len(text_data)} lines.")
    except Exception as e:
        logging.error(f"❌ Failed to build Markov model: {e}")
        raise

def generate_markov_text(prompt: str = "", num_samples: int = 1) -> list[str]:
    if _MARKOV_MODEL is None:
        raise RuntimeError("Markov model not initialized. Call load_markov_model first.")

    results = []
    for _ in range(num_samples):
        try:
            sentence = None
            if prompt and len(prompt.split()) <= 2:
                sentence = _MARKOV_MODEL.make_sentence_with_start(prompt, strict=False)

            if not sentence:
                sentence = _MARKOV_MODEL.make_sentence()

            results.append((sentence or prompt).strip())
        except Exception as e:
            logging.warning(f"⚠️ Failed to generate sentence from prompt '{prompt}': {e}")
            results.append(prompt or "")
    return results


def generate_markov_df(df: pd.DataFrame, column: str, num_samples: int = 10) -> pd.DataFrame:
    """
    Generate a DataFrame of synthetic text using Markov model.

    Args:
        df (pd.DataFrame): Original data to train the model.
        column (str): Column to train and generate from.
        num_samples (int): Number of synthetic rows.

    Returns:
        pd.DataFrame: New DataFrame with generated values.
    """
    load_markov_model(df, column)
    prompt_examples = df[column].dropna().astype(str).tolist()
    
    if not prompt_examples:
        return pd.DataFrame({column: [""] * num_samples})

    results = []
    i = 0
    while len(results) < num_samples:
        prompt = prompt_examples[i % len(prompt_examples)]
        result = generate_markov_text(prompt, 1)[0]
        results.append(result)
        i += 1

    return pd.DataFrame({column: results})
# Exportable function for external imports
def generate_text(
    prompt: str, 
    num_samples: int = 1, 
    model_name: str = None, 
    df: pd.DataFrame = None, 
    column: str = None, 
    **kwargs
) -> list[str]:
    global _MARKOV_MODEL
    if _MARKOV_MODEL is None:
        if df is None or column is None:
            raise ValueError("Markov model not initialized and no df/column provided to initialize.")
        load_markov_model(df, column)

    return generate_markov_text(prompt, num_samples)

