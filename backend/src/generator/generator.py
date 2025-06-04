import pandas as pd
import logging

logger = logging.getLogger(__name__)

def generate_synthetic(df_col: pd.DataFrame, model_type: str, gsize: int, task: str, col: str) -> pd.DataFrame:
    """Generate synthetic data for a single column using the specified model."""
    # 1. Validate input DataFrame and column
    if df_col is None or col not in df_col.columns:
        logger.error("Column '%s' not found in the input DataFrame.", col)
        return pd.DataFrame(columns=[col])
    if df_col.empty:
        logger.error("Input DataFrame for column '%s' is empty; cannot generate data.", col)
        return pd.DataFrame(columns=[col])
    
    try:
        # 2. Dispatch to appropriate generation function based on task
        if task == "text":
            logger.info("Generating synthetic text data for column '%s' using model '%s'...", col, model_type)
            result = generate_text(df_col, model_type, gsize, col)
        elif task == "numeric":
            logger.info("Generating synthetic numeric data for column '%s' using model '%s'...", col, model_type)
            result = generate_numeric(df_col, model_type, gsize)
        else:
            logger.error("Unknown task '%s' for column '%s'. Expected 'text' or 'numeric'.", task, col)
            return pd.DataFrame(columns=[col])
    except Exception as e:
        # Log the exception with traceback and return empty result
        logger.exception("Error generating synthetic data for column '%s' with model '%s': %s", col, model_type, e)
        return pd.DataFrame(columns=[col])
    
    # 3. Standardize and validate the output DataFrame format
    if result is None:
        logger.error("Generation function returned None for column '%s'.", col)
        result_df = pd.DataFrame(columns=[col])
    elif isinstance(result, pd.DataFrame):
        result_df = result.copy()
    else:
        # If result is a Series, list, or array, convert to DataFrame
        result_df = pd.DataFrame(result)
    
    # If the DataFrame has no column name or a wrong name, set the correct one
    if result_df.shape[1] != 1 or result_df.columns[0] != col:
        # Rename the first column to the expected column name
        result_df = result_df.rename(columns={result_df.columns[0]: col})
        # If multiple columns exist (unexpected), reduce to one
        result_df = result_df[[col]]
    
    # 4. Optionally, ensure the number of samples matches gsize
    if len(result_df) != gsize:
        logger.warning("Generated %d samples for column '%s' (expected %d).", len(result_df), col, gsize)
    
    logger.info("Successfully generated synthetic data for '%s' (%d rows).", col, len(result_df))
    return result_df
