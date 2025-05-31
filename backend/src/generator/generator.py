import pandas as pd
import logging
import asyncio
from src.generator.text_generator import generate_text
from src.generator.numeric_generator import generate_numeric

async def generate_synthetic(df, model_type, samples, task, column):
    try:
        print("🧪 generate_synthetic called")
        print(f"➡️  DataFrame head:\n{df.head(3)}")
        print(f"📦 Using model: {model_type} | task: {task} | column: {column}")

        # ✅ Safety checks
        if df.empty or column not in df.columns:
            raise ValueError(f"Invalid DataFrame or missing column: {column}")
        if samples <= 0:
            raise ValueError("Sample size must be a positive integer.")
        if task not in {"text", "numeric"}:
            raise ValueError(f"Unsupported task type: {task}")

        # ✅ Run generation in background thread if needed
        if task == "text":
            result_df = await generate_text_async(df, model_type, samples, column)
        elif task == "numeric":
            result_df = await generate_numeric(df, model_type, samples)



        # ✅ Standardize column name if needed
        if column not in result_df.columns:
            result_df.columns = [column]

        return result_df

    except Exception as e:
        logging.exception(f"❌ Generation failed for column '{column}' with model '{model_type}': {e}")
        raise
