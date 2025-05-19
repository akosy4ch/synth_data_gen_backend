import pandas as pd
import logging

from src.generator.text_generator import generate_text
from src.generator.numeric_generator import generate_numeric

async def generate_synthetic(df, model_type, samples, task, column):
    logging.info("🧪 generate_synthetic called")
    logging.info("➡️  DataFrame head:\n", df.head(3))
    logging.info("📦 Using model:", model_type, "| task:", task, "| column:", column)
    if task == "text":
        return await generate_text(df, model_type, samples, column)
    elif task == "numeric":
        return await generate_numeric(df, model_type, samples)
    else:
        raise ValueError(f"Unsupported task type: {task}")
