import pandas as pd

from src.generator.text_generator import generate_text
from src.generator.numeric_gemerator import generate_numeric

async def generate_synthetic(df, model_type, samples, task, column):
    print("🧪 generate_synthetic called")
    print("➡️  DataFrame head:\n", df.head(3))
    print("📦 Using model:", model_type, "| task:", task, "| column:", column)
    if task == "text":
        return await generate_text(df, model_type, samples, column)
    else:
        return await generate_numeric(df, model_type, samples)
