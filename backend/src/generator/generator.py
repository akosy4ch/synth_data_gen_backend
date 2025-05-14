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

def postprocess_and_fill(df, column, samples):
    df[column] = df[column].fillna("").astype(str)
    df = df[df[column].str.strip() != ""]
    while len(df) < samples:
        df = pd.concat([df, df]).head(samples)
    return df.reset_index(drop=True)
