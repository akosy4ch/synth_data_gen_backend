import pandas as pd
import asyncio
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# ===============================
#          ЧИСЛОВЫЕ ДАННЫЕ
# ===============================

async def generate_numeric(df, model_type, samples):
    return await asyncio.to_thread(_generate_numeric_sync, df, model_type, samples)

def _generate_numeric_sync(df, model_type, samples):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    if model_type == "CTGAN":
        model = CTGANSynthesizer(metadata, epochs=20)
    elif model_type == "TVAE":
        model = TVAESynthesizer(metadata, epochs=20)
    elif model_type == "GMM":
        model = GaussianCopulaSynthesizer(metadata)
    else:
        raise ValueError(f"Unsupported numeric model type: {model_type}")

    model.fit(df)
    return model.sample(samples)
