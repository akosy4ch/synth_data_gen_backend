import pandas as pd
import asyncio
import logging
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from src.generator.utils import validate_model_for_columns

logger = logging.getLogger(__name__)

# ===============================
#        ЧИСЛОВЫЕ ДАННЫЕ
# ===============================

async def generate_numeric(df: pd.DataFrame, model_type: str, samples: int) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    return await asyncio.to_thread(_generate_numeric_sync, df, model_type, samples)


def _generate_numeric_sync(df: pd.DataFrame, model_type: str, samples: int) -> pd.DataFrame:
    logger.info(f"Generating synthetic numeric data using model: {model_type} | Rows: {len(df)}")
    model_type = validate_model_for_columns("numeric", model_type, df)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    model = _select_numeric_model(model_type, metadata)
    model.fit(df)

    logger.info("Model training completed. Sampling synthetic data...")
    return model.sample(samples)


def _select_numeric_model(model_type: str, metadata: SingleTableMetadata):
    """
    Selects and returns the appropriate numeric model.
    """
    if model_type == "CTGAN":
        return CTGANSynthesizer(metadata, epochs=20)
    elif model_type == "TVAE":
        return TVAESynthesizer(metadata, epochs=20)
    elif model_type == "GMM":
        return GaussianCopulaSynthesizer(metadata)
    else:
        raise ValueError(f"Unsupported numeric model type: {model_type}")
