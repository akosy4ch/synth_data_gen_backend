import pandas as pd
import asyncio
import logging
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from src.generator.utils import validate_model_for_columns
import logging

logger = logging.getLogger(__name__)

# ===============================
#        NUMERIC DATA SYNTH
# ===============================

async def generate_numeric(df: pd.DataFrame, model_type: str, samples: int) -> pd.DataFrame:
    """
    Main async interface to generate synthetic numeric data using SDV models.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    return await asyncio.to_thread(_generate_numeric_sync, df, model_type, samples)


def _generate_numeric_sync(df: pd.DataFrame, model_type: str, samples: int) -> pd.DataFrame:
    logger.info(f"Generating synthetic numeric data using model: {model_type} | Rows: {len(df)}")

    if df.isnull().any().any():
        logger.warning("Input contains NaNs. Filling with median.")
        df = df.fillna(df.median(numeric_only=True))

    model_type = validate_model_for_columns("numeric", model_type, df).upper()
    logger.info(f"Validated model: {model_type}")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    try:
        model = _select_numeric_model(model_type, metadata)
        model.fit(df)
        logger.info("Model training completed. Sampling synthetic data...")
        result = model.sample(samples)

        if result.empty:
            logger.warning("⚠️ Generated DataFrame is empty.")
            return pd.DataFrame({df.columns[0]: [""] * samples})
        return result

    except Exception as e:
        logger.error(f"🚨 Exception in numeric model ({model_type}): {e}", exc_info=True)
        return pd.DataFrame({df.columns[0]: [""] * samples})


def _select_numeric_model(model_type: str, metadata: SingleTableMetadata):
    if model_type == "CTGAN":
        return CTGANSynthesizer(metadata, epochs=20)
    elif model_type == "TVAE":
        return TVAESynthesizer(metadata, epochs=20)
    elif model_type == "GMM":
        return GaussianCopulaSynthesizer(metadata)
    else:
        raise ValueError(f"Unsupported numeric model type: {model_type}")
