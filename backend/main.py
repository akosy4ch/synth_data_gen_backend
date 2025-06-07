import os
import io
import uuid
import html
import math
import logging
import asyncio
from pathlib import Path
from typing import List
import logging

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from fastapi.staticfiles import StaticFiles

from minio import Minio
from minio.error import S3Error

from src.auth.dependencies import get_current_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from src.auth.database_models import User


from sqlalchemy.ext.asyncio import AsyncSession
from src.generator.generator import generate_synthetic

from src.data_processing.data_processing import load_dataset
from src.text_analysis.text_analysis import analyze_text_statistics, compare_datasets
from src.auth.routes import router as auth_router
from src.database.db import init_db

from fastapi import Depends
from pandas.api.types import is_string_dtype
from sqlalchemy.future import select
from src.database.models import Dataset, UploadedFile
from src.database.models import UploadedFile
from src.database.db import get_db
from src.storage.minio_utils import upload_fileobj
from src.database.utils import save_file_record_to_db

# ----------------------------------------------------------
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
# Configure logging: INFO level by default, with timestamps and level name
logging.basicConfig(
    filename=log_dir / "app.log",
    filemode="a",
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("synthetic")

app = FastAPI()

# Set up a FIFO queue for generation tasks and a background worker
app.generation_queue = asyncio.Queue()

# ----------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    await init_db()
    logging.info("Database tables ensured")
    await asyncio.to_thread(init_minio_bucket)
    logging.info("MinIO bucket ensured")
    asyncio.create_task(generation_worker())
    logger.info("Synthetic data generation worker started.")

def init_minio_bucket():
    client = get_minio_client()
    bucket = os.getenv("S3_BUCKET")
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logging.info(f"Created MinIO bucket: {bucket}")
        else:
            logging.info(f"Bucket already exists: {bucket}")
    except S3Error as err:
        if err.code == "BucketAlreadyOwnedByYou" or err.code == "BucketAlreadyExists":
            logging.info(f"Bucket {bucket} already present")
        else:
            logging.error(f"Error initializing bucket {bucket}: {err}")
            raise

async def generation_worker():
    """Background task that processes the generation queue continuously."""
    while True:
        task = await app.generation_queue.get()
        try:
            await task()
        except Exception as e:
            logger.error("Unhandled exception in generation worker: %s", e, exc_info=True)
        finally:
            app.generation_queue.task_done()

# ----------------------------------------------------------
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # update in production to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ----------------------------------------------------------
# Analysis: Column Analysis Endpoint
def read_upload(file: UploadFile) -> bytes:
    """
    Read entire upload into memory and return bytes.
    """
    return asyncio.get_event_loop().run_until_complete(file.read())



@app.post("/analyze-columns/")
async def analyze_columns(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    contents = await file.read()
    # Preserve original extension
    orig_ext = os.path.splitext(file.filename)[1].lower() or ".csv"
    object_name = f"uploads/original/{uuid.uuid4()}{orig_ext}"


    try:
        key = await upload_fileobj(contents, object_name)
        await save_file_record_to_db(
            filename=file.filename,
            s3_path=key,
            file_type="original",
            owner_id=current_user.id 
        )
        logging.info(f"Uploaded {file.filename} to {key}")
    except Exception as err:
        logging.error(f"Storage or DB error: {err}", exc_info=True)
        raise HTTPException(status_code=502, detail="Storage error")

    try:
        # Use load_dataset for robust file reading
        df, _ = load_dataset(io.BytesIO(contents))
        df.columns = df.columns.str.strip()

        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            sample_values = df[col].dropna().astype(str).unique().tolist()[:5]
            avg_len = df[col].dropna().astype(str).apply(len).mean() if dtype == 'object' else None

            column_info.append({
                "name": col,
                "dtype": dtype,
                "unique_values": unique_count,
                "sample": sample_values,
                "avg_length": avg_len
            })

        logging.info(f"Analyzed columns for file: {file.filename}")
        return {"columns": column_info, "rows": len(df)}

    except Exception as e:
        logging.error(f"Error analyzing columns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------------
@app.post("/generate-synthetic/")
async def create_synthetic_data(
    file: UploadFile = File(...),
    columns: List[str] = Form(...),
    models: List[str] = Form(...),
    samples: int = Form(...),
    preserve_other_columns: bool = Form(True),
    current_user: User = Depends(get_current_user)
):
    logger.info("Received request: columns=%s, models=%s, samples=%d, preserve_other_columns=%s",
                columns, models, samples, preserve_other_columns)

    contents = await file.read()
    orig_ext = os.path.splitext(file.filename)[1].lower() or ".csv"
    orig_filename = Path(file.filename).stem
    synthetic_filename = f"{orig_filename}_synthetic{orig_ext}"
    object_name = f"uploads/synthetic/{uuid.uuid4()}_{synthetic_filename}"

    loop = asyncio.get_event_loop()
    result_future = loop.create_future()

    try:
        key = await upload_fileobj(contents, object_name)
        await save_file_record_to_db(
            filename=synthetic_filename,
            s3_path=key,
            file_type="synthetic",
            owner_id=current_user.id
        )
        logger.info(f"Uploaded {synthetic_filename} to {key}")
    except Exception as err:
        logger.error(f"Storage or DB error: {err}", exc_info=True)
        raise HTTPException(status_code=502, detail="Storage error")

    try:
        df, _ = load_dataset(io.BytesIO(contents))
        columns = [html.unescape(c.strip()) for c in columns]
        for col in columns:
            if col not in df.columns:
                logger.error("Requested column '%s' not found in input data.", col)
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in input data.")
    except Exception as e:
        logger.error("Failed to read input file: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail="Could not read input file. Please provide a valid file.")

    # 🧠 Вложенная задача (замыкание)
    async def generation_task():
        try:
            synthetic_data = {}
            skipped_columns = []

            for col, model_name in zip(columns, models):
                logger.info(f"Generating synthetic data for column '{col}' using model {model_name}...")
                try:
                    gen_df = await generate_synthetic(
                        df[[col]], model_name, samples,
                        'text' if is_string_dtype(df[col]) else 'numeric', col
                    )
                    synthetic_col = gen_df[col].tolist() if col in gen_df else []
                except Exception as e:
                    logger.error("Generation failed for column '%s': %s", col, e)
                    synthetic_col = []

                if not synthetic_col or len(synthetic_col) < samples:
                    logger.warning("Model %s generated %d values for column '%s' (expected %d).",
                                model_name, len(synthetic_col), col, samples)
                    if synthetic_col:
                        while len(synthetic_col) < samples:
                            synthetic_col += synthetic_col
                    synthetic_col = synthetic_col[:samples] or [""] * samples

                if len(synthetic_col) == samples:
                    synthetic_data[col] = synthetic_col
                else:
                    skipped_columns.append(col)
                    logger.warning(f"⚠️ Skipping column {col} due to length mismatch.")

            if not synthetic_data:
                raise ValueError("No columns were successfully generated.")

            synthetic_df = pd.DataFrame(synthetic_data)

            if preserve_other_columns:
                output_df = df.copy()
                for col in synthetic_df.columns:
                    output_df[col] = synthetic_df[col]
            else:
                output_df = synthetic_df

            result_payload = {
                "synthetic": output_df.to_dict(orient="records"),
                "analysis": {},
            }

            if skipped_columns:
                result_payload["warning"] = f"Some columns were skipped: {skipped_columns}"

            logger.info("✅ Synthetic data generation successful. Generated %d rows for columns %s.",
                        len(output_df), list(synthetic_data.keys()))
            result_future.set_result(result_payload)

        except Exception as e:
            logger.error("❌ Generation task failed for columns %s: %s", columns, e, exc_info=True)
            result_future.set_result({
                "synthetic": [],
                "analysis": {},
                "warning": f"No columns were generated due to an error: {e}"
            })

    # ✅ Важно: ДОБАВИТЬ В ОЧЕРЕДЬ
    await app.generation_queue.put(generation_task)

    try:
        response_data = await asyncio.wait_for(result_future, timeout=60.0)
    except asyncio.TimeoutError:
        logger.error("Synthetic generation timed out after 60 seconds for columns %s.", columns)
        response_data = {
            "synthetic": [],
            "analysis": {},
            "warning": "Generation timed out. Please try again with smaller input or later."
        }

    return response_data

# ---------------------------------------------------------- 
@app.post("/detect-best-config/")
async def detect_best_config(file: UploadFile = File(...)):
    contents = await file.read()
    object_name = f"uploads/{uuid.uuid4()}.csv"
    client = get_minio_client()
    bucket = os.getenv("S3_BUCKET")

    try:
        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=io.BytesIO(contents),
            length=len(contents),
            content_type=file.content_type
        )
        logging.info(f"Uploaded {file.filename} to {bucket}/{object_name}")
    except S3Error as err:
        logging.error(f"MinIO upload error: {err}")
        raise HTTPException(status_code=502, detail="Storage error")

    try:
        df = pd.read_csv(io.BytesIO(contents), encoding_errors="ignore", on_bad_lines="skip")
        recommendations = []
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue

            col_info = {
                "column": col,
                "dtype": str(series.dtype),
                "unique_values": int(series.nunique()),
                "sample_values": series.astype(str).unique().tolist()[:5],
                "original_samples": int(len(series)),
                "default_sample_size": int(len(series)),
                "s3_path": f"{bucket}/{object_name}"
            }

            
            if series.dtype == object:
                avg_len = series.astype(str).apply(len).mean()
                # выбор моделей для текстовых данных
                if avg_len < 30 and col_info["unique_values"] < 100:
                    suggested_models = ["MARKOV", "FLAN-T5", "GPT-J"]
                elif avg_len < 100:
                    suggested_models = ["FLAN-T5", "GPT-J", "LLAMA-2-7B"]
                else:
                    suggested_models = ["GPT-J", "LLAMA-2-7B", "DEEPSEEK"]
                task = "text"
            elif pd.api.types.is_numeric_dtype(series):
                # выбор моделей для числовых данных
                suggested_models = ["CTGAN", "TVAE", "GMM"]
                task = "numeric"
            else:
                suggested_models = ["CTGAN", "TVAE"]
                task = "mixed"

            col_info["task"] = task
            col_info["suggested_models"] = suggested_models
            # опциональная модель по умолчанию
            col_info["recommended_model"] = suggested_models[0]

            recommendations.append(col_info)

        logging.info(f"Advanced auto-detected config for: {file.filename}")
        return {'recommendations': recommendations}
    except Exception as e:
        logging.error(f"Error in auto config detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------------

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from minio.error import S3Error
from src.text_analysis.text_analysis import analyze_text_statistics, compare_datasets
from src.storage.minio_utils import get_minio_client
import logging

router = APIRouter()

class TextEvaluationFromStorageRequest(BaseModel):
    original_s3_path: str
    synthetic_s3_path: str

@router.post("/evaluate-texts/")
async def evaluate_texts_from_storage(payload: TextEvaluationFromStorageRequest):
    try:
        client = get_minio_client()

        # Validate s3_path format
        if "/" not in payload.original_s3_path or "/" not in payload.synthetic_s3_path:
            raise HTTPException(status_code=400, detail="Invalid s3_path format. Expected 'bucket/object_name'.")

        orig_bucket, orig_path = payload.original_s3_path.split("/", 1)
        synth_bucket, synth_path = payload.synthetic_s3_path.split("/", 1)

        # Try to retrieve original file
        try:
            original_obj = client.get_object(orig_bucket, orig_path)
        except S3Error as e:
            logging.error(f"MinIO error for original file: {e.code} - {e.message}")
            raise HTTPException(status_code=502, detail=f"Could not access original file: {e.message}")

        # Try to retrieve synthetic file
        try:
            synthetic_obj = client.get_object(synth_bucket, synth_path)
        except S3Error as e:
            logging.error(f"MinIO error for synthetic file: {e.code} - {e.message}")
            raise HTTPException(status_code=502, detail=f"Could not access synthetic file: {e.message}")

        # Load CSV files
        df_original = pd.read_csv(original_obj, encoding_errors="ignore", on_bad_lines="skip")
        df_synthetic = pd.read_csv(synthetic_obj, encoding_errors="ignore", on_bad_lines="skip")

        # Select first available text column
        text_cols_orig = df_original.select_dtypes(include="object").columns
        text_cols_synth = df_synthetic.select_dtypes(include="object").columns

        if text_cols_orig.empty or text_cols_synth.empty:
            raise HTTPException(status_code=400, detail="No text column found in one of the files")

        text_col_orig = text_cols_orig[0]
        text_col_synth = text_cols_synth[0]

        orig_texts = df_original[text_col_orig].dropna().astype(str).tolist()
        synth_texts = df_synthetic[text_col_synth].dropna().astype(str).tolist()

        if not orig_texts or not synth_texts:
            raise HTTPException(status_code=400, detail="Text columns are empty")

        # Limit comparison to minimum shared length
        limit = min(len(orig_texts), len(synth_texts))
        orig_texts = orig_texts[:limit]
        synth_texts = synth_texts[:limit]

        stats_original = analyze_text_statistics(orig_texts)
        stats_synthetic = analyze_text_statistics(synth_texts)
        comparison = compare_datasets(orig_texts, synth_texts)

        return {
            "original_column": text_col_orig,
            "synthetic_column": text_col_synth,
            "original_stats": stats_original,
            "synthetic_stats": stats_synthetic,
            "comparison": comparison,
            "num_compared_samples": limit
        }

    except S3Error as e:
        logging.error(f"\u274c MinIO access error: {e.code} - {e.message}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"MinIO access error: {e.message}")

    except Exception as e:
        logging.error(f"\u274c Unexpected error in /evaluate-texts/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(auth_router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "This is the running backend."}

@app.get("/routes")
def list_routes():
    return [route.path for route in app.routes]

@app.get("/list-objects/")
async def list_minio_objects(prefix: str = "uploads/"):
    try:
        client = get_minio_client()
        bucket = os.getenv("S3_BUCKET")
        objects = client.list_objects(bucket, prefix=prefix, recursive=True)
        return {"objects": [f"{bucket}/{obj.object_name}" for obj in objects]}
    except Exception as e:
        logging.error(f"Failed to list MinIO objects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list MinIO files")

@app.get("/user-files/")
async def get_files(
    file_type: str = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = select(UploadedFile).where(UploadedFile.owner_id == current_user.id)
    if file_type == "original":
        query = query.where(UploadedFile.s3_path.contains("uploads/original/"))
    elif file_type == "synthetic":
        query = query.where(UploadedFile.s3_path.contains("uploads/synthetic/"))
    result = await db.execute(query)
    files = result.scalars().all()
    return [{"filename": f.filename, "s3_path": f.s3_path} for f in files]



# End of file    