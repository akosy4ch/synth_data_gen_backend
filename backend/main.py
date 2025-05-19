import os
import io
import uuid
import html
import math
import logging
import asyncio
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from fastapi.staticfiles import StaticFiles

from minio import Minio
from minio.error import S3Error

from src.auth.dependencies import get_current_user
from src.auth.database_models import User


from sqlalchemy.ext.asyncio import AsyncSession
from src.generator import generate_synthetic

from src.data_processing.data_processing import load_dataset
from src.text_analysis.text_analysis import analyze_text_statistics, compare_datasets
from src.auth.routes import router as auth_router
from src.database.db import init_db

from fastapi import Depends
from sqlalchemy.future import select
from src.database.models import Dataset, UploadedFile
from src.database.models import UploadedFile
from src.database.db import get_db
from src.storage.minio_utils import upload_fileobj
from src.database.utils import save_file_record_to_db
# ----------------------------------------------------------
# logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ----------------------------------------------------------
app = FastAPI()

# ----------------------------------------------------------
def get_minio_client() -> Minio:
    return Minio(
        endpoint=os.getenv("S3_ENDPOINT"),
        access_key=os.getenv("S3_ACCESS_KEY"),  # Ensure these match MinIO root credentials
        secret_key=os.getenv("S3_SECRET_KEY"),  # Ensure these match MinIO root credentials
        secure=False
    )

# ----------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    await init_db()
    logging.info("Database tables ensured")
    await asyncio.to_thread(init_minio_bucket)
    logging.info("MinIO bucket ensured")
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

# ----------------------------------------------------------
app.include_router(auth_router)

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
async def analyze_columns(file: UploadFile = File(...),
current_user: User = Depends(get_current_user)):
    contents = await file.read()
    object_name = f"uploads/{uuid.uuid4()}.csv"

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
        df = pd.read_csv(io.BytesIO(contents), encoding_errors="ignore", on_bad_lines="skip")
        df.columns = df.columns.str.strip()  # удаление пробелов в названиях колонок

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
    contents = await file.read()
    object_name = f"uploads/{uuid.uuid4()}.csv"

    try:
        key = await upload_fileobj(contents, object_name)
        await save_file_record_to_db(
            filename=file.filename,
            s3_path=key,
            file_type="synthetic",
            owner_id=current_user.id
        )
        logging.info(f"Uploaded {file.filename} to {key}")
    except Exception as err:
        logging.error(f"Storage or DB error: {err}", exc_info=True)
        raise HTTPException(status_code=502, detail="Storage error")

    try:
        df_original, _ = load_dataset(io.BytesIO(contents))
        columns = [html.unescape(c.strip()) for c in columns]
        missing = [c for c in columns if c not in df_original.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")

        df_result = pd.DataFrame()
        analysis_results = {}
        min_len = min(len(df_original[c].dropna()) for c in columns)
        gen_size = min(samples, min_len)
        queue = asyncio.Queue()
        for col, model_type in zip(columns, models):
            task = 'text' if df_original[col].dtype == object else 'numeric'
            await queue.put((df_original[[col]], model_type, gen_size, task, col))

        async def worker():
            while not queue.empty():
                df_col, mtype, gsize, task, col = await queue.get()
                synth_df = await generate_synthetic(df_col, mtype, gsize, task, col)
                synth_df[col] = synth_df[col].fillna('').astype(str)
                synth_df = synth_df[synth_df[col].str.strip() != '']
                while len(synth_df) < gsize:
                    synth_df = pd.concat([synth_df, synth_df]).head(gsize)
                df_result[col] = synth_df[col].reset_index(drop=True)

                if task == 'text':
                    orig_texts = df_col[col].dropna().astype(str).tolist()
                    synth_texts = synth_df[col].dropna().astype(str).tolist()
                    analysis_results[col] = {
                        'stats_original': analyze_text_statistics(orig_texts),
                        'stats_synthetic': analyze_text_statistics(synth_texts),
                        'comparison': compare_datasets(orig_texts, synth_texts)
                    }
                queue.task_done()

        await asyncio.gather(*[worker() for _ in range(3)])
        if preserve_other_columns:
            for col in columns:
                df_original[col] = df_result[col]
            output_df = df_original
        else:
            output_df = df_result

        return {
            'synthetic': output_df.to_dict(orient='records'),
            'analysis': analysis_results
        }

    except Exception as e:
        logging.error(f"Error generating synthetic data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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

from pydantic import BaseModel

class TextEvaluationFromStorageRequest(BaseModel):
    original_s3_path: str
    synthetic_s3_path: str


@app.post("/evaluate-texts/")
async def evaluate_texts_from_storage(payload: TextEvaluationFromStorageRequest):
    try:
        client = get_minio_client()

        # parse bucket and object names
        orig_bucket, orig_path = payload.original_s3_path.split("/", 1)
        synth_bucket, synth_path = payload.synthetic_s3_path.split("/", 1)

        # download files from MinIO
        original_obj = client.get_object(orig_bucket, orig_path)
        synthetic_obj = client.get_object(synth_bucket, synth_path)

        df_original = pd.read_csv(original_obj, encoding_errors="ignore", on_bad_lines="skip")
        df_synthetic = pd.read_csv(synthetic_obj, encoding_errors="ignore", on_bad_lines="skip")

        # take the first column that contains text
        text_col_orig = df_original.select_dtypes(include="object").columns[0]
        text_col_synth = df_synthetic.select_dtypes(include="object").columns[0]

        orig_texts = df_original[text_col_orig].dropna().astype(str).tolist()
        synth_texts = df_synthetic[text_col_synth].dropna().astype(str).tolist()

        if not orig_texts or not synth_texts:
            raise HTTPException(status_code=400, detail="Text columns are empty")

        stats_original = analyze_text_statistics(orig_texts)
        stats_synthetic = analyze_text_statistics(synth_texts)
        comparison = compare_datasets(orig_texts, synth_texts)

        return {
            "original_stats": stats_original,
            "synthetic_stats": stats_synthetic,
            "comparison": comparison
        }

    except S3Error as e:
        logging.error(f"MinIO access error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="MinIO file retrieval error")
    except Exception as e:
        logging.error(f"Error in /evaluate-texts/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
# ----------------------------------------------------------
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
    current_user: User = Depends(get_current_user)  # ✅
):
    query = select(UploadedFile).where(UploadedFile.owner_id == current_user.id)
    if file_type:
        query = query.where(UploadedFile.file_type == file_type)
    result = await db.execute(query)
    files = result.scalars().all()
    return [f"{f.s3_path}" for f in files]


# End of file