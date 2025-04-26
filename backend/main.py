from pathlib import Path
import os
import html
import pandas as pd
import logging
import math
import aiofiles
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List

from backend.src.generator import generate_synthetic
from backend.src.data_processing import load_dataset
from backend.src.text_analysis import analyze_text_statistics, compare_datasets
from backend.src.auth.routes import router as auth_router

app = FastAPI()

app.include_router(auth_router)

import nltk
nltk.download("punkt", quiet=True)

app.mount

# ➡️ Функция очистки NaN, inf
def clean_floats(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: clean_floats(v) for v in obj.items()}
    elif isinstance(obj, list):
        return [clean_floats(v) for v in obj]
    else:
        return obj

# ➡️ Асинхронное сохранение загруженного файла
async def save_upload_file(upload_file: UploadFile, destination: str):
    async with aiofiles.open(destination, 'wb') as out_file:
        while content := await upload_file.read(1024):
            await out_file.write(content)

# ➡️ Логирование
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ➡️ Инициализация FastAPI приложения


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
#if not os.path.exists(FRONTEND_DIR):
#    raise RuntimeError(f"Frontend folder not found at {FRONTEND_DIR}")


# ----------------------------------------------------------
# Анализ колонок
@app.post("/analyze-columns/")
async def analyze_columns(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    try:
        await save_upload_file(file, filepath)

        df = pd.read_csv(filepath, encoding_errors="ignore", on_bad_lines="skip")
        column_info = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            sample_values = df[col].dropna().astype(str).unique().tolist()[:5]
            avg_len = (
                df[col].dropna().astype(str).apply(len).mean()
                if dtype == "object" else None
            )
            column_info.append({
                "name": html.escape(col.replace("'", "&#39;")),
                "dtype": dtype,
                "unique_values": unique_count,
                "sample": sample_values,
                "avg_length": avg_len
            })

        logging.info(f"✅ Columns analyzed from file: {file.filename}")
        return {"columns": column_info, "rows": len(df)}

    except Exception as e:
        logging.error(f"❌ Error analyzing columns from {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ----------------------------------------------------------
# Генерация синтетических данных
@app.post("/generate-synthetic/")
async def create_synthetic_data(
    file: UploadFile = File(...),
    columns: List[str] = Form(...),
    models: List[str] = Form(...),
    samples: int = Form(...),
):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    try:
        await save_upload_file(file, filepath)

        df_original, _ = load_dataset(filepath)

        columns = [html.unescape(col.strip()) for col in columns]
        df_cols_set = set(df_original.columns)
        missing_cols = [col for col in columns if col not in df_cols_set]
        if missing_cols:
            logging.error(f"❌ Columns not found: {missing_cols}")
            raise HTTPException(status_code=400, detail=f"❌ Columns not found: {missing_cols}")

        results = {}
        df_result = pd.DataFrame()

        min_input_length = min(len(df_original[col].dropna()) for col in columns)
        gen_size = min(samples, min_input_length)

        queue = asyncio.Queue()

        # ➡️ Заполняем очередь заданиями на генерацию
        for col, model_type in zip(columns, models):
            df_column = df_original[[col]]
            task = "text" if df_column[col].dtype == object else "numeric"
            await queue.put((df_column, model_type, gen_size, task, col))

        # ➡️ Воркер для обработки одной задачи
        async def worker():
            while not queue.empty():
                df_column, model_type, gen_size, task, col = await queue.get()
                synthetic_df = await generate_synthetic(df_column, model_type, gen_size, task, col)

                synthetic_df[col] = synthetic_df[col].fillna("").astype(str)
                synthetic_df = synthetic_df[synthetic_df[col].str.strip() != ""]

                while len(synthetic_df) < gen_size:
                    synthetic_df = pd.concat([synthetic_df, synthetic_df]).head(gen_size)

                df_result[col] = synthetic_df[col].reset_index(drop=True)

                if task == "text":
                    original_texts = df_column[col].dropna().astype(str).tolist()
                    synthetic_texts = synthetic_df[col].dropna().astype(str).tolist()
                    results[col] = clean_floats({
                        "stats_original": analyze_text_statistics(original_texts),
                        "stats_synthetic": analyze_text_statistics(synthetic_texts),
                        "comparison": compare_datasets(original_texts, synthetic_texts)
                    })

                queue.task_done()

        # ➡️ Запускаем несколько параллельных воркеров
        await asyncio.gather(*[worker() for _ in range(3)])

        logging.info(f"✅ Synthetic generation completed for file: {file.filename}")
        return clean_floats({
            "synthetic": df_result.to_dict(orient="records"),
            "analysis": results
        })

    except Exception as e:
        logging.error(f"❌ Error generating synthetic data from {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ----------------------------------------------------------
# Автоопределение конфигурации
@app.post("/detect-best-config/")
async def detect_best_config(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    try:
        await save_upload_file(file, filepath)

        df = pd.read_csv(filepath, encoding="utf-8-sig", on_bad_lines="skip")
        result = []

        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue

            if series.dtype == object:
                task = "text"
                model = "GPT-J"
            elif pd.api.types.is_numeric_dtype(series):
                task = "numeric"
                model = "CTGAN"
            else:
                task = "mixed"
                model = "CTGAN"

            result.append({
                "column": col,
                "task": task,
                "recommended_model": model,
                "original_samples": int(len(series)),
                "default_sample_size": int(len(series))
            })

        logging.info(f"📊 Auto-detected config for file: {file.filename}")
        return {"recommendations": result}

    except Exception as e:
        logging.error(f"❌ Error in auto config detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ➡️ Монтируем фронтенд
#app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
