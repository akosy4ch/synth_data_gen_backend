import os
import pandas as pd
from typing import Union, IO, Tuple


def load_dataset(
    filepath: Union[str, IO],
    target_column: str = None,
    delimiter: str = ","
) -> Tuple[pd.DataFrame, str]:
    """
    Загружает датасет из CSV, TXT, Excel (.xls/.xlsx), JSON файла.
    Поддерживает как путь к файлу, так и file-like объект (например, BytesIO).

    Параметры:
        filepath (str или file-like): Путь к файлу или file-like объект.
        target_column (str, optional): Если указан, анализируется только эта колонка.
        delimiter (str, optional): Разделитель для CSV/TXT (по умолчанию — запятая).

    Возвращает:
        (DataFrame, str): Загруженный DataFrame и тип задачи ('text', 'numeric', 'mixed').
    """
    ext = ""
    if hasattr(filepath, "read"):
        # Попробуем определить расширение, если есть .name
        ext = os.path.splitext(getattr(filepath, "name", ""))[1].lower()
        if not ext:
            ext = ".csv"  # По умолчанию
    else:
        ext = os.path.splitext(filepath)[1].lower()

    # --- Чтение файла по типу ---
    if ext in [".csv", ".txt"]:
        for enc in ["utf-8-sig", "cp1252", "cp1251"]:
            try:
                df = pd.read_csv(filepath, encoding=enc, delimiter=delimiter, on_bad_lines="skip")
                break
            except UnicodeDecodeError:
                continue
    elif ext in [".xlsx", ".xls"]:
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        df = pd.read_excel(filepath, engine=engine)
    elif ext == ".json":
        try:
            df = pd.read_json(filepath, encoding="utf-8-sig", lines=True)
        except ValueError:
            df = pd.read_json(filepath, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file extension '{ext}'")

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    # --- Чистим названия колонок ---
    df.columns = df.columns.str.strip()

    # --- Определяем тип задачи ---
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset.")
        dtype = df[target_column].dtype
        task = _detect_task_type(dtype)
    else:
        numeric_count = df.select_dtypes(include=["number"]).shape[1]
        object_count = df.select_dtypes(include=["object"]).shape[1]
        total_count = df.shape[1]

        if numeric_count == total_count:
            task = "numeric"
        elif object_count == total_count:
            task = "text"
        else:
            task = "mixed"

    return df, task


def _detect_task_type(dtype) -> str:
    """Вспомогательная функция для определения типа задачи по dtype"""
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    elif pd.api.types.is_string_dtype(dtype):
        return "text"
    else:
        return "mixed"
