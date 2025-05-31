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
    # --- Определение расширения ---
    ext = ""
    if hasattr(filepath, "read"):
        ext = os.path.splitext(getattr(filepath, "name", ""))[1].lower()
        if not ext:
            ext = ".csv"  # по умолчанию, если нет имени
    else:
        ext = os.path.splitext(filepath)[1].lower()

    if hasattr(filepath, "read"):
        size = len(filepath.read())
        filepath.seek(0)
        print("File size:", size)  # Debug

    # --- Проверка бинарности ---
    if hasattr(filepath, "read"):
        peek = filepath.read(2048)
        filepath.seek(0)
        if b'\x00' in peek:
            raise ValueError("Binary file detected. Please upload a valid text-based dataset.")
        
    if hasattr(filepath, "read"):
        filepath.seek(0)  # <<< важно
        for enc in ["utf-8-sig", "cp1252", "cp1251"]:
            try:
                df = pd.read_csv(filepath, encoding=enc, delimiter=delimiter, on_bad_lines="skip")
                break
            except UnicodeDecodeError:
                filepath.seek(0)  # обязательно сброс перед новой попыткой

    # --- Чтение по расширению ---
    if ext in [".csv", ".txt"]:
        df = None
        for enc in ["utf-8-sig", "cp1252", "cp1251"]:
            try:
                if hasattr(filepath, "seek"):
                    filepath.seek(0)
                df = pd.read_csv(filepath, encoding=enc, delimiter=delimiter, on_bad_lines="skip")
                break
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                if hasattr(filepath, "seek"):
                    filepath.seek(0)
                continue
        if df is None:
            raise ValueError("Failed to decode CSV/TXT file. Try saving it in UTF-8.")
       
    elif ext in [".xlsx", ".xls"]:
        try:
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            df = pd.read_excel(filepath, engine=engine)
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {e}")
    elif ext == ".json":
        try:
            df = pd.read_json(filepath, encoding="utf-8-sig", lines=True)
        except ValueError:
            try:
                df = pd.read_json(filepath, encoding="utf-8-sig")
            except Exception as e:
                raise ValueError(f"Failed to read JSON file: {e}")
    else:
        raise ValueError(f"Unsupported file extension '{ext}'")

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    # --- Очистка названий колонок ---
    df.columns = df.columns.str.strip()

    # --- Определение типа задачи ---
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset.")
        task = _detect_task_type(df[target_column].dtype)
    else:
        task = _infer_task_type(df)

    return df, task


def _detect_task_type(dtype) -> str:
    """Определение типа задачи по типу данных"""
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    elif pd.api.types.is_string_dtype(dtype):
        return "text"
    else:
        return "mixed"


def _infer_task_type(df: pd.DataFrame) -> str:
    """Определение типа задачи по структуре всего датафрейма"""
    numeric_count = df.select_dtypes(include=["number"]).shape[1]
    text_count = df.select_dtypes(include=["object"]).shape[1]
    total = df.shape[1]

    if numeric_count == total:
        return "numeric"
    elif text_count == total:
        return "text"
    else:
        return "mixed"
