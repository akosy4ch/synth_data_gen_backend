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
        target_column (str, optional): Если указан, возвращается только эта колонка.
        delimiter (str, optional): Разделитель для CSV/TXT (по умолчанию — запятая).

    Возвращает:
        (DataFrame, str): Загруженный DataFrame и тип задачи ('text', 'numeric', 'mixed').
    """
    
    # Определяем расширение
    if hasattr(filepath, "read"):
        ext = os.path.splitext(getattr(filepath, "name", ""))[1].lower() or ".csv"
    else:
        ext = os.path.splitext(filepath)[1].lower()

    # Чтение файла
    if ext in [".csv", ".txt"]:
        # Пробуем разные кодировки
        try:
            df = pd.read_csv(filepath, encoding="utf-8-sig", delimiter=delimiter, on_bad_lines="skip")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, encoding="cp1252", delimiter=delimiter, on_bad_lines="skip")
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding="cp1251", delimiter=delimiter, on_bad_lines="skip")

    elif ext in [".xlsx", ".xls"]:
        if ext == ".xls":
            df = pd.read_excel(filepath, engine="xlrd")
        else:
            df = pd.read_excel(filepath, engine="openpyxl")

    elif ext == ".json":
        try:
            df = pd.read_json(filepath, encoding="utf-8-sig", lines=True)
        except ValueError:
            df = pd.read_json(filepath, encoding="utf-8-sig")
    
    else:
        raise ValueError(f"Unsupported file extension '{ext}'")

    # Удаляем пробелы в названиях колонок
    df.columns = df.columns.str.strip()

    # Обрабатываем только target_column, если указан
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset")
    
        dtype = df[target_column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            task = "numeric"
        elif dtype == object:
            task = "text"
        else:
            task = "mixed"

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
