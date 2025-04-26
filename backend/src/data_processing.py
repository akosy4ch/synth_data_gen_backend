import os
import pandas as pd


def load_dataset(filepath, target_column=None):
    """
    Автоматически определяет тип файла (CSV, TXT, XLSX, XLS, JSON) по расширению
    и читает его подходящим методом. Если нужно другое поведение, 
    можно доработать логику (разделитель, дополнительную проверку кодировки и т.д.).

    Если target_column указан, возвращает только эту колонку.
    По итогам данных определяет 'task': "text", "numeric" или "mixed".
    """

    # Извлекаем расширение (пример: .csv, .xls, .json)
    ext = os.path.splitext(filepath)[1].lower()

    # 1. Определяем формат файла
    if ext in [".csv", ".txt"]:
        # Для csv/txt пробуем сначала utf-8-sig, если ловим UnicodeDecodeError —
        # пробуем cp1252, затем при ошибке cp1251. 
        # (Расширяем при необходимости.)
        try:
            df = pd.read_csv(filepath, encoding="utf-8-sig", on_bad_lines="skip")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, encoding="cp1252", on_bad_lines="skip")
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding="cp1251", on_bad_lines="skip")

    elif ext in [".xlsx", ".xls"]:
        # Для .xls обычно engine="xlrd", для .xlsx — "openpyxl".
        # Если нужно, добавляем fallback. Здесь самый простой вариант:
        if ext == ".xls":
            df = pd.read_excel(filepath, engine="xlrd")
        else:
            df = pd.read_excel(filepath, engine="openpyxl")

    elif ext == ".json":
        # Если JSON в формате JSON lines (по строкам), нужно lines=True.
        # Если это обычный JSON-массив, убираем lines=True.
        # Для универсальности попробуем сначала JSON lines.
        try:
            df = pd.read_json(filepath, encoding="utf-8-sig", lines=True)
        except ValueError:
            # Если не получилось, значит обычный JSON
            df = pd.read_json(filepath, encoding="utf-8-sig")

    else:
        raise ValueError(f"Unsupported file extension '{ext}'")
    
    # 🔧 Удаляем пробелы по краям названий колонок
    df.columns = df.columns.str.strip()

    # 2. Если указан target_column, обрабатываем только его
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset")

        target_series = df[target_column]
        # Проверяем тип данных в выбранной колонке:
        if target_series.dtype == object:
            task = "text"
            df = df[[target_column]]
        elif pd.api.types.is_numeric_dtype(target_series):
            task = "numeric"
            df = df[[target_column]]
        else:
            # Если в этой колонке не чистые object/числа, пусть будет 'mixed'
            task = "mixed"
            df = df[[target_column]]

    else:
        # Если target_column не задан, определяем тип по всему DataFrame
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
