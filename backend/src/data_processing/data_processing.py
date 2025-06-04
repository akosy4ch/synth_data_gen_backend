import os
import pandas as pd
from typing import Union, IO, Tuple

def load_dataset(
    filepath: Union[str, IO],
    target_column: str = None,
    delimiter: str = ","
) -> Tuple[pd.DataFrame, str]:
    """
    Loads a dataset from CSV, TXT, Excel (.xls/.xlsx), or JSON.
    Supports both file paths and file-like objects (e.g., BytesIO).

    Args:
        filepath (str or file-like): File path or open file-like object.
        target_column (str, optional): Column name to analyze.
        delimiter (str, optional): Delimiter for CSV/TXT files.

    Returns:
        Tuple[pd.DataFrame, str]: Loaded DataFrame and task type ('text', 'numeric', or 'mixed').
    """
    # --- Determine extension ---
    ext = ""
    if hasattr(filepath, "read"):
        ext = os.path.splitext(getattr(filepath, "name", ""))[1].lower()
        if not ext:
            ext = ".csv"
    else:
        ext = os.path.splitext(filepath)[1].lower()

    df = None

    # --- Handle file types ---
    if ext in [".csv", ".txt"]:
        # Only check for binary in text files
        if hasattr(filepath, "read"):
            peek = filepath.read(2048)
            filepath.seek(0)
            if b'\x00' in peek:
                raise ValueError("Binary file detected. Please upload a valid text-based dataset.")

        for enc in ["utf-8-sig", "cp1252", "cp1251"]:
            try:
                if hasattr(filepath, "seek"):
                    filepath.seek(0)
                df = pd.read_csv(filepath, encoding=enc, delimiter=delimiter, on_bad_lines="skip")
                break
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
        if df is None:
            raise ValueError("Failed to read CSV/TXT file. Try saving it in UTF-8 encoding.")

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
        raise ValueError(f"Unsupported file extension '{ext}'.")

    # --- Final checks ---
    if df is None or df.empty:
        raise ValueError("Loaded dataset is empty or invalid.")

    df.columns = df.columns.str.strip()

    # --- Determine task type ---
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset.")
        task = _detect_task_type(df[target_column].dtype)
    else:
        task = _infer_task_type(df)

    return df, task


def _detect_task_type(dtype) -> str:
    """Determine task type from a single column's dtype."""
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    elif pd.api.types.is_string_dtype(dtype):
        return "text"
    else:
        return "mixed"


def _infer_task_type(df: pd.DataFrame) -> str:
    """Infer task type from overall DataFrame structure."""
    numeric_count = df.select_dtypes(include=["number"]).shape[1]
    text_count = df.select_dtypes(include=["object"]).shape[1]
    total = df.shape[1]

    if numeric_count == total:
        return "numeric"
    elif text_count == total:
        return "text"
    else:
        return "mixed"
