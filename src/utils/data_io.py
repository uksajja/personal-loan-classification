import sys
from pathlib import Path
import pandas as pd    
    

def read_batch(file_path: str) -> pd.DataFrame:

    file_path = Path(file_path)   
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    fmt = file_path.suffix.lower().lstrip(".")
    if fmt == "csv":
        return pd.read_csv(file_path)
    elif fmt == "json":
        return pd.read_json(file_path, orient="records", lines=True)
    elif fmt == "parquet":
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported batch input format: {fmt}")

def write_batch(df: pd.DataFrame, file_path: str):
    file_path = Path(file_path)   
    fmt = Path(file_path).suffix.lower().lstrip(".")
    if fmt == "csv":
        df.to_csv(file_path, index=False)
    elif fmt == "json":
        df.to_json(file_path, orient="records", lines=True)
    elif fmt == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported batch output format: {fmt}")