    
import sys
from pathlib import Path
import pandas as pd
import pandera.pandas as pa

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import Logger 

"""
sample input for single-record inference:
{
  "ID": 2389,
  "Age": 64,
  "Experience": 39,
  "Income": 23,
  "ZIPCode": 91768,
  "Family": 3,
  "CCAvg": 0.5,
  "Education": 1,
  "Mortgage": 0,
  "Securities_Account": 1,
  "CD_Account": 0,
  "Online": 0,
  "CreditCard": 0
}
"""
"""
SCHEMA_DTYPES = {
            'ID': 'int64', 'Age': 'int64', 'Experience': 'int64', 'Income': 'int64',
            'ZIPCode': 'int64', 'Family': 'int64', 'CCAvg': 'float64',
            'Education': 'int64', 'Mortgage': 'int64', 'Securities_Account': 'int64',
            'CD_Account': 'int64', 'Online': 'int64', 'CreditCard': 'int64'
        }

INPUT_SCHEMA = pa.DataFrameSchema( {
        "ID": pa.Column(int, pa.Check.ge(0), nullable=False),
        "Age": pa.Column(int, pa.Check.ge(0), nullable=False),
        "Experience": pa.Column(int, pa.Check.ge(0), nullable=False),
        "Income": pa.Column(int, nullable=False),
        "ZIPCode": pa.Column(int, pa.Check.ge(0), nullable=False),
        "Family": pa.Column(int, pa.Check.ge(0), nullable=False),
        "CCAvg": pa.Column(float, pa.Check.ge(0), nullable=False),
        "Education": pa.Column(int, pa.Check.ge(0), nullable=False),
        "Mortgage": pa.Column(int, pa.Check.ge(0), nullable=False),
        "Securities_Account": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "CD_Account": pa.Column(int, pa.Check.isin([0, 1]),nullable=False),
        "Online": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "CreditCard": pa.Column(int, pa.Check.isin([0, 1]), nullable=False)
    })
"""

def run_schema_validation(df: pd.DataFrame, schema: pa.DataFrameSchema) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Core validation logic, shared by batch and single-record paths.
    Returns (valid_rows, rejected_rows_with_reason).
    """
    try:
        validated = schema.validate(df, lazy=True)
        return validated, pd.DataFrame(columns=list(df.columns) + ["error"])
    except pa.errors.SchemaErrors as e:
        bad_indices = e.failure_cases["index"].dropna().unique()
        rejected = df.loc[df.index.isin(bad_indices)].copy()
        rejected["error"] = "failed validation"
        valid = df.loc[~df.index.isin(bad_indices)].copy()
        return valid, rejected


def validate_input(df: pd.DataFrame, schema: pa.DataFrameSchema) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch path: tolerates partial failure, quarantines bad rows, keeps going."""
    valid, rejected = run_schema_validation(df, schema)
    if not rejected.empty:
        Logger.warning(f"{len(rejected)}/{len(df)} rows failed validation and were quarantined")
    return valid, rejected