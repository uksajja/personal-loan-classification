# transformers/column_ops.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops a fixed list of columns. No fitting needed, but kept fit/transform for Pipeline compatibility."""
    def __init__(self, drop_cols_cfg: dict | None = None):
        self.drop_cols_cfg = drop_cols_cfg #Required for BaseEstimator, but not used directly.
        if drop_cols_cfg is None:
            self.columns = []
        else:
            self.columns = drop_cols_cfg.get("columns", [])


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df.drop(columns=[c for c in self.columns if c in df.columns])


class ColumnRenamer(BaseEstimator, TransformerMixin):
    """Renames columns using a fixed mapping."""
    def __init__(self, rename_cols_cfg: dict | None = None):
        self.rename_cols_cfg = rename_cols_cfg #Required for BaseEstimator, but not used directly.
        if rename_cols_cfg is None:
            self.column_mapping = {}
        else:
            self.column_mapping = rename_cols_cfg.get("column_mapping", {})

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df.rename(columns=self.column_mapping) if self.column_mapping else df


class ZipcodeTransformer(BaseEstimator, TransformerMixin):
    """Reduces ZIPCode precision (e.g. 94582 -> '94') by integer division."""
    def __init__(self, zipcode_config: dict | None = None):
        self.zipcode_config = zipcode_config #Required for BaseEstimator, but not used directly.
        if zipcode_config is None:
            self.column = "ZIPCode"
            self.divisor = 1000
        else:
            self.column = zipcode_config.get("column", "ZIPCode")
            self.divisor = zipcode_config.get("divisor", 1000)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.column in df.columns:
            df[self.column] = (df[self.column] // self.divisor).astype(str)
        return df
