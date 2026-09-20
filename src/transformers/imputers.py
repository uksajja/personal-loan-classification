
# transformers/imputers.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class SimpleImputerWrapper(BaseEstimator, TransformerMixin):
    """
    Thin wrapper around sklearn's SimpleImputer that preserves DataFrame
    column names/dtypes (SimpleImputer normally returns a raw numpy array),
    and only touches the configured columns.
    """
    def __init__(self, strategy: str = "mean", columns: list | None = None, **kwargs):
        self.strategy = strategy
        self.columns = columns
        self.kwargs = kwargs  # forwards any extra SimpleImputer args, e.g. fill_value for strategy='constant'

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.columns_ = self.columns or df.columns[df.isnull().any()].tolist()
        if self.columns_:
            self.imputer_ = SimpleImputer(strategy=self.strategy, **self.kwargs)
            self.imputer_.fit(df[self.columns_])
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.columns_:
            existing = [c for c in self.columns_ if c in df.columns]
            if existing:
                df[existing] = self.imputer_.transform(df[existing])
        return df


class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Handles imputation strategies SimpleImputer doesn't natively support:
    ffill, bfill, and per-column mixed strategies (e.g. {'Age': 'mean', 'City': 'mode'}).
    """
    def __init__(self, strategy: str = "mean", columns: list | None = None, per_column: dict | None = None):
        self.strategy = strategy
        self.columns = columns
        self.per_column = per_column

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        mapping = self.per_column or {
            col: self.strategy for col in (self.columns or df.columns[df.isnull().any()].tolist())
        }
        self.impute_values_ = {}
        for col, strat in mapping.items():
            if col not in df.columns:
                raise KeyError(f"Column not found for imputation: {col}")
            self.impute_values_[col] = (strat, self._compute_fill_value(df[col], strat))
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col, (strat, value) in self.impute_values_.items():
            if col not in df.columns:
                continue
            if strat in {"ffill", "bfill"}:
                df[col] = df[col].fillna(method=strat)
            else:
                df[col] = df[col].fillna(value)
        return df

    @staticmethod
    def _compute_fill_value(series: pd.Series, strategy: str):
        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series):
                raise TypeError(f"Cannot apply mean imputation to non-numeric column: {series.name}")
            return series.mean()
        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(series):
                raise TypeError(f"Cannot apply median imputation to non-numeric column: {series.name}")
            return series.median()
        elif strategy == "mode":
            mode_vals = series.mode(dropna=True)
            return mode_vals.iloc[0] if not mode_vals.empty else None
        elif strategy in {"ffill", "bfill"}:
            return strategy
        else:
            raise ValueError(f"Unsupported imputation strategy: {strategy}")
        
class DtypeBasedImputer(BaseEstimator, TransformerMixin):
    """
    No column list needed. Numeric columns get one strategy, categorical
    columns get another — applied to whichever columns match at fit time.
    """
    def __init__(self,         
                numeric_strategy: str = "mean",
                categorical_strategy: str = "mode"
                ):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        numeric_cols = df.select_dtypes(include="number").columns
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

        self.impute_values_ = {}
        for col in numeric_cols:
            if df[col].isnull().any():
                val = df[col].mean() if self.numeric_strategy == "mean" else df[col].median()
                self.impute_values_[col] = val
        for col in categorical_cols:
            if df[col].isnull().any():
                mode = df[col].mode(dropna=True)
                self.impute_values_[col] = mode.iloc[0] if not mode.empty else None
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col, val in self.impute_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df
    
class ImputerFactory:
    """Selects and builds the appropriate imputer from missing_value_treatment config."""

    _IMPUTERS = {
        "simple_imputer": SimpleImputerWrapper,
        "dtype_based": DtypeBasedImputer,
    }

    @classmethod
    def build(cls, missing_value_config: dict):
        config = dict(missing_value_config)  # copy — don't mutate the loaded config
        method = config.pop("method", "dtype_based")

        imputer_cls = cls._IMPUTERS.get(method)
        if imputer_cls is None:
            raise ValueError(f"Unsupported imputation method: {method}. Choose from: {list(cls._IMPUTERS)}")

        return imputer_cls(**config)
    


