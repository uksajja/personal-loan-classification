# transformers/outliers.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """
    Fits IQR bounds (and, for 'replace_with_*' methods, fill values) on
    training data only; transform() applies the *stored* bounds/values —
    never recomputed — so a single inference row transforms consistently
    with however training data was treated.
    """
    def __init__(self, columns: list | None = None, factor: float = 1.5,
                 detection_method: str = "iqr", treatment_method: str = "clip", exclude_columns: list | None = None):
        self.columns = columns
        self.factor = factor
        self.detection_method = detection_method
        self.treatment_method = treatment_method
        self.exclude_columns = exclude_columns

    def fit(self, X, y=None):
        if self.detection_method != "iqr":
            raise ValueError("Unsupported outlier detection method. Only 'iqr' is supported.")

        df = pd.DataFrame(X)
        cols = self.columns or df.select_dtypes(include=["number"]).columns.tolist()

        # Exclude specified columns
        if self.exclude_columns:
            cols = [col for col in cols if col not in self.exclude_columns]

        self.bounds_ = {}
        self.fill_values_ = {}
        for col in cols:
            if col not in df.columns:
                raise KeyError(f"Column not found for outlier treatment: {col}")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Outlier treatment only supports numeric columns: {col}")

            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - self.factor * iqr, q3 + self.factor * iqr
            self.bounds_[col] = (lower, upper)

            if self.treatment_method in {"replace_with_mean", "replace_with_median", "replace_with_mode"}:
                mask = (df[col] < lower) | (df[col] > upper)
                non_outliers = df.loc[~mask, col]
                if self.treatment_method == "replace_with_mean":
                    self.fill_values_[col] = non_outliers.mean()
                elif self.treatment_method == "replace_with_median":
                    self.fill_values_[col] = non_outliers.median()
                else:
                    mode_vals = non_outliers.mode(dropna=True)
                    self.fill_values_[col] = mode_vals.iloc[0] if not mode_vals.empty else None

        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            if col not in df.columns:
                continue
            mask = (df[col] < lower) | (df[col] > upper)

            if self.treatment_method == "clip":
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif self.treatment_method == "remove":
                df = df.loc[~mask]  # only sensible during training fit_transform; a no-op is safer at live inference
            elif self.treatment_method in {"replace_with_mean", "replace_with_median", "replace_with_mode"}:
                df.loc[mask, col] = self.fill_values_[col]
            else:
                raise ValueError(f"Unsupported outlier treatment method: {self.treatment_method}")

        return df