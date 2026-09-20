# transformers/scalers.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer, PowerTransformer, QuantileTransformer
)


class ScalerFactory:
    """Selects a scaler instance from config. No wrapping needed — sklearn's
    scalers are already full transformers; this is just a name -> class lookup."""

    _SCALERS = {
        "standard": StandardScaler,
        "min-max": MinMaxScaler,
        "robust": RobustScaler,
        "normalizer": Normalizer,
        "power": PowerTransformer,
        "quantile": QuantileTransformer,
    }

    @classmethod
    def build(cls, scaler_config: dict):
        scaler_type = scaler_config.get("type", "standard")
        if scaler_type not in cls._SCALERS:
            raise ValueError(
                f"Unsupported scaler type: {scaler_type}. Choose from: {list(cls._SCALERS)}"
            )
        return cls._SCALERS[scaler_type]()
    

class ScalingTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a configured scaler to specified columns only (or all numeric
    columns if none specified), leaving other columns untouched. Fitted
    once on training data; transform() reuses the fitted scaler — same
    train/inference-consistency guarantee as the other transformers.
    """
    def __init__(self, type: str = "standard", columns: list | None = None):
        self.type = type
        self.columns = columns

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.columns_ = self.columns or df.select_dtypes(include=["number"]).columns.tolist()
        self.scaler_ = ScalerFactory.build({"type": self.type})
        if self.columns_:
            self.scaler_.fit(df[self.columns_])
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.columns_:
            existing = [c for c in self.columns_ if c in df.columns]
            if existing:
                df[existing] = self.scaler_.transform(df[existing])
        return df