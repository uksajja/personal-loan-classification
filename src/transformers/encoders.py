# transformers/encoders.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps either sklearn's OneHotEncoder or pd.get_dummies behind one
    interface, selected via `method`. Categories/output schema are learned
    once at fit() and reapplied identically at transform() — unseen
    categories at inference are handled per-method (see below).
    """
    def __init__(self, method: str = "onehotencoder", columns: list | None = None,
                 infer_columns: bool = True, handle_unknown: str = "ignore"):
        self.method = method
        self.columns = columns
        self.infer_columns = infer_columns
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        if self.columns:
            cols = self.columns
        elif self.infer_columns:
            cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        else:
            cols = []
        self.columns_ = cols

        if not cols:
            self.encoder_ = None
            self.output_columns_ = df.columns.tolist()
            return self

        if self.method == "onehotencoder":
            # unseen categories at inference -> all-zero row for that feature (handle_unknown='ignore')
            self.encoder_ = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)
            self.encoder_.fit(df[cols])
        elif self.method == "get_dummies":
            # unseen categories at inference -> dropped by reindex; missing ones -> zero-filled
            self.output_columns_ = pd.get_dummies(df, columns=cols).columns.tolist()
        else:
            raise ValueError(f"Unsupported one-hot method: {self.method}. Choose 'onehotencoder' or 'get_dummies'.")

        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        if not self.columns_:
            return df

        if self.method == "onehotencoder":
            encoded = self.encoder_.transform(df[self.columns_])
            encoded_df = pd.DataFrame(
                encoded, columns=self.encoder_.get_feature_names_out(self.columns_), index=df.index
            )
            return pd.concat([df.drop(columns=self.columns_), encoded_df], axis=1)

        elif self.method == "get_dummies":
            encoded = pd.get_dummies(df, columns=self.columns_)
            return encoded.reindex(columns=self.output_columns_, fill_value=0)
        

class OrdinalEncodingTransformer(BaseEstimator, TransformerMixin):
    """
    Encodes categorical columns as integer codes (not one-hot). Learns
    categories once at fit() from training data; unseen categories at
    inference are mapped to a fixed sentinel value rather than crashing.

    Note: this is what people usually mean by "label encoding" for
    feature columns. sklearn's LabelEncoder is 1D-only and intended for
    the target y, not for X — it has no handle_unknown support and will
    raise on any category not seen during fit.
    """
    def __init__(self, columns: list | None = None, infer_columns: bool = True,
                 unknown_value: int = -1):
        self.columns = columns
        self.infer_columns = infer_columns
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        if self.columns:
            cols = self.columns
        elif self.infer_columns:
            cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        else:
            cols = []
        self.columns_ = cols

        if cols:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=self.unknown_value,
            )
            self.encoder_.fit(df[cols])
        else:
            self.encoder_ = None

        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if not self.columns_:
            return df
        df[self.columns_] = self.encoder_.transform(df[self.columns_])
        return df

    def get_category_mapping(self) -> dict:
        """Inspect learned categories per column — e.g. {'Education': ['Undergrad', 'Graduate', 'Advanced']}."""
        if not self.columns_:
            return {}
        return dict(zip(self.columns_, self.encoder_.categories_))
        

# EncoderFactory — dispatch between OneHotEncodingTransformer and OrdinalEncodingTransformer
class EncoderFactory:
    _ENCODERS = {
        "onehotencoder": OneHotEncodingTransformer,
        "get_dummies": OneHotEncodingTransformer,   # both handled inside OneHotEncodingTransformer via its own 'method' param
        "ordinal": OrdinalEncodingTransformer,
        "label": OrdinalEncodingTransformer,          # alias, since this is what people usually mean by "label encoding" for X
    }

    @classmethod
    def build(cls, encoding_config: dict):
        config = dict(encoding_config)
        method = config.pop("method", "onehotencoder")

        encoder_cls = cls._ENCODERS.get(method)
        if encoder_cls is None:
            raise ValueError(f"Unsupported encoding method: {method}. Choose from: {list(cls._ENCODERS)}")

        # OneHotEncodingTransformer still needs its own internal method param
        # (onehotencoder vs get_dummies) — pass it back in for that class only
        if encoder_cls is OneHotEncodingTransformer:
            config["method"] = method

        return encoder_cls(**config)