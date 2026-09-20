from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TargetTransformer:
    """Single source of truth for everything related to the target column:
    column identification, optional label mapping, missing-label handling,
    and positive-class resolution for scorers and reporting.

    Config shape (all keys under `target:` in your params/config file):

        column: str
            Required. Name of the target column in the raw dataframe.

        mapping: dict[str, int], optional
            Maps raw categorical labels -> numeric labels for training.
            Omit if the target is already numeric.

        display_mapping: dict[int, str], optional
            Maps numeric labels -> human-readable strings, for reporting.
            If omitted and `mapping` is present, falls back to inverting
            `mapping`. If omitted and there's no `mapping` either, decode()
            is a no-op.

        positive_class: str | int, optional
            The positive class, written in whichever space is most natural:
            - if `mapping` is set, write it in raw/human terms (e.g. "Certified")
            - if no `mapping`, write it already in encoded/numeric terms (e.g. 1)
            Used to resolve pos_label for scorers (`positive_label_encoded`)
            and a human-readable form for logging (`positive_label_display`).

    Typical usage:

        target_transformer = TargetTransformer(config.target)

        train_df = target_transformer.filter_valid(train_df)
        test_df = target_transformer.filter_valid(test_df)

        y_train = target_transformer.prepare(train_df)
        X_train = train_df.drop(columns=[target_transformer.column])

        scorer = make_scorer(f1_score, pos_label=target_transformer.positive_label_encoded)

        logger.info(target_handler.summary())
        mlflow.log_param("positive_class", target_handler.positive_label_display)

        # at inference/reporting time:
        readable_preds = target_handler.decode(pd.Series(y_pred))
    """

    def __init__(self, target_config: dict[str, Any]):
        if "column_name" not in target_config:
            raise KeyError("Target config is missing required key: 'column_name'")

        self.column: str = target_config["column_name"]
        self.mapping: Optional[dict] = target_config.get("mapping")
        self.positive_class: Optional[Any] = target_config.get("positive_class")

        if self.mapping is not None and len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError("Mapping values must be unique.")

        # display_mapping is independent of mapping. If not explicitly
        # configured, fall back to inverting `mapping` (if present) as a
        # convenience default — this is just a default, not a coupling.
        display_mapping = target_config.get("display_mapping")
        if display_mapping is None and self.mapping is not None:
            display_mapping = {v: k for k, v in self.mapping.items()}
        self.display_mapping: Optional[dict] = display_mapping

    # ------------------------------------------------------------------ #
    # Column access
    # ------------------------------------------------------------------ #

    def extract(self, df: pd.DataFrame) -> pd.Series:
        """Pulls the raw target column out of a dataframe."""
        if self.column not in df.columns:
            raise KeyError(f"Target column '{self.column}' not found in dataframe.")
        return df[self.column]

    # ------------------------------------------------------------------ #
    # Missing-label handling — drop, never impute
    # ------------------------------------------------------------------ #

    def filter_valid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops rows with missing target labels and returns the filtered
        dataframe, keeping X and y in sync. Missing targets are dropped,
        never imputed — guessing a label injects noise or leakage.

        Logs how many/what fraction of rows were dropped. Call this before
        splitting X/y apart, and ideally before train/test split.
        """
        y = self.extract(df)
        is_missing = y.isna()
        n_missing = int(is_missing.sum())

        if n_missing > 0:
            pct = 100 * n_missing / len(df)
            logger.warning(
                "Dropping %d/%d rows (%.2f%%) with missing target '%s'. "
                "If this is unexpectedly high, investigate whether missingness "
                "is random or systematic before proceeding — systematic "
                "missingness can bias the training distribution.",
                n_missing, len(df), pct, self.column,
            )

        return df.loc[~is_missing].reset_index(drop=True)

    def get_missing_target_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns only the rows with missing target — for inspection/audit,
        not for training.
        """
        return df.loc[self.extract(df).isna()]

    # ------------------------------------------------------------------ #
    # Encode / decode — independent of each other
    # ------------------------------------------------------------------ #

    def encode(self, y: pd.Series) -> pd.Series:
        """Raw target -> numeric, for training. Uses `mapping` if configured;
        otherwise assumes the target is already numeric.

        Expects missing values to already be handled (via filter_valid).
        Raises if it still finds any, so a forgotten filter_valid() call
        fails loudly instead of silently training on NaN-encoded rows.
        """
        if y.isna().any():
            raise ValueError(
                f"Target '{self.column}' still contains missing values at encode() "
                "time. Call filter_valid() on the dataframe before encoding."
            )

        if self.mapping is not None:
            unseen = set(y.unique()) - set(self.mapping)
            if unseen:
                raise ValueError(
                    f"Unseen target label(s) {unseen}. Expected one of {list(self.mapping)}."
                )
            return y.map(self.mapping).astype(int)

        return y

    def decode(self, y: pd.Series) -> pd.Series:
        """Numeric target/prediction -> human-readable label, for
        reporting/serving. Uses `display_mapping` if available; otherwise
        a no-op. Independent of whether encode() does anything — this lets
        an already-numeric target still get meaningful display labels.
        """
        if self.display_mapping is None:
            return y
        return y.map(self.display_mapping)

    def prepare(self, df: pd.DataFrame) -> pd.Series:
        """Convenience: extract + encode in one call."""
        return self.encode(self.extract(df))

    # ------------------------------------------------------------------ #
    # Positive class resolution — for scorers and logging
    # ------------------------------------------------------------------ #

    @property
    def positive_label_encoded(self) -> Optional[Any]:
        """positive_class resolved into encoded (numeric) space, for scorers
        (e.g. `make_scorer(f1_score, pos_label=target_handler.positive_label_encoded)`).
        """
        if self.positive_class is None:
            return None
        if self.mapping is not None:
            if self.positive_class not in self.mapping:
                raise ValueError(
                    f"positive_class '{self.positive_class}' not found in "
                    f"target mapping keys {list(self.mapping)}."
                )
            return self.mapping[self.positive_class]
        return self.positive_class

    @property
    def positive_label_display(self) -> Optional[Any]:
        """positive_class resolved into human-readable display form, for
        logging/reporting (e.g. MLflow run params).
        """
        if self.positive_class is None:
            return None
        encoded = self.positive_label_encoded
        if self.display_mapping is not None and encoded in self.display_mapping:
            return self.display_mapping[encoded]
        return self.positive_class

    def summary(self) -> str:
        """One-line human-readable summary, handy for logging/MLflow params."""
        if self.positive_class is None:
            return f"target='{self.column}' (no positive_class configured)"
        return (
            f"target='{self.column}', positive_class="
            f"{self.positive_label_display!r} (encoded={self.positive_label_encoded})"
        )
