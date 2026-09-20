# transformers/multicollinearity.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MultiCollinearityHandler(BaseEstimator, TransformerMixin):
    """
    Detects and drops multicollinear numeric columns. Which columns to
    drop is decided ONCE at fit() time from training data only; transform()
    simply drops that fixed, stored list — never recomputes VIF/correlation
    on inference data, so a single inference row is handled identically to
    the training set (no matrix inversion or correlation needed on a
    single row, which would be undefined).

    Config shape:
        multicollinearity_treatment:
          method: 'vif'                # 'vif' | 'correlation' | 'none'
          vif_threshold: 5.0            # common convention: 5 (strict) or 10 (lenient)
          correlation_threshold: 0.9
          exclude_columns: []           # numeric columns to never consider for dropping
          protected_columns: []         # columns to keep even if flagged (domain knowledge overrides)
          max_iterations: 20            # safety cap for iterative VIF elimination
    """

    def __init__(
        self,
        method: str = "vif",
        vif_threshold: float = 5.0,
        correlation_threshold: float = 0.9,
        exclude_columns: list | None = None,
        protected_columns: list | None = None,
        max_iterations: int = 20,
    ):
        self.method = method
        self.vif_threshold = vif_threshold
        self.correlation_threshold = correlation_threshold
        self.exclude_columns = exclude_columns
        self.protected_columns = protected_columns
        self.max_iterations = max_iterations


    # ------------------------------------------------------------------
    # FIT — decide once which columns to drop, from training data only
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in self.exclude_columns
        ]

        if self.method == "none" or len(numeric_cols) < 2:
            self.columns_to_drop_ = []
            self.report_ = pd.DataFrame(columns=["column", "vif", "dropped"])
            return self

        if self.method == "vif":
            self.columns_to_drop_, self.report_ = self._fit_vif(df[numeric_cols])
        elif self.method == "correlation":
            self.columns_to_drop_, self.report_ = self._fit_correlation(df[numeric_cols])
        else:
            raise ValueError(
                f"Unsupported multicollinearity method: {self.method}. Choose 'vif', 'correlation', or 'none'."
            )

        return self

    def _fit_vif(self, df: pd.DataFrame):
        """
        Iteratively computes VIF for all remaining numeric columns, drops
        the single worst offender above threshold, repeats. Iterative
        because removing one column changes every other column's VIF.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        working_df = df.copy().dropna()
        dropped = []
        history = []

        for _ in range(self.max_iterations):
            if working_df.shape[1] < 2:
                break

            vif_values = pd.Series(
                [variance_inflation_factor(working_df.values, i) for i in range(working_df.shape[1])],
                index=working_df.columns,
            )

            droppable = vif_values.drop(
                index=[c for c in self.protected_columns if c in vif_values.index], errors="ignore"
            )

            if droppable.empty or droppable.max() <= self.vif_threshold:
                for col, vif in vif_values.items():
                    history.append({"column": col, "vif": vif, "dropped": False})
                break

            worst_col = droppable.idxmax()
            history.append({"column": worst_col, "vif": droppable.max(), "dropped": True})
            working_df = working_df.drop(columns=[worst_col])
            dropped.append(worst_col)

        return dropped, pd.DataFrame(history)

    def _fit_correlation(self, df: pd.DataFrame):
        """
        Pairwise correlation above threshold: for each correlated pair,
        drops the one with higher mean absolute correlation to all other
        columns (i.e. the more redundant one), unless protected.
        """
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        dropped = []
        history = []
        mean_corr = corr_matrix.mean()

        for col in upper.columns:
            correlated_with = upper.index[upper[col] > self.correlation_threshold].tolist()
            for other_col in correlated_with:
                if col in dropped or other_col in dropped:
                    continue
                candidates = [c for c in (col, other_col) if c not in self.protected_columns]
                if not candidates:
                    continue
                to_drop = max(candidates, key=lambda c: mean_corr[c])
                dropped.append(to_drop)
                history.append({
                    "column": to_drop,
                    "correlation": upper.loc[other_col, col],
                    "dropped": True,
                    "correlated_with": other_col if to_drop == col else col,
                })

        return dropped, pd.DataFrame(history)

    # ------------------------------------------------------------------
    # TRANSFORM — apply the fixed, stored drop list only
    # ------------------------------------------------------------------
    def transform(self, X):
        df = pd.DataFrame(X).copy()
        return df.drop(columns=[c for c in self.columns_to_drop_ if c in df.columns])

    def get_report(self) -> pd.DataFrame:
        """Inspect which columns were dropped and why — useful for audit logging."""
        return self.report_