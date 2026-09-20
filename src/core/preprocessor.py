# preprocessor.py
import pandas as pd
from typing import Any
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split

from src.transformers.target_transformer import TargetTransformer
from src.transformers.column_ops import ColumnDropper, ColumnRenamer, ZipcodeTransformer
from src.transformers.imputers import ImputerFactory
from src.transformers.outliers import IQROutlierClipper
from src.transformers.encoders import EncoderFactory, OneHotEncodingTransformer, OrdinalEncodingTransformer
from src.transformers.samplers import SamplerFactory
from src.transformers.scalers import ScalingTransformer
from src.transformers.multicollinearity import MultiCollinearityHandler


class PreProcessor:
    """
    Builds a fitted preprocessing (+ optional sampling + model) Pipeline
    from preprocessing_config + a training_config entry's data_config.
    Pure composition — assembles independently-defined transformers,
    doesn't implement any transform logic itself.
    """

    def __init__(self):
        pass

    @staticmethod
    def filter_missing_target_rows(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Filters out rows with missing values in the target column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_column (str): Name of the target column.

        Returns:
            pd.DataFrame: DataFrame with rows containing missing target values removed.
        """
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in DataFrame.")
        
        filtered_df = df.dropna(subset=[target_column])
        return filtered_df

    @staticmethod
    def prepare_features_target(df: pd.DataFrame, preprocessing_config: dict) -> pd.DataFrame:
        target_config = preprocessing_config.get('target', {})
        target_column = target_config.get('column_name')
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

        # Initialize the target mapper
        if 'mapping' in target_config:
            target_transformer = TargetTransformer(target_config)
        else:
            target_transformer = None
          
        X = df.drop(columns=[target_column]) #Features Only
        y = df[target_column]                #Target Only
        if target_transformer:
            y = target_transformer.encode(y)  # Apply encoding if a mapping is provided

        return X, y, target_transformer
    
    # ---------------------------------------------------------------------
    # Dataset splitting
    # ---------------------------------------------------------------------
    @staticmethod
    def split_data(X: pd.DataFrame, y: pd.Series, preprocessing_config: dict, random_state: int) -> Any:
        """
        Prepares training and testing datasets by applying a target transformation
        and splitting by test_size fraction.

        args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target Series.
            preprocessing_config (dict): Configuration for preprocessing steps.
            random_state (int): Random state for reproducibility.

        Returns:
            tuple containing:
                - X_train (pd.DataFrame): Training features
                - X_test (pd.DataFrame): Testing features
                - y_train (pd.Series): Training target
                - y_test (pd.Series): Testing target
        """
        
        """Split the dataset, optionally using stratification on the target column."""

        try:
            split_params = preprocessing_config.get('split', {})
            stratify =  split_params.get('stratify', True)
            test_size = split_params.get('test_size', 0.2)
            val_size = split_params.get('val_size', 0.0)
        
        except Exception as e:
            print(f"Error in split_data: {e}. Using default split parameters.")
            test_size = 0.2
            val_size = 0.0
            stratify = True 

        X_train, X_test, y_train, y_test =  train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None,
        )
        return X_train, X_test, y_train, y_test

    @staticmethod
    def build_preprocessing_steps(preprocessing_config: dict) -> list:
        steps = []

        drop_cols_cfg = preprocessing_config.get("drop_columns", None)
        if drop_cols_cfg:
            steps.append(("drop_columns", ColumnDropper(drop_cols_cfg=drop_cols_cfg)))

        rename_cols_cfg = preprocessing_config.get("rename_columns", None)
        if rename_cols_cfg:
            steps.append(("rename", ColumnRenamer(rename_cols_cfg=rename_cols_cfg)))

        zipcode_cfg = preprocessing_config.get("zipcode_config", None)
        if zipcode_cfg:
            steps.append(("zipcode", ZipcodeTransformer(zipcode_config=zipcode_cfg)))

        missing_cfg = preprocessing_config.get("missing_value_treatment", None)
        if missing_cfg:
            steps.append(("imputer", ImputerFactory.build(missing_cfg)))          # factory — different classes

        outlier_cfg = preprocessing_config.get("outlier_treatment", None)
        if outlier_cfg:
            steps.append(("outlier_clip", IQROutlierClipper(**outlier_cfg)))       # direct — one class, internal method switch

        onehot_cfg = preprocessing_config.get("one_hot_encoding", None)
        if onehot_cfg:
            steps.append(("encoder", OneHotEncodingTransformer(**onehot_cfg)))     # direct

        multicollinearity_cfg = preprocessing_config.get("multicollinearity_treatment", None)
        if multicollinearity_cfg and multicollinearity_cfg.get("method", "none") != "none":
            steps.append(("multicollinearity", MultiCollinearityHandler(**multicollinearity_cfg)))  # direct

        scaler_cfg = preprocessing_config.get("scaler", None)
        if scaler_cfg:
            steps.append(("scaler", ScalingTransformer(**scaler_cfg)))             # direct — internally uses ScalerFactory for the sklearn scaler object itself

        return steps

    @staticmethod
    def resolve(step_name: str, preprocessing_config: dict, data_config: dict | None):

        if data_config is None:
            data_config = {}

        excluded = data_config.get("exclude_steps") or []
        if step_name in excluded:
            return None  # skip entirely for this entry

        global_cfg = preprocessing_config.get(step_name)
        override_cfg = data_config.get(step_name)
        
        if global_cfg is None and override_cfg is None:
            return None  # step not configured at all, globally or per-entry
        
        for cfg, source in [
            (global_cfg, "preprocessing_config"),
            (override_cfg, "data_config"),
        ]:
            if cfg is not None and not isinstance(cfg, dict):
                raise TypeError(
                    f"Step '{step_name}' in {source} must be a dict for merge-based resolution, "
                    f"got {type(cfg).__name__}: {cfg!r}."
                )

        # shallow merge: override_cfg's keys win over global_cfg's keys,
        # any key global_cfg has that override_cfg doesn't is kept as-is
        merged = dict(global_cfg or {})
        merged.update(override_cfg or {})
        return merged
    
    @staticmethod
    def build_resolved_preprocessing_steps(preprocessing_config: dict, data_config: dict | None) -> list:

        if data_config is None:
            return PreProcessor.build_preprocessing_steps(preprocessing_config)  # no overrides, just use the global config 

        steps = []
        resolve_step = lambda name: PreProcessor.resolve(name, preprocessing_config, data_config)

        drop_cols_cfg = resolve_step("drop_columns")
        if drop_cols_cfg:
            steps.append(("drop_columns", ColumnDropper(drop_cols_cfg=drop_cols_cfg)))

        rename_cols_cfg = resolve_step ("rename_columns")
        if rename_cols_cfg:
            steps.append(("rename", ColumnRenamer(rename_cols_cfg=rename_cols_cfg)))

        zipcode_cfg = resolve_step("zipcode_config")
        if zipcode_cfg:
            steps.append(("zipcode", ZipcodeTransformer(zipcode_config=zipcode_cfg)))
        
        missing_cfg = resolve_step("missing_value_treatment")
        if missing_cfg:
            steps.append(("imputer", ImputerFactory.build(missing_cfg)))

        outlier_cfg = resolve_step("outlier_treatment")
        if outlier_cfg:
            steps.append(("outlier_clip", IQROutlierClipper(**outlier_cfg)))

        onehot_cfg = resolve_step("one_hot_encoding")
        if onehot_cfg:
            steps.append(("encoder", EncoderFactory.build(onehot_cfg)))

        multicollinearity_cfg = resolve_step("multicollinearity_treatment")
        if multicollinearity_cfg and multicollinearity_cfg.get("method", "none") != "none":
            steps.append(("multicollinearity", MultiCollinearityHandler(**multicollinearity_cfg)))

        scaler_cfg = resolve_step("scaler")
        if scaler_cfg:
            steps.append(("scaler", ScalingTransformer(**scaler_cfg)))


        return steps
    
