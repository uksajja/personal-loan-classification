# src/pipelines/inference.py
import os
import sys
from pathlib import Path
import json
import time
import pandas as pd
import mlflow


# Do not change the location. 
# Project root should be established before we import project specific modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


STAGE_NAME = "Inference"

from src.core.config_manager import get_config,  CONFIG_PATH, PROJECT_ROOT
from src.core.schema import SCHEMA_DTYPES, validate_input
from src.core.mlflow_manager import MLFlowManager
from src.utils.data_io import read_batch, write_batch
from src.utils.check_mlflow_ready import wait_for_mlflow
from src.utils.logger import Logger  # or `from logger_setup import logger` per your earlier module-level setup


class InferencePipeline:
    """
    Loads the current champion model once (via alias, so it always tracks
    promotions without code changes) and serves both single-record and
    batch predictions. No preprocessing logic lives here at all — the
    loaded pipeline already contains its own fitted imputer/encoder/
    scaler/etc., learned during training and reused unchanged here.
    """
    
    def __init__(self, config: dict | None = None, champion_model_alias: str = "champion"):
        self.config = get_config(CONFIG_PATH)
        self.raw_data_folder = self.config["data_ingestion_config"]["raw_data_folder"]
        # Wait for MLflow server to be ready. If mlflow server is not started, an error will be raised
        wait_for_mlflow(tracking_uri=self.config['mlflow_config']['tracking_uri'], timeout=self.config['mlflow_config']['timeout'], interval=self.config['mlflow_config']['interval'])
        self.mlflow_manager = MLFlowManager(self.config["mlflow_config"])
        self.inference_config = self.config["inference_config"]
        self.champion_model_alias = champion_model_alias

        self.model_uri = f"models:/{self.mlflow_manager.experiment_model_name}@{self.mlflow_manager.champion_model_alias}"
        self.model = self.load_model(self.model_uri)

        self.single_inference = self.inference_config.get("single_inference", {})
        self.batch_inference  = self.inference_config.get("batch_inference", {})
        self.batch_output_path = self.batch_inference.get("batch_predictions_path", {}) 
        self.batch_size = self.batch_inference.get("batch_size", 100)

    # ------------------------------------------------------------------
    def load_model(self, model_uri):
        Logger.info(f"Loading model from {model_uri}")
        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except mlflow.exceptions.MlflowException as e:
            raise RuntimeError(
                f"Could not load model at '{model_uri}' "
                f"promoted yet? Original error: {e}"
            ) from e
        Logger.info(f"Model loaded successfully from {model_uri}")
        return model


    # ------------------------------------------------------------------
    # SINGLE INFERENCE
    # ------------------------------------------------------------------
    def predict_single(self, record: dict) -> dict:
        """
        record: a flat dict of raw feature values, e.g. {"Age": 34, "Income": 82, ...}
        Returns: {"prediction": int, "probability": float, "latency_ms": float}
        """
        
        start = time.perf_counter()

        df = pd.DataFrame([record])
        df = df.astype(SCHEMA_DTYPES)
        valid_df, rejected_df = validate_input(df)
        if not rejected_df.empty:
            Logger.info(f"The input is invalid. Prediction is NOT posssible.")
            return {}

        prediction = self.model.predict(valid_df)
        result = {"prediction": int(prediction[0])}

        # probability is only available if the underlying pipeline's final
        # estimator supports predict_proba; pyfunc wraps this transparently
        # for sklearn pipelines via the unwrapped model when needed
        proba = self._predict_proba_safe(df)
        if proba is not None:
            result["probability"] = float(proba[0])

        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        Logger.info(f"Single inference: input_keys={list(record.keys())}, result={result}")
        return result

    def _predict_proba_safe(self, df: pd.DataFrame):
        try:
            unwrapped = self.model.unwrap_python_model() if hasattr(self.model, "unwrap_python_model") else None
            underlying = unwrapped or self.model._model_impl.sklearn_model  # sklearn flavor internal
            if hasattr(underlying, "predict_proba"):
                return underlying.predict_proba(df)[:, 1]
        except Exception as e:
            Logger.warning(f"Could not compute prediction probability: {e}")
        return None

    # ------------------------------------------------------------------
    # BATCH INFERENCE
    # ------------------------------------------------------------------
    def predict_batch(self, input_file: str, output_file: str, batch_size: int = 100) -> pd.DataFrame:

        start = time.perf_counter()

        df = read_batch(input_file)
        df = df.astype(SCHEMA_DTYPES)
        valid_df, rejected_df = validate_input(df)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
 
        if not rejected_df.empty:
            rejected_df.to_csv(output_file.replace(".csv", "_rejected.csv"), index=False)
    
        Logger.info(f"Batch inference started: {len(valid_df)} valid rows (of {len(df)} total)")
    
        first_write = True
        for start in range(0, len(valid_df), batch_size):
            chunk = valid_df.iloc[start:start + batch_size].copy()
            preds = self.model.predict(chunk)
            proba = self._predict_proba_safe(chunk)
    
            chunk["prediction"] = preds
            chunk["probability"] = proba if proba is not None else [None] * len(chunk)
            chunk.to_csv(output_file, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
    
        Logger.info(f"Batch inference completed: {len(valid_df)} rows written to {output_path}")


if __name__ == "__main__":

    pipeline = InferencePipeline()
   
    # single inference
    single_record_path =  pipeline.single_inference["single_record"]
    with open(single_record_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    features = record["features"]
    actual_target = record["target"]
    result = pipeline.predict_single(features)

    Logger.info(f" Actual target: {actual_target}, Inference result: {result}")
 
    # batch inference
    batch_features_path = pipeline.batch_inference["batch_features"]
    pipeline.predict_batch(f"{batch_features_path}", f"{pipeline.batch_output_path}", batch_size=pipeline.batch_size)

