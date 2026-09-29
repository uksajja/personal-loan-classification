# src/pipelines/inference.py
import os
import sys
from pathlib import Path
import json
import time
import pandas as pd
import mlflow
import pandera.pandas as pa


# Do not change the location. 
# Project root should be established before we import project specific modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


STAGE_NAME = "Inference"

from src.core.config_manager import get_config,  CONFIG_PATH, PROJECT_ROOT
from src.core.schema import validate_input_against_schema
from src.core.mlflow_manager import MLFlowManager
from src.model_loaders.factory import get_loader
from src.model_loaders.local_loader import LocalModelLoader
from src.model_loaders.mlflow_loader import MLflowModelLoader
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
    
    def __init__(self, config: dict | None = None):
        self.config = get_config(CONFIG_PATH)
        self.raw_data_folder = self.config["data_ingestion_config"]["raw_data_folder"]
        
        self.inference_config = self.config["inference_config"]
        self.inference_type = self.inference_config.get("inference_type", "local")  # or "batch"

        if self.inference_type == "local":
            # Load model from local artifacts
            model_loader = LocalModelLoader(self.config)
            dst = model_loader.model_source
            new_artifact = model_loader.load(dst=dst)
            self.model, self.schema, self.metadata =new_artifact.model, new_artifact.input_schema, new_artifact.metadata
            Logger.info(f"Champion loaded (metadata={self.metadata})")
        elif self.inference_type == "mlflow":       # Wait for MLflow server to be ready. If mlflow server is not started, an error will be raised
            #mlflow_tracking_uri = self.config['mlflow_config']['tracking_uri']
            mlflow_tracking_uri = "http://localhost:5000"
            model_name = self.config['mlflow_config']["registered_model_name"]
            model_alias = self.config['mlflow_config']["champion_model_alias"]
            #wait_for_mlflow(tracking_uri=self.config['mlflow_config']['tracking_uri'], timeout=self.config['mlflow_config']['timeout'], interval=self.config['mlflow_config']['interval'])
            
            model_loader = MLflowModelLoader(mlflow_tracking_uri, model_name, model_alias)
            dst = Path("./mlflow")
            new_artifact = model_loader.load(dst=dst)
            self.model, self.schema, self.metadata = new_artifact.model, new_artifact.input_schema, new_artifact.metadata
            Logger.info(f"Champion loaded (metadata={self.metadata})")
            #self.model, self.schema = self.mlflow_manager.load_model_and_schema_by_alias(self.inference_model_name, self.inference_model_alias)
            #Logger.info(f"Model {self.inference_model_name} with alias {self.inference_model_alias} loaded successfully")
        else:
            raise ValueError("Invalid inference type. Must be 'local' or 'mlflow'")

        self.single_inference = self.inference_config.get("single_inference", {})
        self.batch_inference  = self.inference_config.get("batch_inference", {})
        self.batch_output_path = self.batch_inference.get("batch_predictions_path", {}) 
        self.batch_size = self.batch_inference.get("batch_size", 100)

    
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

        valid_df, rejected_df = validate_input_against_schema(df, self.schema)
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

        valid_df, rejected_df = validate_input_against_schema(df, self.schema)

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

