# src/model_loaders/mlflow_loader.py
"""
Fetches the champion from the MLflow registry and assembles the layout
that ModelLoader.load() expects:

    dst/
    ├── model/                     <- models:/<name>/<version>
    ├── schema/input_schema.yaml   <- runs:/<run_id>/schema
    └── metadata.json
"""

import datetime
import json
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.model_loaders.base import ModelLoader
from src.utils.logger import Logger


class MLflowModelLoader(ModelLoader):
    def __init__(self, tracking_uri: str, model_name: str, champion_alias: str):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.champion_alias = champion_alias

    def fetch(self, dst: Path) -> Path:
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)

        try:
            version = client.get_model_version_by_alias(self.model_name, self.champion_alias)
        except mlflow.exceptions.MlflowException as e:
            raise ValueError(
                f"No '{self.champion_alias}' alias found for model '{self.model_name}'."
            ) from e

        run = client.get_run(version.run_id)

        # Pin to the resolved version so model, schema and metadata all
        # come from the same version even if the alias moves mid-load.
        Logger.info(f"Resolved alias '{self.champion_alias}' to version {version.version}")

        # Model
        Logger.info(f"Downloading model version {version.version} (run_id={version.run_id})")
        model_dir = dst / "model"
        model_dir.mkdir(parents=True, exist_ok=True)  # download_artifacts requires dst_path to exist
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{self.model_name}/{version.version}",
            dst_path=str(model_dir),
        )

        # Schema
        Logger.info("Downloading input schema")
        schema_dir = dst / "schema"
        schema_dir.mkdir(parents=True, exist_ok=True)
        downloaded = Path(mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{version.run_id}/schema/input_schema.yaml", 
            dst_path=str(schema_dir))
        )

        # Metadata
        Logger.info("Writing metadata")
        metadata = {
            "model_name": self.model_name,
            "alias": self.champion_alias,
            "version": version.version,
            "run_id": version.run_id,
            "cycle_id": run.data.tags.get("cycle_id"),
            "downloaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": run.data.params,
            "metrics": run.data.metrics,
        }
        (dst / "metadata.json").write_text(json.dumps(metadata, indent=2))

        Logger.info(f"Fetched champion version {version.version} (run_id={version.run_id}) into {dst}")
        return dst
