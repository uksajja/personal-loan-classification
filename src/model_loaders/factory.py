# src/core/model_loaders/factory.py

"""
Required env vars:
    MODEL_SOURCE=mlflow
        MLFLOW_TRACKING_URI
 
    MODEL_SOURCE=s3
        CHAMPION_S3_BUCKET
        CHAMPION_S3_PREFIX
"""

import os
from pathlib import Path
from src.core.config_manager import get_config, CONFIG_PATH
from src.model_loaders.base import ModelLoader
from src.model_loaders.local_loader import LocalModelLoader
from src.model_loaders.mlflow_loader import MLflowModelLoader


def _require_env(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError(
            f"Missing required environment variable '{name}' for the "
            f"selected MODEL_SOURCE. Check docker-compose.yml / .env."
        ) from None
    
def get_loader() -> ModelLoader:
    source = os.environ["MODEL_SOURCE"]  # "mlflow" or "s3"
    if(source == "local"):
        return LocalModelLoader()
    elif source == "mlflow":
        mlflow_config = get_config(CONFIG_PATH)["mlflow_config"]
        return MLflowModelLoader(
            tracking_uri=_require_env("MLFLOW_TRACKING_URI"),
            model_name=mlflow_config["registered_model_name"],
            champion_alias=mlflow_config["champion_model_alias"]
        )
    elif(source == "s3"):
        raise NotImplementedError("S3 model loader not implemented yet.")
    else:  
        raise ValueError(f"Unknown MODEL_SOURCE: {source}")