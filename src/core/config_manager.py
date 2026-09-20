from functools import lru_cache
import os
import sys
from pathlib import Path
from src.utils.read_yaml import read_yaml
from src.utils.logger import Logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config/config.yaml"

REQUIRED_SECTIONS = [
    "global_config",
    "data_ingestion_config",
    "preprocessing_config",
    "training_config",
    "artifacts_config",
    "inference_config",
    "mlflow_config",
]

@lru_cache(maxsize=None)
def load_config(config_path):    
    config = read_yaml(config_path)

    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(
            f"config.yaml at {config_path} is missing required section(s): {missing}. "
            "Check the file for typos or incomplete sections before running any pipeline."
        )
    return config

def get_config(config_path=CONFIG_PATH) -> dict:
    """Public accessor — use this everywhere instead of importing module-level globals.

    If the environment variable `MLFLOW_TRACKING_URI` is set (for example when
    running in Docker Compose), use it to override the `mlflow_config.tracking_uri`
    value from the YAML so services inside containers can address the MLflow
    server by its service name instead of `localhost`.
    """
    config = load_config(config_path)
    tracking_uri_env = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri_env:
        return config

    # load_config is lru_cache'd, so `config` is a shared object — mutating it
    # in place would leak this override into every other caller, including
    # ones that read load_config() directly. Copy the layers we touch instead.
    config = dict(config)
    config['mlflow_config'] = dict(config.get('mlflow_config', {}))
    config['mlflow_config']['tracking_uri'] = tracking_uri_env

    Logger.info(f"Overriding mlflow_config.tracking_uri from MLFLOW_TRACKING_URI env var: {tracking_uri_env}")
    return config



    """
    Notes:
    config = dict(config) — shallow-copies the top-level mapping, so reassigning config['mlflow_config'] doesn't touch the cached original. 
    Other nested sections (data_ingestion_config, training_config, etc.) still point at the same shared sub-objects, which is fine — you're not mutating those.
    config['mlflow_config'] = dict(config.get('mlflow_config', {})) — copies just that one nested dict before writing tracking_uri into it, so the cached mlflow_config dict is never touched.
    .strip() and the early-return — folds in the whitespace guard from before, and makes the "no override" path a one-liner that returns the cached object untouched (no unnecessary copying when there's nothing to override).
    """



