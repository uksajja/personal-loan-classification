import sys
import datetime
import json
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Do not change the location. 
# Project root should be established before we import project specific modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.core.config_manager import get_config,  CONFIG_PATH, PROJECT_ROOT
from src.model_loaders.base import ModelLoader
from src.utils.logger import Logger


class LocalModelLoader(ModelLoader):

    def __init__(self, config: dict | None = None):
        self.config = get_config(CONFIG_PATH)
        self.inference_config = self.config["inference_config"]
        self.model_source = self.inference_config["model_source"]

    def fetch(self, dst: Path) -> Path:
        return Path(dst)

    
