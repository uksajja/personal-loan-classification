import os
import sys
from pathlib import Path

# Do not change the location. 
# Project root should be established before we import project specific modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from src.core.config_manager import get_config, CONFIG_PATH, PROJECT_ROOT
from src.core.data_ingestor import DataIngestor
from src.utils.logger import Logger

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionPipeline:
    def __init__(self):
        self.config = get_config(CONFIG_PATH)
        self.data_ingestion_config = self.config['data_ingestion_config']
        self.data_ingestor = DataIngestor(self.data_ingestion_config)

    def run(self):
        self.df = self.data_ingestor.load_data()


if __name__ == '__main__':
    try:
        Logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.run()
        print(f"df.shape: {obj.df.shape}")
        print(f"Info: {obj.df.info}")
        Logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        Logger.exception(e)
        raise e