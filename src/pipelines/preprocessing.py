import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

# Do not change the location. 
# Project root should be established before we import project specific modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from src.core.config_manager import get_config, CONFIG_PATH, PROJECT_ROOT
from src.core.data_ingestor import DataIngestor
from src.core.preprocessor  import PreProcessor
from src.utils.data_io import write_batch
from src.utils.logger import Logger



STAGE_NAME = "Data Preprocessing stage"


class PreProcessingPipeline:
    """
    A pipeline class for data preprocessing steps, including handling missing values,
    encoding categorical variables, and feature scaling.
    """
    def __init__(self):
        self.config = get_config(CONFIG_PATH)
        self.random_state = self.config['global_config']['random_state']
        self.data_ingestion_config = self.config['data_ingestion_config']
        data_ingestor = DataIngestor(config=self.data_ingestion_config)
        self.preprocessing_config = self.config['preprocessing_config']

        self.raw_data_folder = self.data_ingestion_config['raw_data_folder']

        self.original_df = data_ingestor.load_data()
    
    def _get_processed_data_folder(self) -> Path:
        folder = Path(self.data_ingestion_config.get("processed_data_folder", "data/processed_data/"))
        return folder.expanduser()

    def _get_supported_processed_files(self) -> list[Path]:
        folder = self._get_processed_data_folder()
        if not os.path.exists(folder) or not os.path.isdir(folder):
            return []
        supported_extensions = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".json", ".h5", ".hdf", ".hdf5"}
        return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in supported_extensions]

    def _processed_data_exists(self) -> bool:
        return bool(self._get_supported_processed_files())

    def _load_processed_data(self) -> pd.DataFrame | None:
        files = sorted(self._get_supported_processed_files(), key=lambda p: p.name.lower())
        if not files:
            return None
        file_path = files[0]
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(file_path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        if suffix == ".json":
            return pd.read_json(file_path)
        if suffix in {".h5", ".hdf", ".hdf5"}:
            return pd.read_hdf(file_path)
        return None

    def _save_raw_data(self, X_train_raw, X_test_raw, y_train_raw, y_test_raw):
        """
        Saves the split datasets to CSV files as specified in the preprocessing_config.

        args:
            df (pd.DataFrame): Input DataFrame with features and target.
            preprocessing_config (dict): Configuration dictionary containing file paths.
        """
        
        try:
            train_features_path = f"{self.raw_data_folder}/X_train_raw.csv"
            test_features_path = f"{self.raw_data_folder}/X_test_raw.csv"
            train_target_path = f"{self.raw_data_folder}/y_train_raw.csv"
            test_target_path = f"{self.raw_data_folder}/y_test_raw.csv"

            X_train_raw.to_csv(train_features_path, index=False)
            X_test_raw.to_csv(test_features_path, index=False)
            y_train_raw.to_csv(train_target_path, index=False)
            y_test_raw.to_csv(test_target_path, index=False)

        except Exception as e:
            print(f"Error saving split data: {e}. Please check the file paths and permissions.")
    
    def _save_processed_data(self, df: pd.DataFrame) -> None:
        folder = self._get_processed_data_folder()
        folder.mkdir(parents=True, exist_ok=True)
        output_file = f"{folder}/Loan_Modeling_Processed_Data.csv"
        write_batch(df, output_file)
        Logger.info(f"Saved processed data to {output_file}")

    def run(self):

        processed_folder = self._get_processed_data_folder()
        if self._processed_data_exists():          
            processed_df = self._load_processed_data()
            Logger.info(f"Processed data already exists in {processed_folder}; skipping preprocessing.")
            return processed_df

        raw_df = self.original_df.copy()

        target = self.preprocessing_config.get('target', {})
        if not target:
            raise ValueError("Target configuration is missing.")
        target_column = target.get('column_name')
        df = PreProcessor.filter_missing_target_rows(raw_df, target_column)

        Logger.info(f"Preparing features and target...")
        X, y, target_transformer = PreProcessor.prepare_features_target(df, self.preprocessing_config)
        Logger.info(f"Splitting data into training and testing sets...")

        X_train_raw, X_test_raw, y_train_raw, y_test_raw = PreProcessor.split_data(X, y, self.preprocessing_config, self.random_state)
              
        self._save_raw_data(X_train_raw, X_test_raw, y_train_raw, y_test_raw)

        preprocess_steps = PreProcessor.build_preprocessing_steps(self.preprocessing_config)

        preprocessed_df = Pipeline(preprocess_steps).fit_transform(X_train_raw, y_train_raw)

        self._save_processed_data(preprocessed_df)
        
        return preprocessed_df

if __name__ == '__main__':
    try:
        Logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PreProcessingPipeline()
        obj.run()
        print(f"df.shape: {obj.original_df.shape}")
        print(f"Info: {obj.original_df.info}")
        Logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        Logger.exception(e)
        raise e