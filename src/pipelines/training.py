import os
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import  make_scorer, accuracy_score, precision_score, recall_score, f1_score, get_scorer
import mlflow
from mlflow.models import infer_signature
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.core.config_manager import get_config, CONFIG_PATH, PROJECT_ROOT
from src.core.data_ingestor import DataIngestor
from src.core.preprocessor import PreProcessor
from src.core.mlflow_manager import MLFlowManager
from src.core.model_factory import ModelFactory
from src.transformers.samplers import SamplerFactory
from src.core.optuna_tuner import OptunaTuner
from src.core.model_exporter import ModelExporter
from src.core.postprocessor import PostProcessor

from src.utils.check_mlflow_ready import wait_for_mlflow
from src.utils.logger import Logger


STAGE_NAME = "Training"

class TrainingPipeline:
    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize the pipeline runner and its pipeline components.

        Args:
            config: Full pipeline configuration. If omitted, it's loaded
                from CONFIG_PATH -- this keeps `TrainingPipeline()` usable
                as a script entry point while still allowing a config to be
                injected directly (e.g. in tests).
        """
        self.config = config if config is not None else get_config(CONFIG_PATH)
        self.global_config = self.config['global_config']
        self.random_state = self.global_config ['random_state']

        self.data_ingestion_config = self.config['data_ingestion_config']
        self.preprocessing_config = self.config['preprocessing_config']
        self.training_config = self.config['training_config']
        self.mlflow_config = self.config['mlflow_config']
        self.artifact_config = self.config['artifacts_config']           

        self.data_ingestor = DataIngestor(self.data_ingestion_config)
        self.mlflow_manager = MLFlowManager(self.mlflow_config)

        self.artifacts_path = self.artifact_config['artifacts_path']
        self.challenger_model_path = self.artifact_config['challenger_model_path']
        self.champion_model_path = self.artifact_config['champion_model_path']
        self.model_save_format = self.artifact_config['model_save_format']

        os.makedirs(Path(self.artifacts_path), exist_ok=True)
        os.makedirs(Path( self.challenger_model_path), exist_ok=True)
        os.makedirs(Path( self.champion_model_path), exist_ok=True)

    def load_data(self):
        # Initialize individual pipeline components
        df = self.data_ingestor.load_data()
        return df
   
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        #filter out rows with missing target values
        target = self.preprocessing_config.get('target', {})
        target_column = target.get('column_name')
        return PreProcessor.filter_missing_target_rows(df, target_column)

    def build_train_pipeline(self, preprocessing_config: dict, data_config: dict, model_config:dict, random_state: int ):
            """
            Assembles preprocessing steps + optional sampler + model into one
            pipeline. Uses imblearn's Pipeline (not sklearn's) whenever a
            sampler is configured, since only imblearn's Pipeline correctly
            no-ops resampling at predict()/transform() time while still
            applying it during fit().
            """
    
            if data_config is None:
                steps = PreProcessor.build_preprocessing_steps(preprocessing_config)
            else:
                steps = PreProcessor.build_resolved_preprocessing_steps(preprocessing_config, data_config)
                   
            sampler = SamplerFactory.build(data_config, random_state=random_state)
    
            model = ModelFactory.get_model(model_config['model_type'], model_config['hyperparameters'])
    
            if sampler is not None:
                steps.append(("sampler", sampler))
                steps.append(("model", model))
                return ImbPipeline(steps)
            else:
                steps.append(("model", model))
                return SkPipeline(steps)

   
    
    def train(self, df: pd.DataFrame):

        # ------------------------------------------------------------------
        # 1. SPLIT FIRST — before anything else is fit
        # ------------------------------------------------------------------
        Logger.info(f"Preparing features and target...")
        X, y, target_transformer = PreProcessor.prepare_features_target(df, self.preprocessing_config)

        Logger.info(f"Splitting data into training and testing sets...")
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = PreProcessor.split_data(X, y, self.preprocessing_config, self.random_state)

        Logger.info(f"X_train.shape: {X_train_raw.shape}    y_train.shape: {y_train_raw.shape}")
        Logger.info(f"X_test.shape: {X_test_raw.shape}    y_test.shape: {y_test_raw.shape}")
        Logger.info(f"Finished data preparation and splitting.")


        for training_config in self.config['training_config']:
            if 'model_name' in training_config:
                model_name = training_config['model_name']
            else:
                model_name = None

            if 'data_config' in training_config:
                data_config = training_config['data_config'] 
            else:
                data_config = None

            if 'model_config' in training_config:
                model_config = training_config['model_config']
            else:
                model_config = None

            Logger.info(f"==============================================================================================")
            Logger.info(f"Training model: {model_name}")
            Logger.info(f"data_config:    {data_config}")
            Logger.info(f"model_config:   {model_config}")
           
            if model_config is None:
                if model_name is None:
                    raise ValueError("No model_config or model_name provided in training_config.")
                else:
                    Logger.error("No model_config provided in training_config.")
                    raise ValueError("No model_config provided in training_config for the specified model: {model_name}.")
                   
            model_type = model_config.get("model_type")
            hyperparameters = model_config.get('hyperparameters', {})
            scoring_metric = model_config.get('scoring_metric', 'f1_score')
            model_fine_tuning = model_config.get('fine_tuning', None)
            fine_tuning_method = model_fine_tuning.get('method', 'none') if model_fine_tuning else None
            search_space = model_fine_tuning.get('search_space', {}) if model_fine_tuning else {}
                       
            Logger.info(f"scoring metric:      {scoring_metric}")
            scorer = PostProcessor.custom_scorer(scoring_metric)

            Logger.info(f"fine_tuning_method:  {fine_tuning_method}")
            Logger.info(f"search space: {search_space}")

            if fine_tuning_method is not None:
                if not search_space:
                    Logger.error("search_space must be provided for optuna fine-tuning.")
                    Logger.info(f"skipping fine-tuning for model: {model_type}")

            if fine_tuning_method == 'optuna':                          

                Logger.info(f"Starting hyperparameter tuning using Optuna...")
                tuner = OptunaTuner(self, X_train_raw, y_train_raw, self.preprocessing_config, data_config, model_config, self.mlflow_manager, self.random_state)
                best_params, train_score = tuner.optimize()
                best_model_config = {}
                best_model_config['model_type'] = model_type
                best_model_config['hyperparameters'] = best_params
                best_pipeline = self.build_train_pipeline(self.preprocessing_config, data_config, best_model_config, self.random_state)
                best_pipeline.fit(X_train_raw, y_train_raw)

                if model_name is not None:                   
                    logged_model_name = model_name
                else:
                    logged_model_name = f"{training_config['model_config']['model_type']}_optuna_best_model"

                logged_pipeline = best_pipeline
                logged_parameters = best_params


            elif fine_tuning_method == 'GridSearchCV':

                Logger.info(f"Starting hyperparameter tuning using GridSearchCV...")              
                cv = model_fine_tuning.get('cv', 5) 

                pipeline = self.build_train_pipeline(self.preprocessing_config, data_config, model_config, self.random_state)
                param_search_space = ModelFactory.get_model_search_space(search_space)

                grid_search = GridSearchCV(estimator=pipeline, param_grid=param_search_space, cv=cv, scoring=scorer)
                grid_search.fit(X_train_raw, y_train_raw)

                if model_name is not None:
                    logged_model_name = model_name
                else:
                    logged_model_name = f"{training_config['model_config']['model_type']}_gridsearchcv_best_model"

                logged_pipeline = grid_search.best_estimator_
                logged_parameters = grid_search.best_params_

            elif fine_tuning_method == 'RandomizedSearchCV':

                Logger.info(f"Starting hyperparameter tuning using RandomizedSearchCV...")
                cv = model_fine_tuning.get('cv', 5)
                n_iter = model_fine_tuning.get('n_iter', 5)

                pipeline = self.build_train_pipeline(self.preprocessing_config, data_config, model_config, self.random_state)
                param_search_space = ModelFactory.get_model_search_space(search_space)

                randomized_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_search_space, cv=cv, scoring=scorer,n_iter=n_iter)
                randomized_search.fit(X_train_raw, y_train_raw)

                #best_estimator_ returns the estimator (pipeline) with the best found parameters
                #while best_params_ returns the parameter setting that gave the best results on the hold out data.

                if model_name is not None:
                    logged_model_name = model_name
                else:
                    logged_model_name = f"{training_config['model_config']['model_type']}_randomizedsearchcv_best_model"

                logged_pipeline = randomized_search.best_estimator_  
                logged_parameters = randomized_search.best_params_

            else:
                 # Train Base Model
                Logger.info(f"---------------------------------------------------------------------------------------")
                Logger.info(f"Training base model: {model_type}")
                Logger.info(f"hyperparameters:     {hyperparameters}")

                pipeline = self.build_train_pipeline(self.preprocessing_config, data_config, model_config, self.random_state)
                pipeline.fit(X_train_raw, y_train_raw)
                
                if model_name is not None:
                    logged_model_name = model_name
                else:
                    logged_model_name = f"{training_config['model_config']['model_type']}_base_model"

                logged_pipeline = pipeline
                logged_parameters = hyperparameters     

            logged_train_score = cross_val_score(logged_pipeline, X_train_raw, y_train_raw, cv=5, scoring=scorer).mean()
            logged_test_score = scorer(logged_pipeline, X_test_raw, y_test_raw)

            logged_train_metrics = PostProcessor.classification_metrics(logged_pipeline, X_train_raw, y_train_raw, prefix='train')
            logged_test_metrics = PostProcessor.classification_metrics(logged_pipeline, X_test_raw, y_test_raw, prefix='test')

            input_example = X_train_raw[:1]
            prediction = logged_pipeline.predict(input_example)
            signature = infer_signature(input_example, prediction)

            payload = {
                'model': logged_pipeline,
                'model_name': logged_model_name,
                'params': logged_parameters,
                'fine_tuning_method': fine_tuning_method,
                'run_type': 'candidate',
                'scoring_metric': scoring_metric,
                'train_score': logged_train_score,
                'test_score': logged_test_score,
                'train_metrics': logged_train_metrics,
                'test_metrics': logged_test_metrics,
                'signature': signature,
                'input_example': input_example,
                'artifact_name': 'model',
                'model_logger': 'sklearn'
            }
            Logger.info(f"Logging model to MLflow: {logged_model_name} | fine_tuning_method: {fine_tuning_method} | scoring_metric: {scoring_metric}")  
            self.mlflow_manager.log_model_to_mlflow(payload)
            
            Logger.info(f"Logging model to artifacts folder as: {self.artifacts_path}/{logged_model_name}.{self.model_save_format}")
            artifact_path = Path(f"{self.artifacts_path}/{logged_model_name}.{self.model_save_format}")
            ModelExporter.save_model(logged_pipeline, artifact_path)
            
            output = {  'model_name': logged_model_name, 
                        "parameters": logged_parameters, 
                        "scoring_metric": scoring_metric, 
                        'train_score': float(logged_train_score),
                        'test_score': float(logged_test_score),
                        'train_metrics': logged_train_metrics,
                        'test_metrics': logged_test_metrics,
                    }
            Logger.info(f"Saving model parameters and scores to {self.artifacts_path}/{logged_model_name}_params_score.yaml")
            Logger.info(f"Model parameters and scores: {output}")
            with open(f"{self.artifacts_path}/{logged_model_name}_params_score.yaml", "w") as f:
                yaml.dump(output, f)
                

    def evaluate_and_promote_cycle_best_model(self, mlflow_manager: MLFlowManager, champion_metric: str='test_score'):
    
        experiment_name = mlflow_manager.experiment_name
        print("Experiment Name:", experiment_name if experiment_name else "NOT FOUND")
        experiment_id = mlflow_manager.experiment_id
        print("Experiment ID:", experiment_id if experiment_id else "NOT FOUND")
        
        all_runs = mlflow.search_runs(experiment_ids=[experiment_id])
        print("Total runs in experiment:", len(all_runs))
        print("Columns:", all_runs.columns.tolist())

        if not all_runs.empty:
            print(all_runs[[c for c in all_runs.columns if 'tag' in c.lower() or 'metric' in c.lower()]])
        
        
        best_run = mlflow_manager.pick_best_run('test_score')
        best_run_id = best_run.get('run_id')
        best_model_name = best_run.get('tags.model_name')

        challenger_version, challenger_score = mlflow_manager.register_challenger(best_run_id, champion_metric)

        challenger_model_uri = f"models:/{mlflow_manager.experiment_model_name}/{challenger_version.version}"
        try:
            challenger_model = mlflow.pyfunc.load_model(challenger_model_uri)
            Logger.info(f"Loaded challenger model successfully from {challenger_model_uri}")
        except Exception as e:
            # Log helpful diagnostics: the model URI, registered version source, and run artifacts
            try:
                src = getattr(challenger_version, 'source', None)
            except Exception:
                src = None
            Logger.info(f"Failed loading challenger model URI={challenger_model_uri}, source={src}: {e}")
            # Try to list artifacts for the run (if available)
            try:
                run_id = getattr(challenger_version, 'run_id', None)
                if run_id:
                    arts = mlflow.artifacts.list_artifacts(f"runs:/{run_id}/")
                    art_paths = [a.path for a in arts]
                    Logger.info(f"Artifacts for run {run_id}: {art_paths}")
            except Exception as e2:
                Logger.info(f"Failed to list run artifacts: {e2}")
            raise
        
        ModelExporter.save_model(challenger_model, Path(f"{self.challenger_model_path}/{best_model_name}.{self.model_save_format}"))

        champion_score, candidate_score, is_promoted = mlflow_manager.promote_if_better(challenger_version, challenger_score, champion_metric='test_score')

        Logger.info(f"==============================================================================================")
        Logger.info(f"Best run ID: {best_run_id} | Best model name: {best_model_name}")
        Logger.info(f"Champion score: {champion_score} | Candidate score: {candidate_score} | Promoted: {is_promoted}")

        Logger.info(
            f"Cycle {mlflow_manager.cycle_id}: challenger_score={challenger_score}, "
            f"champion_score={champion_score}, promoted={is_promoted}"
        )
        Logger.info(f"==============================================================================================")
        # Download the model files directly into your custom folder
        local_path = mlflow_manager.download_champion_artifact(dst_path=self.champion_model_path)

    def run(self):
        Logger.info(f"==============================================================================================")
        Logger.info(f"Loading data for training...")
        df = self.load_data()
        Logger.info(f"df.shape (Ingestion): {df.shape}")
        Logger.info(f"Filtering out rows with missing target values...")
        df = self.prepare_training_data(df)
        Logger.info(f"df.shape (After filtering missing targets): {df.shape}")

        Logger.info(f"==============================================================================================")
        Logger.info(f"Training Models...")
        # Wait for MLflow server to be ready before starting training
        wait_for_mlflow(tracking_uri=self.config['mlflow_config']['tracking_uri'], timeout=self.config['mlflow_config']['timeout'], interval=self.config['mlflow_config']['interval'])
        self.train(df)
        Logger.info(f"Training completed successfully.")
        Logger.info(f"==============================================================================================")
        Logger.info(f"Evaluating and promoting the best model...")
        self.evaluate_and_promote_cycle_best_model(self.mlflow_manager, champion_metric='test_score')

if __name__ == '__main__':
    try:
        Logger.info(f"***********************************************************************************************")
        Logger.info(f">>>>>> STAGE: {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.run()
        Logger.info(f">>>>>> STAGE: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        Logger.exception(e)
        raise e
        
