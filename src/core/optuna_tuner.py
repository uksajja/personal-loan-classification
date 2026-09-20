import mlflow
import optuna
from sklearn.model_selection import cross_val_score

from src.core.model_factory import ModelFactory
from src.core.postprocessor import PostProcessor



class OptunaTuner():
    """Manages the Optuna study optimization loop and syncs trials with MLflow."""

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def __init__(self, train_pipeline, X_train, y_train, preprocessing_config: dict, data_config: dict, model_config: dict, mlflow_manager=None, random_state: int = 42):
        self.X_train, self.y_train = X_train, y_train       
        self.preprocessing_config = preprocessing_config
        self.data_config = data_config
        self.model_config = model_config
        self.mlflow_manager = mlflow_manager
        self.random_state = random_state

        self.model_type = model_config.get('model_type')
        self.hyperparameters = model_config.get('hyperparameters')
        self.model_fine_tuning = model_config.get('fine_tuning') or {}
        self.scoring_metric = model_config.get('scoring_metric')
        self.fine_tuning_method = self.model_fine_tuning.get('method')
        self.param_search_space = self.model_fine_tuning.get('search_space') or {}
        self.direction = self.model_fine_tuning.get('direction', 'maximize')
        self.n_trials = self.model_fine_tuning.get('n_trials', 20)
        self.cv = self.model_fine_tuning.get('cv', 3)

        self.scorer = PostProcessor.custom_scorer(self.scoring_metric)
        self.train_pipeline = train_pipeline

    #--------------------------------------------------------------------------
    # Optuna objective function: defines the search space, trains the model, evaluates it, and logs results to MLflow.
    #--------------------------------------------------------------------------
    def _objective(self, trial: optuna.Trial) -> float:
        # 1. Define search spaces polymorphically based on architecture
        params = self.hyperparameters or {}
        search_space = self.param_search_space
        for param_name, param_range in search_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Continuous or integer range
                if all(isinstance(x, int) for x in param_range):
                    params[param_name] = trial.suggest_int(param_name, *param_range)
                else:
                    params[param_name] = trial.suggest_float(param_name, *param_range)
            elif isinstance(param_range, list):
                # Categorical choices
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            else:
                raise ValueError(f"Unsupported search space format for {param_name}: {param_range}")

        with self.mlflow_manager.start_run(run_name=f"{self.model_type}_optuna_trial_{trial.number}", nested=True):
            model = ModelFactory.get_model(self.model_type, params)
            model_config = self.model_config.copy()
            model_config['hyperparameters'] = params  # Update model config with trial parameters

            pipeline = self.train_pipeline.build_train_pipeline(
                        preprocessing_config=self.preprocessing_config,
                        data_config=self.data_config,
                        model_config=model_config,
                        random_state=self.random_state
                    )
            scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=3, scoring=self.scorer)
            score = scores.mean()

            self.mlflow_manager.set_tags({
                'model_type': self.model_type,
                'fine_tuning_method': self.fine_tuning_method,
                'run_type': 'trial',
            })
            self.mlflow_manager.log_params(params)
            pipeline.fit(self.X_train, self.y_train)
            metrics = PostProcessor.classification_metrics(pipeline, self.X_train, self.y_train, prefix='train')
            self.mlflow_manager.log_metrics(metrics)
            self.mlflow_manager.log_metric('train_score', score)
        
        return score

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def optimize(self, n_trials: int = 50) -> dict:
        """Executes the study loop and saves the winning parameters to disk."""
        study = optuna.create_study(direction=self.direction)
        study.optimize(lambda trial: self._objective(trial), n_trials=self.n_trials)
        #study.optimize(self._objective, n_trials=self.n_trials)

        return study.best_params, study.best_value