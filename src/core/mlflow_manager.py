import datetime
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow.tracking import MlflowClient
from src.utils.logger import Logger


class MLFlowManager:

    # ─────────────────────────────────────────────────────────────────────────────
    # Constructor — call once at process startup
    # ─────────────────────────────────────────────────────────────────────────────
    def __init__(self, mlflow_config: dict):
        if not isinstance(mlflow_config, dict):
            raise TypeError("mlflow_config must be a dictionary")
        try:
            self.raw_tracking_uri = mlflow_config['tracking_uri']
            self.experiment_name = mlflow_config['experiment_name']
        except KeyError as e:
            raise TypeError(f"config must provide {e} or be a dict")

        self.training_date = datetime.date.today().isoformat()
        self.cycle_id = f"cycle_{self.training_date}"
                
        self.experiment_model_name = mlflow_config.get('registered_model_name', f"{self.experiment_name}_Model")
        self.challenger_model_alias = mlflow_config.get('challenger_model_alias', 'challenger') 
        self.champion_model_alias = mlflow_config.get('champion_model_alias', 'champion')   

        # Initialise MLflow once at construction time
        self.tracking_uri = self._normalize_tracking_uri(self.raw_tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment = mlflow.set_experiment(self.experiment_name)
        self.experiment_id = self.experiment.experiment_id

        # One shared client for all registry operations
        self._client = MlflowClient(tracking_uri=self.tracking_uri)
    
    def __repr__(self) -> str:
        return (
            f"MLFlowManager(experiment='{self.experiment}', "
            f"cycle='{self.cycle_id}', "
            f"tracking='{self.tracking_uri}')"
        )

    @staticmethod
    def _normalize_tracking_uri(uri: str) -> str:
        parsed = urlparse(uri)
        if not parsed.scheme or len(parsed.scheme) == 1:
            return Path(uri).resolve().as_uri()
        return uri
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Standard tag set — every run gets these so cross-run queries work uniformly
    # ─────────────────────────────────────────────────────────────────────────────
    def base_tags(self, model_config) -> dict:
        """
        Returns the standard tag dict that every trainer must set.
        Consistent tags are what make search_runs() filters reliable
        across modules.
        """
        return {
            "model_name"   : model_config.model_name,
            "model_type"   : model_config.model_type,
            "scoring_metric" : model_config.scoring_metric,
            "fine_tuning_method" : model_config.fine_tuning_method,

            "run_type"     : "candidate",   # marks this as a promotable run
            "training_date": self.training_date,
            "cycle_id"     : self.cycle_id,      # groups all runs from one session
        }
    
    # ──────────────────────────────────────────────────────────────────────────
    # Logging — params, metrics, models
    # ──────────────────────────────────────────────────────────────────────────
    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float) -> None:
        mlflow.log_metric(key, value)

    def log_metrics(self, metrics: dict) -> None:
        mlflow.log_metrics(metrics)
    
    def log_model_sklearn(self, payload: dict) -> None:
        mlflow.sklearn.log_model(
            sk_model=payload['model'],
            name=payload['model_name'],
            signature=payload.get('signature'),
            input_example=payload.get('input_example'),
            **payload.get('extra_kwargs', {}),  # optional: forward anything flavor-specific
        )
  
    def start_run(self, run_name: str, nested: bool = False):
        """Start an MLflow run with the given name."""
        return mlflow.start_run(run_name=run_name, nested=nested)

    def _build_tags(self, payload: dict, parent_run_id: str = "") -> dict:
        tags = dict(payload.get('tags', {}))
        tags.update({
            'model_name': payload.get('model_name') or payload.get('name') or '',
            'model_type': payload.get('model_type', 'unknown'),
            'fine_tuning_method': payload.get('fine_tuning_method', 'base_model'),
            'run_type': payload.get('run_type', 'candidate'),
            'scoring_metric': payload.get('scoring_metric') or '',
            'parent_run_id': parent_run_id,
            'cycle_id': payload.get('cycle_id', self.cycle_id)
        })
        return tags

    def _get_run_value(self, run, key: str, default=None):
        """Extract a value from an MLflow run row, dict, or object in a tolerant way."""
        if run is None:
            return default

        if hasattr(run, "get"):
            value = run.get(key, default)
            if value is not None:
                return value

        if hasattr(run, "data"):
            value = getattr(run.data, key, default)
            if value is not None:
                return value

        if hasattr(run, "__getitem__"):
            try:
                value = run[key]
                if value is not None:
                    return value
            except (KeyError, TypeError, IndexError):
                return default

        return default
    
    def set_tags(self, tags: dict):
        """Set tags for the current MLflow run."""
        mlflow.set_tags(tags)


    def log_model_to_mlflow(self, payload: dict):
        """Log a trained model to MLflow using a payload-based contract."""
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")        

        # --- required identifiers (fail loudly if missing, not silently None) ---
        model = payload['model']
        model_name = payload['model_name']
        model_logger = payload.get('model_logger')

        # --- optional / defaulted fields ---
        params = payload.get('params', {})
        signature = payload.get('signature')
        input_example = payload.get('input_example')
        artifact_name = payload.get('artifact_name', 'model')  # renamed from `name` — this is the artifact folder name inside the run, not the model's identifier
        train_score = payload.get('train_score')
        test_score = payload.get('test_score')
        run_type = payload.get('run_type', 'candidate')
        train_metrics = payload.get('train_metrics', {})
        test_metrics = payload.get('test_metrics', {})
       
        model_logger_kwargs = payload.get('model_logger_kwargs', {})

        with self.start_run(run_name=model_name, nested=False) as active_run:
            parent_run_id = ""
            if active_run and getattr(active_run, "info", None) and getattr(active_run.info, "parent_run_id", None):
                parent_run_id = active_run.info.parent_run_id or ""

            tags = self._build_tags(payload, parent_run_id)
            self.set_tags(tags)
            self.log_params(params)

            if train_score is not None:
                self.log_metric('train_score', train_score)
            if test_score is not None:
                self.log_metric('test_score', test_score)

            # keys should be prefixed to avoid collisions between train/test metrics of the same name (e.g. both dicts having "accuracy")
            # if they are not already prefixed, use f'train_{k}' instead of f'{k}'
            if train_metrics:
                self.log_metrics({f'{k}': v for k, v in train_metrics.items()})
            if test_metrics:
                self.log_metrics({f'{k}': v for k, v in test_metrics.items()})

            if callable(model_logger):
                model_logger(**payload)
            elif model_logger == 'sklearn':
                self.log_model_sklearn(payload)


        return {
            'model_name': model_name,
            'train_score': train_score,
            'test_score': test_score,
            'run_type': run_type,
        }
    
    def pick_best_run(self, champion_metric: str = "test_score"):
        """Pick the best model from the MLflow experiment based on the scoring metric."""
        metric_column = f"metrics.{champion_metric}"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=(f"tags.run_type = 'candidate' AND tags.cycle_id = '{self.cycle_id}'"),
            order_by=[f"{metric_column} DESC"],
        )
        if runs.empty:
            raise ValueError(f"No candidate runs found in experiment '{self.experiment_name}'.")

        best_run = runs.loc[runs[metric_column].idxmax()]
        return best_run
    
    def register_challenger(self, candidate_run_id, champion_metric='test_score'):
        """
        Register the cycle's best candidate model regardless of outcome,
        tag it with cycle metadata, and alias it as 'challenger'.
        Returns the registered ModelVersion and its score.
        """
        candidate_run = mlflow.get_run(candidate_run_id)
        candidate_score = candidate_run.data.metrics.get(champion_metric)

        if candidate_score is None:
            raise ValueError(
                f"Candidate run {candidate_run_id} has no metric '{champion_metric}' logged."
            )

        # Register the candidate regardless of outcome
        # MLflow 3.x: models are LoggedModel entities, not run artifacts —
        # "runs:/<run_id>/model" no longer resolves to real content. Find the
        # LoggedModel this run actually produced instead.
        logged_models = mlflow.search_logged_models(
            experiment_ids=[self.experiment_id],
            output_format="list",
        )
        matches = [m for m in logged_models if m.source_run_id == candidate_run_id]
        if not matches:
            raise ValueError(
                f"No LoggedModel found for run {candidate_run_id}. "
                f"Was mlflow.sklearn.log_model(...) actually called in this run?"
            )
        model_uri = f"models:/{matches[0].model_id}"
        new_version = mlflow.register_model(model_uri, self.experiment_model_name)

        # Tag the registered version with cycle metadata for traceability
        self._client.set_model_version_tag(
            self.experiment_model_name, new_version.version, "cycle_id", self.cycle_id
        )
        self._client.set_model_version_tag(
            self.experiment_model_name, new_version.version, champion_metric, str(candidate_score)
        )
        # Always alias the latest cycle's best candidate as 'challenger'
        self._client.set_registered_model_alias(
            self.experiment_model_name, self.challenger_model_alias, new_version.version
        )

        return new_version, candidate_score
    
    def promote_if_better(self, challenger_version, challenger_score, champion_metric='test_score'):
        """
        Compare the candidate's score against the current champion's score
        (if one exists) and promote the candidate to 'champion' if it wins.
        Returns (champion_score, candidate_score, is_promoted).
        """
        try:
            champion_version = self._client.get_model_version_by_alias(
                self.experiment_model_name, self.champion_model_alias
            )
            champion_run = mlflow.get_run(champion_version.run_id)
            champion_score = champion_run.data.metrics.get(champion_metric)
            if champion_score is None:
                raise ValueError(
                    f"Champion run {champion_version.run_id} has no metric '{champion_metric}' logged."
                )
        except mlflow.exceptions.MlflowException:
            champion_score = None  # no champion registered yet

        is_promoted = champion_score is None or challenger_score > champion_score

        if is_promoted:
            self._client.set_registered_model_alias(
                self.experiment_model_name, self.champion_model_alias, challenger_version.version
            )
        return champion_score, challenger_score, is_promoted
    
    

    def get_registered_model_versions(self, registered_model_name: str):
        """Get all versions of a registered model."""
        return self._client.get_latest_versions(registered_model_name)
    
    def download_champion_artifact(self, dst_path="champion_model_path"):
        """
        Download the model artifact currently aliased as 'champion'
        to a local directory. Returns the local path.
        """
        try:
            champion_version = self._client.get_model_version_by_alias(
                self.experiment_model_name, self.champion_model_alias
            )
        except mlflow.exceptions.MlflowException as e:
            raise ValueError(
                f"No champion alias found for model '{self.experiment_model_name}'."
            ) from e

        model_uri = f"models:/{self.experiment_model_name}@{self.champion_model_alias}"
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri, dst_path=dst_path
        )

        Logger.info(
            f"Downloaded champion model (version {champion_version.version}, "
            f"run_id={champion_version.run_id}) to {local_path}"
        ) if hasattr(self, "_logger") else None

        return local_path
    
    def load_model_from_run(self, run_id, artifact_path="model", flavor="pyfunc"):
        """
        Load a model object directly from a run's artifacts.
        flavor: 'pyfunc' (generic) or a specific flavor module name like 'sklearn'.
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        loader = getattr(mlflow, flavor)
        return loader.load_model(model_uri)
    
    
    """
    #Load Models

    model_uri = f"runs:/{best_run_id}/model"   # 'model' = the artifact_path you used when logging
    model = mlflow.pyfunc.load_model(model_uri)


    model = mlflow.sklearn.load_model(model_uri)   # or mlflow.statsmodels, etc.


    #If best_run_id has already gone through register_challenger, you can load via the alias instead of the run:
    model = mlflow.pyfunc.load_model(f"models:/{self.experiment_model_name}@challenger")
    # or
    model = mlflow.sklearn.load_model(f"models:/{self.experiment_model_name}@challenger")


    candidate_version.version    # e.g. '7' — model version number
    candidate_version.run_id     # the run_id that produced it
    candidate_version.source     # the artifact URI mlflow registered from, e.g. 'runs:/<run_id>/model'
    
    
    """
        