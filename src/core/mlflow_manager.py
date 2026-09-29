import datetime
from pathlib import Path
from urllib.parse import urlparse
import pandera.pandas as pa
import shutil

import json
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
        self.cycle_id = None
    
    def __repr__(self) -> str:
        return (
            f"MLFlowManager(experiment='{self.experiment}', "
            f"cycle='{self.cycle_id}', "
            f"tracking='{self.tracking_uri}')"
        )

    def start_cycle(self) -> str:
        """Call this once, explicitly, at the start of a training cycle -
        assigns a fresh cycle_id that every run in this cycle will share."""
        self.cycle_id = self.new_cycle_id()
        Logger.info(f"Started cycle {self.cycle_id}")
        return self.cycle_id

    def new_cycle_id(self) -> str:
        """Sequence number derived from today's existing run tags - resets
        naturally each day since it only counts cycles tagged with today's
        date prefix. No separate counter file to keep in sync."""
        today_date = datetime.date.today().strftime("%Y-%m-%d")

        runs = self._client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.cycle_id LIKE '{today_date}_%'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            sequence_number = 1
        else:
            last_cycle_id = runs[0].data.tags["cycle_id"]
            last_sequence = int(last_cycle_id.split("_")[1])
            sequence_number = last_sequence + 1

        return f"{today_date}_{sequence_number}"

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
    # Logging — params, metrics, models, artifacts
    # ──────────────────────────────────────────────────────────────────────────
    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float) -> None:
        mlflow.log_metric(key, value)

    def log_metrics(self, metrics: dict) -> None:
        mlflow.log_metrics(metrics)
    
    def log_model_sklearn(self, payload: dict) -> None:
        model_info = mlflow.sklearn.log_model(
            sk_model=payload['model'],
            name=payload['model_name'],
            signature=payload.get('signature'),
            input_example=payload.get('input_example'),
            **payload.get('extra_kwargs', {}),  # optional: forward anything flavor-specific
        )
        mlflow.log_param("model_uri", model_info.model_uri)  # correct URI for MLflow 3
        mlflow.log_param("model_id", model_info.model_id)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log an artifact (file) to the current MLflow run."""
        mlflow.log_artifact(local_path, artifact_path)
  
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
    
    def set_tags(self, tags: dict):
        """Set tags for the current MLflow run."""
        mlflow.set_tags(tags)

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

    def get_model_uri_for_run(self, run_id: str) -> str:
        """Resolves a run_id to its LoggedModel's model_uri — the MLflow 3"""
        logged_models = self._client.search_logged_models(
            experiment_ids=[self._client.get_run(run_id).info.experiment_id]
        )
        matches = [m for m in logged_models if m.source_run_id == run_id]
        if not matches:
            raise ValueError(
                f"No LoggedModel found for run {run_id}. "
                f"Was mlflow.sklearn.log_model(...) actually called in this run?"
            )
        return f"models:/{matches[0].model_id}"
    
    def _has_schema_artifact(self, run_id: str) -> bool:
        """Checks whether a run has a schema/input_schema.yaml artifact,
        without downloading it - used as a gate before selection/registration."""
        try:
            artifacts = self._client.list_artifacts(run_id, path="schema")
            return any(a.path == "schema/input_schema.yaml" for a in artifacts)
        except mlflow.exceptions.MlflowException:
            return False


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
        input_schema = payload.get('input_schema')
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

            if model_logger == 'sklearn':
                self.log_model_sklearn(payload)
            elif callable(model_logger):
                model_logger(**payload)

                    # Schema is data-contract metadata, not flavor-specific — logged
            # once here regardless of which model_logger branch ran above.
            if input_schema is not None:
                schema_path = "input_schema.yaml"
                input_schema.to_yaml(schema_path)
                self.log_artifact(schema_path, artifact_path="schema")
                Path(schema_path).unlink()

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

        # Exclude runs with no schema logged - they can't be safely promoted
        # downstream even if their score is best.
        runs = runs[runs["run_id"].apply(self._has_schema_artifact)]
        if runs.empty:
            raise ValueError(
                f"No candidate runs with both a '{champion_metric}' score and a logged "
                f"schema found in experiment '{self.experiment_name}'."
            )

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

        # Ensure the candidate has a schema artifact — this is the key signal
        # that the model was logged correctly and is loadable
        if not self._has_schema_artifact(candidate_run_id):
            raise ValueError(
                f"Refusing to register run {candidate_run_id} as challenger: "
                f"no input_schema.yaml artifact found for this run."
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
    
    

    def get_registered_model_versions(self, registered_model_name: str, cycle_id: str | None = None):
        """Audit/reporting utility - not used in the per-cycle promotion
        decision, which relies solely on the champion/challenger aliases.
        Optionally scope to one cycle via the cycle_id tag set in
        register_challenger().
        """
        filter_string = f"name='{registered_model_name}'"
        if cycle_id is not None:
            filter_string += f" AND tags.cycle_id = '{cycle_id}'"
        versions = self._client.search_model_versions(filter_string)
        return sorted(versions, key=lambda v: int(v.version), reverse=True)
       
    def load_model_schema_by_run(self, run_id: str):
            Logger.info(f"Loading model from run {run_id}")
            try:
                model_uri = self.get_model_uri_for_run(run_id)  # MLflow 3: resolves run_id -> models:/{model_id}
                model = mlflow.pyfunc.load_model(model_uri)
                schema_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/schema/input_schema.yaml")
            except mlflow.exceptions.MlflowException as e:
                raise RuntimeError(
                    f"Could not load model/schema for run_id '{run_id}'. "
                    f"Was this run fully logged/promoted yet? Original error: {e}"
                ) from e
    
            schema = pa.DataFrameSchema.from_yaml(schema_path)
            Logger.info(f"Model and schema loaded successfully from run {run_id}")
            return model, schema
    
    def load_model_and_schema_by_alias(self, model_name, model_alias):
        """Loads a model by registry alias, plus the input schema logged
        alongside it in the same training run."""
        
        model_uri = f"models:/{model_name}@{model_alias}"
        
        Logger.info(f"Loading model from {model_uri}")
        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except mlflow.exceptions.MlflowException as e:
            raise RuntimeError(
                f"Could not load model at '{model_uri}' "
                f"promoted yet? Original error: {e}"
            ) from e
        Logger.info(f"Model loaded successfully from {model_uri}")

        model_version = self._client.get_model_version_by_alias(model_name, model_alias)
        run_id = model_version.run_id

        schema_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/schema/input_schema.yaml")
        schema = pa.DataFrameSchema.from_yaml(schema_path)

        return model, schema
    
    
    def download_champion_artifacts(self, dst_path="artifacts"):
        """
        Download the champion model (subfolder), its input schema
        (subfolder), and a metadata snapshot (root) to a local directory.

            dst_path/
            |
            └── champion/
                |
                ├── model/            <- model.pkl, MLmodel, conda.yaml, ...
                ├── schema/
                │   └── input_schema.yaml
                └── metadata.json

        Returns (model_local_path, schema_local_path, metadata_local_path).
        """
        dst = Path(dst_path)/"champion"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        try:
            champion_version = self._client.get_model_version_by_alias(
                self.experiment_model_name, self.champion_model_alias
            )
        except mlflow.exceptions.MlflowException as e:
            raise ValueError(
                f"No champion alias found for model '{self.experiment_model_name}'."
            ) from e

        run = self._client.get_run(champion_version.run_id)

        model_uri = f"models:/{self.experiment_model_name}@{self.champion_model_alias}"
        model_local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri, dst_path=str(dst / "model")
        )
        schema_local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{champion_version.run_id}/schema/input_schema.yaml", dst_path=str(dst / "schema")
        )

        metadata = {
            "model_name": self.experiment_model_name,
            "alias": self.champion_model_alias,
            "version": champion_version.version,
            "run_id": champion_version.run_id,
            "cycle_id": run.data.tags.get("cycle_id"),
            "downloaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": run.data.params,
            "metrics": run.data.metrics,
        }
        metadata_local_path = dst / "metadata.json"
        with open(metadata_local_path, "w") as f:
            json.dump(metadata, f, indent=2)

        Logger.info(
            f"Downloaded champion model (version {champion_version.version}, "
            f"run_id={champion_version.run_id}) to {model_local_path}, "
            f"schema to {schema_local_path}, metadata to {metadata_local_path}"
        )

        return model_local_path, schema_local_path, str(metadata_local_path)
        