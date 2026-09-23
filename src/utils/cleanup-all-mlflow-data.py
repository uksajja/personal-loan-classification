"""
Safe MLflow cleanup: checks existence at each level before attempting
deletion, so re-running this script (or running it when some things
have already been cleaned up) doesn't error out partway through.
"""
 
import logging
 
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
 
logger = logging.getLogger(__name__)
 
 
def cleanup_experiment_runs(client: MlflowClient, experiment_name: str) -> None:
    """Soft-delete all runs under an experiment, if the experiment exists."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.info(f"Experiment '{experiment_name}' does not exist - nothing to clean up.")
        return
 
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,  # skip runs already soft-deleted
        max_results=50000,
    )
    if not runs:
        logger.info(f"Experiment '{experiment_name}' has no active runs to delete.")
        return
 
    for run in runs:
        client.delete_run(run.info.run_id)
    logger.info(f"Deleted {len(runs)} run(s) from experiment '{experiment_name}'.")
 
 
def cleanup_experiment(client: MlflowClient, experiment_name: str, delete_experiment: bool = False) -> None:
    """Cleans up runs, and optionally the experiment itself."""
    cleanup_experiment_runs(client, experiment_name)
 
    if not delete_experiment:
        return
 
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.info(f"Experiment '{experiment_name}' already gone - nothing further to delete.")
        return
    if experiment.lifecycle_stage == "deleted":
        logger.info(f"Experiment '{experiment_name}' is already soft-deleted.")
        return
 
    client.delete_experiment(experiment.experiment_id)
    logger.info(f"Deleted experiment '{experiment_name}'.")
 
 
def cleanup_registered_model(client: MlflowClient, model_name: str) -> None:
    """Deletes all versions of a registered model, then the registered
    model itself, if it exists. This is a hard delete - no undo."""
    try:
        client.get_registered_model(model_name)
    except MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e.error_code):
            logger.info(f"Registered model '{model_name}' does not exist - nothing to clean up.")
            return
        raise  # unexpected error - don't swallow it
 
    versions = client.search_model_versions(f"name='{model_name}'")
    if versions:
        for v in versions:
            client.delete_model_version(name=model_name, version=v.version)
        logger.info(f"Deleted {len(versions)} version(s) of model '{model_name}'.")
    else:
        logger.info(f"Registered model '{model_name}' has no versions.")
 
    client.delete_registered_model(name=model_name)
    logger.info(f"Deleted registered model '{model_name}'.")
 
 
def full_cleanup(
    experiment_name: str,
    model_name: str | None = None,
    delete_experiment: bool = False,
    delete_registered_model: bool = False,
) -> None:
    """One entry point for a full pre-training reset. Each piece is
    optional and independently existence-checked."""
    client = MlflowClient()
 
    cleanup_experiment(client, experiment_name, delete_experiment=delete_experiment)
 
    if delete_registered_model:
        if model_name is None:
            raise ValueError("model_name is required when delete_registered_model=True")
        cleanup_registered_model(client, model_name)
 
 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    full_cleanup(
        experiment_name="PersonalLoan_Classification",
        model_name="PersonalLoan_Classification_Model",
        delete_experiment=False,        # set True to also remove the experiment itself
        delete_registered_model=False,  # set True to also wipe the model registry entry (hard delete)
    )
 
