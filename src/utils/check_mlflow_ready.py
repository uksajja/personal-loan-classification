# src/utils/mlflow_ready.py
import time
import requests

def wait_for_mlflow(tracking_uri: str, timeout: int = 10, interval: float = 1.0) -> None:
    """Fail fast with a clear message if MLflow isn't reachable, instead of
    letting the first MLflow API call surface a multi-frame connection traceback."""
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            requests.get(tracking_uri, timeout=2).raise_for_status()
            return
        except Exception as e:
            last_error = e
            time.sleep(interval)
    raise RuntimeError(
        f"Could not reach MLflow server at '{tracking_uri}' after {timeout}s.\n"
        f"Start it first:\n"
        f"  docker compose up -d mlflow init-permissions   (recommended — shared store for local + Docker)\n"
        f"  or: mlflow server --host 0.0.0.0 --port 5000   (standalone local-only)\n"
        f"Last error: {last_error}"
    ) from last_error