
"""
HTTP serving entrypoint for the personal loan classification model.

Run with gunicorn, e.g.:
    gunicorn src.pipelines.serving_app:app --bind 0.0.0.0:8080 --workers 2 --threads 4

Where the champion model (model/, schema/, metadata.json) comes from is
entirely controlled by config (model_source.type: "mlflow" or "s3"),
resolved in src/model_loaders/factory.py.

This module doesn't know or care which source is active -- it just asks
the loader for a LoadedArtifact (model + input_schema + metadata) and
serves requests against whatever it gets back.
"""

import os
import sys
import tempfile
from pathlib import Path
import logging
from datetime import datetime
import threading

import mlflow
import pandas as pd
from flask import Flask, render_template, request, jsonify

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT))


from src.core.config_manager import get_config, CONFIG_PATH
from src.core.schema import partition_by_schema, validate_input_against_schema
from src.model_loaders.factory import get_loader
from src.utils.logger import Logger
from src.utils.schema_to_form import build_form_fields


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# ---------------------------------------------------------------------------
# Champion artifact: loaded once at startup, swappable via /reload.
# ---------------------------------------------------------------------------

class ChampionHolder:
    """Thread-safe holder for the currently-served champion artifact.
    Keeps serving requests against the old artifact while a reload is
    in progress, and only swaps in the new one once it's fully loaded."""

    def __init__(self):
        self._lock = threading.Lock()
        self._artifact = None

    def load(self):
        """Fetches and loads the current champion, then atomically swaps
        it in. Raises if the fetch/load fails -- caller decides what to
        do (crash at startup, or keep serving the old one on /reload)."""
        dst = Path(tempfile.mkdtemp(prefix="champion_"))
        Logger.info(f"Loading champion model into {dst}")
        new_artifact = get_loader().load(dst=dst)
        Logger.info(f"Champion loaded (metadata={new_artifact.metadata})")
        with self._lock:
            self._artifact = new_artifact

    @property
    def artifact(self):
        with self._lock:
            if self._artifact is None:
                raise RuntimeError("Champion model has not been loaded yet.")
            return self._artifact


champion = ChampionHolder()
champion.load()  # fail fast at startup if the champion can't be loaded


# ---------------------------------------------------------------------------
# Request parsing + validation helpers
# ---------------------------------------------------------------------------

def _parse_request_df(req) -> pd.DataFrame:
    """Builds a DataFrame from a CSV upload, a JSON list of records, or a
    single JSON object. Raises ValueError on anything else."""
    if "file" in req.files:
        return pd.read_csv(req.files["file"])

    payload = req.get_json(silent=True)
    if payload is None:
        payload = req.form.to_dict()

    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    raise ValueError("Expected a JSON object, a JSON list of records, or a CSV file upload.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    schema_yaml = champion.artifact.input_schema
    fields = build_form_fields(schema_yaml)
    return render_template("index.html", fields=fields)


@app.get("/ping")
def ping():
    """Lightweight liveness check for HEALTHCHECK / load balancer probes.
    """
    return jsonify({"status": "ok"}), 200


@app.get("/health")
def health():
    """Readiness check -- confirms a champion is actually loaded and
    reports which one."""
    return jsonify({"status": "ok", "champion_metadata": champion.artifact.metadata})


@app.post("/predict")
def predict():
    artifact = champion.artifact

    try:
        df = _parse_request_df(request)
    except ValueError as e:
        return jsonify({"error": "invalid_request", "detail": str(e)}), 400

    if df.empty:
        return jsonify({"error": "invalid_request", "detail": "No records provided."}), 400

    valid_df, rejected_df = partition_by_schema(df, artifact.input_schema)
    if not rejected_df.empty:
        Logger.info("Input failed schema validation; rejecting request.")
        return jsonify({
            "error": "schema_validation_failed",
            "detail": rejected_df.to_dict(orient="records"),
        }), 422

    prediction = artifact.model.predict(valid_df)
    return jsonify({
        "prediction": prediction.tolist(),
        "model_version": artifact.metadata.get("version"),
    })


@app.post("/predict_batch")
def predict_batch():
    artifact = champion.artifact

    try:
        df = _parse_request_df(request)
    except ValueError as e:
        return jsonify({"prediction": [], "error": "invalid_request", "detail": str(e)}), 400

    valid_df, rejected_df = validate_input_against_schema(df, artifact.input_schema)
    if not rejected_df.empty:
        if valid_df.empty:
            return jsonify({
                "prediction": [],
                "error": "schema_validation_failed",
                "rejected": rejected_df.to_dict(orient="records"),
            }), 400
        # Partial batch: predict on the valid rows, surface the rejections too.
        prediction = artifact.model.predict(valid_df)
        return jsonify({
            "prediction": prediction.tolist(),
            "rejected": rejected_df.to_dict(orient="records"),
        }), 207  # Multi-Status: mixed success/failure

    prediction = artifact.model.predict(valid_df)
    return jsonify({"prediction": prediction.tolist()})


@app.post("/reload")
def reload_champion():
    """Re-fetches and swaps in the latest champion without restarting the
    process. Lock this down (internal network / API key) before exposing
    it outside the cluster -- it triggers a full model fetch on demand."""
    try:
        champion.load()
    except Exception as e:
        Logger.info(f"Champion reload failed, keeping current model: {e}")
        return jsonify({"error": "reload_failed", "detail": str(e)}), 500
    return jsonify({"status": "reloaded", "champion_metadata": champion.artifact.metadata})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)