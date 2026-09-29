# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Personal Loan  Classifier — Serving image
# Build context MUST be the project root, e.g.:
#   docker build -f deployment/serving/Dockerfile -t personal-loan-classifier-seving:latest .
# ---------------------------------------------------------------------------

# ---- Stage 1: build wheels in a full image (has a C compiler) ----
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/serve.txt ./requirements/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements/serve.txt

# ---- Stage 2: slim runtime image ----
FROM python:3.12-slim AS runtime

COPY --from=builder /install /usr/local


WORKDIR /app

# Only what inference actually needs: config, shared src/ (the Flask app + template are inside src/pipelines)
COPY config ./config
COPY src ./src


RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

USER appuser

# SERVING_CONTAINER_PORT is set at runtime via docker-compose.yml's
# `environment:` block, same value used in `ports:` -- single source of
# truth, no hardcoded port here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"SERVING_CONTAINER_PORT\"]}/ping')"]

# --chdir keeps templates/ and the relative "logs/" path resolving the same way
# they would when the original `python app.py` is run from deployment/serving.
CMD ["/bin/sh", "-c", "exec gunicorn src.pipelines.serving_app:app --bind 0.0.0.0:${SERVING_CONTAINER_PORT} --workers 2 --threads 2 --timeout 120"]
