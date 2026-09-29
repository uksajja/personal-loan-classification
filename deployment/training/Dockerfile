# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Training image
# Build context MUST be the project root, e.g.:
#   docker build -f deployment/training/Dockerfile \
#     -t ${PROJECT_NAME}-training:latest .
#
# Permissions on mounted volumes (data/, artifacts/, mlflow/) are handled by
# the `init-permissions` service in docker-compose.yml (chmod a+rwx), so this
# image runs as a fixed, arbitrary non-root user -- no host UID/GID matching
# needed, and the image builds identically on every machine.
#
# Readiness on mlflow is handled by compose's `depends_on: condition:
# service_healthy`, not by this image, so no extra readiness tooling
# (curl, wait loops) is baked in here.
# ---------------------------------------------------------------------------

# ---- build stage -----------------------------------------------------------

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/train.txt ./requirements/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements/train.txt


# ---- runtime stage ----------------------------------------------------------

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local

COPY config ./config
COPY src ./src

# Notebooks are typically dev-time only. Uncomment if the training
# entrypoint genuinely executes something from this directory at runtime;
# otherwise leave excluded to keep the image smaller.

# COPY notebooks ./notebooks

# arbitrary non-root user -- safe regardless of host UID because
# init-permissions makes bind-mounted paths world-writable (chmod a+rwx).
# chown here covers the baked-in config/ and src/ only.
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appuser /app

# Training pipeline entrypoint module — override via --build-arg for a
# different project's pipeline without touching this file.
ARG TRAINING_MODULE=src.pipelines.training

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TRAINING_MODULE=${TRAINING_MODULE}

USER appuser

ENTRYPOINT ["/bin/sh", "-c",  "exec python -m ${TRAINING_MODULE}"]