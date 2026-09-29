# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# MLflow tracking server image — mlflow is installed at BUILD time, not on
# every container start. This makes container startup near-instant instead
# of paying a ~30-90s `pip install` cost on every `docker compose up`.
# ---------------------------------------------------------------------------

# ---- build stage -----------------------------------------------------------

FROM python:3.12-slim As builder

RUN pip install --no-cache-dir --prefix /install mlflow==3.13.0

# ---- runtime stage ----------------------------------------------------------

FROM python:3.12-slim As runtime

# Bring in only the installed packages, not pip's build cache/wheels
COPY --from=builder /install /usr/local

# Non-root by default. Additional safety to avoid running by root
# Since permissions on the mount are opened up by init-permissions (chmod a+rwx), 
#this UID doesn't need to match anything on the host -- any non-root UID works.
RUN useradd --create-home --uid 1000 mlflow
USER mlflow

WORKDIR /app/mlflow

## MLFLOW_HOST / MLFLOW_CONTAINER_PORT are resolved at container *runtime* by
# the shell below, from this service's `environment:` block in
# docker-compose.yml -- they are NOT build-time ARGs. Defaults are provided
# via shell parameter expansion (${VAR:- default}) in case they're unset.
CMD ["sh", "-c", "\
    mlflow server \
    --backend-store-uri sqlite:////app/mlflow/mlflow.db \
    --default-artifact-root /app/mlflow/artifacts \
    --host ${MLFLOW_HOST:- 0.0.0.0} \
    --port ${MLFLOW_CONTAINER_PORT:- 5000} \
    --allowed-hosts '*' \
    --cors-allowed-origins '*' \
    --workers ${MLFLOW_WORKERS:- 4} \
"]
