# src/core/model_loaders/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
import mlflow
import pandera as pa


@dataclass
class LoadedArtifact:
    model: object
    input_schema: pa.DataFrameSchema
    metadata: dict


class ModelLoader(ABC):
    """Fetches the champion/ directory (model/, schema/, metadata.json)
    from wherever it lives, then loads it into memory. Subclasses only
    need to implement `fetch` — everything else is shared."""

    @abstractmethod
    def fetch(self, dst: Path) -> Path:
        """Downloads/copies champion/ to dst. Returns the local path
        containing model/, schema/, and metadata.json."""

    def load(self, dst: Path) -> LoadedArtifact:
        artifact_dir = self.fetch(dst)
        model = mlflow.pyfunc.load_model(str(artifact_dir / "model"))
        input_schema = pa.DataFrameSchema.from_yaml(
            artifact_dir / "schema" / "input_schema.yaml"
        )
        metadata = json.loads((artifact_dir / "metadata.json").read_text())
        return LoadedArtifact(model=model, input_schema=input_schema, metadata=metadata)