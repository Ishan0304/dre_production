"""Artifact registry and run manifest structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from core.contracts import ArtifactRecord, DatasetDescriptor, RunArtifactType, RunContextRecord


@dataclass(slots=True)
class ArtifactRegistry:
    """Ordered registry of artifacts produced during a pipeline run."""

    artifacts: list[ArtifactRecord] = field(default_factory=list)

    def register(self, record: ArtifactRecord) -> None:
        """Register one artifact and preserve insertion order."""

        self.artifacts.append(record)

    def register_many(self, records: list[ArtifactRecord]) -> None:
        """Register multiple artifacts in the provided order."""

        self.artifacts.extend(records)

    def list_by_type(self, artifact_type: RunArtifactType) -> list[ArtifactRecord]:
        """Return artifacts matching the requested artifact type."""

        return [
            artifact
            for artifact in self.artifacts
            if artifact.artifact_type is artifact_type
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return _to_serializable(asdict(self))


@dataclass(slots=True)
class RunManifest:
    """Structured record of datasets and artifacts for one pipeline run."""

    run_context: RunContextRecord
    datasets: list[DatasetDescriptor] = field(default_factory=list)
    artifact_registry: ArtifactRegistry = field(default_factory=ArtifactRegistry)
    notes: list[str] = field(default_factory=list)

    def add_dataset(self, dataset: DatasetDescriptor) -> None:
        """Add one dataset descriptor to the manifest."""

        self.datasets.append(dataset)

    def add_note(self, note: str) -> None:
        """Add one manifest note."""

        self.notes.append(note)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return _to_serializable(asdict(self))


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    return value
