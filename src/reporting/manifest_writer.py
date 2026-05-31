"""JSON writers for run manifests and artifact registries."""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from core.artifacts import ArtifactRegistry, RunManifest
from core.contracts import DatasetDescriptor


class ManifestWriter:
    """Write manifest-related structures to deterministic JSON files."""

    def write_manifest(self, manifest: RunManifest, output_path: str | Path) -> Path:
        """Write a run manifest to JSON and return the resolved path."""

        path = self._prepare_output_path(output_path)
        self._write_json(manifest.to_dict(), path)
        return path

    def write_artifact_registry(
        self,
        registry: ArtifactRegistry,
        output_path: str | Path,
    ) -> Path:
        """Write an artifact registry to JSON and return the resolved path."""

        path = self._prepare_output_path(output_path)
        self._write_json(registry.to_dict(), path)
        return path

    def write_dataset_descriptors(
        self,
        datasets: list[DatasetDescriptor],
        output_path: str | Path,
    ) -> Path:
        """Write dataset descriptors to JSON and return the resolved path."""

        path = self._prepare_output_path(output_path)
        self._write_json(
            [_to_serializable(dataset.to_dict()) for dataset in datasets],
            path,
        )
        return path

    @staticmethod
    def _prepare_output_path(output_path: str | Path) -> Path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_json(payload: object, output_path: Path) -> None:
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


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
    return value
