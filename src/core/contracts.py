"""Shared typed contracts for pipeline metadata and runtime records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EvidenceStatus(Enum):
    """Status describing how evidence is represented in the source record."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    MISSING = "missing"


class ModalityType(Enum):
    """Clinical data modality used by the pipeline."""

    EHR = "ehr"
    MRI = "mri"
    EEG = "eeg"


class SplitType(Enum):
    """Dataset split designation for modeling and inference workflows."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    INFERENCE = "inference"


class RunArtifactType(Enum):
    """Supported artifact categories emitted by pipeline runs."""

    TABLE = "table"
    PLOT = "plot"
    JSON = "json"
    MODEL = "model"
    LOG = "log"


@dataclass(slots=True)
class ArtifactRecord:
    """Metadata for an artifact produced by a pipeline run."""

    artifact_name: str
    artifact_type: RunArtifactType
    relative_path: str
    description: str | None = None
    created_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class DatasetDescriptor:
    """Metadata describing an input or derived dataset."""

    dataset_name: str
    modality: ModalityType
    source_format: str
    row_count: int | None = None
    patient_count: int | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class TimeWindow:
    """Optional temporal bounds for evidence, cohorts, or reporting periods."""

    start: datetime | None = None
    end: datetime | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class ColumnSpec:
    """Column-level schema expectation for tabular datasets."""

    column_name: str
    dtype_hint: str | None = None
    required: bool = True
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class RunContextRecord:
    """Metadata identifying a reproducible pipeline run context."""

    run_id: str
    project_name: str
    stage_name: str
    timestamp_utc: datetime
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)
