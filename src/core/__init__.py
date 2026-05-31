"""Shared contracts and runtime primitives for the DRE production pipeline."""

from core.config import ConfigLoader, LoadedConfig, PathConfig
from core.contracts import (
    ArtifactRecord,
    ColumnSpec,
    DatasetDescriptor,
    EvidenceStatus,
    ModalityType,
    RunArtifactType,
    RunContextRecord,
    SplitType,
    TimeWindow,
)

__all__ = [
    "ArtifactRecord",
    "ColumnSpec",
    "ConfigLoader",
    "DatasetDescriptor",
    "EvidenceStatus",
    "LoadedConfig",
    "ModalityType",
    "PathConfig",
    "RunArtifactType",
    "RunContextRecord",
    "SplitType",
    "TimeWindow",
]
