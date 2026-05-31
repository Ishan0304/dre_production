"""EEG pipeline orchestration for recording and patient feature datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.artifacts import ArtifactRegistry, RunManifest
from core.contracts import (
    ArtifactRecord,
    DatasetDescriptor,
    ModalityType,
    RunArtifactType,
    RunContextRecord,
)
from datasets.eeg_patient_dataset_builder import EEGPatientDatasetBuilder
from features import EEGFeatureBuilder
from ingestion import EEGDatasetLoadConfig, EEGLoader
from insights import DatasetProfile, DatasetProfiler
from reporting import ManifestWriter


@dataclass(slots=True)
class EEGPipelineInputConfig:
    """Input and output configuration for one EEG pipeline run."""

    dataset_name: str
    dataset_root: str
    output_dir: str
    recording_feature_filename: str = "eeg_recording_features.csv"
    patient_dataset_filename: str = "eeg_patient_dataset.csv"
    profile_filename: str = "eeg_dataset_profile.json"
    error_filename: str = "eeg_feature_errors.csv"
    registry_filename: str = "eeg_artifact_registry.json"
    manifest_filename: str = "eeg_run_manifest.json"

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class EEGPipelineRunResult:
    """Structured metadata returned by one EEG pipeline run."""

    run_context: RunContextRecord
    recording_feature_path: str | None
    patient_dataset_path: str | None
    dataset_profile_path: str | None
    error_table_path: str | None
    artifact_registry_path: str | None
    run_manifest_path: str | None
    recording_row_count: int
    patient_row_count: int
    error_count: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return _to_serializable(asdict(self))


class EEGPipelineRunner:
    """Run EEG discovery, feature extraction, aggregation, and manifest writing."""

    def __init__(
        self,
        loader: EEGLoader | None = None,
        feature_builder: EEGFeatureBuilder | None = None,
        dataset_builder: EEGPatientDatasetBuilder | None = None,
        dataset_profiler: DatasetProfiler | None = None,
        manifest_writer: ManifestWriter | None = None,
    ) -> None:
        self.loader = loader or EEGLoader()
        self.feature_builder = feature_builder or EEGFeatureBuilder()
        self.dataset_builder = dataset_builder or EEGPatientDatasetBuilder()
        self.dataset_profiler = dataset_profiler or DatasetProfiler()
        self.manifest_writer = manifest_writer or ManifestWriter()

    def run(
        self,
        input_config: EEGPipelineInputConfig,
        run_context: RunContextRecord,
    ) -> EEGPipelineRunResult:
        """Execute the EEG feature pipeline."""

        output_dir = Path(input_config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        notes: list[str] = []

        inventory_df, load_result = self.loader.build_file_inventory(
            EEGDatasetLoadConfig(dataset_root=input_config.dataset_root)
        )
        notes.extend(load_result.notes)
        recording_df, error_df = self.feature_builder.build_recording_feature_table(inventory_df)
        patient_df, build_result = self.dataset_builder.build_patient_dataset(
            recording_feature_df=recording_df,
            error_df=error_df,
            dataset_name=input_config.dataset_name,
        )
        notes.extend(build_result.notes)

        recording_path = self._write_csv(recording_df, output_dir / input_config.recording_feature_filename)
        patient_path = self._write_csv(patient_df, output_dir / input_config.patient_dataset_filename)
        profile = self.dataset_profiler.profile_dataframe(
            df=patient_df,
            dataset_name=input_config.dataset_name,
            patient_id_col="patient_id",
        )
        profile_path = self._write_profile(profile, output_dir / input_config.profile_filename)

        error_path: Path | None = None
        if not error_df.empty:
            error_path = self._write_csv(error_df, output_dir / input_config.error_filename)

        registry_path = output_dir / input_config.registry_filename
        manifest_path = output_dir / input_config.manifest_filename
        registry = ArtifactRegistry()
        registry.register(
            self._artifact_record(
                "eeg_recording_features",
                RunArtifactType.TABLE,
                recording_path,
                output_dir,
            )
        )
        registry.register(
            self._artifact_record(
                "eeg_patient_dataset",
                RunArtifactType.TABLE,
                patient_path,
                output_dir,
            )
        )
        registry.register(
            self._artifact_record(
                "eeg_dataset_profile",
                RunArtifactType.JSON,
                profile_path,
                output_dir,
            )
        )
        if error_path is not None:
            registry.register(
                self._artifact_record(
                    "eeg_feature_errors",
                    RunArtifactType.TABLE,
                    error_path,
                    output_dir,
                )
            )
        registry.register(
            self._artifact_record(
                "eeg_artifact_registry",
                RunArtifactType.JSON,
                registry_path,
                output_dir,
            )
        )
        registry.register(
            self._artifact_record(
                "eeg_run_manifest",
                RunArtifactType.JSON,
                manifest_path,
                output_dir,
            )
        )

        descriptor = DatasetDescriptor(
            dataset_name=input_config.dataset_name,
            modality=ModalityType.EEG,
            source_format="csv",
            row_count=len(patient_df),
            patient_count=(
                int(patient_df["patient_id"].nunique())
                if "patient_id" in patient_df.columns
                else len(patient_df)
            ),
            description="Patient-level handcrafted EEG feature dataset.",
            metadata={"dataset_root": str(Path(input_config.dataset_root).resolve())},
        )
        manifest = RunManifest(
            run_context=run_context,
            datasets=[descriptor],
            artifact_registry=registry,
            notes=list(dict.fromkeys(notes)),
        )
        written_registry_path = self.manifest_writer.write_artifact_registry(registry, registry_path)
        written_manifest_path = self.manifest_writer.write_manifest(manifest, manifest_path)

        return EEGPipelineRunResult(
            run_context=run_context,
            recording_feature_path=str(recording_path),
            patient_dataset_path=str(patient_path),
            dataset_profile_path=str(profile_path),
            error_table_path=str(error_path) if error_path else None,
            artifact_registry_path=str(written_registry_path),
            run_manifest_path=str(written_manifest_path),
            recording_row_count=len(recording_df),
            patient_row_count=len(patient_df),
            error_count=len(error_df),
            notes=list(dict.fromkeys(notes)),
        )

    @staticmethod
    def _write_csv(df: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path.resolve()

    @staticmethod
    def _write_profile(profile: DatasetProfile, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_to_serializable(profile.to_dict()), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path.resolve()

    @staticmethod
    def _artifact_record(
        name: str,
        artifact_type: RunArtifactType,
        path: Path,
        output_dir: Path,
    ) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_name=name,
            artifact_type=artifact_type,
            relative_path=path.resolve().relative_to(output_dir).as_posix(),
            description=f"EEG pipeline artifact: {name}",
            created_by="EEGPipelineRunner",
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
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    return value
