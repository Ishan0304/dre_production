"""MRI pipeline orchestration for subject-level feature datasets."""

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
from datasets.mri_subject_dataset_builder import MRISubjectDatasetBuilder
from features import MRIFeatureBuilder
from ingestion import MRIDatasetLoadConfig, MRILoader
from insights import DatasetProfile, DatasetProfiler
from reporting import ManifestWriter


@dataclass(slots=True)
class MRIPipelineInputConfig:
    """Input and output configuration for one MRI pipeline run."""

    dataset_name: str
    dataset_root: str
    output_dir: str
    subject_dataset_filename: str = "mri_subject_dataset.csv"
    profile_filename: str = "mri_dataset_profile.json"
    error_filename: str = "mri_feature_errors.csv"
    registry_filename: str = "mri_artifact_registry.json"
    manifest_filename: str = "mri_run_manifest.json"

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class MRIPipelineRunResult:
    """Structured metadata returned by one MRI pipeline run."""

    run_context: RunContextRecord
    subject_dataset_path: str | None
    dataset_profile_path: str | None
    error_table_path: str | None
    artifact_registry_path: str | None
    run_manifest_path: str | None
    subject_row_count: int
    error_count: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return _to_serializable(asdict(self))


class MRIPipelineRunner:
    """Run MRI loading, feature extraction, profiling, and manifest writing."""

    def __init__(
        self,
        loader: MRILoader | None = None,
        feature_builder: MRIFeatureBuilder | None = None,
        dataset_builder: MRISubjectDatasetBuilder | None = None,
        dataset_profiler: DatasetProfiler | None = None,
        manifest_writer: ManifestWriter | None = None,
    ) -> None:
        self.loader = loader or MRILoader()
        self.feature_builder = feature_builder or MRIFeatureBuilder()
        self.dataset_builder = dataset_builder or MRISubjectDatasetBuilder()
        self.dataset_profiler = dataset_profiler or DatasetProfiler()
        self.manifest_writer = manifest_writer or ManifestWriter()

    def run(
        self,
        input_config: MRIPipelineInputConfig,
        run_context: RunContextRecord,
    ) -> MRIPipelineRunResult:
        """Execute the MRI subject-level feature pipeline."""

        output_dir = Path(input_config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        notes: list[str] = []

        participants_df, inventory_df, load_result = self.loader.scan_dataset(
            MRIDatasetLoadConfig(dataset_root=input_config.dataset_root)
        )
        notes.extend(load_result.notes)
        feature_df, error_df = self.feature_builder.build_feature_table(inventory_df, participants_df)
        subject_df, build_result = self.dataset_builder.build_subject_dataset(
            feature_df=feature_df,
            error_df=error_df,
            dataset_name=input_config.dataset_name,
        )
        notes.extend(build_result.notes)

        subject_dataset_path = self._write_csv(subject_df, output_dir / input_config.subject_dataset_filename)
        profile = self.dataset_profiler.profile_dataframe(
            df=subject_df,
            dataset_name=input_config.dataset_name,
            patient_id_col="subject_id",
        )
        profile_path = self._write_profile(profile, output_dir / input_config.profile_filename)

        error_table_path: Path | None = None
        if not error_df.empty:
            error_table_path = self._write_csv(error_df, output_dir / input_config.error_filename)

        registry_path = output_dir / input_config.registry_filename
        manifest_path = output_dir / input_config.manifest_filename
        registry = ArtifactRegistry()
        registry.register(
            self._artifact_record(
                "mri_subject_dataset",
                RunArtifactType.TABLE,
                subject_dataset_path,
                output_dir,
            )
        )
        registry.register(
            self._artifact_record(
                "mri_dataset_profile",
                RunArtifactType.JSON,
                profile_path,
                output_dir,
            )
        )
        if error_table_path is not None:
            registry.register(
                self._artifact_record(
                    "mri_feature_errors",
                    RunArtifactType.TABLE,
                    error_table_path,
                    output_dir,
                )
            )
        registry.register(
            self._artifact_record(
                "mri_artifact_registry",
                RunArtifactType.JSON,
                registry_path,
                output_dir,
            )
        )
        registry.register(
            self._artifact_record(
                "mri_run_manifest",
                RunArtifactType.JSON,
                manifest_path,
                output_dir,
            )
        )

        descriptor = DatasetDescriptor(
            dataset_name=input_config.dataset_name,
            modality=ModalityType.MRI,
            source_format="csv",
            row_count=len(subject_df),
            patient_count=(
                int(subject_df["subject_id"].nunique())
                if "subject_id" in subject_df.columns
                else len(subject_df)
            ),
            description="Subject-level handcrafted MRI feature dataset.",
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

        return MRIPipelineRunResult(
            run_context=run_context,
            subject_dataset_path=str(subject_dataset_path),
            dataset_profile_path=str(profile_path),
            error_table_path=str(error_table_path) if error_table_path else None,
            artifact_registry_path=str(written_registry_path),
            run_manifest_path=str(written_manifest_path),
            subject_row_count=len(subject_df),
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
            description=f"MRI pipeline artifact: {name}",
            created_by="MRIPipelineRunner",
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
