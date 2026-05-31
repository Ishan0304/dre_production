"""End-to-end orchestration for the EHR patient dataset pipeline."""

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
from datasets.ehr_patient_dataset_builder import (
    EHRPatientDatasetBuilder,
    PatientDatasetBuildResult,
)
from features import EHRColumnConfig, EHREvidenceBuilder
from ingestion import EHRLoader, TableLoadRequest
from insights import DatasetProfile, DatasetProfiler
from reporting import ManifestWriter


@dataclass(slots=True)
class EHRPipelineInputConfig:
    """Input and output configuration for one EHR pipeline run."""

    dataset_name: str
    diagnoses_request: TableLoadRequest | None
    medications_request: TableLoadRequest | None
    seizure_events_request: TableLoadRequest | None
    output_dir: str
    patient_dataset_filename: str = "ehr_patient_dataset.csv"
    profile_manifest_filename: str = "ehr_dataset_profile.json"
    registry_filename: str = "ehr_artifact_registry.json"
    run_manifest_filename: str = "ehr_run_manifest.json"

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class EHRPipelineRunResult:
    """Structured output metadata for one EHR pipeline run."""

    run_context: RunContextRecord
    patient_dataset_path: str | None
    dataset_profile_path: str | None
    artifact_registry_path: str | None
    run_manifest_path: str | None
    patient_dataset_row_count: int
    likely_dre_positive_count: int
    likely_dre_negative_count: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return _to_serializable(asdict(self))


class EHRPipelineRunner:
    """Run the EHR loading, labeling, profiling, and manifest workflow."""

    def __init__(
        self,
        loader: EHRLoader | None = None,
        dataset_builder: EHRPatientDatasetBuilder | None = None,
        dataset_profiler: DatasetProfiler | None = None,
        manifest_writer: ManifestWriter | None = None,
    ) -> None:
        self.loader = loader or EHRLoader()
        self.dataset_builder = dataset_builder or EHRPatientDatasetBuilder(
            evidence_builder=EHREvidenceBuilder(
                column_config=EHRColumnConfig(
                    diagnosis_time_col="diagnosis_time",
                    medication_start_col="medication_start",
                    seizure_event_time_col="event_time",
                    seizure_event_type_col="event_type",
                )
            )
        )
        self.dataset_profiler = dataset_profiler or DatasetProfiler()
        self.manifest_writer = manifest_writer or ManifestWriter()

    def run(
        self,
        input_config: EHRPipelineInputConfig,
        run_context: RunContextRecord,
    ) -> EHRPipelineRunResult:
        """Execute the EHR pipeline and write structured artifacts."""

        output_dir = Path(input_config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        diagnoses_df, medications_df, seizure_events_df, notes = self.load_input_tables(input_config)
        patient_df, build_result = self.dataset_builder.build_patient_dataset(
            dataset_name=input_config.dataset_name,
            diagnoses_df=diagnoses_df,
            medications_df=medications_df,
            seizure_events_df=seizure_events_df,
        )
        notes.extend(build_result.notes)

        patient_dataset_path = self._write_patient_dataset(
            patient_df,
            output_dir / input_config.patient_dataset_filename,
        )
        dataset_profile = self.dataset_profiler.profile_dataframe(
            df=patient_df,
            dataset_name=input_config.dataset_name,
            patient_id_col="patient_id",
            label_col="likely_dre",
        )
        dataset_profile_path = self._write_profile_json(
            dataset_profile,
            output_dir / input_config.profile_manifest_filename,
        )

        dataset_descriptor = self.build_dataset_descriptor(
            dataset_name=input_config.dataset_name,
            patient_df=patient_df,
        )
        artifact_registry_path = output_dir / input_config.registry_filename
        run_manifest_path = output_dir / input_config.run_manifest_filename

        artifact_registry = ArtifactRegistry()
        artifact_registry.register_many(
            [
                self._artifact_record(
                    artifact_name="patient_dataset",
                    artifact_type=RunArtifactType.TABLE,
                    path=patient_dataset_path,
                    output_dir=output_dir,
                    description="Patient-level EHR dataset.",
                    created_by="EHRPipelineRunner",
                ),
                self._artifact_record(
                    artifact_name="dataset_profile",
                    artifact_type=RunArtifactType.JSON,
                    path=dataset_profile_path,
                    output_dir=output_dir,
                    description="Structured profile of the patient-level EHR dataset.",
                    created_by="EHRPipelineRunner",
                ),
                self._artifact_record(
                    artifact_name="artifact_registry",
                    artifact_type=RunArtifactType.JSON,
                    path=artifact_registry_path,
                    output_dir=output_dir,
                    description="Artifact registry for this run.",
                    created_by="EHRPipelineRunner",
                ),
                self._artifact_record(
                    artifact_name="run_manifest",
                    artifact_type=RunArtifactType.JSON,
                    path=run_manifest_path,
                    output_dir=output_dir,
                    description="Run manifest for this pipeline run.",
                    created_by="EHRPipelineRunner",
                ),
            ]
        )

        run_manifest = self.build_run_manifest(
            run_context=run_context,
            dataset_descriptor=dataset_descriptor,
            artifact_registry=artifact_registry,
            notes=notes,
        )
        written_registry_path = self.manifest_writer.write_artifact_registry(
            artifact_registry,
            artifact_registry_path,
        )
        written_manifest_path = self.manifest_writer.write_manifest(
            run_manifest,
            run_manifest_path,
        )

        return EHRPipelineRunResult(
            run_context=run_context,
            patient_dataset_path=str(patient_dataset_path),
            dataset_profile_path=str(dataset_profile_path),
            artifact_registry_path=str(written_registry_path),
            run_manifest_path=str(written_manifest_path),
            patient_dataset_row_count=build_result.row_count,
            likely_dre_positive_count=build_result.likely_dre_positive_count,
            likely_dre_negative_count=build_result.likely_dre_negative_count,
            notes=list(dict.fromkeys(notes)),
        )

    def load_input_tables(
        self,
        input_config: EHRPipelineInputConfig,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, list[str]]:
        """Load configured input tables and return dataframes plus notes."""

        diagnoses_df, diagnoses_notes = self._load_optional_table(
            "diagnoses",
            input_config.diagnoses_request,
        )
        medications_df, medications_notes = self._load_optional_table(
            "medications",
            input_config.medications_request,
        )
        seizure_events_df, seizure_events_notes = self._load_optional_table(
            "seizure_events",
            input_config.seizure_events_request,
        )

        notes = diagnoses_notes + medications_notes + seizure_events_notes
        return diagnoses_df, medications_df, seizure_events_df, notes

    @staticmethod
    def build_dataset_descriptor(
        dataset_name: str,
        patient_df: pd.DataFrame,
    ) -> DatasetDescriptor:
        """Build a descriptor for the persisted patient-level EHR dataset."""

        patient_count = (
            int(patient_df["patient_id"].nunique(dropna=True))
            if "patient_id" in patient_df.columns
            else len(patient_df)
        )
        return DatasetDescriptor(
            dataset_name=dataset_name,
            modality=ModalityType.EHR,
            source_format="csv",
            row_count=len(patient_df),
            patient_count=patient_count,
            description="Patient-level EHR dataset produced by the EHR pipeline runner.",
            metadata={"label_column": "likely_dre"},
        )

    @staticmethod
    def build_run_manifest(
        run_context: RunContextRecord,
        dataset_descriptor: DatasetDescriptor,
        artifact_registry: ArtifactRegistry,
        notes: list[str],
    ) -> RunManifest:
        """Create a run manifest from descriptors, artifacts, and notes."""

        return RunManifest(
            run_context=run_context,
            datasets=[dataset_descriptor],
            artifact_registry=artifact_registry,
            notes=list(dict.fromkeys(notes)),
        )

    def _load_optional_table(
        self,
        table_name: str,
        request: TableLoadRequest | None,
    ) -> tuple[pd.DataFrame | None, list[str]]:
        if request is None:
            return None, [f"{table_name} table request absent"]

        df, result = self.loader.load_table(request)
        return df, [f"{table_name} table loaded from {result.path}"]

    @staticmethod
    def _write_patient_dataset(df: pd.DataFrame, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path.resolve()

    @staticmethod
    def _write_profile_json(profile: DatasetProfile, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_to_serializable(profile.to_dict()), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path.resolve()

    @staticmethod
    def _artifact_record(
        artifact_name: str,
        artifact_type: RunArtifactType,
        path: Path,
        output_dir: Path,
        description: str,
        created_by: str,
    ) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            relative_path=path.resolve().relative_to(output_dir).as_posix(),
            description=description,
            created_by=created_by,
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
