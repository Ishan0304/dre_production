"""End-to-end multimodal pipeline orchestration."""

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
from datasets.eeg_pipeline_runner import (
    EEGPipelineInputConfig,
    EEGPipelineRunner,
    EEGPipelineRunResult,
)
from datasets.ehr_pipeline_runner import (
    EHRPipelineInputConfig,
    EHRPipelineRunner,
    EHRPipelineRunResult,
)
from datasets.mri_pipeline_runner import (
    MRIPipelineInputConfig,
    MRIPipelineRunner,
    MRIPipelineRunResult,
)
from datasets.multimodal_fusion_dataset_builder import (
    MultimodalFusionDatasetBuilder,
    MultimodalFusionResult,
)
from insights import DatasetProfile, DatasetProfiler
from modeling import MultimodalBaselinePipeline, MultimodalTrainingResult
from normalization import IdentityLinkageConfig, IdentityLinker
from reporting import ManifestWriter, RunSummaryWriter


@dataclass(slots=True)
class MultimodalPipelineInputConfig:
    """Input and output configuration for one multimodal pipeline run."""

    dataset_name: str
    output_dir: str
    ehr_input_config: EHRPipelineInputConfig | None = None
    mri_input_config: MRIPipelineInputConfig | None = None
    eeg_input_config: EEGPipelineInputConfig | None = None
    run_multimodal_model: bool = True
    fused_dataset_filename: str = "multimodal_fused_dataset.csv"
    fused_profile_filename: str = "multimodal_fused_profile.json"
    registry_filename: str = "multimodal_artifact_registry.json"
    manifest_filename: str = "multimodal_run_manifest.json"
    reporting_bundle_filename: str = "multimodal_reporting_bundle.json"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class MultimodalPipelineRunResult:
    """Structured metadata returned by one multimodal orchestration run."""

    run_context: RunContextRecord
    fused_dataset_path: str | None
    fused_profile_path: str | None
    artifact_registry_path: str | None
    run_manifest_path: str | None
    reporting_bundle_path: str | None
    multimodal_model_path: str | None
    multimodal_metrics_path: str | None
    fused_row_count: int
    patients_with_ehr: int
    patients_with_mri: int
    patients_with_eeg: int
    patients_with_all_modalities: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return _to_serializable(asdict(self))


class MultimodalPipelineRunner:
    """Orchestrate modality pipelines, fusion, profiling, reporting, and modeling."""

    def __init__(
        self,
        ehr_runner: EHRPipelineRunner | None = None,
        mri_runner: MRIPipelineRunner | None = None,
        eeg_runner: EEGPipelineRunner | None = None,
        fusion_builder: MultimodalFusionDatasetBuilder | None = None,
        dataset_profiler: DatasetProfiler | None = None,
        multimodal_model_pipeline: MultimodalBaselinePipeline | None = None,
        identity_linker: IdentityLinker | None = None,
        manifest_writer: ManifestWriter | None = None,
        run_summary_writer: RunSummaryWriter | None = None,
    ) -> None:
        self.ehr_runner = ehr_runner or EHRPipelineRunner()
        self.mri_runner = mri_runner or MRIPipelineRunner()
        self.eeg_runner = eeg_runner or EEGPipelineRunner()
        self.fusion_builder = fusion_builder or MultimodalFusionDatasetBuilder()
        self.dataset_profiler = dataset_profiler or DatasetProfiler()
        self.multimodal_model_pipeline = (
            multimodal_model_pipeline or MultimodalBaselinePipeline()
        )
        self.identity_linker = identity_linker or IdentityLinker()
        self.manifest_writer = manifest_writer or ManifestWriter()
        self.run_summary_writer = run_summary_writer or RunSummaryWriter()

    def run(
        self,
        input_config: MultimodalPipelineInputConfig,
        run_context: RunContextRecord,
    ) -> MultimodalPipelineRunResult:
        """Execute the end-to-end multimodal orchestration path."""

        self._validate_has_modality_config(input_config)
        output_dir = Path(input_config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        notes: list[str] = []
        if input_config.notes:
            notes.append(input_config.notes)

        (
            _ehr_result,
            _mri_result,
            _eeg_result,
            modality_dfs,
            modality_notes,
        ) = self.run_modality_pipelines(input_config, run_context)
        notes.extend(modality_notes)
        self._validate_has_usable_modality_data(modality_dfs)

        fused_df, fusion_result, profile, fused_notes = self.build_fused_outputs(
            dataset_name=input_config.dataset_name,
            ehr_df=modality_dfs["ehr_df"],
            mri_df=modality_dfs["mri_df"],
            eeg_df=modality_dfs["eeg_df"],
        )
        notes.extend(fused_notes)
        notes.extend(fusion_result.notes)

        fused_dataset_path = self._write_csv(
            fused_df,
            output_dir / input_config.fused_dataset_filename,
        )
        fused_profile_path = self._write_profile_json(
            profile,
            output_dir / input_config.fused_profile_filename,
        )

        model_result: MultimodalTrainingResult | None = None
        if input_config.run_multimodal_model:
            self._ensure_model_output_dir(output_dir)
            model_result = self.multimodal_model_pipeline.run(fused_df)
            notes.extend(model_result.notes)
        else:
            notes.append("multimodal baseline model run was skipped")

        registry_path = output_dir / input_config.registry_filename
        manifest_path = output_dir / input_config.manifest_filename
        reporting_bundle_path = output_dir / input_config.reporting_bundle_filename
        registry = self._build_artifact_registry(
            output_dir=output_dir,
            fused_dataset_path=fused_dataset_path,
            fused_profile_path=fused_profile_path,
            registry_path=registry_path,
            manifest_path=manifest_path,
            reporting_bundle_path=reporting_bundle_path,
            model_result=model_result,
        )
        descriptor = self.build_dataset_descriptor(input_config.dataset_name, fused_df)
        manifest = RunManifest(
            run_context=run_context,
            datasets=[descriptor],
            artifact_registry=registry,
            notes=list(dict.fromkeys(notes)),
        )

        written_registry_path = self.manifest_writer.write_artifact_registry(
            registry,
            registry_path,
        )
        written_manifest_path = self.manifest_writer.write_manifest(
            manifest,
            manifest_path,
        )
        bundle = self.build_reporting_bundle(
            input_config=input_config,
            run_context=run_context,
            fusion_result=fusion_result,
            profile=profile,
            model_result=model_result,
            artifact_paths={
                "fused_dataset": str(fused_dataset_path),
                "fused_profile": str(fused_profile_path),
                "artifact_registry": str(written_registry_path),
                "run_manifest": str(written_manifest_path),
            },
            notes=list(dict.fromkeys(notes)),
        )
        written_reporting_bundle_path = self.run_summary_writer.write_reporting_bundle(
            bundle,
            reporting_bundle_path,
        )

        return MultimodalPipelineRunResult(
            run_context=run_context,
            fused_dataset_path=str(fused_dataset_path),
            fused_profile_path=str(fused_profile_path),
            artifact_registry_path=str(written_registry_path),
            run_manifest_path=str(written_manifest_path),
            reporting_bundle_path=str(written_reporting_bundle_path),
            multimodal_model_path=model_result.model_path if model_result else None,
            multimodal_metrics_path=model_result.metrics_path if model_result else None,
            fused_row_count=len(fused_df),
            patients_with_ehr=fusion_result.patients_with_ehr,
            patients_with_mri=fusion_result.patients_with_mri,
            patients_with_eeg=fusion_result.patients_with_eeg,
            patients_with_all_modalities=fusion_result.patients_with_all_modalities,
            notes=list(dict.fromkeys(notes)),
        )

    def run_modality_pipelines(
        self,
        input_config: MultimodalPipelineInputConfig,
        run_context: RunContextRecord,
    ) -> tuple[
        EHRPipelineRunResult | None,
        MRIPipelineRunResult | None,
        EEGPipelineRunResult | None,
        dict[str, pd.DataFrame | None],
        list[str],
    ]:
        """Run configured modality pipelines and load their patient-level outputs."""

        notes: list[str] = []
        ehr_result: EHRPipelineRunResult | None = None
        mri_result: MRIPipelineRunResult | None = None
        eeg_result: EEGPipelineRunResult | None = None
        modality_dfs: dict[str, pd.DataFrame | None] = {
            "ehr_df": None,
            "mri_df": None,
            "eeg_df": None,
        }

        if input_config.ehr_input_config is None:
            notes.append("EHR input config was not provided")
        else:
            ehr_result = self.ehr_runner.run(input_config.ehr_input_config, run_context)
            notes.extend(ehr_result.notes)
            modality_dfs["ehr_df"] = self._load_csv_if_exists(
                ehr_result.patient_dataset_path,
                "EHR patient dataset",
                notes,
            )

        if input_config.mri_input_config is None:
            notes.append("MRI input config was not provided")
        else:
            mri_result = self.mri_runner.run(input_config.mri_input_config, run_context)
            notes.extend(mri_result.notes)
            mri_df = self._load_csv_if_exists(
                mri_result.subject_dataset_path,
                "MRI subject dataset",
                notes,
            )
            modality_dfs["mri_df"] = self._link_mri_patient_id(mri_df, notes)

        if input_config.eeg_input_config is None:
            notes.append("EEG input config was not provided")
        else:
            eeg_result = self.eeg_runner.run(input_config.eeg_input_config, run_context)
            notes.extend(eeg_result.notes)
            modality_dfs["eeg_df"] = self._load_csv_if_exists(
                eeg_result.patient_dataset_path,
                "EEG patient dataset",
                notes,
            )

        return ehr_result, mri_result, eeg_result, modality_dfs, list(dict.fromkeys(notes))

    def build_fused_outputs(
        self,
        dataset_name: str,
        ehr_df: pd.DataFrame | None,
        mri_df: pd.DataFrame | None,
        eeg_df: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, MultimodalFusionResult, DatasetProfile, list[str]]:
        """Build and profile a fused multimodal patient-level dataset."""

        fused_df, fusion_result = self.fusion_builder.build_fused_dataset(
            dataset_name=dataset_name,
            ehr_df=ehr_df,
            mri_df=mri_df,
            eeg_df=eeg_df,
        )
        profile = self.dataset_profiler.profile_dataframe(
            df=fused_df,
            dataset_name=dataset_name,
            patient_id_col="patient_id",
            label_col="ehr_likely_dre" if "ehr_likely_dre" in fused_df.columns else None,
        )
        notes = ["fused multimodal dataset was built and profiled"]
        return fused_df, fusion_result, profile, notes

    @staticmethod
    def build_dataset_descriptor(
        dataset_name: str,
        fused_df: pd.DataFrame,
    ) -> DatasetDescriptor:
        """Build a descriptor for the persisted fused multimodal dataset."""

        patient_count = (
            int(fused_df["patient_id"].nunique(dropna=True))
            if "patient_id" in fused_df.columns
            else len(fused_df)
        )
        return DatasetDescriptor(
            dataset_name=dataset_name,
            modality=ModalityType.EHR,
            source_format="csv",
            row_count=len(fused_df),
            patient_count=patient_count,
            description="Patient-level fused EHR, MRI, and EEG feature dataset.",
            metadata={"modality_detail": "multimodal"},
        )

    def build_reporting_bundle(
        self,
        input_config: MultimodalPipelineInputConfig,
        run_context: RunContextRecord,
        fusion_result: MultimodalFusionResult,
        profile: DatasetProfile,
        model_result: MultimodalTrainingResult | None,
        artifact_paths: dict[str, str],
        notes: list[str],
    ) -> Any:
        """Build a concise reporting bundle for a multimodal run."""

        pipeline_summary = {
            "run_id": run_context.run_id,
            "project_name": run_context.project_name,
            "stage_name": run_context.stage_name,
            "dataset_name": input_config.dataset_name,
            "fused_row_count": fusion_result.row_count,
            "patient_count": fusion_result.patient_count,
            "patients_with_ehr": fusion_result.patients_with_ehr,
            "patients_with_mri": fusion_result.patients_with_mri,
            "patients_with_eeg": fusion_result.patients_with_eeg,
            "patients_with_all_modalities": fusion_result.patients_with_all_modalities,
            "artifact_paths": artifact_paths,
            "notes": list(dict.fromkeys(notes)),
        }
        dataset_summary = self.run_summary_writer.build_dataset_profile_summary(profile)
        model_summary = (
            self.run_summary_writer.build_model_summary(model_result)
            if model_result is not None
            else None
        )
        return self.run_summary_writer.build_reporting_bundle(
            pipeline_summary=pipeline_summary,
            model_summary=model_summary,
            dataset_profile_summary=dataset_summary,
            notes=list(dict.fromkeys(notes)),
        )

    @staticmethod
    def _validate_has_modality_config(input_config: MultimodalPipelineInputConfig) -> None:
        if not any(
            [
                input_config.ehr_input_config,
                input_config.mri_input_config,
                input_config.eeg_input_config,
            ]
        ):
            raise ValueError("At least one modality input config must be provided.")

    @staticmethod
    def _validate_has_usable_modality_data(
        modality_dfs: dict[str, pd.DataFrame | None],
    ) -> None:
        usable = [
            df
            for df in modality_dfs.values()
            if df is not None and not df.empty
        ]
        if not usable:
            raise ValueError("No usable patient-level modality outputs were produced.")

    @staticmethod
    def _load_csv_if_exists(
        path_value: str | None,
        artifact_label: str,
        notes: list[str],
    ) -> pd.DataFrame | None:
        if path_value is None:
            notes.append(f"{artifact_label} path was not produced")
            return None
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"{artifact_label} path does not exist: {path}")
        df = pd.read_csv(path)
        if df.empty:
            notes.append(f"{artifact_label} was empty")
        return df

    def _link_mri_patient_id(
        self,
        df: pd.DataFrame | None,
        notes: list[str],
    ) -> pd.DataFrame | None:
        if df is None or "patient_id" in df.columns or "subject_id" not in df.columns:
            return df
        linked_df, linkage_result = self.identity_linker.link_dataframe(
            df=df,
            config=IdentityLinkageConfig(
                source_id_col="subject_id",
                output_patient_id_col="patient_id",
                normalization_mode="subject_to_patient",
                keep_original_id_col=True,
                notes="MRI subject identifiers linked through identity linkage layer",
            ),
            source_modality="mri",
        )
        notes.extend(linkage_result.notes)
        return linked_df

    @staticmethod
    def _write_csv(df: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path.resolve()

    @staticmethod
    def _write_profile_json(profile: DatasetProfile, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_to_serializable(profile.to_dict()), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path.resolve()

    def _ensure_model_output_dir(self, output_dir: Path) -> None:
        if self.multimodal_model_pipeline.config.output_dir is None:
            self.multimodal_model_pipeline.config.output_dir = str(output_dir)

    def _build_artifact_registry(
        self,
        output_dir: Path,
        fused_dataset_path: Path,
        fused_profile_path: Path,
        registry_path: Path,
        manifest_path: Path,
        reporting_bundle_path: Path,
        model_result: MultimodalTrainingResult | None,
    ) -> ArtifactRegistry:
        registry = ArtifactRegistry()
        registry.register_many(
            [
                self._artifact_record(
                    "multimodal_fused_dataset",
                    RunArtifactType.TABLE,
                    fused_dataset_path,
                    output_dir,
                    "Fused patient-level multimodal dataset.",
                ),
                self._artifact_record(
                    "multimodal_fused_profile",
                    RunArtifactType.JSON,
                    fused_profile_path,
                    output_dir,
                    "Structured profile of the fused multimodal dataset.",
                ),
                self._artifact_record(
                    "multimodal_artifact_registry",
                    RunArtifactType.JSON,
                    registry_path,
                    output_dir,
                    "Artifact registry for the multimodal orchestration run.",
                ),
                self._artifact_record(
                    "multimodal_run_manifest",
                    RunArtifactType.JSON,
                    manifest_path,
                    output_dir,
                    "Run manifest for the multimodal orchestration run.",
                ),
                self._artifact_record(
                    "multimodal_reporting_bundle",
                    RunArtifactType.JSON,
                    reporting_bundle_path,
                    output_dir,
                    "Reporting bundle for the multimodal orchestration run.",
                ),
            ]
        )
        if model_result and model_result.model_path:
            registry.register(
                self._artifact_record(
                    "multimodal_baseline_model",
                    RunArtifactType.MODEL,
                    Path(model_result.model_path),
                    output_dir,
                    "Fitted multimodal baseline classifier.",
                )
            )
        if model_result and model_result.metrics_path:
            registry.register(
                self._artifact_record(
                    "multimodal_baseline_metrics",
                    RunArtifactType.JSON,
                    Path(model_result.metrics_path),
                    output_dir,
                    "Multimodal baseline model metrics.",
                )
            )
        return registry

    @staticmethod
    def _artifact_record(
        artifact_name: str,
        artifact_type: RunArtifactType,
        path: Path,
        output_dir: Path,
        description: str,
    ) -> ArtifactRecord:
        resolved_path = path.resolve()
        resolved_output_dir = output_dir.resolve()
        try:
            relative_path = resolved_path.relative_to(resolved_output_dir).as_posix()
        except ValueError:
            relative_path = resolved_path.as_posix()
        return ArtifactRecord(
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            relative_path=relative_path,
            description=description,
            created_by="MultimodalPipelineRunner",
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
