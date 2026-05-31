import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from core.contracts import RunContextRecord
from datasets import (
    EEGPipelineRunResult,
    EHRPipelineRunResult,
    MRIPipelineRunResult,
    MultimodalPipelineInputConfig,
    MultimodalPipelineRunner,
)
from modeling import MultimodalLeakageCheckResult, MultimodalTrainingResult


class FakeEHRRunner:
    def run(self, input_config, run_context):
        output_dir = Path(input_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "ehr_patient_dataset.csv"
        pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3", "p4"],
                "likely_dre": [0, 1, 0, 1],
                "ehr_safe_feature": [0.1, 0.9, 0.2, 0.8],
            }
        ).to_csv(path, index=False)
        return EHRPipelineRunResult(
            run_context=run_context,
            patient_dataset_path=str(path),
            dataset_profile_path=None,
            artifact_registry_path=None,
            run_manifest_path=None,
            patient_dataset_row_count=4,
            likely_dre_positive_count=2,
            likely_dre_negative_count=2,
            notes=["EHR fake run complete"],
        )


class FakeMRIRunner:
    def run(self, input_config, run_context):
        output_dir = Path(input_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "mri_subject_dataset.csv"
        pd.DataFrame(
            {
                "subject_id": ["p1", "p3"],
                "lesion_score": [0.4, 0.7],
            }
        ).to_csv(path, index=False)
        return MRIPipelineRunResult(
            run_context=run_context,
            subject_dataset_path=str(path),
            dataset_profile_path=None,
            error_table_path=None,
            artifact_registry_path=None,
            run_manifest_path=None,
            subject_row_count=2,
            error_count=0,
            notes=["MRI fake run complete"],
        )


class FakeEEGRunner:
    def run(self, input_config, run_context):
        output_dir = Path(input_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "eeg_patient_dataset.csv"
        pd.DataFrame(
            {
                "patient_id": ["p2", "p4"],
                "bandpower_alpha_mean": [1.2, 1.8],
            }
        ).to_csv(path, index=False)
        return EEGPipelineRunResult(
            run_context=run_context,
            recording_feature_path=None,
            patient_dataset_path=str(path),
            dataset_profile_path=None,
            error_table_path=None,
            artifact_registry_path=None,
            run_manifest_path=None,
            recording_row_count=2,
            patient_row_count=2,
            error_count=0,
            notes=["EEG fake run complete"],
        )


class EmptyEHRRunner:
    def run(self, input_config, run_context):
        output_dir = Path(input_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "empty_ehr_patient_dataset.csv"
        pd.DataFrame(columns=["patient_id"]).to_csv(path, index=False)
        return EHRPipelineRunResult(
            run_context=run_context,
            patient_dataset_path=str(path),
            dataset_profile_path=None,
            artifact_registry_path=None,
            run_manifest_path=None,
            patient_dataset_row_count=0,
            likely_dre_positive_count=0,
            likely_dre_negative_count=0,
            notes=["EHR empty fake run complete"],
        )


class FakeMultimodalModelPipeline:
    def __init__(self) -> None:
        self.config = type("Config", (), {"output_dir": None})()

    def run(self, df):
        output_dir = Path(self.config.output_dir)
        model_path = output_dir / "fake_multimodal_model.joblib"
        metrics_path = output_dir / "fake_multimodal_metrics.json"
        model_path.write_text("model", encoding="utf-8")
        metrics_path.write_text('{"test": {"accuracy": 1.0}}', encoding="utf-8")
        return MultimodalTrainingResult(
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            train_row_count=max(len(df) - 2, 1),
            val_row_count=1,
            test_row_count=1,
            feature_cols_used=["mri_lesion_score"],
            leakage_check=MultimodalLeakageCheckResult(
                candidate_feature_cols=["mri_lesion_score"],
                excluded_feature_cols=[],
                excluded_for_leakage=["ehr_likely_dre"],
                excluded_for_identifier=["patient_id"],
                excluded_for_non_numeric=[],
                excluded_for_config=[],
                notes=[],
            ),
            modality_flag_cols_used=["has_mri"],
            metrics={"test": {"accuracy": 1.0}},
            notes=["fake model run complete"],
        )


def _run_context() -> RunContextRecord:
    return RunContextRecord(
        run_id="run-mm-001",
        project_name="dre_production",
        stage_name="multimodal_pipeline",
        timestamp_utc=datetime(2026, 5, 30, tzinfo=UTC),
    )


def _input_config(tmp_path, run_model: bool = True) -> MultimodalPipelineInputConfig:
    return MultimodalPipelineInputConfig(
        dataset_name="multimodal_dataset",
        output_dir=str(tmp_path / "multimodal"),
        ehr_input_config=type(
            "EHRConfig",
            (),
            {"output_dir": str(tmp_path / "ehr")},
        )(),
        mri_input_config=type(
            "MRIConfig",
            (),
            {"output_dir": str(tmp_path / "mri")},
        )(),
        eeg_input_config=type(
            "EEGConfig",
            (),
            {"output_dir": str(tmp_path / "eeg")},
        )(),
        run_multimodal_model=run_model,
    )


def test_multimodal_pipeline_happy_path_writes_outputs(tmp_path) -> None:
    runner = MultimodalPipelineRunner(
        ehr_runner=FakeEHRRunner(),
        mri_runner=FakeMRIRunner(),
        eeg_runner=FakeEEGRunner(),
        multimodal_model_pipeline=FakeMultimodalModelPipeline(),
    )

    result = runner.run(_input_config(tmp_path), _run_context())

    assert result.fused_row_count == 4
    assert result.patients_with_ehr == 4
    assert result.patients_with_mri == 2
    assert result.patients_with_eeg == 2
    assert result.multimodal_model_path is not None
    assert Path(result.fused_dataset_path).exists()
    assert Path(result.fused_profile_path).exists()


def test_multimodal_pipeline_succeeds_with_missing_eeg(tmp_path) -> None:
    config = _input_config(tmp_path, run_model=False)
    config.eeg_input_config = None
    runner = MultimodalPipelineRunner(
        ehr_runner=FakeEHRRunner(),
        mri_runner=FakeMRIRunner(),
        eeg_runner=FakeEEGRunner(),
    )

    result = runner.run(config, _run_context())

    assert result.patients_with_eeg == 0
    assert "EEG input config was not provided" in result.notes


def test_multimodal_pipeline_raises_without_modality_configs(tmp_path) -> None:
    config = MultimodalPipelineInputConfig(
        dataset_name="empty",
        output_dir=str(tmp_path),
        run_multimodal_model=False,
    )

    with pytest.raises(ValueError, match="At least one modality input config"):
        MultimodalPipelineRunner().run(config, _run_context())


def test_multimodal_pipeline_raises_without_usable_outputs(tmp_path) -> None:
    config = MultimodalPipelineInputConfig(
        dataset_name="empty",
        output_dir=str(tmp_path / "multimodal"),
        ehr_input_config=type(
            "EHRConfig",
            (),
            {"output_dir": str(tmp_path / "ehr")},
        )(),
        run_multimodal_model=False,
    )
    runner = MultimodalPipelineRunner(ehr_runner=EmptyEHRRunner())

    with pytest.raises(ValueError, match="No usable patient-level modality outputs"):
        runner.run(config, _run_context())


def test_multimodal_pipeline_artifact_registry_contains_expected_artifacts(tmp_path) -> None:
    runner = MultimodalPipelineRunner(
        ehr_runner=FakeEHRRunner(),
        mri_runner=FakeMRIRunner(),
        eeg_runner=FakeEEGRunner(),
        multimodal_model_pipeline=FakeMultimodalModelPipeline(),
    )

    result = runner.run(_input_config(tmp_path), _run_context())
    registry = json.loads(Path(result.artifact_registry_path).read_text(encoding="utf-8"))
    names = {artifact["artifact_name"] for artifact in registry["artifacts"]}

    assert "multimodal_fused_dataset" in names
    assert "multimodal_reporting_bundle" in names
    assert "multimodal_baseline_model" in names
    assert "multimodal_baseline_metrics" in names


def test_multimodal_pipeline_reporting_bundle_is_valid_json(tmp_path) -> None:
    runner = MultimodalPipelineRunner(
        ehr_runner=FakeEHRRunner(),
        mri_runner=FakeMRIRunner(),
        eeg_runner=FakeEEGRunner(),
        multimodal_model_pipeline=FakeMultimodalModelPipeline(),
    )

    result = runner.run(_input_config(tmp_path), _run_context())
    bundle = json.loads(Path(result.reporting_bundle_path).read_text(encoding="utf-8"))

    assert bundle["pipeline_summary"]["dataset_name"] == "multimodal_dataset"
    assert bundle["dataset_profile_summary"]["row_count"] == 4
    assert bundle["model_summary"]["model_path"] == result.multimodal_model_path


def test_multimodal_pipeline_can_skip_model(tmp_path) -> None:
    runner = MultimodalPipelineRunner(
        ehr_runner=FakeEHRRunner(),
        mri_runner=FakeMRIRunner(),
        eeg_runner=FakeEEGRunner(),
    )

    result = runner.run(_input_config(tmp_path, run_model=False), _run_context())
    registry = json.loads(Path(result.artifact_registry_path).read_text(encoding="utf-8"))
    names = {artifact["artifact_name"] for artifact in registry["artifacts"]}

    assert result.multimodal_model_path is None
    assert "multimodal_baseline_model" not in names
    assert "multimodal baseline model run was skipped" in result.notes
