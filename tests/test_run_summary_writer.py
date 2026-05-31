import json
from datetime import UTC, datetime

from core.contracts import RunContextRecord
from datasets import EHRPipelineRunResult
from insights import (
    ClassBalanceRecord,
    DatasetProfile,
    MissingnessRecord,
    SplitBalanceRecord,
)
from modeling import BaselineTrainingResult, LeakageCheckResult
from reporting import RunSummaryWriter


def _run_context() -> RunContextRecord:
    return RunContextRecord(
        run_id="run-001",
        project_name="dre_production",
        stage_name="ehr_pipeline",
        timestamp_utc=datetime(2026, 5, 30, tzinfo=UTC),
    )


def _pipeline_result() -> EHRPipelineRunResult:
    return EHRPipelineRunResult(
        run_context=_run_context(),
        patient_dataset_path="reports/ehr_patient_dataset.csv",
        dataset_profile_path="reports/ehr_dataset_profile.json",
        artifact_registry_path="reports/ehr_artifact_registry.json",
        run_manifest_path="reports/ehr_run_manifest.json",
        patient_dataset_row_count=10,
        likely_dre_positive_count=3,
        likely_dre_negative_count=7,
        notes=["pipeline complete"],
    )


def _training_result() -> BaselineTrainingResult:
    leakage = LeakageCheckResult(
        candidate_feature_cols=["age", "distinct_asm_count"],
        excluded_feature_cols=[],
        excluded_for_leakage=["distinct_asm_count"],
        excluded_for_identifier=["patient_id"],
        excluded_for_non_numeric=[],
        notes=["leakage excluded"],
    )
    return BaselineTrainingResult(
        model_path="artifacts/model.joblib",
        metrics_path="artifacts/metrics.json",
        train_row_count=6,
        val_row_count=2,
        test_row_count=2,
        feature_cols_used=["age"],
        leakage_check=leakage,
        metrics={
            "test": {
                "accuracy": 0.75,
                "f1": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "roc_auc": None,
                "average_precision": 0.6,
            }
        },
        notes=["trained baseline"],
    )


def _dataset_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_name="patient_dataset",
        row_count=10,
        column_count=5,
        patient_count=10,
        missingness=[
            MissingnessRecord("age", 1, 0.1),
            MissingnessRecord("score", 4, 0.4),
            MissingnessRecord("flag", 0, 0.0),
        ],
        class_balance=[
            ClassBalanceRecord("False", 7, 0.7),
            ClassBalanceRecord("True", 3, 0.3),
        ],
        split_balance=[
            SplitBalanceRecord("train", 6, 0.6),
            SplitBalanceRecord("test", 4, 0.4),
        ],
        notes=["profile note"],
    )


def test_build_pipeline_summary_captures_counts_and_artifacts() -> None:
    summary = RunSummaryWriter().build_pipeline_summary(
        run_result=_pipeline_result(),
        run_context=_run_context(),
        dataset_name="patient_dataset",
    )

    assert summary.run_id == "run-001"
    assert summary.dataset_name == "patient_dataset"
    assert summary.patient_dataset_row_count == 10
    assert summary.likely_dre_positive_count == 3
    assert summary.artifact_paths["patient_dataset"] == "reports/ehr_patient_dataset.csv"


def test_build_model_summary_extracts_metrics_and_leakage() -> None:
    summary = RunSummaryWriter().build_model_summary(_training_result())

    assert summary.model_type == "logistic_regression"
    assert summary.key_metrics["test_accuracy"] == 0.75
    assert summary.key_metrics["test_roc_auc"] is None
    assert summary.leakage_excluded_cols == ["distinct_asm_count"]


def test_build_dataset_profile_summary_condenses_profile() -> None:
    summary = RunSummaryWriter().build_dataset_profile_summary(
        _dataset_profile(),
        top_n_missing=2,
    )

    assert summary.class_balance == {"False": 7, "True": 3}
    assert summary.split_balance == {"train": 6, "test": 4}
    assert [item["column_name"] for item in summary.top_missing_columns] == ["score", "age"]


def test_reporting_bundle_combines_summaries() -> None:
    writer = RunSummaryWriter()
    bundle = writer.build_reporting_bundle(
        pipeline_summary=writer.build_pipeline_summary(_pipeline_result(), _run_context()),
        model_summary=writer.build_model_summary(_training_result()),
        dataset_profile_summary=writer.build_dataset_profile_summary(_dataset_profile()),
        notes=["bundle note"],
    )

    as_dict = bundle.to_dict()

    assert as_dict["pipeline_summary"]["run_id"] == "run-001"
    assert as_dict["model_summary"]["feature_cols_used"] == ["age"]
    assert as_dict["notes"] == ["bundle note"]


def test_write_summary_and_bundle_json(tmp_path) -> None:
    writer = RunSummaryWriter()
    pipeline_summary = writer.build_pipeline_summary(_pipeline_result(), _run_context())
    bundle = writer.build_reporting_bundle(pipeline_summary=pipeline_summary)

    summary_path = writer.write_summary_json(
        pipeline_summary,
        tmp_path / "summaries" / "pipeline.json",
    )
    bundle_path = writer.write_reporting_bundle(
        bundle,
        tmp_path / "summaries" / "bundle.json",
    )

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert summary_payload["run_id"] == "run-001"
    assert bundle_payload["pipeline_summary"]["run_id"] == "run-001"
