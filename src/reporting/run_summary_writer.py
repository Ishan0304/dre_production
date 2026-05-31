"""Concise run summary builders and JSON writers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.contracts import RunContextRecord

if TYPE_CHECKING:
    from datasets import EHRPipelineRunResult
    from insights import DatasetProfile
    from modeling import BaselineTrainingResult


@dataclass(slots=True)
class PipelineRunSummary:
    """Concise summary of an end-to-end EHR pipeline run."""

    run_id: str
    project_name: str
    stage_name: str
    dataset_name: str | None
    patient_dataset_row_count: int | None
    likely_dre_positive_count: int | None
    likely_dre_negative_count: int | None
    artifact_paths: dict[str, str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class ModelRunSummary:
    """Concise summary of one baseline model training run."""

    model_type: str
    train_row_count: int
    val_row_count: int
    test_row_count: int
    feature_cols_used: list[str]
    model_path: str | None
    metrics_path: str | None
    key_metrics: dict[str, float | int | str | None]
    leakage_excluded_cols: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class DatasetProfileSummary:
    """Condensed summary of a full dataset profile."""

    dataset_name: str
    row_count: int
    column_count: int
    patient_count: int | None
    class_balance: dict[str, int]
    split_balance: dict[str, int]
    top_missing_columns: list[dict[str, float | int | str]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class ReportingBundle:
    """Bundle of available reporting summaries for a run."""

    pipeline_summary: PipelineRunSummary | None = None
    model_summary: ModelRunSummary | None = None
    dataset_profile_summary: DatasetProfileSummary | None = None
    notes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class RunSummaryWriter:
    """Build concise reporting summaries and write them as JSON."""

    def build_pipeline_summary(
        self,
        run_result: EHRPipelineRunResult,
        run_context: RunContextRecord,
        dataset_name: str | None = None,
    ) -> PipelineRunSummary:
        """Convert an EHR pipeline run result into a concise summary."""

        artifact_paths = {
            name: path
            for name, path in {
                "patient_dataset": run_result.patient_dataset_path,
                "dataset_profile": run_result.dataset_profile_path,
                "artifact_registry": run_result.artifact_registry_path,
                "run_manifest": run_result.run_manifest_path,
            }.items()
            if path is not None
        }

        return PipelineRunSummary(
            run_id=run_context.run_id,
            project_name=run_context.project_name,
            stage_name=run_context.stage_name,
            dataset_name=dataset_name,
            patient_dataset_row_count=run_result.patient_dataset_row_count,
            likely_dre_positive_count=run_result.likely_dre_positive_count,
            likely_dre_negative_count=run_result.likely_dre_negative_count,
            artifact_paths=artifact_paths,
            notes=list(run_result.notes),
        )

    def build_model_summary(
        self,
        training_result: BaselineTrainingResult,
        model_type: str = "logistic_regression",
    ) -> ModelRunSummary:
        """Convert a baseline training result into a concise model summary."""

        return ModelRunSummary(
            model_type=model_type,
            train_row_count=training_result.train_row_count,
            val_row_count=training_result.val_row_count,
            test_row_count=training_result.test_row_count,
            feature_cols_used=list(training_result.feature_cols_used),
            model_path=training_result.model_path,
            metrics_path=training_result.metrics_path,
            key_metrics=self._extract_test_metrics(training_result.metrics),
            leakage_excluded_cols=list(training_result.leakage_check.excluded_for_leakage),
            notes=list(training_result.notes),
        )

    def build_dataset_profile_summary(
        self,
        profile: DatasetProfile,
        top_n_missing: int = 10,
    ) -> DatasetProfileSummary:
        """Condense a dataset profile into reporting-friendly fields."""

        return DatasetProfileSummary(
            dataset_name=profile.dataset_name,
            row_count=profile.row_count,
            column_count=profile.column_count,
            patient_count=profile.patient_count,
            class_balance=self._summarize_class_balance(profile),
            split_balance=self._summarize_split_balance(profile),
            top_missing_columns=self._top_missingness(profile, top_n_missing),
            notes=list(profile.notes),
        )

    @staticmethod
    def build_reporting_bundle(
        pipeline_summary: PipelineRunSummary | None = None,
        model_summary: ModelRunSummary | None = None,
        dataset_profile_summary: DatasetProfileSummary | None = None,
        notes: list[str] | None = None,
    ) -> ReportingBundle:
        """Build a bundle from available reporting summaries."""

        return ReportingBundle(
            pipeline_summary=pipeline_summary,
            model_summary=model_summary,
            dataset_profile_summary=dataset_profile_summary,
            notes=notes or [],
        )

    def write_summary_json(self, summary_obj: Any, output_path: str | Path) -> Path:
        """Write any summary dataclass or dictionary-like object as JSON."""

        path = self._safe_resolved_path(output_path)
        payload = summary_obj.to_dict() if hasattr(summary_obj, "to_dict") else summary_obj
        if is_dataclass(payload):
            payload = asdict(payload)
        path.write_text(
            json.dumps(_to_serializable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def write_reporting_bundle(self, bundle: ReportingBundle, output_path: str | Path) -> Path:
        """Write a full reporting bundle as JSON."""

        return self.write_summary_json(bundle, output_path)

    @staticmethod
    def _extract_test_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str | None]:
        test_metrics = metrics.get("test", {})
        return {
            "test_accuracy": test_metrics.get("accuracy"),
            "test_f1": test_metrics.get("f1"),
            "test_precision": test_metrics.get("precision"),
            "test_recall": test_metrics.get("recall"),
            "test_roc_auc": test_metrics.get("roc_auc"),
            "test_average_precision": test_metrics.get("average_precision"),
        }

    @staticmethod
    def _summarize_class_balance(profile: DatasetProfile) -> dict[str, int]:
        return {record.label_value: record.count for record in profile.class_balance}

    @staticmethod
    def _summarize_split_balance(profile: DatasetProfile) -> dict[str, int]:
        return {record.split_value: record.count for record in profile.split_balance}

    @staticmethod
    def _top_missingness(
        profile: DatasetProfile,
        top_n_missing: int,
    ) -> list[dict[str, float | int | str]]:
        sorted_records = sorted(
            profile.missingness,
            key=lambda record: (-record.missing_fraction, record.column_name),
        )
        return [
            {
                "column_name": record.column_name,
                "missing_count": record.missing_count,
                "missing_fraction": record.missing_fraction,
            }
            for record in sorted_records[:top_n_missing]
        ]

    @staticmethod
    def _safe_resolved_path(output_path: str | Path) -> Path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


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
