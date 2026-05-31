"""EEG patient-level dataset aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass(slots=True)
class EEGPatientDatasetBuildConfig:
    """Configuration for EEG patient-level aggregation."""

    dataset_name: str
    aggregation_strategy: str = "mean"
    include_recording_count: bool = True
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class EEGPatientDatasetBuildResult:
    """Structured metadata for an EEG patient dataset build."""

    dataset_name: str
    row_count: int
    patient_count: int
    error_count: int
    column_names: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class EEGPatientDatasetBuilder:
    """Aggregate EEG recording features into patient-level datasets."""

    def __init__(self, config: EEGPatientDatasetBuildConfig | None = None) -> None:
        self.config = config

    def build_patient_dataset(
        self,
        recording_feature_df: pd.DataFrame,
        error_df: pd.DataFrame | None = None,
        dataset_name: str | None = None,
    ) -> tuple[pd.DataFrame, EEGPatientDatasetBuildResult]:
        """Aggregate recording-level EEG features to one row per patient."""

        resolved_name = dataset_name or (self.config.dataset_name if self.config else "eeg_patient_dataset")
        notes: list[str] = []
        if self.config and self.config.notes:
            notes.append(self.config.notes)
        aggregation_strategy = self.config.aggregation_strategy if self.config else "mean"
        include_recording_count = True if self.config is None else self.config.include_recording_count
        if aggregation_strategy != "mean":
            raise ValueError(f"Unsupported EEG aggregation_strategy: {aggregation_strategy}")

        if recording_feature_df.empty or "patient_id" not in recording_feature_df.columns:
            output_df = pd.DataFrame(columns=["patient_id"])
        else:
            numeric_cols = [
                column
                for column in recording_feature_df.columns
                if column != "patient_id" and is_numeric_dtype(recording_feature_df[column])
            ]
            output_df = (
                recording_feature_df.groupby("patient_id", as_index=False)[numeric_cols]
                .mean()
                .sort_values("patient_id")
                .reset_index(drop=True)
            )
            if include_recording_count:
                counts = (
                    recording_feature_df.groupby("patient_id")
                    .size()
                    .rename("recording_count")
                    .reset_index()
                )
                output_df = output_df.merge(counts, on="patient_id", how="left")

        error_count = 0 if error_df is None else len(error_df)
        if error_count:
            notes.append(f"{error_count} recordings had feature extraction errors")

        result = EEGPatientDatasetBuildResult(
            dataset_name=resolved_name,
            row_count=len(output_df),
            patient_count=int(output_df["patient_id"].nunique()) if "patient_id" in output_df.columns else 0,
            error_count=error_count,
            column_names=[str(column) for column in output_df.columns],
            notes=notes,
        )
        return output_df, result
