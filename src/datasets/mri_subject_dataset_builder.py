"""MRI subject-level dataset finalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class MRISubjectDatasetBuildConfig:
    """Configuration for MRI subject dataset construction."""

    dataset_name: str
    include_metadata_columns: bool = True
    include_error_summary: bool = True
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class MRISubjectDatasetBuildResult:
    """Structured metadata for an MRI subject dataset build."""

    dataset_name: str
    row_count: int
    subject_count: int
    error_count: int
    column_names: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class MRISubjectDatasetBuilder:
    """Finalize MRI feature tables into subject-level datasets."""

    def __init__(self, config: MRISubjectDatasetBuildConfig | None = None) -> None:
        self.config = config

    def build_subject_dataset(
        self,
        feature_df: pd.DataFrame,
        error_df: pd.DataFrame | None = None,
        dataset_name: str | None = None,
    ) -> tuple[pd.DataFrame, MRISubjectDatasetBuildResult]:
        """Finalize an MRI subject-level dataset and return build metadata."""

        resolved_name = dataset_name or (self.config.dataset_name if self.config else "mri_subject_dataset")
        notes: list[str] = []
        if self.config and self.config.notes:
            notes.append(self.config.notes)

        output_df = feature_df.copy()
        if not output_df.empty and "subject_id" in output_df.columns:
            output_df = output_df.sort_values("subject_id").reset_index(drop=True)
        error_count = 0 if error_df is None else len(error_df)
        if error_count:
            notes.append(f"{error_count} subjects had feature extraction errors")

        result = MRISubjectDatasetBuildResult(
            dataset_name=resolved_name,
            row_count=len(output_df),
            subject_count=(
                int(output_df["subject_id"].nunique())
                if "subject_id" in output_df.columns
                else len(output_df)
            ),
            error_count=error_count,
            column_names=[str(column) for column in output_df.columns],
            notes=notes,
        )
        return output_df, result
