"""Structured pandas-based dataset profiling utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


@dataclass(slots=True)
class MissingnessRecord:
    """Missing value summary for one column."""

    column_name: str
    missing_count: int
    missing_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class CategoryCardinalityRecord:
    """Cardinality and top value summary for one categorical column."""

    column_name: str
    unique_count: int
    top_values: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class NumericSummaryRecord:
    """Numeric distribution summary for one column."""

    column_name: str
    non_null_count: int
    mean: float | None
    std: float | None
    min_value: float | None
    p25: float | None
    median: float | None
    p75: float | None
    max_value: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class TemporalCoverageRecord:
    """Temporal coverage summary for one datetime-like column."""

    column_name: str
    min_time: datetime | None
    max_time: datetime | None
    non_null_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class ClassBalanceRecord:
    """Class count and fraction for one label value."""

    label_value: str
    count: int
    fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class SplitBalanceRecord:
    """Split count and fraction for one split value."""

    split_value: str
    count: int
    fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class DatasetProfile:
    """Structured profile for a dataframe used in pipeline workflows."""

    dataset_name: str
    row_count: int
    column_count: int
    patient_count: int | None
    missingness: list[MissingnessRecord] = field(default_factory=list)
    numeric_summaries: list[NumericSummaryRecord] = field(default_factory=list)
    category_cardinality: list[CategoryCardinalityRecord] = field(default_factory=list)
    class_balance: list[ClassBalanceRecord] = field(default_factory=list)
    split_balance: list[SplitBalanceRecord] = field(default_factory=list)
    temporal_coverage: list[TemporalCoverageRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class DatasetProfiler:
    """Profile tabular datasets with structured, reusable outputs."""

    def profile_dataframe(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        patient_id_col: str | None = None,
        label_col: str | None = None,
        split_col: str | None = None,
        time_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        numeric_cols: list[str] | None = None,
        top_n_categories: int = 10,
    ) -> DatasetProfile:
        """Profile a dataframe and return a structured dataset profile."""

        notes = self._collect_missing_column_notes(
            df=df,
            patient_id_col=patient_id_col,
            label_col=label_col,
            split_col=split_col,
            time_cols=time_cols,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
        )

        return DatasetProfile(
            dataset_name=dataset_name,
            row_count=len(df),
            column_count=len(df.columns),
            patient_count=self.compute_patient_count(df, patient_id_col),
            missingness=self.compute_missingness(df),
            numeric_summaries=self.compute_numeric_summaries(df, numeric_cols),
            category_cardinality=self.compute_category_cardinality(
                df=df,
                categorical_cols=categorical_cols,
                top_n=top_n_categories,
            ),
            class_balance=self.compute_class_balance(df, label_col),
            split_balance=self.compute_split_balance(df, split_col),
            temporal_coverage=self.compute_temporal_coverage(df, time_cols),
            notes=notes,
        )

    @staticmethod
    def compute_missingness(df: pd.DataFrame) -> list[MissingnessRecord]:
        """Compute missing counts and fractions for every dataframe column."""

        row_count = len(df)
        records: list[MissingnessRecord] = []
        for column_name in df.columns:
            missing_count = int(df[column_name].isna().sum())
            records.append(
                MissingnessRecord(
                    column_name=str(column_name),
                    missing_count=missing_count,
                    missing_fraction=_fraction(missing_count, row_count),
                )
            )
        return records

    @staticmethod
    def compute_patient_count(df: pd.DataFrame, patient_id_col: str | None) -> int | None:
        """Compute distinct patient count when a patient identifier is available."""

        if patient_id_col is None or patient_id_col not in df.columns:
            return None
        return int(df[patient_id_col].nunique(dropna=True))

    @staticmethod
    def compute_class_balance(
        df: pd.DataFrame,
        label_col: str | None,
    ) -> list[ClassBalanceRecord]:
        """Compute label balance for a requested label column."""

        if label_col is None or label_col not in df.columns:
            return []

        row_count = len(df)
        counts = df[label_col].value_counts(dropna=False)
        return [
            ClassBalanceRecord(
                label_value=str(label_value),
                count=int(count),
                fraction=_fraction(int(count), row_count),
            )
            for label_value, count in counts.items()
        ]

    @staticmethod
    def compute_split_balance(
        df: pd.DataFrame,
        split_col: str | None,
    ) -> list[SplitBalanceRecord]:
        """Compute split balance for a requested split column."""

        if split_col is None or split_col not in df.columns:
            return []

        row_count = len(df)
        counts = df[split_col].value_counts(dropna=False)
        return [
            SplitBalanceRecord(
                split_value=str(split_value),
                count=int(count),
                fraction=_fraction(int(count), row_count),
            )
            for split_value, count in counts.items()
        ]

    @staticmethod
    def compute_numeric_summaries(
        df: pd.DataFrame,
        numeric_cols: list[str] | None = None,
    ) -> list[NumericSummaryRecord]:
        """Compute numeric distribution summaries for selected or inferred columns."""

        columns = _existing_columns(df, numeric_cols) if numeric_cols else _infer_numeric_columns(df)
        records: list[NumericSummaryRecord] = []

        for column_name in columns:
            numeric_series = pd.to_numeric(df[column_name], errors="coerce").dropna()
            records.append(
                NumericSummaryRecord(
                    column_name=column_name,
                    non_null_count=int(numeric_series.count()),
                    mean=_optional_float(numeric_series.mean()),
                    std=_optional_float(numeric_series.std()),
                    min_value=_optional_float(numeric_series.min()),
                    p25=_optional_float(numeric_series.quantile(0.25)),
                    median=_optional_float(numeric_series.median()),
                    p75=_optional_float(numeric_series.quantile(0.75)),
                    max_value=_optional_float(numeric_series.max()),
                )
            )

        return records

    @staticmethod
    def compute_category_cardinality(
        df: pd.DataFrame,
        categorical_cols: list[str] | None = None,
        top_n: int = 10,
    ) -> list[CategoryCardinalityRecord]:
        """Compute cardinality and top values for selected or inferred columns."""

        columns = (
            _existing_columns(df, categorical_cols)
            if categorical_cols
            else _infer_categorical_columns(df)
        )
        records: list[CategoryCardinalityRecord] = []

        for column_name in columns:
            counts = df[column_name].value_counts(dropna=True).head(top_n)
            records.append(
                CategoryCardinalityRecord(
                    column_name=column_name,
                    unique_count=int(df[column_name].nunique(dropna=True)),
                    top_values={str(value): int(count) for value, count in counts.items()},
                )
            )

        return records

    @staticmethod
    def compute_temporal_coverage(
        df: pd.DataFrame,
        time_cols: list[str] | None = None,
    ) -> list[TemporalCoverageRecord]:
        """Compute min and max timestamps for selected or inferred time columns."""

        columns = _existing_columns(df, time_cols) if time_cols else _infer_time_columns(df)
        records: list[TemporalCoverageRecord] = []

        for column_name in columns:
            times = pd.to_datetime(df[column_name], errors="coerce").dropna()
            records.append(
                TemporalCoverageRecord(
                    column_name=column_name,
                    min_time=_optional_datetime(times.min()),
                    max_time=_optional_datetime(times.max()),
                    non_null_count=int(times.count()),
                )
            )

        return records

    @staticmethod
    def _collect_missing_column_notes(
        df: pd.DataFrame,
        patient_id_col: str | None,
        label_col: str | None,
        split_col: str | None,
        time_cols: list[str] | None,
        categorical_cols: list[str] | None,
        numeric_cols: list[str] | None,
    ) -> list[str]:
        notes: list[str] = []

        requested_columns = {
            "patient_id_col": [patient_id_col] if patient_id_col else [],
            "label_col": [label_col] if label_col else [],
            "split_col": [split_col] if split_col else [],
            "time_cols": time_cols or [],
            "categorical_cols": categorical_cols or [],
            "numeric_cols": numeric_cols or [],
        }

        for argument_name, columns in requested_columns.items():
            for column_name in columns:
                if column_name not in df.columns:
                    notes.append(f"requested {argument_name} missing: {column_name}")

        return notes


def _fraction(count: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return count / denominator


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column_name for column_name in columns if column_name in df.columns]


def _infer_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(column_name)
        for column_name in df.columns
        if is_numeric_dtype(df[column_name]) and df[column_name].dtype != bool
    ]


def _infer_categorical_columns(df: pd.DataFrame) -> list[str]:
    categorical_dtypes = ["object", "category", "bool"]
    return [str(column_name) for column_name in df.select_dtypes(include=categorical_dtypes).columns]


def _infer_time_columns(df: pd.DataFrame) -> list[str]:
    return [str(column_name) for column_name in df.columns if is_datetime64_any_dtype(df[column_name])]


def _optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _optional_datetime(value: Any) -> datetime | None:
    if pd.isna(value):
        return None
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    return None
