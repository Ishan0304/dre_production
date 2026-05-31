"""Generic pandas-based EHR table loading utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


SUPPORTED_FORMATS = {"csv", "csv.gz", "parquet"}


class RequiredColumnsMissingError(ValueError):
    """Raised when required columns are absent from a loaded table."""


@dataclass(slots=True)
class TableLoadRequest:
    """Request configuration for loading one tabular EHR extract."""

    path: str
    file_format: str | None = None
    columns: list[str] | None = None
    required_columns: list[str] | None = None
    chunk_size: int | None = None
    dtype_overrides: dict[str, str] | None = None
    parse_date_columns: list[str] | None = None
    dataset_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class TableLoadResult:
    """Structured metadata describing a table load operation."""

    path: str
    file_format: str
    row_count: int | None
    column_names: list[str]
    loaded_columns: list[str]
    missing_required_columns: list[str]
    used_chunking: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class EHRLoader:
    """Load generic EHR tables from CSV and parquet sources."""

    def infer_file_format(self, path: str) -> str:
        """Infer a supported file format from the path extension."""

        normalized_path = path.lower()
        if normalized_path.endswith(".csv.gz"):
            return "csv.gz"
        if normalized_path.endswith(".csv"):
            return "csv"
        if normalized_path.endswith(".parquet"):
            return "parquet"
        raise ValueError(f"Unsupported file format for path: {path}")

    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame,
        required_columns: list[str] | None,
    ) -> list[str]:
        """Return required columns that are absent from the dataframe."""

        if not required_columns:
            return []
        existing_columns = set(df.columns)
        return [column for column in dict.fromkeys(required_columns) if column not in existing_columns]

    def load_table(self, request: TableLoadRequest) -> tuple[pd.DataFrame, TableLoadResult]:
        """Load a full CSV or parquet table and return data plus metadata."""

        self._validate_path_exists(request.path)
        file_format, notes = self._resolve_file_format(request)
        columns_to_load, column_notes = self._resolve_columns_to_load(
            path=request.path,
            file_format=file_format,
            columns=request.columns,
        )
        notes.extend(column_notes)
        dtype_overrides = self._filter_dtype_overrides(request.dtype_overrides, columns_to_load)
        parse_date_columns = self._filter_parse_date_columns(
            request.parse_date_columns,
            columns_to_load,
        )

        if file_format in {"csv", "csv.gz"}:
            df = pd.read_csv(
                request.path,
                usecols=columns_to_load,
                dtype=dtype_overrides,
                parse_dates=parse_date_columns,
            )
        elif file_format == "parquet":
            df = pd.read_parquet(request.path, columns=columns_to_load)
            if dtype_overrides:
                df = df.astype(dtype_overrides)
            df = self._coerce_date_columns(df, parse_date_columns)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        notes.extend(self._request_notes(request, used_chunking=False))
        missing_required_columns = self.validate_required_columns(df, request.required_columns)
        if missing_required_columns:
            raise RequiredColumnsMissingError(
                self._missing_required_columns_message(request.path, missing_required_columns)
            )

        result = self._build_result(
            request=request,
            file_format=file_format,
            df=df,
            missing_required_columns=missing_required_columns,
            used_chunking=False,
            notes=notes,
        )
        return df, result

    def iter_csv_chunks(
        self,
        request: TableLoadRequest,
    ) -> Iterator[tuple[pd.DataFrame, TableLoadResult]]:
        """Yield CSV chunks with structured metadata for each chunk."""

        self._validate_path_exists(request.path)
        file_format, notes = self._resolve_file_format(request)
        columns_to_load, column_notes = self._resolve_columns_to_load(
            path=request.path,
            file_format=file_format,
            columns=request.columns,
        )
        notes.extend(column_notes)
        dtype_overrides = self._filter_dtype_overrides(request.dtype_overrides, columns_to_load)
        parse_date_columns = self._filter_parse_date_columns(
            request.parse_date_columns,
            columns_to_load,
        )

        if file_format not in {"csv", "csv.gz"}:
            raise ValueError("Chunked iteration is only supported for CSV inputs.")
        if request.chunk_size is None or request.chunk_size <= 0:
            raise ValueError("chunk_size must be set to a positive integer for chunked CSV loading.")

        chunk_notes = notes + self._request_notes(request, used_chunking=True)
        reader = pd.read_csv(
            request.path,
            usecols=columns_to_load,
            dtype=dtype_overrides,
            parse_dates=parse_date_columns,
            chunksize=request.chunk_size,
        )

        for chunk in reader:
            missing_required_columns = self.validate_required_columns(chunk, request.required_columns)
            if missing_required_columns:
                raise RequiredColumnsMissingError(
                    self._missing_required_columns_message(request.path, missing_required_columns)
                )
            yield chunk, self._build_result(
                request=request,
                file_format=file_format,
                df=chunk,
                missing_required_columns=missing_required_columns,
                used_chunking=True,
                notes=chunk_notes,
            )

    def peek_columns(self, path: str, file_format: str | None = None) -> list[str]:
        """Return table column names without loading full table data when practical."""

        self._validate_path_exists(path)
        resolved_format = self._normalize_file_format(file_format) if file_format else self.infer_file_format(path)

        if resolved_format in {"csv", "csv.gz"}:
            return [str(column) for column in pd.read_csv(path, nrows=0).columns]
        if resolved_format == "parquet":
            return [str(column) for column in pd.read_parquet(path).columns]
        raise ValueError(f"Unsupported file format: {resolved_format}")

    @staticmethod
    def _validate_path_exists(path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Table path does not exist: {path}")

    def _resolve_file_format(self, request: TableLoadRequest) -> tuple[str, list[str]]:
        notes: list[str] = []
        if request.file_format is None:
            notes.append("file format inferred from path")
            return self.infer_file_format(request.path), notes
        return self._normalize_file_format(request.file_format), notes

    @staticmethod
    def _normalize_file_format(file_format: str) -> str:
        normalized = file_format.lower().lstrip(".")
        if normalized == "gz":
            normalized = "csv.gz"
        if normalized not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_format}")
        return normalized

    @staticmethod
    def _request_notes(request: TableLoadRequest, used_chunking: bool) -> list[str]:
        notes: list[str] = []
        if used_chunking:
            notes.append("chunking was used")
        if request.parse_date_columns:
            notes.append("date parsing was requested")
        if request.columns:
            notes.append("selected columns were used")
        return notes

    def _resolve_columns_to_load(
        self,
        path: str,
        file_format: str,
        columns: list[str] | None,
    ) -> tuple[list[str] | None, list[str]]:
        if not columns:
            return None, []

        available_columns = set(self.peek_columns(path, file_format))
        deduped_columns = list(dict.fromkeys(columns))
        selected_columns = [column for column in deduped_columns if column in available_columns]
        missing_columns = [column for column in deduped_columns if column not in available_columns]

        notes = [
            f"requested selected column missing: {column}"
            for column in missing_columns
        ]
        return selected_columns, notes

    @staticmethod
    def _filter_dtype_overrides(
        dtype_overrides: dict[str, str] | None,
        columns_to_load: list[str] | None,
    ) -> dict[str, str] | None:
        if not dtype_overrides or columns_to_load is None:
            return dtype_overrides
        loadable_columns = set(columns_to_load)
        return {
            column: dtype
            for column, dtype in dtype_overrides.items()
            if column in loadable_columns
        }

    @staticmethod
    def _filter_parse_date_columns(
        parse_date_columns: list[str] | None,
        columns_to_load: list[str] | None,
    ) -> list[str] | None:
        if not parse_date_columns or columns_to_load is None:
            return parse_date_columns
        loadable_columns = set(columns_to_load)
        return [column for column in parse_date_columns if column in loadable_columns]

    @staticmethod
    def _coerce_date_columns(
        df: pd.DataFrame,
        parse_date_columns: list[str] | None,
    ) -> pd.DataFrame:
        if not parse_date_columns:
            return df
        converted = df.copy()
        for column in parse_date_columns:
            if column in converted.columns:
                converted[column] = pd.to_datetime(converted[column], errors="coerce")
        return converted

    @staticmethod
    def _build_result(
        request: TableLoadRequest,
        file_format: str,
        df: pd.DataFrame,
        missing_required_columns: list[str],
        used_chunking: bool,
        notes: list[str],
    ) -> TableLoadResult:
        return TableLoadResult(
            path=request.path,
            file_format=file_format,
            row_count=len(df),
            column_names=[str(column) for column in df.columns],
            loaded_columns=[str(column) for column in df.columns],
            missing_required_columns=missing_required_columns,
            used_chunking=used_chunking,
            notes=list(dict.fromkeys(notes)),
        )

    @staticmethod
    def _missing_required_columns_message(path: str, missing_columns: list[str]) -> str:
        missing = ", ".join(missing_columns)
        return f"Required columns missing from {path}: {missing}"
