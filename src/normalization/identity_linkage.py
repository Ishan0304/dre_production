"""Deterministic patient identity linkage utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import pandas as pd


class IdentityMatchStatus(Enum):
    """Status for one modality-specific identity mapping."""

    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"


@dataclass(slots=True)
class IdentityLinkRecord:
    """Mapping from a source identifier to a canonical patient identifier."""

    source_modality: str
    source_id: str
    canonical_patient_id: str | None
    match_status: IdentityMatchStatus
    match_method: str
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class IdentityLinkageResult:
    """Structured result for linking identifiers across one dataframe."""

    source_modality: str
    row_count: int
    matched_count: int
    unmatched_count: int
    ambiguous_count: int
    output_patient_id_col: str
    notes: list[str]
    link_records: list[IdentityLinkRecord]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class IdentityLinkageConfig:
    """Configuration for deterministic patient identity linkage."""

    source_id_col: str
    output_patient_id_col: str = "patient_id"
    explicit_mapping: dict[str, str] | None = None
    normalization_mode: str = "identity"
    keep_original_id_col: bool = True
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class IdentityLinker:
    """Link modality-specific identifiers to canonical patient identifiers."""

    SUPPORTED_NORMALIZATION_MODES = {"identity", "lower", "subject_to_patient"}

    def normalize_identifier(
        self,
        identifier: str | None,
        mode: str = "identity",
    ) -> str | None:
        """Normalize a source identifier with deterministic rules only."""

        if mode not in self.SUPPORTED_NORMALIZATION_MODES:
            raise ValueError(f"Unsupported identity normalization_mode: {mode}")
        if identifier is None or pd.isna(identifier):
            return None
        normalized = str(identifier).strip()
        if not normalized:
            return None
        if mode == "lower":
            return normalized.lower()
        if mode == "subject_to_patient":
            return normalized
        return normalized

    def apply_explicit_mapping(
        self,
        df: pd.DataFrame,
        config: IdentityLinkageConfig,
        source_modality: str,
    ) -> tuple[pd.DataFrame, IdentityLinkageResult]:
        """Map source IDs to canonical patient IDs using an explicit mapping."""

        self._validate_source_column(df, config.source_id_col)
        output_df = df.copy()
        notes = self._initial_notes(config)
        mapping, ambiguous_keys = self._normalized_mapping(config)
        canonical_values: list[str | None] = []
        link_records: list[IdentityLinkRecord] = []

        for source_value in output_df[config.source_id_col].tolist():
            normalized_source_id = self.normalize_identifier(
                source_value,
                config.normalization_mode,
            )
            if normalized_source_id is None:
                canonical_id = None
                status = IdentityMatchStatus.UNMATCHED
                note = "source identifier was missing"
            elif normalized_source_id in ambiguous_keys:
                canonical_id = None
                status = IdentityMatchStatus.AMBIGUOUS
                note = "source identifier had conflicting explicit mappings"
            elif normalized_source_id in mapping:
                canonical_id = mapping[normalized_source_id]
                status = IdentityMatchStatus.MATCHED
                note = None
            else:
                canonical_id = None
                status = IdentityMatchStatus.UNMATCHED
                note = "source identifier was absent from explicit mapping"

            canonical_values.append(canonical_id)
            link_records.append(
                IdentityLinkRecord(
                    source_modality=source_modality,
                    source_id=normalized_source_id or "",
                    canonical_patient_id=canonical_id,
                    match_status=status,
                    match_method="explicit_mapping",
                    notes=note,
                )
            )

        output_df[config.output_patient_id_col] = canonical_values
        output_df = self._drop_source_if_requested(output_df, config)
        return output_df, self._build_result(
            source_modality=source_modality,
            output_patient_id_col=config.output_patient_id_col,
            notes=notes,
            link_records=link_records,
        )

    def apply_normalization(
        self,
        df: pd.DataFrame,
        config: IdentityLinkageConfig,
        source_modality: str,
    ) -> tuple[pd.DataFrame, IdentityLinkageResult]:
        """Link source IDs by deterministic normalization."""

        self._validate_source_column(df, config.source_id_col)
        output_df = df.copy()
        notes = self._initial_notes(config)
        canonical_values: list[str | None] = []
        link_records: list[IdentityLinkRecord] = []

        for source_value in output_df[config.source_id_col].tolist():
            canonical_id = self.normalize_identifier(
                source_value,
                config.normalization_mode,
            )
            status = (
                IdentityMatchStatus.MATCHED
                if canonical_id is not None
                else IdentityMatchStatus.UNMATCHED
            )
            canonical_values.append(canonical_id)
            link_records.append(
                IdentityLinkRecord(
                    source_modality=source_modality,
                    source_id=canonical_id or "",
                    canonical_patient_id=canonical_id,
                    match_status=status,
                    match_method=f"normalization:{config.normalization_mode}",
                    notes=None if canonical_id is not None else "source identifier was missing",
                )
            )

        output_df[config.output_patient_id_col] = canonical_values
        output_df = self._drop_source_if_requested(output_df, config)
        return output_df, self._build_result(
            source_modality=source_modality,
            output_patient_id_col=config.output_patient_id_col,
            notes=notes,
            link_records=link_records,
        )

    def link_dataframe(
        self,
        df: pd.DataFrame,
        config: IdentityLinkageConfig,
        source_modality: str,
    ) -> tuple[pd.DataFrame, IdentityLinkageResult]:
        """Link a dataframe using explicit mapping or deterministic normalization."""

        self._validate_source_column(df, config.source_id_col)
        if config.explicit_mapping is not None:
            return self.apply_explicit_mapping(df, config, source_modality)
        return self.apply_normalization(df, config, source_modality)

    @staticmethod
    def summarize_results(results: list[IdentityLinkageResult]) -> dict[str, object]:
        """Build a compact JSON-ready summary across linkage results."""

        modalities = []
        notes: list[str] = []
        totals = {
            "row_count": 0,
            "matched_count": 0,
            "unmatched_count": 0,
            "ambiguous_count": 0,
        }
        for result in results:
            modalities.append(
                {
                    "source_modality": result.source_modality,
                    "row_count": result.row_count,
                    "matched_count": result.matched_count,
                    "unmatched_count": result.unmatched_count,
                    "ambiguous_count": result.ambiguous_count,
                    "output_patient_id_col": result.output_patient_id_col,
                }
            )
            notes.extend(result.notes)
            totals["row_count"] += result.row_count
            totals["matched_count"] += result.matched_count
            totals["unmatched_count"] += result.unmatched_count
            totals["ambiguous_count"] += result.ambiguous_count

        return {
            "modality_count": len(results),
            "totals": totals,
            "modalities": modalities,
            "notes": list(dict.fromkeys(notes)),
        }

    @staticmethod
    def _validate_source_column(df: pd.DataFrame, source_id_col: str) -> None:
        if source_id_col not in df.columns:
            raise ValueError(f"Missing source identifier column: {source_id_col}")

    @staticmethod
    def _initial_notes(config: IdentityLinkageConfig) -> list[str]:
        return [config.notes] if config.notes else []

    def _normalized_mapping(
        self,
        config: IdentityLinkageConfig,
    ) -> tuple[dict[str, str], set[str]]:
        mapping: dict[str, str] = {}
        ambiguous_keys: set[str] = set()
        for source_id, canonical_id in (config.explicit_mapping or {}).items():
            normalized_source_id = self.normalize_identifier(
                source_id,
                config.normalization_mode,
            )
            normalized_canonical_id = self.normalize_identifier(canonical_id, "identity")
            if normalized_source_id is None or normalized_canonical_id is None:
                continue
            existing = mapping.get(normalized_source_id)
            if existing is not None and existing != normalized_canonical_id:
                ambiguous_keys.add(normalized_source_id)
            mapping[normalized_source_id] = normalized_canonical_id
        return mapping, ambiguous_keys

    @staticmethod
    def _drop_source_if_requested(
        df: pd.DataFrame,
        config: IdentityLinkageConfig,
    ) -> pd.DataFrame:
        if config.keep_original_id_col or config.source_id_col == config.output_patient_id_col:
            return df
        return df.drop(columns=[config.source_id_col])

    @staticmethod
    def _build_result(
        source_modality: str,
        output_patient_id_col: str,
        notes: list[str],
        link_records: list[IdentityLinkRecord],
    ) -> IdentityLinkageResult:
        matched_count = sum(
            record.match_status is IdentityMatchStatus.MATCHED
            for record in link_records
        )
        unmatched_count = sum(
            record.match_status is IdentityMatchStatus.UNMATCHED
            for record in link_records
        )
        ambiguous_count = sum(
            record.match_status is IdentityMatchStatus.AMBIGUOUS
            for record in link_records
        )
        if unmatched_count:
            notes.append(f"{unmatched_count} {source_modality} rows were unmatched")
        if ambiguous_count:
            notes.append(f"{ambiguous_count} {source_modality} rows were ambiguous")
        return IdentityLinkageResult(
            source_modality=source_modality,
            row_count=len(link_records),
            matched_count=int(matched_count),
            unmatched_count=int(unmatched_count),
            ambiguous_count=int(ambiguous_count),
            output_patient_id_col=output_patient_id_col,
            notes=list(dict.fromkeys(notes)),
            link_records=link_records,
        )
