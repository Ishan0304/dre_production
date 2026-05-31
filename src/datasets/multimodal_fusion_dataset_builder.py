"""Patient-level multimodal fusion dataset construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import reduce
from typing import Any

import pandas as pd


@dataclass(slots=True)
class MultimodalFusionConfig:
    """Configuration for patient-level multimodal dataset fusion."""

    patient_id_col: str = "patient_id"
    ehr_prefix: str = "ehr"
    mri_prefix: str = "mri"
    eeg_prefix: str = "eeg"
    include_modality_flags: bool = True
    require_patient_id: bool = True
    sort_output_by_patient_id: bool = True
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class MultimodalFusionResult:
    """Structured metadata for one multimodal fusion build."""

    dataset_name: str
    row_count: int
    patient_count: int
    column_names: list[str]
    ehr_row_count: int
    mri_row_count: int
    eeg_row_count: int
    patients_with_ehr: int
    patients_with_mri: int
    patients_with_eeg: int
    patients_with_all_modalities: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class MultimodalFusionDatasetBuilder:
    """Align patient-level EHR, MRI, and EEG datasets into one feature table."""

    def __init__(self, config: MultimodalFusionConfig | None = None) -> None:
        self.config = config or MultimodalFusionConfig()

    def build_fused_dataset(
        self,
        dataset_name: str,
        ehr_df: pd.DataFrame | None = None,
        mri_df: pd.DataFrame | None = None,
        eeg_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, MultimodalFusionResult]:
        """Build a patient-level fused feature dataset using outer joins."""

        notes: list[str] = []
        if self.config.notes:
            notes.append(self.config.notes)

        modality_specs = [
            ("ehr", self.config.ehr_prefix, ehr_df),
            ("mri", self.config.mri_prefix, mri_df),
            ("eeg", self.config.eeg_prefix, eeg_df),
        ]
        prepared_frames: list[pd.DataFrame] = []
        source_row_counts: dict[str, int] = {}

        for modality_name, prefix, df in modality_specs:
            notes.extend(self.validate_modality_df(df, modality_name))
            source_row_counts[f"{modality_name}_row_count"] = (
                len(df) if df is not None else 0
            )
            if df is None or df.empty or self.config.patient_id_col not in df.columns:
                continue

            self._raise_on_duplicate_patient_ids(df, modality_name)
            namespaced_df = self.namespace_modality_columns(df, prefix)
            namespaced_df[f"has_{modality_name}"] = True
            prepared_frames.append(namespaced_df)

        if not prepared_frames:
            fused_df = pd.DataFrame(columns=[self.config.patient_id_col])
            counts = {
                "patients_with_ehr": 0,
                "patients_with_mri": 0,
                "patients_with_eeg": 0,
                "patients_with_all_modalities": 0,
            }
        else:
            fused_with_flags = reduce(
                lambda left, right: left.merge(
                    right,
                    on=self.config.patient_id_col,
                    how="outer",
                ),
                prepared_frames,
            )
            fused_with_flags = self._fill_modality_flags(fused_with_flags)
            counts = self.compute_modality_counts(fused_with_flags)
            fused_df = (
                fused_with_flags
                if self.config.include_modality_flags
                else self._drop_modality_flags(fused_with_flags)
            )
            fused_df = fused_df[self._ordered_columns(fused_df)]
            if self.config.sort_output_by_patient_id:
                fused_df = fused_df.sort_values(
                    self.config.patient_id_col
                ).reset_index(drop=True)

        result = MultimodalFusionResult(
            dataset_name=dataset_name,
            row_count=len(fused_df),
            patient_count=self._safe_patient_count(fused_df),
            column_names=[str(column) for column in fused_df.columns],
            ehr_row_count=source_row_counts.get("ehr_row_count", 0),
            mri_row_count=source_row_counts.get("mri_row_count", 0),
            eeg_row_count=source_row_counts.get("eeg_row_count", 0),
            patients_with_ehr=counts["patients_with_ehr"],
            patients_with_mri=counts["patients_with_mri"],
            patients_with_eeg=counts["patients_with_eeg"],
            patients_with_all_modalities=counts["patients_with_all_modalities"],
            notes=list(dict.fromkeys(notes)),
        )
        return fused_df, result

    def validate_modality_df(
        self,
        df: pd.DataFrame | None,
        modality_name: str,
    ) -> list[str]:
        """Return validation notes for one modality dataframe."""

        notes: list[str] = []
        patient_id_col = self.config.patient_id_col
        if df is None:
            return [f"{modality_name} dataframe was not provided"]
        if df.empty:
            notes.append(f"{modality_name} dataframe was empty")
        if df.empty and patient_id_col not in df.columns:
            return notes
        if patient_id_col not in df.columns:
            message = (
                f"{modality_name} dataframe is missing required column: {patient_id_col}"
            )
            if self.config.require_patient_id:
                raise ValueError(message)
            notes.append(message)
        elif df[patient_id_col].isna().any():
            notes.append(f"{modality_name} dataframe contains missing patient identifiers")
        return notes

    def namespace_modality_columns(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Prefix modality-specific columns while preserving patient_id."""

        patient_id_col = self.config.patient_id_col
        ordered_columns = [patient_id_col] + [
            column for column in df.columns if column != patient_id_col
        ]
        copied_df = df.loc[:, ordered_columns].copy()
        rename_map = {
            column: f"{prefix}_{column}"
            for column in copied_df.columns
            if column != patient_id_col
        }
        return copied_df.rename(columns=rename_map)

    def compute_modality_counts(self, fused_df: pd.DataFrame) -> dict[str, int]:
        """Compute patient counts by available modality."""

        if fused_df.empty:
            return {
                "patients_with_ehr": 0,
                "patients_with_mri": 0,
                "patients_with_eeg": 0,
                "patients_with_all_modalities": 0,
            }

        ehr_mask = self._flag_mask(fused_df, "has_ehr")
        mri_mask = self._flag_mask(fused_df, "has_mri")
        eeg_mask = self._flag_mask(fused_df, "has_eeg")
        all_mask = ehr_mask & mri_mask & eeg_mask
        return {
            "patients_with_ehr": int(ehr_mask.sum()),
            "patients_with_mri": int(mri_mask.sum()),
            "patients_with_eeg": int(eeg_mask.sum()),
            "patients_with_all_modalities": int(all_mask.sum()),
        }

    def _raise_on_duplicate_patient_ids(self, df: pd.DataFrame, modality_name: str) -> None:
        patient_id_col = self.config.patient_id_col
        duplicate_count = int(df[patient_id_col].duplicated().sum())
        if duplicate_count:
            raise ValueError(
                f"{modality_name} dataframe contains duplicate patient_id values: {duplicate_count}"
            )

    def _fill_modality_flags(self, fused_df: pd.DataFrame) -> pd.DataFrame:
        copied_df = fused_df.copy()
        for flag in ("has_ehr", "has_mri", "has_eeg"):
            if flag not in copied_df.columns:
                copied_df[flag] = False
            else:
                copied_df[flag] = copied_df[flag].fillna(False).astype(bool)
        return copied_df

    @staticmethod
    def _drop_modality_flags(fused_df: pd.DataFrame) -> pd.DataFrame:
        return fused_df.drop(columns=["has_ehr", "has_mri", "has_eeg"], errors="ignore")

    def _ordered_columns(self, fused_df: pd.DataFrame) -> list[str]:
        patient_id_col = self.config.patient_id_col
        flag_columns = [
            column
            for column in ("has_ehr", "has_mri", "has_eeg")
            if column in fused_df.columns
        ]
        remaining_columns = [
            column
            for column in fused_df.columns
            if column != patient_id_col and column not in flag_columns
        ]
        return [patient_id_col, *flag_columns, *remaining_columns]

    def _safe_patient_count(self, fused_df: pd.DataFrame) -> int:
        patient_id_col = self.config.patient_id_col
        if fused_df.empty or patient_id_col not in fused_df.columns:
            return 0
        return int(fused_df[patient_id_col].nunique(dropna=True))

    @staticmethod
    def _flag_mask(fused_df: pd.DataFrame, flag_column: str) -> pd.Series:
        if flag_column not in fused_df.columns:
            return pd.Series(False, index=fused_df.index)
        return fused_df[flag_column].fillna(False).astype(bool)
