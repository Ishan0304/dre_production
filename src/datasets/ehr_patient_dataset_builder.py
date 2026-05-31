"""Build patient-level EHR datasets from evidence bundles and definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from definitions import LikelyDREDefinition, LikelyDREResult, PatientEvidenceBundle
from features import EHRColumnConfig, EHREvidenceBuilder


@dataclass(slots=True)
class PatientDatasetBuildConfig:
    """Configuration for patient-level EHR dataset construction."""

    patient_id_col: str = "patient_id"
    include_reason_columns: bool = True
    include_missing_evidence_columns: bool = True
    include_source_modality_columns: bool = True
    sort_output_by_patient_id: bool = True
    definition_version_override: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class PatientDatasetBuildResult:
    """Metadata describing a patient-level dataset build."""

    dataset_name: str
    row_count: int
    patient_count: int
    column_names: list[str]
    likely_dre_positive_count: int
    likely_dre_negative_count: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class EHRPatientDatasetBuilder:
    """Orchestrate EHR evidence building into patient-level tabular outputs."""

    def __init__(
        self,
        evidence_builder: EHREvidenceBuilder | None = None,
        definition_engine: LikelyDREDefinition | None = None,
        config: PatientDatasetBuildConfig | None = None,
    ) -> None:
        self.config = config or PatientDatasetBuildConfig()
        self.evidence_builder = evidence_builder or EHREvidenceBuilder(
            column_config=EHRColumnConfig(patient_id_col=self.config.patient_id_col)
        )
        if self.config.definition_version_override is not None:
            self.definition_engine = LikelyDREDefinition(
                definition_version=self.config.definition_version_override
            )
        else:
            self.definition_engine = definition_engine or LikelyDREDefinition()

    def build_patient_dataset(
        self,
        dataset_name: str,
        diagnoses_df: pd.DataFrame | None = None,
        medications_df: pd.DataFrame | None = None,
        seizure_events_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, PatientDatasetBuildResult]:
        """Build a patient-level dataframe and structured build metadata."""

        notes: list[str] = []
        if self.config.notes:
            notes.append(self.config.notes)

        patient_ids = self.collect_patient_ids(
            diagnoses_df=diagnoses_df,
            medications_df=medications_df,
            seizure_events_df=seizure_events_df,
        )
        if not patient_ids:
            notes.append("no patients found in provided EHR tables")
            empty_df = pd.DataFrame(columns=self._output_columns())
            return empty_df, self._build_result(dataset_name, empty_df, notes)

        rows: list[dict[str, object]] = []
        for patient_id in patient_ids:
            bundle = self.evidence_builder.build_patient_bundle(
                patient_id=patient_id,
                diagnoses_df=diagnoses_df,
                medications_df=medications_df,
                seizure_events_df=seizure_events_df,
            )
            result = self.definition_engine.evaluate(bundle)
            rows.append(self.build_patient_row(bundle=bundle, result=result))

        output_df = pd.DataFrame(rows, columns=self._output_columns())
        if self.config.sort_output_by_patient_id and not output_df.empty:
            output_df = output_df.sort_values("patient_id").reset_index(drop=True)

        return output_df, self._build_result(dataset_name, output_df, notes)

    def collect_patient_ids(
        self,
        diagnoses_df: pd.DataFrame | None = None,
        medications_df: pd.DataFrame | None = None,
        seizure_events_df: pd.DataFrame | None = None,
    ) -> list[str]:
        """Collect unique patient IDs across provided source tables."""

        patient_ids: set[str] = set()
        for df in (diagnoses_df, medications_df, seizure_events_df):
            patient_ids.update(self._extract_patient_ids_from_df(df))

        collected = list(patient_ids)
        if self.config.sort_output_by_patient_id:
            return sorted(collected)
        return collected

    def build_patient_row(
        self,
        bundle: PatientEvidenceBundle,
        result: LikelyDREResult,
    ) -> dict[str, object]:
        """Convert evidence and definition outputs into one flat row."""

        row: dict[str, object] = {
            "patient_id": bundle.patient_id,
            "likely_dre": result.likely_dre,
            "definition_version": result.definition_version,
            "evidence_completeness_score": result.evidence_completeness_score,
            "epilepsy_evidence_status": bundle.epilepsy_evidence.status.value,
            "asm_evidence_status": bundle.asm_treatment_evidence.status.value,
            "seizure_burden_status": bundle.seizure_burden_evidence.status.value,
            "has_epilepsy_diagnosis": bundle.epilepsy_evidence.has_epilepsy_diagnosis,
            "has_recurrent_seizure_care": bundle.epilepsy_evidence.has_recurrent_seizure_care,
            "distinct_asm_count": bundle.asm_treatment_evidence.distinct_asm_count,
            "has_two_or_more_distinct_asms": (
                bundle.asm_treatment_evidence.has_two_or_more_distinct_asms
            ),
            "second_asm_start_time": self._format_datetime_for_output(
                bundle.asm_treatment_evidence.second_asm_start_time
            ),
            "post_second_asm_event_count": (
                bundle.seizure_burden_evidence.post_second_asm_event_count
            ),
            "has_persistent_seizure_burden": (
                bundle.seizure_burden_evidence.has_persistent_seizure_burden
            ),
        }

        if self.config.include_reason_columns:
            row["reasons"] = self._serialize_list_field(result.reasons)
        if self.config.include_missing_evidence_columns:
            row["missing_evidence"] = self._serialize_list_field(result.missing_evidence)
        if self.config.include_source_modality_columns:
            row["modality_sources"] = self._serialize_list_field(
                [modality.value for modality in bundle.modality_sources]
            )

        return row

    def _extract_patient_ids_from_df(self, df: pd.DataFrame | None) -> set[str]:
        if df is None or df.empty or self.config.patient_id_col not in df.columns:
            return set()
        values = df[self.config.patient_id_col].dropna().astype(str)
        return set(values)

    @staticmethod
    def _format_datetime_for_output(value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat()

    @staticmethod
    def _serialize_list_field(values: list[str]) -> str:
        return "|".join(str(value) for value in values)

    def _output_columns(self) -> list[str]:
        columns = [
            "patient_id",
            "likely_dre",
            "definition_version",
            "evidence_completeness_score",
            "epilepsy_evidence_status",
            "asm_evidence_status",
            "seizure_burden_status",
            "has_epilepsy_diagnosis",
            "has_recurrent_seizure_care",
            "distinct_asm_count",
            "has_two_or_more_distinct_asms",
            "second_asm_start_time",
            "post_second_asm_event_count",
            "has_persistent_seizure_burden",
        ]
        if self.config.include_reason_columns:
            columns.append("reasons")
        if self.config.include_missing_evidence_columns:
            columns.append("missing_evidence")
        if self.config.include_source_modality_columns:
            columns.append("modality_sources")
        return columns

    @staticmethod
    def _build_result(
        dataset_name: str,
        df: pd.DataFrame,
        notes: list[str],
    ) -> PatientDatasetBuildResult:
        positive_count = int(df["likely_dre"].sum()) if "likely_dre" in df.columns else 0
        row_count = len(df)
        return PatientDatasetBuildResult(
            dataset_name=dataset_name,
            row_count=row_count,
            patient_count=row_count,
            column_names=[str(column) for column in df.columns],
            likely_dre_positive_count=positive_count,
            likely_dre_negative_count=row_count - positive_count,
            notes=list(dict.fromkeys(notes)),
        )
