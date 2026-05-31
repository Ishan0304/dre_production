"""Build clinical evidence bundles from normalized EHR-like tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from core import EvidenceStatus, ModalityType
from definitions import (
    ASMScheduleRecord,
    ASMTreatmentEvidence,
    EpilepsyEvidence,
    EvidenceCompleteness,
    PatientEvidenceBundle,
    SeizureBurdenEvent,
    SeizureBurdenEvidence,
)


@dataclass(slots=True)
class EHRColumnConfig:
    """Column mapping for normalized EHR-like source tables."""

    patient_id_col: str = "patient_id"
    diagnosis_code_col: str = "diagnosis_code"
    diagnosis_time_col: str | None = None
    encounter_time_col: str | None = None
    encounter_type_col: str | None = None
    medication_name_col: str = "medication_name"
    medication_start_col: str | None = None
    medication_end_col: str | None = None
    medication_source_col: str | None = None
    seizure_event_time_col: str | None = None
    seizure_event_type_col: str | None = None


@dataclass(slots=True)
class EHRBuilderConfig:
    """Evidence builder settings and source-independent clinical patterns."""

    epilepsy_code_prefixes: tuple[str, ...] = ("G40", "R56", "345", "780")
    seizure_code_prefixes: tuple[str, ...] = ("R56", "780", "345", "G40")
    asm_name_patterns: tuple[str, ...] = (
        "levetiracetam",
        "keppra",
        "valpro",
        "divalproex",
        "lamotrigine",
        "carbamazepine",
        "oxcarbazepine",
        "phenytoin",
        "topiramate",
        "zonisamide",
        "lacosamide",
        "clobazam",
        "clonazepam",
        "phenobarbital",
    )
    recurrent_seizure_event_threshold: int = 2
    require_post_second_asm_for_persistent_burden: bool = True
    notes: str | None = None


class EHREvidenceBuilder:
    """Construct patient evidence bundles from generic longitudinal EHR tables."""

    def __init__(
        self,
        column_config: EHRColumnConfig | None = None,
        builder_config: EHRBuilderConfig | None = None,
    ) -> None:
        self.column_config = column_config or EHRColumnConfig()
        self.builder_config = builder_config or EHRBuilderConfig()

    def build_patient_bundle(
        self,
        patient_id: str,
        diagnoses_df: pd.DataFrame | None = None,
        medications_df: pd.DataFrame | None = None,
        seizure_events_df: pd.DataFrame | None = None,
    ) -> PatientEvidenceBundle:
        """Build a complete evidence bundle without applying the final definition."""

        epilepsy_evidence = self.build_epilepsy_evidence(
            patient_id=patient_id,
            diagnoses_df=diagnoses_df,
            seizure_events_df=seizure_events_df,
        )
        asm_treatment_evidence = self.build_asm_treatment_evidence(
            patient_id=patient_id,
            medications_df=medications_df,
        )
        seizure_burden_evidence = self.build_seizure_burden_evidence(
            patient_id=patient_id,
            seizure_events_df=seizure_events_df,
            second_asm_start_time=asm_treatment_evidence.second_asm_start_time,
            diagnoses_df=diagnoses_df,
        )
        completeness = self.build_completeness(
            epilepsy_evidence=epilepsy_evidence,
            asm_treatment_evidence=asm_treatment_evidence,
            seizure_burden_evidence=seizure_burden_evidence,
        )

        return PatientEvidenceBundle(
            patient_id=patient_id,
            epilepsy_evidence=epilepsy_evidence,
            asm_treatment_evidence=asm_treatment_evidence,
            seizure_burden_evidence=seizure_burden_evidence,
            completeness=completeness,
            modality_sources=[ModalityType.EHR],
            source_records={},
            notes="Evidence bundle built from EHR-like source tables.",
        )

    def build_epilepsy_evidence(
        self,
        patient_id: str,
        diagnoses_df: pd.DataFrame | None = None,
        seizure_events_df: pd.DataFrame | None = None,
    ) -> EpilepsyEvidence:
        """Build evidence for epilepsy or recurrent seizure-related care."""

        notes: list[str] = []
        evidence_sources: list[str] = []
        diagnosis_codes: list[str] = []
        evidence_times: list[datetime] = []

        diagnosis_rows = self._filter_patient_rows(diagnoses_df, patient_id)
        seizure_rows = self._filter_patient_rows(seizure_events_df, patient_id)

        has_diagnosis_data = diagnosis_rows is not None and not diagnosis_rows.empty
        has_seizure_event_data = seizure_rows is not None and not seizure_rows.empty

        epilepsy_code_rows = pd.DataFrame()
        seizure_code_rows = pd.DataFrame()
        if has_diagnosis_data and self.column_config.diagnosis_code_col in diagnosis_rows.columns:
            codes = diagnosis_rows[self.column_config.diagnosis_code_col].astype(str)
            epilepsy_mask = codes.map(
                lambda code: self._matches_any_prefix(
                    code,
                    self.builder_config.epilepsy_code_prefixes,
                )
            )
            seizure_mask = codes.map(
                lambda code: self._matches_any_prefix(
                    code,
                    self.builder_config.seizure_code_prefixes,
                )
            )
            epilepsy_code_rows = diagnosis_rows[epilepsy_mask]
            seizure_code_rows = diagnosis_rows[seizure_mask]
            diagnosis_codes = sorted(
                {
                    str(code)
                    for code in epilepsy_code_rows[self.column_config.diagnosis_code_col].dropna()
                }
            )
            if not epilepsy_code_rows.empty or not seizure_code_rows.empty:
                evidence_sources.append("diagnoses")
                matched_rows = pd.concat([epilepsy_code_rows, seizure_code_rows]).drop_duplicates()
                evidence_times.extend(self._extract_times(matched_rows, self._diagnosis_time_columns()))
        elif diagnoses_df is not None:
            notes.append("diagnosis code column unavailable")

        seizure_event_count = 0 if seizure_rows is None else len(seizure_rows)
        seizure_code_count = len(seizure_code_rows)
        recurrent_count = seizure_event_count + seizure_code_count
        has_epilepsy_diagnosis = not epilepsy_code_rows.empty
        has_recurrent_seizure_care = (
            recurrent_count >= self.builder_config.recurrent_seizure_event_threshold
        )

        if has_seizure_event_data:
            evidence_sources.append("seizure_events")
            evidence_times.extend(
                self._extract_times(seizure_rows, [self.column_config.seizure_event_time_col])
            )

        status = (
            EvidenceStatus.OBSERVED
            if has_epilepsy_diagnosis or has_recurrent_seizure_care
            else EvidenceStatus.MISSING
        )
        if not has_diagnosis_data and not has_seizure_event_data:
            notes.append("no usable epilepsy or seizure care source rows available")

        return EpilepsyEvidence(
            status=status,
            has_epilepsy_diagnosis=has_epilepsy_diagnosis,
            has_recurrent_seizure_care=has_recurrent_seizure_care,
            diagnosis_codes=diagnosis_codes,
            evidence_sources=sorted(set(evidence_sources)),
            first_evidence_time=min(evidence_times) if evidence_times else None,
            last_evidence_time=max(evidence_times) if evidence_times else None,
            notes="; ".join(notes) if notes else None,
        )

    def build_asm_treatment_evidence(
        self,
        patient_id: str,
        medications_df: pd.DataFrame | None = None,
    ) -> ASMTreatmentEvidence:
        """Build normalized antiseizure medication treatment evidence."""

        notes: list[str] = []
        medication_rows = self._filter_patient_rows(medications_df, patient_id)
        if medication_rows is None or medication_rows.empty:
            return ASMTreatmentEvidence(
                status=EvidenceStatus.MISSING,
                schedules=[],
                distinct_asm_count=0,
                has_two_or_more_distinct_asms=False,
                second_asm_start_time=None,
                evidence_sources=[],
                notes="no usable medication source rows available",
            )
        if self.column_config.medication_name_col not in medication_rows.columns:
            return ASMTreatmentEvidence(
                status=EvidenceStatus.MISSING,
                schedules=[],
                distinct_asm_count=0,
                has_two_or_more_distinct_asms=False,
                second_asm_start_time=None,
                evidence_sources=[],
                notes="medication name column unavailable",
            )

        schedules: list[ASMScheduleRecord] = []
        for _, row in medication_rows.iterrows():
            medication_name = row.get(self.column_config.medication_name_col)
            if pd.isna(medication_name):
                continue
            medication_name_text = str(medication_name)
            normalized_name = self._normalize_medication_name(medication_name_text)
            if not normalized_name:
                continue

            schedules.append(
                ASMScheduleRecord(
                    medication_name=medication_name_text,
                    normalized_medication_name=normalized_name,
                    start_time=self._row_time(row, self.column_config.medication_start_col),
                    end_time=self._row_time(row, self.column_config.medication_end_col),
                    source=self._row_string(row, self.column_config.medication_source_col),
                    status=EvidenceStatus.OBSERVED,
                )
            )

        schedules = sorted(
            schedules,
            key=lambda schedule: (
                schedule.start_time is None,
                schedule.start_time or datetime.max,
                schedule.normalized_medication_name,
            ),
        )
        distinct_names = list(dict.fromkeys(schedule.normalized_medication_name for schedule in schedules))
        second_asm_start_time = self._second_distinct_asm_start_time(schedules)
        if not schedules:
            notes.append("no configured ASM medication patterns found")

        return ASMTreatmentEvidence(
            status=EvidenceStatus.OBSERVED if schedules else EvidenceStatus.MISSING,
            schedules=schedules,
            distinct_asm_count=len(distinct_names),
            has_two_or_more_distinct_asms=len(distinct_names) >= 2,
            second_asm_start_time=second_asm_start_time,
            evidence_sources=["medications"] if schedules else [],
            notes="; ".join(notes) if notes else None,
        )

    def build_seizure_burden_evidence(
        self,
        patient_id: str,
        seizure_events_df: pd.DataFrame | None = None,
        second_asm_start_time: datetime | None = None,
        diagnoses_df: pd.DataFrame | None = None,
    ) -> SeizureBurdenEvidence:
        """Build seizure burden evidence from event rows or seizure-coded diagnoses."""

        notes: list[str] = []
        evidence_sources: list[str] = []
        event_rows = self._filter_patient_rows(seizure_events_df, patient_id)
        events: list[SeizureBurdenEvent] = []

        if event_rows is not None and not event_rows.empty:
            events.extend(self._events_from_seizure_rows(event_rows))
            evidence_sources.append("seizure_events")
        else:
            diagnosis_rows = self._filter_patient_rows(diagnoses_df, patient_id)
            fallback_events = self._events_from_seizure_diagnoses(diagnosis_rows)
            if fallback_events:
                events.extend(fallback_events)
                evidence_sources.append("diagnoses")

        events = sorted(
            events,
            key=lambda event: (
                event.event_time is None,
                event.event_time or datetime.max,
                event.event_type,
            ),
        )

        post_second_events = self._post_second_asm_events(events, second_asm_start_time)
        first_post_second_event_time = (
            min(event.event_time for event in post_second_events if event.event_time is not None)
            if post_second_events
            else None
        )

        if self.builder_config.require_post_second_asm_for_persistent_burden:
            has_persistent_burden = len(post_second_events) >= 1
            if second_asm_start_time is None:
                notes.append("second ASM temporal anchor unavailable")
        else:
            has_persistent_burden = len(events) >= 1

        if not events:
            notes.append("no seizure burden event evidence found")

        return SeizureBurdenEvidence(
            status=EvidenceStatus.OBSERVED if events else EvidenceStatus.MISSING,
            events=events,
            has_persistent_seizure_burden=has_persistent_burden,
            post_second_asm_event_count=len(post_second_events),
            first_post_second_asm_event_time=first_post_second_event_time,
            evidence_sources=sorted(set(evidence_sources)),
            notes="; ".join(notes) if notes else None,
        )

    @staticmethod
    def build_completeness(
        epilepsy_evidence: EpilepsyEvidence,
        asm_treatment_evidence: ASMTreatmentEvidence,
        seizure_burden_evidence: SeizureBurdenEvidence,
    ) -> EvidenceCompleteness:
        """Summarize evidence completeness across required domains."""

        statuses = [
            epilepsy_evidence.status,
            asm_treatment_evidence.status,
            seizure_burden_evidence.status,
        ]
        missing_fields: list[str] = []
        if epilepsy_evidence.status is EvidenceStatus.MISSING:
            missing_fields.append("epilepsy_evidence")
        if asm_treatment_evidence.status is EvidenceStatus.MISSING:
            missing_fields.append("asm_treatment_evidence")
        if seizure_burden_evidence.status is EvidenceStatus.MISSING:
            missing_fields.append("seizure_burden_evidence")

        present_count = sum(status is not EvidenceStatus.MISSING for status in statuses)
        completeness_score = present_count / len(statuses)

        return EvidenceCompleteness(
            epilepsy_evidence_status=epilepsy_evidence.status,
            asm_evidence_status=asm_treatment_evidence.status,
            seizure_burden_status=seizure_burden_evidence.status,
            missing_fields=missing_fields,
            completeness_score=completeness_score,
            notes="Completeness is the proportion of evidence domains that are not missing.",
        )

    def _filter_patient_rows(
        self,
        df: pd.DataFrame | None,
        patient_id: str,
    ) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        patient_col = self.column_config.patient_id_col
        if patient_col not in df.columns:
            return None
        return df[df[patient_col].astype(str) == str(patient_id)].copy()

    def _diagnosis_time_columns(self) -> list[str | None]:
        return [self.column_config.diagnosis_time_col, self.column_config.encounter_time_col]

    @staticmethod
    def _extract_times(df: pd.DataFrame, candidate_columns: list[str | None]) -> list[datetime]:
        times: list[datetime] = []
        for column in candidate_columns:
            if not column or column not in df.columns:
                continue
            parsed = pd.to_datetime(df[column], errors="coerce").dropna()
            for value in parsed:
                times.append(_to_python_datetime(value))
        return times

    @staticmethod
    def _row_time(row: pd.Series, column_name: str | None) -> datetime | None:
        if not column_name or column_name not in row.index:
            return None
        value = pd.to_datetime(row[column_name], errors="coerce")
        if pd.isna(value):
            return None
        return _to_python_datetime(value)

    @staticmethod
    def _row_string(row: pd.Series, column_name: str | None) -> str | None:
        if not column_name or column_name not in row.index or pd.isna(row[column_name]):
            return None
        return str(row[column_name])

    @staticmethod
    def _matches_any_prefix(value: str, prefixes: tuple[str, ...]) -> bool:
        normalized = value.strip().upper().replace(".", "")
        return any(normalized.startswith(prefix.upper().replace(".", "")) for prefix in prefixes)

    def _normalize_medication_name(self, medication_name: str) -> str | None:
        normalized = " ".join(medication_name.lower().strip().split())
        if not self._matches_any_asm_pattern(normalized):
            return None
        if "keppra" in normalized or "levetiracetam" in normalized:
            return "levetiracetam"
        if "divalproex" in normalized or "valpro" in normalized:
            return "valproate"
        return next(
            pattern
            for pattern in self.builder_config.asm_name_patterns
            if pattern in normalized
        )

    def _matches_any_asm_pattern(self, medication_name: str) -> bool:
        return any(pattern in medication_name for pattern in self.builder_config.asm_name_patterns)

    @staticmethod
    def _second_distinct_asm_start_time(schedules: list[ASMScheduleRecord]) -> datetime | None:
        seen_names: set[str] = set()
        for schedule in schedules:
            if schedule.normalized_medication_name in seen_names:
                continue
            seen_names.add(schedule.normalized_medication_name)
            if len(seen_names) == 2:
                return schedule.start_time
        return None

    def _events_from_seizure_rows(self, rows: pd.DataFrame) -> list[SeizureBurdenEvent]:
        events: list[SeizureBurdenEvent] = []
        for _, row in rows.iterrows():
            event_type = (
                self._row_string(row, self.column_config.seizure_event_type_col)
                or "seizure_event"
            )
            events.append(
                SeizureBurdenEvent(
                    event_type=event_type,
                    event_time=self._row_time(row, self.column_config.seizure_event_time_col),
                    source="seizure_events",
                    status=EvidenceStatus.OBSERVED,
                    description="Seizure burden signal from event table.",
                )
            )
        return events

    def _events_from_seizure_diagnoses(self, rows: pd.DataFrame | None) -> list[SeizureBurdenEvent]:
        if rows is None or rows.empty or self.column_config.diagnosis_code_col not in rows.columns:
            return []
        codes = rows[self.column_config.diagnosis_code_col].astype(str)
        seizure_rows = rows[
            codes.map(
                lambda code: self._matches_any_prefix(
                    code,
                    self.builder_config.seizure_code_prefixes,
                )
            )
        ]
        events: list[SeizureBurdenEvent] = []
        for _, row in seizure_rows.iterrows():
            code = str(row[self.column_config.diagnosis_code_col])
            events.append(
                SeizureBurdenEvent(
                    event_type="seizure_coded_diagnosis",
                    event_time=self._row_time_from_candidates(row, self._diagnosis_time_columns()),
                    source="diagnoses",
                    status=EvidenceStatus.OBSERVED,
                    description=f"Seizure-coded diagnosis: {code}",
                )
            )
        return events

    @staticmethod
    def _row_time_from_candidates(
        row: pd.Series,
        candidate_columns: list[str | None],
    ) -> datetime | None:
        for column in candidate_columns:
            if not column or column not in row.index:
                continue
            value = pd.to_datetime(row[column], errors="coerce")
            if not pd.isna(value):
                return _to_python_datetime(value)
        return None

    @staticmethod
    def _post_second_asm_events(
        events: list[SeizureBurdenEvent],
        second_asm_start_time: datetime | None,
    ) -> list[SeizureBurdenEvent]:
        if second_asm_start_time is None:
            return []
        return [
            event
            for event in events
            if event.event_time is not None and event.event_time >= second_asm_start_time
        ]


def _to_python_datetime(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value
