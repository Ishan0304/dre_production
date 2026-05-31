"""Typed clinical evidence models for likely DRE inference inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from core import EvidenceStatus, ModalityType


@dataclass(slots=True)
class EpilepsyEvidence:
    """Evidence of documented epilepsy or recurrent seizure-related care."""

    status: EvidenceStatus
    has_epilepsy_diagnosis: bool
    has_recurrent_seizure_care: bool
    diagnosis_codes: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    first_evidence_time: datetime | None = None
    last_evidence_time: datetime | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class ASMScheduleRecord:
    """Normalized representation of one ASM schedule or regimen attempt."""

    medication_name: str
    normalized_medication_name: str
    status: EvidenceStatus
    start_time: datetime | None = None
    end_time: datetime | None = None
    source: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class ASMTreatmentEvidence:
    """Evidence summarizing normalized antiseizure medication history."""

    status: EvidenceStatus
    distinct_asm_count: int
    has_two_or_more_distinct_asms: bool
    schedules: list[ASMScheduleRecord] = field(default_factory=list)
    second_asm_start_time: datetime | None = None
    evidence_sources: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class SeizureBurdenEvent:
    """One seizure burden signal from a normalized clinical source."""

    event_type: str
    status: EvidenceStatus
    event_time: datetime | None = None
    source: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class SeizureBurdenEvidence:
    """Evidence of persistent seizure burden after treatment escalation."""

    status: EvidenceStatus
    has_persistent_seizure_burden: bool
    post_second_asm_event_count: int
    events: list[SeizureBurdenEvent] = field(default_factory=list)
    first_post_second_asm_event_time: datetime | None = None
    evidence_sources: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class EvidenceCompleteness:
    """Completeness summary for evidence domains required by the definition."""

    epilepsy_evidence_status: EvidenceStatus
    asm_evidence_status: EvidenceStatus
    seizure_burden_status: EvidenceStatus
    completeness_score: float
    missing_fields: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class PatientEvidenceBundle:
    """Patient-level evidence bundle for a future likely DRE definition engine."""

    patient_id: str
    epilepsy_evidence: EpilepsyEvidence
    asm_treatment_evidence: ASMTreatmentEvidence
    seizure_burden_evidence: SeizureBurdenEvidence
    completeness: EvidenceCompleteness
    modality_sources: list[ModalityType] = field(default_factory=list)
    source_records: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class LikelyDREResult:
    """Typed output contract for the future likely DRE definition engine."""

    patient_id: str
    likely_dre: bool
    definition_version: str
    reasons: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    evidence_completeness_score: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)
