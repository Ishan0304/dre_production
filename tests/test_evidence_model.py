from datetime import UTC, datetime

from core import EvidenceStatus, ModalityType
from definitions import (
    ASMScheduleRecord,
    ASMTreatmentEvidence,
    EpilepsyEvidence,
    EvidenceCompleteness,
    LikelyDREResult,
    PatientEvidenceBundle,
    SeizureBurdenEvent,
    SeizureBurdenEvidence,
)


def test_nested_patient_evidence_bundle_construction() -> None:
    diagnosis_time = datetime(2025, 1, 1, tzinfo=UTC)
    second_asm_time = datetime(2025, 2, 1, tzinfo=UTC)
    seizure_time = datetime(2025, 3, 1, tzinfo=UTC)

    epilepsy = EpilepsyEvidence(
        status=EvidenceStatus.OBSERVED,
        has_epilepsy_diagnosis=True,
        has_recurrent_seizure_care=True,
        diagnosis_codes=["G40.909"],
        evidence_sources=["encounters"],
        first_evidence_time=diagnosis_time,
        last_evidence_time=diagnosis_time,
    )
    schedule = ASMScheduleRecord(
        medication_name="Levetiracetam",
        normalized_medication_name="levetiracetam",
        status=EvidenceStatus.OBSERVED,
        start_time=second_asm_time,
        source="medications",
    )
    asm = ASMTreatmentEvidence(
        status=EvidenceStatus.OBSERVED,
        schedules=[schedule],
        distinct_asm_count=2,
        has_two_or_more_distinct_asms=True,
        second_asm_start_time=second_asm_time,
        evidence_sources=["medications"],
    )
    burden_event = SeizureBurdenEvent(
        event_type="seizure_coded_encounter",
        event_time=seizure_time,
        source="encounters",
        status=EvidenceStatus.OBSERVED,
        description="Seizure-coded encounter after second ASM start.",
    )
    burden = SeizureBurdenEvidence(
        status=EvidenceStatus.OBSERVED,
        events=[burden_event],
        has_persistent_seizure_burden=True,
        post_second_asm_event_count=1,
        first_post_second_asm_event_time=seizure_time,
        evidence_sources=["encounters"],
    )
    completeness = EvidenceCompleteness(
        epilepsy_evidence_status=EvidenceStatus.OBSERVED,
        asm_evidence_status=EvidenceStatus.OBSERVED,
        seizure_burden_status=EvidenceStatus.OBSERVED,
        missing_fields=[],
        completeness_score=1.0,
    )

    bundle = PatientEvidenceBundle(
        patient_id="patient-001",
        epilepsy_evidence=epilepsy,
        asm_treatment_evidence=asm,
        seizure_burden_evidence=burden,
        completeness=completeness,
        modality_sources=[ModalityType.EHR],
    )

    assert bundle.patient_id == "patient-001"
    assert bundle.modality_sources == [ModalityType.EHR]
    assert bundle.asm_treatment_evidence.schedules[0].normalized_medication_name == "levetiracetam"


def test_default_factories_are_not_shared() -> None:
    first = EpilepsyEvidence(
        status=EvidenceStatus.MISSING,
        has_epilepsy_diagnosis=False,
        has_recurrent_seizure_care=False,
    )
    second = EpilepsyEvidence(
        status=EvidenceStatus.MISSING,
        has_epilepsy_diagnosis=False,
        has_recurrent_seizure_care=False,
    )

    first.diagnosis_codes.append("G40.909")
    first.evidence_sources.append("encounters")

    assert second.diagnosis_codes == []
    assert second.evidence_sources == []


def test_to_dict_preserves_nested_values_and_enum_references() -> None:
    result = LikelyDREResult(
        patient_id="patient-002",
        likely_dre=False,
        reasons=["insufficient seizure burden evidence"],
        missing_evidence=["seizure_burden"],
        evidence_completeness_score=0.67,
        definition_version="contract-only",
    )

    as_dict = result.to_dict()

    assert as_dict["patient_id"] == "patient-002"
    assert as_dict["likely_dre"] is False
    assert as_dict["reasons"] == ["insufficient seizure burden evidence"]
    assert as_dict["missing_evidence"] == ["seizure_burden"]
    assert as_dict["evidence_completeness_score"] == 0.67
