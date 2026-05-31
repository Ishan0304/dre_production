from datetime import UTC, datetime

from core import EvidenceStatus, ModalityType
from definitions import (
    ASMTreatmentEvidence,
    EpilepsyEvidence,
    EvidenceCompleteness,
    LikelyDREDefinition,
    PatientEvidenceBundle,
    SeizureBurdenEvidence,
)
from definitions.likely_dre_definition import ASM_REASON, EPILEPSY_REASON, SEIZURE_BURDEN_REASON


SECOND_ASM_TIME = datetime(2025, 2, 1, tzinfo=UTC)
SEIZURE_TIME = datetime(2025, 3, 1, tzinfo=UTC)


def _bundle(
    *,
    has_epilepsy_diagnosis: bool = True,
    has_recurrent_seizure_care: bool = False,
    epilepsy_status: EvidenceStatus = EvidenceStatus.OBSERVED,
    has_two_or_more_distinct_asms: bool = True,
    distinct_asm_count: int = 2,
    asm_status: EvidenceStatus = EvidenceStatus.OBSERVED,
    second_asm_start_time: datetime | None = SECOND_ASM_TIME,
    has_persistent_seizure_burden: bool = True,
    post_second_asm_event_count: int = 1,
    seizure_status: EvidenceStatus = EvidenceStatus.OBSERVED,
) -> PatientEvidenceBundle:
    return PatientEvidenceBundle(
        patient_id="patient-001",
        epilepsy_evidence=EpilepsyEvidence(
            status=epilepsy_status,
            has_epilepsy_diagnosis=has_epilepsy_diagnosis,
            has_recurrent_seizure_care=has_recurrent_seizure_care,
        ),
        asm_treatment_evidence=ASMTreatmentEvidence(
            status=asm_status,
            distinct_asm_count=distinct_asm_count,
            has_two_or_more_distinct_asms=has_two_or_more_distinct_asms,
            second_asm_start_time=second_asm_start_time,
        ),
        seizure_burden_evidence=SeizureBurdenEvidence(
            status=seizure_status,
            has_persistent_seizure_burden=has_persistent_seizure_burden,
            post_second_asm_event_count=post_second_asm_event_count,
            first_post_second_asm_event_time=SEIZURE_TIME
            if post_second_asm_event_count
            else None,
        ),
        completeness=EvidenceCompleteness(
            epilepsy_evidence_status=epilepsy_status,
            asm_evidence_status=asm_status,
            seizure_burden_status=seizure_status,
            completeness_score=0.95,
        ),
        modality_sources=[ModalityType.EHR],
    )


def test_happy_path_returns_likely_dre_true() -> None:
    result = LikelyDREDefinition().evaluate(_bundle())

    assert result.likely_dre is True
    assert result.patient_id == "patient-001"
    assert result.evidence_completeness_score == 0.95
    assert result.missing_evidence == []


def test_missing_epilepsy_evidence_returns_false() -> None:
    result = LikelyDREDefinition().evaluate(
        _bundle(
            has_epilepsy_diagnosis=False,
            has_recurrent_seizure_care=False,
            epilepsy_status=EvidenceStatus.MISSING,
        )
    )

    assert result.likely_dre is False
    assert "missing epilepsy evidence" in result.missing_evidence


def test_fewer_than_two_asms_returns_false() -> None:
    result = LikelyDREDefinition().evaluate(
        _bundle(has_two_or_more_distinct_asms=False, distinct_asm_count=1)
    )

    assert result.likely_dre is False
    assert "missing ASM treatment evidence" in result.missing_evidence


def test_no_post_second_asm_seizure_burden_returns_false() -> None:
    result = LikelyDREDefinition().evaluate(
        _bundle(has_persistent_seizure_burden=False, post_second_asm_event_count=0)
    )

    assert result.likely_dre is False
    assert "missing seizure burden evidence" in result.missing_evidence


def test_reasons_include_satisfied_conditions() -> None:
    result = LikelyDREDefinition().evaluate(
        _bundle(has_persistent_seizure_burden=False, post_second_asm_event_count=0)
    )

    assert EPILEPSY_REASON in result.reasons
    assert ASM_REASON in result.reasons
    assert SEIZURE_BURDEN_REASON not in result.reasons


def test_definition_version_is_configurable() -> None:
    result = LikelyDREDefinition(definition_version="v1-test").evaluate(_bundle())

    assert result.definition_version == "v1-test"


def test_evaluation_summary_reports_component_conditions() -> None:
    summary = LikelyDREDefinition().evaluation_summary(
        _bundle(has_two_or_more_distinct_asms=False, distinct_asm_count=1)
    )

    assert summary["epilepsy_condition"] is True
    assert summary["asm_condition"] is False
    assert summary["seizure_burden_condition"] is True
    assert summary["likely_dre"] is False
