import pandas as pd

from core import EvidenceStatus
from features import EHRBuilderConfig, EHRColumnConfig, EHREvidenceBuilder


def _builder() -> EHREvidenceBuilder:
    return EHREvidenceBuilder(
        column_config=EHRColumnConfig(
            diagnosis_time_col="diagnosis_time",
            medication_start_col="medication_start",
            seizure_event_time_col="event_time",
            seizure_event_type_col="event_type",
        )
    )


def test_happy_path_bundle_builds_observed_evidence() -> None:
    diagnoses = pd.DataFrame(
        {
            "patient_id": ["p1"],
            "diagnosis_code": ["G40.909"],
            "diagnosis_time": ["2025-01-01"],
        }
    )
    medications = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "medication_name": ["Keppra", "lamotrigine"],
            "medication_start": ["2025-01-10", "2025-02-01"],
        }
    )
    seizure_events = pd.DataFrame(
        {
            "patient_id": ["p1"],
            "event_time": ["2025-02-15"],
            "event_type": ["ed_visit"],
        }
    )

    bundle = _builder().build_patient_bundle(
        patient_id="p1",
        diagnoses_df=diagnoses,
        medications_df=medications,
        seizure_events_df=seizure_events,
    )

    assert bundle.epilepsy_evidence.status is EvidenceStatus.OBSERVED
    assert bundle.epilepsy_evidence.has_epilepsy_diagnosis is True
    assert bundle.asm_treatment_evidence.has_two_or_more_distinct_asms is True
    assert bundle.seizure_burden_evidence.has_persistent_seizure_burden is True
    assert bundle.completeness.completeness_score == 1.0


def test_no_epilepsy_diagnosis_can_remain_missing() -> None:
    medications = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "medication_name": ["levetiracetam", "lamotrigine"],
            "medication_start": ["2025-01-10", "2025-02-01"],
        }
    )
    seizure_events = pd.DataFrame(
        {
            "patient_id": ["p1"],
            "event_time": ["2025-02-15"],
            "event_type": ["ed_visit"],
        }
    )

    bundle = _builder().build_patient_bundle(
        patient_id="p1",
        medications_df=medications,
        seizure_events_df=seizure_events,
    )

    assert bundle.epilepsy_evidence.status is EvidenceStatus.MISSING
    assert bundle.epilepsy_evidence.has_epilepsy_diagnosis is False


def test_fewer_than_two_asms_does_not_claim_two_attempts() -> None:
    medications = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "medication_name": ["Keppra", "levetiracetam"],
            "medication_start": ["2025-01-10", "2025-02-01"],
        }
    )

    evidence = _builder().build_asm_treatment_evidence("p1", medications)

    assert evidence.distinct_asm_count == 1
    assert evidence.has_two_or_more_distinct_asms is False


def test_no_post_second_asm_event_does_not_mark_persistent_burden() -> None:
    seizure_events = pd.DataFrame(
        {
            "patient_id": ["p1"],
            "event_time": ["2025-01-15"],
            "event_type": ["ed_visit"],
        }
    )

    evidence = _builder().build_seizure_burden_evidence(
        patient_id="p1",
        seizure_events_df=seizure_events,
        second_asm_start_time=pd.Timestamp("2025-02-01").to_pydatetime(),
    )

    assert evidence.status is EvidenceStatus.OBSERVED
    assert evidence.has_persistent_seizure_burden is False
    assert evidence.post_second_asm_event_count == 0


def test_missing_optional_time_columns_do_not_crash() -> None:
    builder = EHREvidenceBuilder()
    diagnoses = pd.DataFrame({"patient_id": ["p1"], "diagnosis_code": ["G40.909"]})

    evidence = builder.build_epilepsy_evidence("p1", diagnoses_df=diagnoses)

    assert evidence.status is EvidenceStatus.OBSERVED
    assert evidence.first_evidence_time is None


def test_medication_aliases_normalize_consistently() -> None:
    builder = _builder()
    medications = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "medication_name": ["Keppra", "levetiracetam"],
        }
    )

    evidence = builder.build_asm_treatment_evidence("p1", medications)

    assert [schedule.normalized_medication_name for schedule in evidence.schedules] == [
        "levetiracetam",
        "levetiracetam",
    ]
    assert evidence.distinct_asm_count == 1


def test_completeness_score_is_deterministic() -> None:
    builder = EHREvidenceBuilder(
        builder_config=EHRBuilderConfig(recurrent_seizure_event_threshold=2)
    )
    epilepsy = builder.build_epilepsy_evidence("p1")
    asm = builder.build_asm_treatment_evidence("p1")
    burden = builder.build_seizure_burden_evidence("p1")

    completeness = builder.build_completeness(epilepsy, asm, burden)

    assert completeness.completeness_score == 0.0
    assert completeness.missing_fields == [
        "epilepsy_evidence",
        "asm_treatment_evidence",
        "seizure_burden_evidence",
    ]
