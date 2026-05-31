import pandas as pd

from datasets import EHRPatientDatasetBuilder, PatientDatasetBuildConfig
from features import EHRColumnConfig, EHREvidenceBuilder


def _builder(config: PatientDatasetBuildConfig | None = None) -> EHRPatientDatasetBuilder:
    evidence_builder = EHREvidenceBuilder(
        column_config=EHRColumnConfig(
            diagnosis_time_col="diagnosis_time",
            medication_start_col="medication_start",
            seizure_event_time_col="event_time",
            seizure_event_type_col="event_type",
        )
    )
    return EHRPatientDatasetBuilder(evidence_builder=evidence_builder, config=config)


def test_happy_path_dataset_build_counts_positive_and_negative() -> None:
    diagnoses = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "diagnosis_code": ["G40.909", "G40.909"],
            "diagnosis_time": ["2025-01-01", "2025-01-01"],
        }
    )
    medications = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "medication_name": ["Keppra", "lamotrigine", "Keppra"],
            "medication_start": ["2025-01-10", "2025-02-01", "2025-01-10"],
        }
    )
    seizure_events = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "event_time": ["2025-02-15", "2025-01-20"],
            "event_type": ["ed_visit", "ed_visit"],
        }
    )

    df, result = _builder().build_patient_dataset(
        dataset_name="patient_labels",
        diagnoses_df=diagnoses,
        medications_df=medications,
        seizure_events_df=seizure_events,
    )

    assert result.row_count == 2
    assert result.patient_count == 2
    assert result.likely_dre_positive_count == 1
    assert result.likely_dre_negative_count == 1
    assert set(df["patient_id"]) == {"p1", "p2"}
    assert bool(df.loc[df["patient_id"] == "p1", "likely_dre"].iloc[0]) is True
    assert "distinct_asm_count" in df.columns


def test_collect_patient_ids_merges_sources_deterministically() -> None:
    diagnoses = pd.DataFrame({"patient_id": ["p2", "p1"]})
    medications = pd.DataFrame({"patient_id": ["p3"]})
    seizure_events = pd.DataFrame({"patient_id": ["p1", "p4"]})

    patient_ids = _builder().collect_patient_ids(
        diagnoses_df=diagnoses,
        medications_df=medications,
        seizure_events_df=seizure_events,
    )

    assert patient_ids == ["p1", "p2", "p3", "p4"]


def test_missing_tables_do_not_crash() -> None:
    medications = pd.DataFrame(
        {
            "patient_id": ["p1"],
            "medication_name": ["Keppra"],
            "medication_start": ["2025-01-10"],
        }
    )

    df, result = _builder().build_patient_dataset(
        dataset_name="medications_only",
        medications_df=medications,
    )

    assert len(df) == 1
    assert result.row_count == 1
    assert result.likely_dre_positive_count == 0
    assert df.loc[0, "epilepsy_evidence_status"] == "missing"


def test_empty_input_returns_empty_dataframe_and_result() -> None:
    df, result = _builder().build_patient_dataset(dataset_name="empty")

    assert df.empty
    assert result.dataset_name == "empty"
    assert result.row_count == 0
    assert result.patient_count == 0
    assert "no patients found in provided EHR tables" in result.notes


def test_row_serialization_is_consistent() -> None:
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
        {"patient_id": ["p1"], "event_time": ["2025-02-15"], "event_type": ["ed_visit"]}
    )

    df, _ = _builder().build_patient_dataset(
        dataset_name="patient_labels",
        diagnoses_df=diagnoses,
        medications_df=medications,
        seizure_events_df=seizure_events,
    )

    row = df.iloc[0]
    assert "epilepsy or recurrent seizure care evidence present" in row["reasons"]
    assert row["missing_evidence"] == ""
    assert row["modality_sources"] == "ehr"
    assert row["second_asm_start_time"] == "2025-02-01T00:00:00"


def test_definition_version_override_is_used() -> None:
    diagnoses = pd.DataFrame({"patient_id": ["p1"], "diagnosis_code": ["G40.909"]})
    config = PatientDatasetBuildConfig(definition_version_override="v1-custom")

    df, _ = _builder(config=config).build_patient_dataset(
        dataset_name="patient_labels",
        diagnoses_df=diagnoses,
    )

    assert df.loc[0, "definition_version"] == "v1-custom"
