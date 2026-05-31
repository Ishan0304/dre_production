import json
from datetime import UTC, datetime

import pandas as pd

from core.contracts import RunContextRecord
from datasets import EHRPipelineInputConfig, EHRPipelineRunner
from ingestion import TableLoadRequest


def _run_context() -> RunContextRecord:
    return RunContextRecord(
        run_id="run-001",
        project_name="dre_production",
        stage_name="ehr_pipeline",
        timestamp_utc=datetime(2026, 5, 30, tzinfo=UTC),
    )


def _write_inputs(tmp_path):
    diagnoses_path = tmp_path / "diagnoses.csv"
    medications_path = tmp_path / "medications.csv"
    seizure_events_path = tmp_path / "seizure_events.csv"

    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "diagnosis_code": ["G40.909", "G40.909"],
        }
    ).to_csv(diagnoses_path, index=False)
    pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "medication_name": ["Keppra", "lamotrigine", "Keppra"],
            "medication_start": ["2025-01-10", "2025-02-01", "2025-01-10"],
        }
    ).to_csv(medications_path, index=False)
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "event_time": ["2025-02-15", "2025-01-20"],
            "event_type": ["ed_visit", "ed_visit"],
        }
    ).to_csv(seizure_events_path, index=False)

    return diagnoses_path, medications_path, seizure_events_path


def _input_config(tmp_path, output_dir=None) -> EHRPipelineInputConfig:
    diagnoses_path, medications_path, seizure_events_path = _write_inputs(tmp_path)
    return EHRPipelineInputConfig(
        dataset_name="ehr_patient_dataset",
        diagnoses_request=TableLoadRequest(path=str(diagnoses_path)),
        medications_request=TableLoadRequest(
            path=str(medications_path),
            parse_date_columns=["medication_start"],
        ),
        seizure_events_request=TableLoadRequest(
            path=str(seizure_events_path),
            parse_date_columns=["event_time"],
        ),
        output_dir=str(output_dir or tmp_path / "outputs"),
    )


def test_happy_path_run_writes_outputs_and_counts_labels(tmp_path) -> None:
    result = EHRPipelineRunner().run(_input_config(tmp_path), _run_context())

    assert result.patient_dataset_row_count == 2
    assert result.likely_dre_positive_count == 1
    assert result.likely_dre_negative_count == 1
    assert result.patient_dataset_path is not None
    assert result.dataset_profile_path is not None
    assert result.artifact_registry_path is not None
    assert result.run_manifest_path is not None


def test_missing_optional_tables_records_notes(tmp_path) -> None:
    diagnoses_path = tmp_path / "diagnoses.csv"
    pd.DataFrame({"patient_id": ["p1"], "diagnosis_code": ["G40.909"]}).to_csv(
        diagnoses_path,
        index=False,
    )
    config = EHRPipelineInputConfig(
        dataset_name="ehr_patient_dataset",
        diagnoses_request=TableLoadRequest(path=str(diagnoses_path)),
        medications_request=None,
        seizure_events_request=None,
        output_dir=str(tmp_path / "outputs"),
    )

    result = EHRPipelineRunner().run(config, _run_context())

    assert result.patient_dataset_row_count == 1
    assert "medications table request absent" in result.notes
    assert "seizure_events table request absent" in result.notes


def test_empty_inputs_still_write_coherent_artifacts(tmp_path) -> None:
    diagnoses_path = tmp_path / "diagnoses.csv"
    pd.DataFrame(columns=["patient_id", "diagnosis_code"]).to_csv(diagnoses_path, index=False)
    config = EHRPipelineInputConfig(
        dataset_name="empty",
        diagnoses_request=TableLoadRequest(path=str(diagnoses_path)),
        medications_request=None,
        seizure_events_request=None,
        output_dir=str(tmp_path / "outputs"),
    )

    result = EHRPipelineRunner().run(config, _run_context())

    assert result.patient_dataset_row_count == 0
    assert result.likely_dre_positive_count == 0
    assert result.run_manifest_path is not None


def test_artifact_registry_contains_expected_artifacts(tmp_path) -> None:
    result = EHRPipelineRunner().run(_input_config(tmp_path), _run_context())

    payload = json.loads(open(result.artifact_registry_path, encoding="utf-8").read())
    by_name = {artifact["artifact_name"]: artifact for artifact in payload["artifacts"]}

    assert by_name["patient_dataset"]["artifact_type"] == "table"
    assert by_name["dataset_profile"]["artifact_type"] == "json"
    assert by_name["artifact_registry"]["artifact_type"] == "json"
    assert by_name["run_manifest"]["artifact_type"] == "json"


def test_run_manifest_and_registry_json_are_created(tmp_path) -> None:
    result = EHRPipelineRunner().run(_input_config(tmp_path), _run_context())

    registry_payload = json.loads(open(result.artifact_registry_path, encoding="utf-8").read())
    manifest_payload = json.loads(open(result.run_manifest_path, encoding="utf-8").read())

    assert registry_payload["artifacts"]
    assert manifest_payload["run_context"]["run_id"] == "run-001"
    assert manifest_payload["datasets"][0]["modality"] == "ehr"
