import json
from datetime import UTC, datetime

import pandas as pd

from core.contracts import RunContextRecord
from datasets import EEGPipelineInputConfig, EEGPipelineRunner


class FakeEEGFeatureBuilder:
    def build_recording_feature_table(self, inventory_df):
        rows = [
            {
                "patient_id": row["patient_id"],
                "edf_path": row["edf_path"],
                "feature_a": 1.0,
            }
            for _, row in inventory_df.iterrows()
            if "bad" not in row["edf_path"]
        ]
        errors = [
            {
                "patient_id": row["patient_id"],
                "edf_path": row["edf_path"],
                "error_type": "ValueError",
                "error_message": "bad recording",
            }
            for _, row in inventory_df.iterrows()
            if "bad" in row["edf_path"]
        ]
        return pd.DataFrame(rows), pd.DataFrame(
            errors,
            columns=["patient_id", "edf_path", "error_type", "error_message"],
        )


def _run_context() -> RunContextRecord:
    return RunContextRecord(
        run_id="run-eeg-001",
        project_name="dre_production",
        stage_name="eeg_pipeline",
        timestamp_utc=datetime(2026, 5, 30, tzinfo=UTC),
    )


def test_eeg_pipeline_run_writes_outputs_and_artifacts(tmp_path) -> None:
    patient_dir = tmp_path / "chb01"
    patient_dir.mkdir()
    (patient_dir / "good.edf").write_text("edf", encoding="utf-8")
    (patient_dir / "bad.edf").write_text("edf", encoding="utf-8")

    result = EEGPipelineRunner(feature_builder=FakeEEGFeatureBuilder()).run(
        EEGPipelineInputConfig(
            dataset_name="eeg_features",
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "outputs"),
        ),
        _run_context(),
    )

    assert result.recording_row_count == 1
    assert result.patient_row_count == 1
    assert result.error_count == 1
    assert result.error_table_path is not None
    manifest = json.loads(open(result.run_manifest_path, encoding="utf-8").read())
    assert manifest["datasets"][0]["modality"] == "eeg"
