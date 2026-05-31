import pandas as pd

from datasets import EEGPatientDatasetBuilder, EEGPatientDatasetBuildConfig


def test_eeg_patient_dataset_aggregates_recording_features() -> None:
    recording_df = pd.DataFrame(
        {
            "patient_id": ["p2", "p1", "p1"],
            "edf_path": ["b.edf", "a.edf", "c.edf"],
            "feature_a": [4.0, 1.0, 3.0],
            "feature_b": [10.0, 2.0, 4.0],
        }
    )
    error_df = pd.DataFrame(
        [{"patient_id": "p3", "edf_path": "bad.edf", "error_type": "Error", "error_message": "bad"}]
    )
    builder = EEGPatientDatasetBuilder(
        EEGPatientDatasetBuildConfig(dataset_name="eeg_features")
    )

    patient_df, result = builder.build_patient_dataset(recording_df, error_df)

    assert patient_df["patient_id"].tolist() == ["p1", "p2"]
    assert patient_df.loc[0, "feature_a"] == 2.0
    assert patient_df.loc[0, "recording_count"] == 2
    assert result.error_count == 1
    assert result.patient_count == 2
