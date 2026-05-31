import pandas as pd

from datasets import MRISubjectDatasetBuilder, MRISubjectDatasetBuildConfig


def test_mri_subject_dataset_builder_sorts_and_reports_counts() -> None:
    feature_df = pd.DataFrame(
        {
            "subject_id": ["sub-002", "sub-001"],
            "feature_a": [2.0, 1.0],
        }
    )
    error_df = pd.DataFrame(
        [{"subject_id": "sub-003", "error_type": "missing_t1", "error_message": "missing"}]
    )
    builder = MRISubjectDatasetBuilder(
        MRISubjectDatasetBuildConfig(dataset_name="mri_features")
    )

    output_df, result = builder.build_subject_dataset(feature_df, error_df)

    assert output_df["subject_id"].tolist() == ["sub-001", "sub-002"]
    assert result.dataset_name == "mri_features"
    assert result.row_count == 2
    assert result.subject_count == 2
    assert result.error_count == 1
