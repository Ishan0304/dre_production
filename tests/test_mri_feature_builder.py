import numpy as np
import pandas as pd
import pytest

from features import MRIFeatureBuildConfig, MRIFeatureBuilder


def test_mri_feature_extraction_on_synthetic_nifti(tmp_path) -> None:
    nib = pytest.importorskip("nibabel")
    image_path = tmp_path / "sub-001_T1w.nii.gz"
    data = np.ones((24, 24, 24), dtype=np.float32)
    data[8:16, 8:16, 8:16] = 10.0
    image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(image, image_path)

    result = MRIFeatureBuilder(
        MRIFeatureBuildConfig(min_foreground_voxels=1, glcm_max_slices=1)
    ).extract_subject_features("sub-001", image_path)

    assert result.subject_id == "sub-001"
    assert result.feature_values["shape_x"] == 24
    assert result.feature_values["foreground_voxel_count"] > 0
    assert "t1_intensity_mean" in result.feature_values
    assert "glcm_contrast_mean" in result.feature_values


def test_mri_feature_table_joins_participant_metadata(monkeypatch) -> None:
    builder = MRIFeatureBuilder()

    def fake_extract(subject_id, t1_path):
        return type(
            "Result",
            (),
            {
                "feature_values": {
                    "subject_id": subject_id,
                    "t1_path": str(t1_path),
                    "feature_a": 1.0,
                }
            },
        )()

    monkeypatch.setattr(builder, "extract_subject_features", fake_extract)
    inventory = pd.DataFrame(
        {
            "subject_id": ["sub-001"],
            "t1_path": ["image.nii.gz"],
            "has_t1": [True],
        }
    )
    participants = pd.DataFrame({"participant_id": ["sub-001"], "age": [42]})

    feature_df, error_df = builder.build_feature_table(inventory, participants)

    assert error_df.empty
    assert feature_df.loc[0, "age"] == 42
    assert feature_df.loc[0, "feature_a"] == 1.0


def test_mri_feature_table_records_errors_for_missing_t1() -> None:
    inventory = pd.DataFrame(
        {
            "subject_id": ["sub-001"],
            "t1_path": [None],
            "has_t1": [False],
        }
    )

    feature_df, error_df = MRIFeatureBuilder().build_feature_table(inventory)

    assert feature_df.empty
    assert error_df.loc[0, "error_type"] == "missing_t1"
