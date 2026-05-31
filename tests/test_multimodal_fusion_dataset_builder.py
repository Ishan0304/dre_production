import pandas as pd
import pytest

from datasets import MultimodalFusionConfig, MultimodalFusionDatasetBuilder


def test_multimodal_fusion_uses_outer_join_and_namespaces_columns() -> None:
    ehr_df = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "age": [40, 55],
            "likely_dre": [True, False],
            "shared": [1.0, 2.0],
        }
    )
    mri_df = pd.DataFrame(
        {
            "patient_id": ["p2", "p3"],
            "lesion_score": [0.3, 0.9],
            "shared": [5.0, 6.0],
        }
    )
    eeg_df = pd.DataFrame(
        {
            "patient_id": ["p1", "p3"],
            "bandpower_alpha_mean": [0.2, 0.4],
            "shared": [9.0, 10.0],
        }
    )

    fused_df, result = MultimodalFusionDatasetBuilder().build_fused_dataset(
        dataset_name="multimodal_features",
        ehr_df=ehr_df,
        mri_df=mri_df,
        eeg_df=eeg_df,
    )

    assert fused_df["patient_id"].tolist() == ["p1", "p2", "p3"]
    assert "ehr_likely_dre" in fused_df.columns
    assert {"ehr_shared", "mri_shared", "eeg_shared"}.issubset(fused_df.columns)
    assert not fused_df.loc[fused_df["patient_id"] == "p1", "has_mri"].item()
    assert not fused_df.loc[fused_df["patient_id"] == "p2", "has_eeg"].item()
    assert result.patient_count == 3
    assert result.patients_with_ehr == 2
    assert result.patients_with_mri == 2
    assert result.patients_with_eeg == 2
    assert result.patients_with_all_modalities == 0


def test_multimodal_fusion_succeeds_with_missing_modalities() -> None:
    ehr_df = pd.DataFrame({"patient_id": ["p1"], "feature": [1.0]})

    fused_df, result = MultimodalFusionDatasetBuilder().build_fused_dataset(
        dataset_name="ehr_only",
        ehr_df=ehr_df,
        mri_df=None,
        eeg_df=None,
    )

    assert fused_df["patient_id"].tolist() == ["p1"]
    assert fused_df["has_ehr"].tolist() == [True]
    assert fused_df["has_mri"].tolist() == [False]
    assert fused_df["has_eeg"].tolist() == [False]
    assert "mri dataframe was not provided" in result.notes
    assert "eeg dataframe was not provided" in result.notes


def test_multimodal_fusion_empty_inputs_return_empty_metadata() -> None:
    fused_df, result = MultimodalFusionDatasetBuilder().build_fused_dataset(
        dataset_name="empty_multimodal",
        ehr_df=pd.DataFrame(),
        mri_df=None,
        eeg_df=None,
    )

    assert fused_df.empty
    assert fused_df.columns.tolist() == ["patient_id"]
    assert result.row_count == 0
    assert result.patient_count == 0
    assert result.patients_with_all_modalities == 0
    assert "ehr dataframe was empty" in result.notes


def test_multimodal_fusion_raises_for_missing_patient_id() -> None:
    bad_df = pd.DataFrame({"feature": [1.0]})

    with pytest.raises(ValueError, match="missing required column"):
        MultimodalFusionDatasetBuilder().build_fused_dataset(
            dataset_name="bad",
            ehr_df=bad_df,
        )


def test_multimodal_fusion_can_omit_modality_flags_from_output() -> None:
    builder = MultimodalFusionDatasetBuilder(
        MultimodalFusionConfig(include_modality_flags=False)
    )

    fused_df, result = builder.build_fused_dataset(
        dataset_name="no_flags",
        ehr_df=pd.DataFrame({"patient_id": ["p1"], "feature": [1.0]}),
        mri_df=pd.DataFrame({"patient_id": ["p1"], "feature": [2.0]}),
        eeg_df=pd.DataFrame({"patient_id": ["p2"], "feature": [3.0]}),
    )

    assert "has_ehr" not in fused_df.columns
    assert {"ehr_feature", "mri_feature", "eeg_feature"}.issubset(fused_df.columns)
    assert result.patients_with_ehr == 1
    assert result.patients_with_mri == 1
    assert result.patients_with_eeg == 1
    assert result.patients_with_all_modalities == 0


def test_multimodal_fusion_prevents_duplicate_patient_rows() -> None:
    duplicate_df = pd.DataFrame({"patient_id": ["p1", "p1"], "feature": [1.0, 2.0]})

    with pytest.raises(ValueError, match="duplicate patient_id"):
        MultimodalFusionDatasetBuilder().build_fused_dataset(
            dataset_name="duplicates",
            ehr_df=duplicate_df,
        )
