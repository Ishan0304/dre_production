import json

import pandas as pd
import pytest

from modeling import MultimodalBaselineModelConfig, MultimodalBaselinePipeline


def _training_df(row_count: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(row_count):
        target = int(index % 2 == 0)
        rows.append(
            {
                "patient_id": f"p{index:03d}",
                "ehr_likely_dre": target,
                "ehr_reasons": "derived reason",
                "ehr_distinct_asm_count": 2 + target,
                "ehr_independent_numeric": float(index % 5),
                "mri_lesion_score": float(target) + (index * 0.01),
                "eeg_bandpower_alpha_mean": None if index % 4 == 0 else float(index),
                "has_ehr": True,
                "has_mri": index % 3 != 0,
                "has_eeg": index % 4 != 0,
            }
        )
    return pd.DataFrame(rows)


def test_multimodal_leakage_checks_exclude_target_derived_columns() -> None:
    df = _training_df()

    leakage = MultimodalBaselinePipeline().run_leakage_checks(df)

    assert "ehr_likely_dre" in leakage.excluded_for_leakage
    assert "ehr_reasons" in leakage.excluded_for_leakage
    assert "ehr_distinct_asm_count" in leakage.excluded_for_leakage
    assert "patient_id" in leakage.excluded_for_identifier
    assert "mri_lesion_score" in leakage.candidate_feature_cols
    assert "eeg_bandpower_alpha_mean" in leakage.candidate_feature_cols
    assert "has_mri" in leakage.candidate_feature_cols


def test_multimodal_baseline_pipeline_trains_on_synthetic_fused_data() -> None:
    result = MultimodalBaselinePipeline().run(_training_df())

    assert result.train_row_count > 0
    assert "mri_lesion_score" in result.feature_cols_used
    assert "has_mri" in result.modality_flag_cols_used
    assert result.metrics["train"]["row_count"] == result.train_row_count


def test_multimodal_baseline_pipeline_respects_existing_split_column() -> None:
    df = _training_df()
    df["split"] = ["train"] * 20 + ["val"] * 5 + ["test"] * 5
    config = MultimodalBaselineModelConfig(split_col="split")

    result = MultimodalBaselinePipeline(config).run(df)

    assert result.train_row_count == 20
    assert result.val_row_count == 5
    assert result.test_row_count == 5
    assert "existing split column was used" in result.notes


def test_multimodal_baseline_random_split_is_deterministic() -> None:
    config = MultimodalBaselineModelConfig(random_state=7)
    first = MultimodalBaselinePipeline(config).build_splits(_training_df())
    second = MultimodalBaselinePipeline(config).build_splits(_training_df())

    assert first[0]["patient_id"].tolist() == second[0]["patient_id"].tolist()
    assert first[1]["patient_id"].tolist() == second[1]["patient_id"].tolist()
    assert first[2]["patient_id"].tolist() == second[2]["patient_id"].tolist()


def test_multimodal_baseline_handles_missing_modalities_with_imputation() -> None:
    df = _training_df()
    df.loc[df.index[:10], "mri_lesion_score"] = None
    df.loc[df.index[10:20], "eeg_bandpower_alpha_mean"] = None

    result = MultimodalBaselinePipeline().run(df)

    assert result.train_row_count > 0
    assert "mri_lesion_score" in result.feature_cols_used


def test_multimodal_baseline_evaluation_handles_one_class_split() -> None:
    df = _training_df()
    df["split"] = ["train"] * 20 + ["val"] * 5 + ["test"] * 5
    df.loc[df["split"] == "val", "ehr_likely_dre"] = 1
    config = MultimodalBaselineModelConfig(split_col="split")

    result = MultimodalBaselinePipeline(config).run(df)

    assert result.metrics["val"]["roc_auc"] is None
    assert result.metrics["val"]["notes"]


def test_multimodal_baseline_saves_model_and_metrics(tmp_path) -> None:
    config = MultimodalBaselineModelConfig(output_dir=str(tmp_path))

    result = MultimodalBaselinePipeline(config).run(_training_df())

    assert result.model_path is not None
    assert result.metrics_path is not None
    assert (tmp_path / config.model_filename).exists()
    metrics = json.loads((tmp_path / config.metrics_filename).read_text(encoding="utf-8"))
    assert metrics["model_type"] == "logistic_regression"


def test_multimodal_baseline_raises_when_no_safe_features_remain() -> None:
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "ehr_likely_dre": [0, 1, 0, 1],
            "ehr_reasons": ["a", "b", "c", "d"],
        }
    )

    with pytest.raises(ValueError, match="No safe numeric or boolean feature columns"):
        MultimodalBaselinePipeline().run(df)
