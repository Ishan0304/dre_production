import json

import pandas as pd
import pytest

from modeling import BaselineModelConfig, EHRBaselinePipeline


def _training_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(12)],
            "likely_dre": [0, 1] * 6,
            "age": [20, 44, 30, 50, 41, 60, 55, 35, 48, 62, 29, 70],
            "encounter_count": [1, 8, 2, 9, 3, 10, 4, 7, 2, 11, 1, 12],
            "has_epilepsy_diagnosis": [True] * 12,
            "distinct_asm_count": [1, 2] * 6,
            "reasons": ["reason"] * 12,
        }
    )


def test_leakage_exclusion_removes_target_derived_columns() -> None:
    result = EHRBaselinePipeline().run_leakage_checks(_training_df())

    assert "patient_id" in result.excluded_for_identifier
    assert "likely_dre" in result.excluded_for_leakage
    assert "has_epilepsy_diagnosis" in result.excluded_for_leakage
    assert "distinct_asm_count" in result.excluded_for_leakage
    assert "reasons" in result.excluded_for_leakage
    assert "age" not in result.excluded_for_leakage


def test_honest_training_run_returns_features_and_metrics() -> None:
    result = EHRBaselinePipeline().run(_training_df())

    assert result.train_row_count > 0
    assert result.val_row_count > 0
    assert result.test_row_count > 0
    assert result.feature_cols_used == ["age", "encounter_count"]
    assert "test" in result.metrics
    assert result.metrics["test"]["row_count"] == result.test_row_count


def test_existing_split_column_is_respected() -> None:
    df = _training_df()
    df["split"] = ["train"] * 8 + ["val"] * 2 + ["test"] * 2
    config = BaselineModelConfig(split_col="split")

    result = EHRBaselinePipeline(config).run(df)

    assert result.train_row_count == 8
    assert result.val_row_count == 2
    assert result.test_row_count == 2
    assert "existing split column was used" in result.notes


def test_random_split_path_creates_valid_splits() -> None:
    train_df, val_df, test_df, notes = EHRBaselinePipeline().build_splits(_training_df())

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert "random train validation test split was created" in notes


def test_undefined_metric_path_does_not_crash() -> None:
    pipeline = EHRBaselinePipeline()
    df = _training_df()
    train_df, _, _test_df, _ = pipeline.build_splits(df)
    features = ["age", "encounter_count"]
    model = pipeline.train_model(
        train_df[features],
        train_df[pipeline.config.target_col].astype(int),
    )
    one_class_df = pd.DataFrame({"age": [20, 30], "encounter_count": [1, 2]})
    one_class_y = pd.Series([0, 0])

    metrics = pipeline.evaluate_model(model, one_class_df, one_class_y, "one_class")

    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["notes"]


def test_save_outputs_writes_model_and_metrics(tmp_path) -> None:
    config = BaselineModelConfig(output_dir=str(tmp_path))
    result = EHRBaselinePipeline(config).run(_training_df())

    assert result.model_path is not None
    assert result.metrics_path is not None
    assert len(result.artifact_records) == 2
    payload = json.loads(open(result.metrics_path, encoding="utf-8").read())
    assert payload["model_type"] == "logistic_regression"


def test_no_safe_features_raises_clear_error() -> None:
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "likely_dre": [0, 1, 0, 1],
            "reasons": ["a", "b", "c", "d"],
            "distinct_asm_count": [1, 2, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="No safe numeric feature columns"):
        EHRBaselinePipeline().run(df)
