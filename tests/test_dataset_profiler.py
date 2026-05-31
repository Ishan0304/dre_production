from datetime import datetime

import pandas as pd

from insights import DatasetProfiler


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3"],
            "label": [1, 0, 1, 1],
            "split": ["train", "train", "val", "test"],
            "age": [10.0, 20.0, None, 40.0],
            "source": ["ehr", "ehr", "mri", "ehr"],
            "event_time": ["2025-01-01", "2025-01-03", None, "2025-02-01"],
        }
    )


def test_basic_profile_counts_rows_columns_and_patients() -> None:
    profile = DatasetProfiler().profile_dataframe(
        df=_sample_dataframe(),
        dataset_name="training_frame",
        patient_id_col="patient_id",
        label_col="label",
        split_col="split",
        time_cols=["event_time"],
        categorical_cols=["source"],
        numeric_cols=["age"],
    )

    assert profile.dataset_name == "training_frame"
    assert profile.row_count == 4
    assert profile.column_count == 6
    assert profile.patient_count == 3


def test_missingness_counts_and_fractions() -> None:
    missingness = DatasetProfiler().compute_missingness(_sample_dataframe())
    age_missingness = next(record for record in missingness if record.column_name == "age")

    assert age_missingness.missing_count == 1
    assert age_missingness.missing_fraction == 0.25


def test_class_balance_counts_and_fractions() -> None:
    balance = DatasetProfiler().compute_class_balance(_sample_dataframe(), "label")
    by_label = {record.label_value: record for record in balance}

    assert by_label["1"].count == 3
    assert by_label["1"].fraction == 0.75
    assert by_label["0"].count == 1
    assert by_label["0"].fraction == 0.25


def test_split_balance_counts_and_fractions() -> None:
    balance = DatasetProfiler().compute_split_balance(_sample_dataframe(), "split")
    by_split = {record.split_value: record for record in balance}

    assert by_split["train"].count == 2
    assert by_split["train"].fraction == 0.5
    assert by_split["val"].count == 1
    assert by_split["test"].count == 1


def test_numeric_summary_for_known_column() -> None:
    summaries = DatasetProfiler().compute_numeric_summaries(_sample_dataframe(), ["age"])
    age_summary = summaries[0]

    assert age_summary.non_null_count == 3
    assert age_summary.mean == 70.0 / 3.0
    assert age_summary.median == 20.0


def test_category_cardinality_top_values() -> None:
    records = DatasetProfiler().compute_category_cardinality(
        _sample_dataframe(),
        categorical_cols=["source"],
        top_n=2,
    )
    source_record = records[0]

    assert source_record.unique_count == 2
    assert source_record.top_values == {"ehr": 3, "mri": 1}


def test_temporal_coverage_parses_requested_time_column() -> None:
    records = DatasetProfiler().compute_temporal_coverage(_sample_dataframe(), ["event_time"])
    event_time_record = records[0]

    assert event_time_record.non_null_count == 3
    assert event_time_record.min_time == datetime(2025, 1, 1)
    assert event_time_record.max_time == datetime(2025, 2, 1)


def test_missing_requested_columns_add_notes_without_crashing() -> None:
    profile = DatasetProfiler().profile_dataframe(
        df=_sample_dataframe(),
        dataset_name="training_frame",
        label_col="missing_label",
        split_col="missing_split",
    )

    assert profile.class_balance == []
    assert profile.split_balance == []
    assert "requested label_col missing: missing_label" in profile.notes
    assert "requested split_col missing: missing_split" in profile.notes


def test_profile_to_dict_returns_nested_structures() -> None:
    profile = DatasetProfiler().profile_dataframe(
        df=_sample_dataframe(),
        dataset_name="training_frame",
        patient_id_col="patient_id",
    )

    as_dict = profile.to_dict()

    assert as_dict["dataset_name"] == "training_frame"
    assert as_dict["patient_count"] == 3
    assert isinstance(as_dict["missingness"], list)
