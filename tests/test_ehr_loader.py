import gzip

import pandas as pd
import pytest

from ingestion import EHRLoader, RequiredColumnsMissingError, TableLoadRequest


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "event_time": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "value": [1, 2, 3],
            "source": ["encounter", "medication", "encounter"],
        }
    )


def test_infer_file_format() -> None:
    loader = EHRLoader()

    assert loader.infer_file_format("events.csv") == "csv"
    assert loader.infer_file_format("events.csv.gz") == "csv.gz"
    assert loader.infer_file_format("events.parquet") == "parquet"

    with pytest.raises(ValueError):
        loader.infer_file_format("events.txt")


def test_load_csv_returns_dataframe_and_metadata(tmp_path) -> None:
    csv_path = tmp_path / "events.csv"
    _sample_dataframe().to_csv(csv_path, index=False)

    df, result = EHRLoader().load_table(
        TableLoadRequest(
            path=str(csv_path),
            required_columns=["patient_id", "event_time"],
            parse_date_columns=["event_time"],
            dataset_name="events",
        )
    )

    assert len(df) == 3
    assert result.row_count == 3
    assert result.file_format == "csv"
    assert result.loaded_columns == ["patient_id", "event_time", "value", "source"]
    assert "file format inferred from path" in result.notes
    assert "date parsing was requested" in result.notes


def test_load_csv_gz(tmp_path) -> None:
    csv_gz_path = tmp_path / "events.csv.gz"
    csv_content = _sample_dataframe().to_csv(index=False)
    with gzip.open(csv_gz_path, "wt", encoding="utf-8") as csv_file:
        csv_file.write(csv_content)

    df, result = EHRLoader().load_table(TableLoadRequest(path=str(csv_gz_path)))

    assert len(df) == 3
    assert result.file_format == "csv.gz"


def test_load_parquet_returns_dataframe_and_metadata(tmp_path) -> None:
    parquet_path = tmp_path / "events.parquet"
    _sample_dataframe().to_parquet(parquet_path, index=False)

    df, result = EHRLoader().load_table(
        TableLoadRequest(path=str(parquet_path), required_columns=["patient_id"])
    )

    assert len(df) == 3
    assert result.row_count == 3
    assert result.file_format == "parquet"
    assert "patient_id" in result.loaded_columns


def test_required_columns_missing_raises(tmp_path) -> None:
    csv_path = tmp_path / "events.csv"
    _sample_dataframe().to_csv(csv_path, index=False)

    with pytest.raises(RequiredColumnsMissingError):
        EHRLoader().load_table(
            TableLoadRequest(path=str(csv_path), required_columns=["missing_column"])
        )


def test_selected_columns_are_loaded(tmp_path) -> None:
    csv_path = tmp_path / "events.csv"
    _sample_dataframe().to_csv(csv_path, index=False)

    df, result = EHRLoader().load_table(
        TableLoadRequest(path=str(csv_path), columns=["patient_id", "value"])
    )

    assert list(df.columns) == ["patient_id", "value"]
    assert result.loaded_columns == ["patient_id", "value"]
    assert "selected columns were used" in result.notes


def test_partially_missing_selected_columns_are_not_fatal(tmp_path) -> None:
    csv_path = tmp_path / "events.csv"
    _sample_dataframe().to_csv(csv_path, index=False)

    df, result = EHRLoader().load_table(
        TableLoadRequest(path=str(csv_path), columns=["patient_id", "missing_column"])
    )

    assert list(df.columns) == ["patient_id"]
    assert "requested selected column missing: missing_column" in result.notes


def test_peek_columns_for_csv_and_parquet(tmp_path) -> None:
    csv_path = tmp_path / "events.csv"
    parquet_path = tmp_path / "events.parquet"
    sample_df = _sample_dataframe()
    sample_df.to_csv(csv_path, index=False)
    sample_df.to_parquet(parquet_path, index=False)

    loader = EHRLoader()

    assert loader.peek_columns(str(csv_path)) == ["patient_id", "event_time", "value", "source"]
    assert loader.peek_columns(str(parquet_path)) == [
        "patient_id",
        "event_time",
        "value",
        "source",
    ]


def test_chunked_csv_iteration_yields_chunk_metadata(tmp_path) -> None:
    csv_path = tmp_path / "events.csv"
    _sample_dataframe().to_csv(csv_path, index=False)

    chunks = list(
        EHRLoader().iter_csv_chunks(
            TableLoadRequest(
                path=str(csv_path),
                chunk_size=2,
                required_columns=["patient_id"],
            )
        )
    )

    assert len(chunks) == 2
    assert [len(chunk) for chunk, _ in chunks] == [2, 1]
    assert all(result.used_chunking for _, result in chunks)
    assert all("chunking was used" in result.notes for _, result in chunks)


def test_chunking_requires_chunk_size(tmp_path) -> None:
    csv_path = tmp_path / "events.csv"
    _sample_dataframe().to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        list(EHRLoader().iter_csv_chunks(TableLoadRequest(path=str(csv_path))))


def test_chunking_rejects_parquet(tmp_path) -> None:
    parquet_path = tmp_path / "events.parquet"
    _sample_dataframe().to_parquet(parquet_path, index=False)

    with pytest.raises(ValueError):
        list(EHRLoader().iter_csv_chunks(TableLoadRequest(path=str(parquet_path), chunk_size=2)))
