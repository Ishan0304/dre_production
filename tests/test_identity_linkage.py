import pandas as pd
import pytest

from datasets import MultimodalPipelineRunner
from normalization import (
    IdentityLinkageConfig,
    IdentityLinker,
    IdentityMatchStatus,
)


def test_identity_normalization_strips_values() -> None:
    linker = IdentityLinker()

    assert linker.normalize_identifier("  P001  ") == "P001"
    assert linker.normalize_identifier("   ") is None
    assert linker.normalize_identifier(None) is None


def test_lower_normalization_strips_and_lowercases_values() -> None:
    linker = IdentityLinker()

    assert linker.normalize_identifier("  Sub-001  ", mode="lower") == "sub-001"


def test_explicit_mapping_links_rows_and_records_unmatched() -> None:
    df = pd.DataFrame({"source_id": ["mri-1", "mri-2", "mri-3"]})
    config = IdentityLinkageConfig(
        source_id_col="source_id",
        explicit_mapping={"mri-1": "p1", "mri-2": "p2"},
    )

    linked_df, result = IdentityLinker().link_dataframe(df, config, "mri")

    assert linked_df["patient_id"].tolist() == ["p1", "p2", None]
    assert result.matched_count == 2
    assert result.unmatched_count == 1
    assert result.ambiguous_count == 0
    assert result.link_records[-1].match_status is IdentityMatchStatus.UNMATCHED


def test_normalization_based_linking_marks_blank_ids_unmatched() -> None:
    df = pd.DataFrame({"subject_id": ["sub-1", "  sub-2  ", ""]})
    config = IdentityLinkageConfig(
        source_id_col="subject_id",
        normalization_mode="subject_to_patient",
    )

    linked_df, result = IdentityLinker().link_dataframe(df, config, "mri")

    assert linked_df["patient_id"].tolist() == ["sub-1", "sub-2", None]
    assert result.matched_count == 2
    assert result.unmatched_count == 1


def test_missing_source_column_raises_clear_error() -> None:
    df = pd.DataFrame({"other_id": ["p1"]})
    config = IdentityLinkageConfig(source_id_col="source_id")

    with pytest.raises(ValueError, match="Missing source identifier column"):
        IdentityLinker().link_dataframe(df, config, "ehr")


def test_summarize_results_returns_compact_json_ready_summary() -> None:
    linker = IdentityLinker()
    _, ehr_result = linker.link_dataframe(
        pd.DataFrame({"patient_id": ["p1", "p2"]}),
        IdentityLinkageConfig(source_id_col="patient_id"),
        "ehr",
    )
    _, mri_result = linker.link_dataframe(
        pd.DataFrame({"subject_id": ["p1", ""]}),
        IdentityLinkageConfig(source_id_col="subject_id"),
        "mri",
    )

    summary = linker.summarize_results([ehr_result, mri_result])

    assert summary["modality_count"] == 2
    assert summary["totals"]["row_count"] == 4
    assert summary["totals"]["matched_count"] == 3
    assert summary["totals"]["unmatched_count"] == 1
    assert summary["modalities"][0]["source_modality"] == "ehr"


def test_multimodal_runner_routes_mri_subject_id_through_identity_linker() -> None:
    df = pd.DataFrame({"subject_id": ["sub-1"], "feature": [1.0]})
    notes: list[str] = []

    linked_df = MultimodalPipelineRunner()._link_mri_patient_id(df, notes)

    assert linked_df["patient_id"].tolist() == ["sub-1"]
    assert linked_df["subject_id"].tolist() == ["sub-1"]
    assert "MRI subject identifiers linked through identity linkage layer" in notes
