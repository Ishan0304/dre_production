from datetime import UTC, datetime

from core import (
    ArtifactRecord,
    DatasetDescriptor,
    EvidenceStatus,
    ModalityType,
    RunArtifactType,
    RunContextRecord,
    SplitType,
)


def test_enum_values_are_stable() -> None:
    assert EvidenceStatus.OBSERVED.value == "observed"
    assert EvidenceStatus.INFERRED.value == "inferred"
    assert EvidenceStatus.MISSING.value == "missing"
    assert ModalityType.EHR.value == "ehr"
    assert ModalityType.MRI.value == "mri"
    assert ModalityType.EEG.value == "eeg"
    assert SplitType.TRAIN.value == "train"
    assert SplitType.VAL.value == "val"
    assert SplitType.TEST.value == "test"
    assert SplitType.INFERENCE.value == "inference"
    assert RunArtifactType.TABLE.value == "table"
    assert RunArtifactType.PLOT.value == "plot"
    assert RunArtifactType.JSON.value == "json"
    assert RunArtifactType.MODEL.value == "model"
    assert RunArtifactType.LOG.value == "log"


def test_dataclass_construction_and_to_dict() -> None:
    timestamp = datetime(2026, 5, 30, tzinfo=UTC)
    record = RunContextRecord(
        run_id="run-001",
        project_name="dre_production",
        stage_name="block_1",
        seed=42,
        timestamp_utc=timestamp,
        metadata={"owner": "core"},
    )

    assert record.run_id == "run-001"
    assert record.to_dict() == {
        "run_id": "run-001",
        "project_name": "dre_production",
        "stage_name": "block_1",
        "seed": 42,
        "timestamp_utc": timestamp,
        "metadata": {"owner": "core"},
    }


def test_default_metadata_factories_are_not_shared() -> None:
    first = ArtifactRecord(
        artifact_name="profile",
        artifact_type=RunArtifactType.JSON,
        relative_path="artifacts/profile.json",
    )
    second = DatasetDescriptor(
        dataset_name="ehr_extract",
        modality=ModalityType.EHR,
        source_format="parquet",
    )

    first.metadata["key"] = "value"

    assert second.metadata == {}
    assert first.metadata is not second.metadata
