import json
from datetime import UTC, datetime

from core.artifacts import ArtifactRegistry, RunManifest
from core.contracts import (
    ArtifactRecord,
    DatasetDescriptor,
    ModalityType,
    RunArtifactType,
    RunContextRecord,
)
from reporting import ManifestWriter


def _run_context() -> RunContextRecord:
    return RunContextRecord(
        run_id="run-001",
        project_name="dre_production",
        stage_name="test",
        timestamp_utc=datetime(2026, 5, 30, tzinfo=UTC),
    )


def _registry() -> ArtifactRegistry:
    return ArtifactRegistry(
        artifacts=[
            ArtifactRecord(
                artifact_name="labels",
                artifact_type=RunArtifactType.TABLE,
                relative_path="artifacts/labels.parquet",
            )
        ]
    )


def _dataset() -> DatasetDescriptor:
    return DatasetDescriptor(
        dataset_name="patient_labels",
        modality=ModalityType.EHR,
        source_format="parquet",
        row_count=2,
        patient_count=2,
    )


def test_write_manifest_creates_valid_json(tmp_path) -> None:
    manifest = RunManifest(
        run_context=_run_context(),
        datasets=[_dataset()],
        artifact_registry=_registry(),
        notes=["written by test"],
    )
    output_path = tmp_path / "nested" / "manifest.json"

    written_path = ManifestWriter().write_manifest(manifest, output_path)
    payload = json.loads(written_path.read_text(encoding="utf-8"))

    assert written_path == output_path.resolve()
    assert payload["run_context"]["run_id"] == "run-001"
    assert payload["datasets"][0]["dataset_name"] == "patient_labels"
    assert payload["artifact_registry"]["artifacts"][0]["artifact_type"] == "table"


def test_write_artifact_registry_creates_valid_json(tmp_path) -> None:
    output_path = tmp_path / "registry" / "artifacts.json"

    written_path = ManifestWriter().write_artifact_registry(_registry(), output_path)
    payload = json.loads(written_path.read_text(encoding="utf-8"))

    assert written_path.exists()
    assert payload["artifacts"][0]["relative_path"] == "artifacts/labels.parquet"


def test_write_dataset_descriptors_creates_valid_json(tmp_path) -> None:
    output_path = tmp_path / "datasets" / "datasets.json"

    written_path = ManifestWriter().write_dataset_descriptors([_dataset()], output_path)
    payload = json.loads(written_path.read_text(encoding="utf-8"))

    assert written_path.exists()
    assert payload[0]["modality"] == "ehr"
    assert payload[0]["patient_count"] == 2


def test_writer_creates_parent_directories(tmp_path) -> None:
    output_path = tmp_path / "a" / "b" / "manifest.json"

    written_path = ManifestWriter().write_manifest(
        RunManifest(run_context=_run_context()),
        output_path,
    )

    assert written_path.exists()
    assert written_path.parent == (tmp_path / "a" / "b").resolve()
