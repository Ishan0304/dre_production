from datetime import UTC, datetime

from core.artifacts import ArtifactRegistry, RunManifest
from core.contracts import (
    ArtifactRecord,
    DatasetDescriptor,
    ModalityType,
    RunArtifactType,
    RunContextRecord,
)


def _artifact(name: str, artifact_type: RunArtifactType) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_name=name,
        artifact_type=artifact_type,
        relative_path=f"artifacts/{name}",
    )


def _run_context() -> RunContextRecord:
    return RunContextRecord(
        run_id="run-001",
        project_name="dre_production",
        stage_name="test",
        timestamp_utc=datetime(2026, 5, 30, tzinfo=UTC),
    )


def test_artifact_registry_registration_preserves_order() -> None:
    registry = ArtifactRegistry()
    table = _artifact("table.parquet", RunArtifactType.TABLE)
    log = _artifact("run.log", RunArtifactType.LOG)

    registry.register(table)
    registry.register(log)

    assert registry.artifacts == [table, log]


def test_artifact_registry_list_by_type_filters_records() -> None:
    registry = ArtifactRegistry()
    registry.register_many(
        [
            _artifact("table.parquet", RunArtifactType.TABLE),
            _artifact("plot.png", RunArtifactType.PLOT),
            _artifact("features.parquet", RunArtifactType.TABLE),
        ]
    )

    table_artifacts = registry.list_by_type(RunArtifactType.TABLE)

    assert [artifact.artifact_name for artifact in table_artifacts] == [
        "table.parquet",
        "features.parquet",
    ]


def test_artifact_registry_to_dict_is_json_ready() -> None:
    registry = ArtifactRegistry()
    registry.register(_artifact("table.parquet", RunArtifactType.TABLE))

    as_dict = registry.to_dict()

    assert as_dict == {
        "artifacts": [
            {
                "artifact_name": "table.parquet",
                "artifact_type": "table",
                "relative_path": "artifacts/table.parquet",
                "description": None,
                "created_by": None,
                "metadata": {},
            }
        ]
    }


def test_run_manifest_includes_datasets_artifacts_and_notes() -> None:
    dataset = DatasetDescriptor(
        dataset_name="patient_labels",
        modality=ModalityType.EHR,
        source_format="parquet",
        row_count=10,
        patient_count=10,
    )
    registry = ArtifactRegistry([_artifact("labels.parquet", RunArtifactType.TABLE)])
    manifest = RunManifest(run_context=_run_context(), artifact_registry=registry)

    manifest.add_dataset(dataset)
    manifest.add_note("test manifest")
    as_dict = manifest.to_dict()

    assert as_dict["run_context"]["timestamp_utc"] == "2026-05-30T00:00:00+00:00"
    assert as_dict["datasets"][0]["modality"] == "ehr"
    assert as_dict["artifact_registry"]["artifacts"][0]["artifact_type"] == "table"
    assert as_dict["notes"] == ["test manifest"]
