from ingestion import EEGDatasetLoadConfig, EEGLoader


def test_eeg_loader_discovers_edf_files_and_infers_patient_ids(tmp_path) -> None:
    patient_dir = tmp_path / "chb01"
    patient_dir.mkdir()
    first = patient_dir / "chb01_01.edf"
    second = patient_dir / "chb01_02.EDF"
    first.write_text("edf", encoding="utf-8")
    second.write_text("edf", encoding="utf-8")

    loader = EEGLoader()
    files = loader.discover_edf_files(EEGDatasetLoadConfig(dataset_root=str(tmp_path)))

    assert files == [second, first] or files == sorted([first, second])
    assert loader.infer_patient_id_from_path(first) == "chb01"


def test_eeg_loader_builds_deterministic_inventory(tmp_path) -> None:
    (tmp_path / "chb02").mkdir()
    (tmp_path / "chb01").mkdir()
    (tmp_path / "chb02" / "b.edf").write_text("edf", encoding="utf-8")
    (tmp_path / "chb01" / "a.edf").write_text("edf", encoding="utf-8")

    inventory, result = EEGLoader().build_file_inventory(
        EEGDatasetLoadConfig(dataset_root=str(tmp_path))
    )

    assert inventory["patient_id"].tolist() == ["chb01", "chb02"]
    assert result.file_count == 2
    assert result.patient_count == 2
