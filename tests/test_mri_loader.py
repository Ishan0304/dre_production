import pandas as pd

from ingestion import MRIDatasetLoadConfig, MRILoader


def test_mri_loader_loads_participants_and_discovers_subjects(tmp_path) -> None:
    dataset_root = tmp_path / "openneuro"
    anat_dir = dataset_root / "sub-001" / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / "sub-001_T1w.nii.gz").write_text("not a real image", encoding="utf-8")
    pd.DataFrame({"participant_id": ["sub-001"], "age": [30]}).to_csv(
        dataset_root / "participants.tsv",
        sep="\t",
        index=False,
    )

    participants, inventory, result = MRILoader().scan_dataset(
        MRIDatasetLoadConfig(dataset_root=str(dataset_root))
    )

    assert participants is not None
    assert result.participants_row_count == 1
    assert result.subject_count == 1
    assert result.subjects_with_t1_count == 1
    assert inventory.loc[0, "subject_id"] == "sub-001"


def test_mri_loader_selects_t1_deterministically(tmp_path) -> None:
    anat_dir = tmp_path / "sub-001" / "anat"
    anat_dir.mkdir(parents=True)
    first = anat_dir / "sub-001_a_T1w.nii.gz"
    second = anat_dir / "sub-001_b_T1w.nii.gz"
    second.write_text("b", encoding="utf-8")
    first.write_text("a", encoding="utf-8")

    selected = MRILoader().find_t1_for_subject(tmp_path / "sub-001")

    assert selected == first
