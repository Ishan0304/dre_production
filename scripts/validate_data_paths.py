"""Validate repo-local public and credentialed dataset paths."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(slots=True)
class DatasetCheck:
    """Status for one dataset input path."""

    name: str
    present: bool
    path: Path
    detail: str


def repo_root() -> Path:
    """Return the repository root from this script location."""

    return Path(__file__).resolve().parents[1]


def count_csv_rows(path: Path) -> int | None:
    """Count CSV rows without loading the full file into memory."""

    if not path.exists():
        return None
    with open_text(path) as file:
        reader = csv.reader(file)
        next(reader, None)
        return sum(1 for _ in reader)


def open_text(path: Path) -> TextIO:
    """Open plain or gzipped text CSV files."""

    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def check_file(name: str, path: Path, required_columns: set[str] | None = None) -> DatasetCheck:
    """Validate one CSV file and optional required columns."""

    if not path.exists():
        return DatasetCheck(name, False, path, "missing file")
    if path.stat().st_size == 0:
        return DatasetCheck(name, False, path, "file is empty")
    if required_columns:
        with open_text(path) as file:
            reader = csv.DictReader(file)
            columns = set(reader.fieldnames or [])
        missing = sorted(required_columns - columns)
        if missing:
            return DatasetCheck(name, False, path, f"missing columns: {missing}")
    row_count = count_csv_rows(path)
    return DatasetCheck(name, True, path, f"present, rows={row_count}")


def check_directory(name: str, path: Path, detail: str) -> DatasetCheck:
    """Validate one directory path."""

    if not path.exists():
        return DatasetCheck(name, False, path, "missing directory")
    if not path.is_dir():
        return DatasetCheck(name, False, path, "path is not a directory")
    return DatasetCheck(name, True, path, detail)


def validate_mimic_demo(root: Path) -> list[DatasetCheck]:
    """Validate MIMIC Demo raw and normalized paths."""

    base = root / "data" / "raw" / "mimic_demo"
    return [
        check_file(
            "mimic_demo_normalized_diagnoses",
            base / "ehr_pipeline" / "diagnoses.csv",
            {"patient_id", "diagnosis_code"},
        ),
        check_file(
            "mimic_demo_normalized_medications",
            base / "ehr_pipeline" / "medications.csv",
            {"patient_id", "medication_name"},
        ),
        check_file(
            "mimic_demo_raw_diagnoses",
            base / "raw" / "diagnoses_icd.csv.gz",
        ),
        check_file(
            "mimic_demo_raw_prescriptions",
            base / "raw" / "prescriptions.csv.gz",
        ),
    ]


def validate_openneuro(root: Path) -> list[DatasetCheck]:
    """Validate OpenNeuro BIDS-like MRI path."""

    base = root / "data" / "raw" / "openneuro"
    dataset_dirs = sorted(path for path in base.glob("ds*") if path.is_dir()) if base.exists() else []
    if not dataset_dirs:
        return [DatasetCheck("openneuro_dataset_root", False, base, "missing OpenNeuro dataset")]

    checks: list[DatasetCheck] = []
    for dataset_dir in dataset_dirs:
        participants = dataset_dir / "participants.tsv"
        subject_dirs = sorted(path for path in dataset_dir.glob("sub-*") if path.is_dir())
        t1_files = sorted(dataset_dir.glob("sub-*/anat/*T1w.nii.gz"))
        detail = (
            f"subjects={len(subject_dirs)}, t1_files={len(t1_files)}, "
            f"participants_tsv={participants.exists()}"
        )
        checks.append(check_directory(f"openneuro_{dataset_dir.name}", dataset_dir, detail))
    return checks


def validate_chbmit(root: Path) -> list[DatasetCheck]:
    """Validate CHB-MIT EDF path."""

    base = root / "data" / "raw" / "chbmit"
    edf_files = sorted(base.glob("**/*.edf")) + sorted(base.glob("**/*.EDF")) if base.exists() else []
    detail = f"edf_files={len(edf_files)}"
    return [check_directory("chbmit_dataset_root", base, detail)]


def print_checks(checks: list[DatasetCheck]) -> None:
    """Print validation results."""

    for check in checks:
        status = "PRESENT" if check.present else "MISSING"
        print(f"{status}: {check.name}")
        print(f"  path: {check.path}")
        print(f"  detail: {check.detail}")


def main() -> None:
    """Validate all standard dataset locations."""

    root = repo_root()
    checks = [
        *validate_mimic_demo(root),
        *validate_openneuro(root),
        *validate_chbmit(root),
    ]
    print_checks(checks)

    missing = [check for check in checks if not check.present]
    print("\nSummary:")
    print(f"  present={len(checks) - len(missing)}")
    print(f"  missing={len(missing)}")
    if missing:
        print("\nMissing datasets can be bootstrapped with:")
        print("  python scripts/bootstrap_public_data.py --dataset mimic_demo")
        print("  python scripts/bootstrap_public_data.py --dataset chbmit")
        print("  python scripts/bootstrap_public_data.py --dataset openneuro")
        print(
            "MIMIC-IV full is credentialed and must be supplied manually "
            "after completing required PhysioNet access."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
