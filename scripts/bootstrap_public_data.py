"""Bootstrap public datasets into repo-local data/raw paths.

This script does not download credentialed MIMIC-IV full data. It can download
MIMIC-IV Demo public files and convert them into normalized EHR CSVs for this
repo. It can also download a small public CHB-MIT EDF subset. For OpenNeuro,
it uses installed public dataset tooling when available.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


MIMIC_DEMO_VERSION = "2.2"
MIMIC_DEMO_BASE_URL = f"https://physionet.org/files/mimic-iv-demo/{MIMIC_DEMO_VERSION}"
MIMIC_DEMO_FILES = {
    "diagnoses_icd.csv.gz": f"{MIMIC_DEMO_BASE_URL}/hosp/diagnoses_icd.csv.gz",
    "prescriptions.csv.gz": f"{MIMIC_DEMO_BASE_URL}/hosp/prescriptions.csv.gz",
}

CHBMIT_BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
CHBMIT_DEFAULT_FILES = (
    "chb01/chb01-summary.txt",
    "chb01/chb01_01.edf",
)

DEFAULT_OPENNEURO_DATASET = "ds000030"
DEFAULT_OPENNEURO_SNAPSHOT = "1.0.0"


def repo_root() -> Path:
    """Return the repository root from this script location."""

    return Path(__file__).resolve().parents[1]


def download_file(url: str, destination: Path) -> Path:
    """Download one public file when it is not already present."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Present: {destination}")
        return destination
    print(f"Downloading: {url}")
    try:
        urlretrieve(url, destination)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    print(f"Saved: {destination}")
    return destination


def bootstrap_mimic_demo(root: Path) -> None:
    """Download MIMIC-IV Demo public files and write normalized EHR CSVs."""

    raw_dir = root / "data" / "raw" / "mimic_demo" / "raw"
    normalized_dir = root / "data" / "raw" / "mimic_demo" / "ehr_pipeline"

    downloaded = {
        name: download_file(url, raw_dir / name)
        for name, url in MIMIC_DEMO_FILES.items()
    }
    normalize_mimic_diagnoses(
        downloaded["diagnoses_icd.csv.gz"],
        normalized_dir / "diagnoses.csv",
    )
    normalize_mimic_prescriptions(
        downloaded["prescriptions.csv.gz"],
        normalized_dir / "medications.csv",
    )
    print("MIMIC-IV Demo bootstrap complete.")
    print(f"Normalized EHR path: {normalized_dir}")
    print(
        "MIMIC-IV full clinical database is credentialed. "
        "Provide approved local files manually if you need the full dataset."
    )


def normalize_mimic_diagnoses(source_path: Path, output_path: Path) -> None:
    """Convert MIMIC Demo diagnoses into normalized EHR diagnosis rows."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source_path, "rt", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=["patient_id", "diagnosis_code"],
            )
            writer.writeheader()
            for row in reader:
                writer.writerow(
                    {
                        "patient_id": row.get("subject_id", ""),
                        "diagnosis_code": row.get("icd_code", ""),
                    }
                )
    print(f"Wrote normalized diagnoses: {output_path}")


def normalize_mimic_prescriptions(source_path: Path, output_path: Path) -> None:
    """Convert MIMIC Demo prescriptions into normalized EHR medication rows."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source_path, "rt", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "patient_id",
                    "medication_name",
                    "medication_start",
                    "medication_end",
                ],
            )
            writer.writeheader()
            for row in reader:
                writer.writerow(
                    {
                        "patient_id": row.get("subject_id", ""),
                        "medication_name": row.get("drug", ""),
                        "medication_start": row.get("starttime", ""),
                        "medication_end": row.get("stoptime", ""),
                    }
                )
    print(f"Wrote normalized medications: {output_path}")


def bootstrap_chbmit(root: Path, full: bool = False) -> None:
    """Download public CHB-MIT files into data/raw/chbmit."""

    target_root = root / "data" / "raw" / "chbmit"
    files = CHBMIT_DEFAULT_FILES
    if full:
        raise RuntimeError(
            "Full CHB-MIT download is intentionally not scripted here because it is large. "
            "Use PhysioNet tooling or extend the file list intentionally."
        )
    for relative_path in files:
        download_file(f"{CHBMIT_BASE_URL}/{relative_path}", target_root / relative_path)
    print(f"CHB-MIT public subset bootstrap complete: {target_root}")


def bootstrap_openneuro(
    root: Path,
    dataset_id: str,
    snapshot: str,
) -> None:
    """Download an OpenNeuro dataset using available public dataset tooling."""

    target_root = root / "data" / "raw" / "openneuro" / dataset_id
    if shutil.which("openneuro"):
        command = [
            "openneuro",
            "download",
            "--snapshot",
            snapshot,
            dataset_id,
            str(target_root),
        ]
    elif shutil.which("datalad"):
        command = [
            "datalad",
            "install",
            "-r",
            "-s",
            f"https://github.com/OpenNeuroDatasets/{dataset_id}.git",
            str(target_root),
        ]
    else:
        raise RuntimeError(
            "OpenNeuro bootstrap requires the openneuro CLI or datalad. "
            "Install one of those tools, then rerun this script. "
            f"Target path will be {target_root}."
        )

    target_root.parent.mkdir(parents=True, exist_ok=True)
    print("Running: " + " ".join(command))
    subprocess.run(command, check=True)
    print(f"OpenNeuro bootstrap complete: {target_root}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Bootstrap public datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "mimic_demo", "chbmit", "openneuro"],
        default="all",
        help="Dataset bootstrap target.",
    )
    parser.add_argument(
        "--openneuro-dataset",
        default=DEFAULT_OPENNEURO_DATASET,
        help="OpenNeuro dataset accession.",
    )
    parser.add_argument(
        "--openneuro-snapshot",
        default=DEFAULT_OPENNEURO_SNAPSHOT,
        help="OpenNeuro snapshot tag.",
    )
    return parser.parse_args()


def main() -> None:
    """Run selected public dataset bootstraps."""

    args = parse_args()
    root = repo_root()
    targets = (
        ["mimic_demo", "chbmit", "openneuro"]
        if args.dataset == "all"
        else [args.dataset]
    )
    failures: list[str] = []

    for target in targets:
        try:
            if target == "mimic_demo":
                bootstrap_mimic_demo(root)
            elif target == "chbmit":
                bootstrap_chbmit(root)
            elif target == "openneuro":
                bootstrap_openneuro(
                    root,
                    dataset_id=args.openneuro_dataset,
                    snapshot=args.openneuro_snapshot,
                )
        except Exception as exc:
            failures.append(f"{target}: {exc}")
            print(f"FAILED {target}: {exc}")

    if failures:
        print("\nBootstrap completed with failures:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("\nBootstrap completed successfully.")


if __name__ == "__main__":
    main()
