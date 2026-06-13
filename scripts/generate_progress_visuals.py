"""Generate the likely DRE class balance chart.

Run from the repository root:

    python scripts/generate_progress_visuals.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_PATH = Path("outputs") / "reports" / "figures" / "likely_dre_class_balance.png"
SEARCH_ROOTS = (Path("outputs"), Path("reports"), Path("artifacts"))
DATASET_PATTERNS = (
    "*ehr*patient*dataset*.csv",
    "*ehr_patient_dataset*.csv",
    "*patient_dataset.csv",
)


def find_candidate_files() -> list[Path]:
    """Find likely EHR patient-level dataset CSV files."""

    candidates: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for pattern in DATASET_PATTERNS:
            candidates.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(candidates), key=dataset_preference_key)


def dataset_preference_key(path: Path) -> tuple[int, str]:
    """Prefer files that clearly look like EHR patient dataset outputs."""

    name = path.name.lower()
    path_text = str(path).lower()
    score = 0
    if "ehr" in path_text:
        score -= 4
    if "patient" in name:
        score -= 3
    if "dataset" in name:
        score -= 2
    if "profile" in name or "registry" in name or "manifest" in name:
        score += 10
    return score, path_text


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows from a patient-level dataset."""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def find_ehr_patient_dataset_with_label() -> tuple[Path, list[dict[str, str]]]:
    """Return the best available EHR patient dataset containing likely_dre."""

    candidates = find_candidate_files()
    if not candidates:
        raise FileNotFoundError(
            "Missing EHR patient-level dataset CSV. "
            "Searched outputs/, reports/, and artifacts/ for patient dataset CSV files."
        )

    checked_without_label: list[Path] = []
    for path in candidates:
        rows = load_csv_rows(path)
        if not rows:
            checked_without_label.append(path)
            continue
        if "likely_dre" in rows[0]:
            return path, rows
        checked_without_label.append(path)

    checked_text = "\n".join(f"- {path}" for path in checked_without_label)
    raise ValueError(
        "Missing likely_dre column in candidate EHR patient-level dataset CSV files:\n"
        f"{checked_text}"
    )


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count likely_dre values for classes 0 and 1."""

    counts = {"0": 0, "1": 0}
    for row in rows:
        value = str(row.get("likely_dre", "")).strip().lower()
        if value in {"1", "true", "yes", "y"}:
            counts["1"] += 1
        elif value in {"0", "false", "no", "n"}:
            counts["0"] += 1
        else:
            raise ValueError(f"Unexpected likely_dre value: {row.get('likely_dre')!r}")
    return counts


def save_class_balance_chart(counts: dict[str, int], output_path: Path) -> Path:
    """Save a presentation-ready class balance bar chart."""

    labels = ["likely_dre = 0", "likely_dre = 1"]
    values = [counts["0"], counts["1"]]

    plt.figure(figsize=(7.5, 4.8))
    bars = plt.bar(labels, values)
    plt.title("Likely DRE class balance")
    plt.xlabel("Class")
    plt.ylabel("Patient count")

    upper = max(values) if values else 0
    plt.ylim(0, max(1, upper * 1.18))
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path.resolve()


def main() -> None:
    """Find the EHR patient dataset and generate the class balance chart."""

    try:
        dataset_path, rows = find_ehr_patient_dataset_with_label()
        counts = class_counts(rows)
        saved_path = save_class_balance_chart(counts, OUTPUT_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    print(f"Read EHR patient-level dataset: {dataset_path.resolve()}")
    print(f"Saved PNG: {saved_path}")


if __name__ == "__main__":
    main()
