"""MRI dataset loading utilities for BIDS-like directory structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class MRIDatasetLoadConfig:
    """Configuration for scanning a BIDS-like MRI dataset."""

    dataset_root: str
    participants_file: str = "participants.tsv"
    require_participants_file: bool = True
    t1_pattern: str = "*T1w.nii.gz"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class MRIDatasetLoadResult:
    """Structured metadata from scanning an MRI dataset."""

    dataset_root: str
    participants_row_count: int | None
    participants_columns: list[str]
    subject_count: int
    subjects_with_t1_count: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class MRILoader:
    """Load participants metadata and scan BIDS-like MRI subject folders."""

    def load_participants(self, config: MRIDatasetLoadConfig) -> pd.DataFrame | None:
        """Load participants metadata when available."""

        participants_path = Path(config.dataset_root) / config.participants_file
        if not participants_path.exists():
            if config.require_participants_file:
                raise FileNotFoundError(f"Missing participants file: {participants_path}")
            return None
        return pd.read_csv(participants_path, sep="\t")

    @staticmethod
    def discover_subject_dirs(config: MRIDatasetLoadConfig) -> list[Path]:
        """Discover BIDS-style subject directories under the dataset root."""

        root = Path(config.dataset_root)
        if not root.exists():
            raise FileNotFoundError(f"MRI dataset root does not exist: {root}")
        return sorted(path for path in root.glob("sub-*") if path.is_dir())

    @staticmethod
    def find_t1_for_subject(subject_dir: Path, t1_pattern: str = "*T1w.nii.gz") -> Path | None:
        """Find one T1w image for a subject using deterministic ordering."""

        anat_dir = subject_dir / "anat"
        if not anat_dir.exists():
            return None
        matches = sorted(path for path in anat_dir.glob(t1_pattern) if path.is_file())
        if not matches:
            return None
        return matches[0]

    def scan_dataset(
        self,
        config: MRIDatasetLoadConfig,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame, MRIDatasetLoadResult]:
        """Scan a BIDS-like dataset and return participants, inventory, and metadata."""

        notes: list[str] = []
        if config.notes:
            notes.append(config.notes)

        participants_df = self.load_participants(config)
        if participants_df is None:
            notes.append("participants file absent")
        subject_dirs = self.discover_subject_dirs(config)

        rows: list[dict[str, object]] = []
        for subject_dir in subject_dirs:
            t1_path = self.find_t1_for_subject(subject_dir, config.t1_pattern)
            rows.append(
                {
                    "subject_id": subject_dir.name,
                    "subject_dir": str(subject_dir),
                    "t1_path": str(t1_path) if t1_path else None,
                    "has_t1": t1_path is not None,
                }
            )

        inventory_df = pd.DataFrame(
            rows,
            columns=["subject_id", "subject_dir", "t1_path", "has_t1"],
        )
        participants_columns = (
            []
            if participants_df is None
            else [str(col) for col in participants_df.columns]
        )
        participants_row_count = None if participants_df is None else len(participants_df)

        result = MRIDatasetLoadResult(
            dataset_root=str(Path(config.dataset_root).resolve()),
            participants_row_count=participants_row_count,
            participants_columns=participants_columns,
            subject_count=len(inventory_df),
            subjects_with_t1_count=int(inventory_df["has_t1"].sum()) if not inventory_df.empty else 0,
            notes=notes,
        )
        return participants_df, inventory_df, result
