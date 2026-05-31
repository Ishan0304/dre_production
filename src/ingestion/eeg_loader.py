"""EEG dataset loading utilities for EDF-based directory structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class EEGDatasetLoadConfig:
    """Configuration for scanning an EDF-based EEG dataset."""

    dataset_root: str
    edf_glob_patterns: tuple[str, ...] = ("**/*.edf", "**/*.EDF")
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class EEGDatasetLoadResult:
    """Structured metadata from scanning an EEG dataset."""

    dataset_root: str
    file_count: int
    patient_count: int | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class EEGLoader:
    """Discover EDF recordings and build deterministic EEG file inventories."""

    @staticmethod
    def discover_edf_files(config: EEGDatasetLoadConfig) -> list[Path]:
        """Recursively discover EDF files under the dataset root."""

        root = Path(config.dataset_root)
        if not root.exists():
            raise FileNotFoundError(f"EEG dataset root does not exist: {root}")

        files: set[Path] = set()
        for pattern in config.edf_glob_patterns:
            files.update(path for path in root.glob(pattern) if path.is_file())
        return sorted(files)

    @staticmethod
    def infer_patient_id_from_path(edf_path: Path) -> str:
        """Infer a stable patient identifier from an EDF path."""

        for parent in [edf_path.parent, *edf_path.parents]:
            name = parent.name
            if name.lower().startswith(("chb", "sub-", "patient", "pt")):
                return name
        return edf_path.stem

    def build_file_inventory(
        self,
        config: EEGDatasetLoadConfig,
    ) -> tuple[pd.DataFrame, EEGDatasetLoadResult]:
        """Build an EDF file inventory and structured load metadata."""

        notes: list[str] = []
        if config.notes:
            notes.append(config.notes)

        edf_files = self.discover_edf_files(config)
        rows = [
            {
                "patient_id": self.infer_patient_id_from_path(path),
                "edf_path": str(path),
                "file_name": path.name,
            }
            for path in edf_files
        ]
        inventory_df = pd.DataFrame(rows, columns=["patient_id", "edf_path", "file_name"])
        patient_count = (
            int(inventory_df["patient_id"].nunique(dropna=True))
            if not inventory_df.empty
            else 0
        )
        result = EEGDatasetLoadResult(
            dataset_root=str(Path(config.dataset_root).resolve()),
            file_count=len(inventory_df),
            patient_count=patient_count,
            notes=notes,
        )
        return inventory_df, result
