"""Handcrafted MRI feature extraction for subject-level tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MRIFeatureBuildConfig:
    """Configuration for handcrafted MRI feature extraction."""

    min_foreground_voxels: int = 5000
    foreground_quantile: float = 20.0
    glcm_levels: int = 64
    glcm_max_slices: int = 3
    preferred_t1_tokens: tuple[str, ...] = ("iso08", "sag111", "mprage", "t1w")
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class MRISubjectFeatureResult:
    """Feature extraction result for one MRI subject."""

    subject_id: str
    t1_path: str
    feature_values: dict[str, object]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class MRIFeatureBuilder:
    """Extract auditable handcrafted features from T1w MRI images."""

    def __init__(self, config: MRIFeatureBuildConfig | None = None) -> None:
        self.config = config or MRIFeatureBuildConfig()

    def extract_subject_features(
        self,
        subject_id: str,
        t1_path: str | Path,
    ) -> MRISubjectFeatureResult:
        """Extract handcrafted subject-level features from one T1w NIfTI image."""

        nib = _import_nibabel()
        path = Path(t1_path)
        image = nib.load(str(path))
        data = np.asarray(image.get_fdata(dtype=np.float32))
        finite_mask = np.isfinite(data)
        nan_fraction = 1.0 - float(finite_mask.mean()) if data.size else 1.0
        clean_data = np.where(finite_mask, data, np.nan)

        foreground_mask = self._foreground_mask(clean_data)
        foreground_values = clean_data[foreground_mask]
        voxel_sizes = tuple(float(value) for value in image.header.get_zooms()[:3])
        voxel_volume = float(np.prod(voxel_sizes))

        notes: list[str] = []
        if foreground_values.size < self.config.min_foreground_voxels:
            notes.append("foreground voxel count below configured minimum")

        features: dict[str, object] = {
            "subject_id": subject_id,
            "t1_path": str(path),
            "shape_x": int(data.shape[0]) if data.ndim >= 1 else None,
            "shape_y": int(data.shape[1]) if data.ndim >= 2 else None,
            "shape_z": int(data.shape[2]) if data.ndim >= 3 else None,
            "voxel_size_x": voxel_sizes[0] if len(voxel_sizes) > 0 else None,
            "voxel_size_y": voxel_sizes[1] if len(voxel_sizes) > 1 else None,
            "voxel_size_z": voxel_sizes[2] if len(voxel_sizes) > 2 else None,
            "voxel_volume": voxel_volume,
            "foreground_voxel_count": int(foreground_values.size),
            "foreground_volume": float(foreground_values.size * voxel_volume),
            "foreground_fraction": float(foreground_values.size / data.size) if data.size else 0.0,
            "image_nan_fraction": nan_fraction,
        }
        features.update(self._intensity_summary(foreground_values))
        features.update(self._glcm_summary(clean_data, foreground_mask))

        return MRISubjectFeatureResult(
            subject_id=subject_id,
            t1_path=str(path),
            feature_values=features,
            notes=notes,
        )

    def build_feature_table(
        self,
        inventory_df: pd.DataFrame,
        participants_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build a feature table and error table from a subject inventory."""

        feature_rows: list[dict[str, object]] = []
        error_rows: list[dict[str, object]] = []
        if inventory_df.empty:
            return pd.DataFrame(), pd.DataFrame(columns=["subject_id", "error_type", "error_message"])

        sorted_inventory = inventory_df.sort_values("subject_id").reset_index(drop=True)
        for _, row in sorted_inventory.iterrows():
            if not bool(row.get("has_t1", False)) or pd.isna(row.get("t1_path")):
                error_rows.append(
                    {
                        "subject_id": str(row.get("subject_id")),
                        "error_type": "missing_t1",
                        "error_message": "subject has no discovered T1w image",
                    }
                )
                continue
            try:
                result = self.extract_subject_features(
                    subject_id=str(row["subject_id"]),
                    t1_path=str(row["t1_path"]),
                )
                feature_rows.append(result.feature_values)
            except Exception as exc:
                error_rows.append(
                    {
                        "subject_id": str(row.get("subject_id")),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

        feature_df = pd.DataFrame(feature_rows)
        if participants_df is not None and not participants_df.empty and not feature_df.empty:
            feature_df = self._join_participants(feature_df, participants_df)
        error_df = pd.DataFrame(error_rows, columns=["subject_id", "error_type", "error_message"])
        return feature_df, error_df

    def _foreground_mask(self, data: np.ndarray) -> np.ndarray:
        finite_values = data[np.isfinite(data)]
        if finite_values.size == 0:
            return np.zeros(data.shape, dtype=bool)
        threshold = float(np.nanpercentile(finite_values, self.config.foreground_quantile))
        return np.isfinite(data) & (data > threshold)

    @staticmethod
    def _intensity_summary(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {
                "t1_intensity_mean": None,
                "t1_intensity_std": None,
                "t1_intensity_p10": None,
                "t1_intensity_p50": None,
                "t1_intensity_p90": None,
                "t1_intensity_min": None,
                "t1_intensity_max": None,
            }
        return {
            "t1_intensity_mean": float(np.nanmean(values)),
            "t1_intensity_std": float(np.nanstd(values)),
            "t1_intensity_p10": float(np.nanpercentile(values, 10)),
            "t1_intensity_p50": float(np.nanpercentile(values, 50)),
            "t1_intensity_p90": float(np.nanpercentile(values, 90)),
            "t1_intensity_min": float(np.nanmin(values)),
            "t1_intensity_max": float(np.nanmax(values)),
        }

    def _glcm_summary(self, data: np.ndarray, foreground_mask: np.ndarray) -> dict[str, float | None]:
        metrics = {"contrast": [], "homogeneity": [], "energy": [], "correlation": []}
        if data.ndim < 3 or foreground_mask.sum() == 0:
            return _empty_glcm_summary()

        candidate_slices = np.where(foreground_mask.sum(axis=(0, 1)) > 0)[0]
        if candidate_slices.size == 0:
            return _empty_glcm_summary()
        center = len(candidate_slices) // 2
        half_window = max(self.config.glcm_max_slices // 2, 0)
        selected_slices = candidate_slices[max(0, center - half_window) : center + half_window + 1]

        foreground_values = data[foreground_mask]
        min_value = float(np.nanmin(foreground_values))
        max_value = float(np.nanmax(foreground_values))
        if min_value == max_value:
            return _empty_glcm_summary()

        for slice_index in selected_slices[: self.config.glcm_max_slices]:
            quantized = _quantize(data[:, :, slice_index], min_value, max_value, self.config.glcm_levels)
            mask = foreground_mask[:, :, slice_index]
            glcm = _glcm_horizontal(quantized, mask, self.config.glcm_levels)
            if glcm.sum() == 0:
                continue
            slice_metrics = _glcm_metrics(glcm)
            for name, value in slice_metrics.items():
                metrics[name].append(value)

        return {
            f"glcm_{name}_{stat}": _optional_stat(values, stat)
            for name, values in metrics.items()
            for stat in ("mean", "std")
        }

    @staticmethod
    def _join_participants(feature_df: pd.DataFrame, participants_df: pd.DataFrame) -> pd.DataFrame:
        participants = participants_df.copy()
        if "participant_id" in participants.columns:
            participants = participants.rename(columns={"participant_id": "subject_id"})
        if "subject_id" not in participants.columns:
            return feature_df
        return feature_df.merge(participants, on="subject_id", how="left")


def _import_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required for MRI feature extraction.") from exc
    return nib


def _empty_glcm_summary() -> dict[str, None]:
    return {
        f"glcm_{name}_{stat}": None
        for name in ("contrast", "homogeneity", "energy", "correlation")
        for stat in ("mean", "std")
    }


def _quantize(slice_data: np.ndarray, min_value: float, max_value: float, levels: int) -> np.ndarray:
    scaled = (slice_data - min_value) / (max_value - min_value)
    clipped = np.clip(scaled, 0.0, 1.0)
    return np.floor(clipped * (levels - 1)).astype(np.int32)


def _glcm_horizontal(quantized: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    glcm = np.zeros((levels, levels), dtype=np.float64)
    left = quantized[:, :-1]
    right = quantized[:, 1:]
    pair_mask = mask[:, :-1] & mask[:, 1:]
    for i, j in zip(left[pair_mask], right[pair_mask], strict=False):
        glcm[int(i), int(j)] += 1.0
    if glcm.sum() > 0:
        glcm = glcm / glcm.sum()
    return glcm


def _glcm_metrics(glcm: np.ndarray) -> dict[str, float]:
    indices = np.arange(glcm.shape[0])
    i_grid, j_grid = np.meshgrid(indices, indices, indexing="ij")
    diff = i_grid - j_grid
    contrast = float(np.sum((diff**2) * glcm))
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(diff))))
    energy = float(np.sqrt(np.sum(glcm**2)))
    mu_i = float(np.sum(i_grid * glcm))
    mu_j = float(np.sum(j_grid * glcm))
    sigma_i = float(np.sqrt(np.sum(((i_grid - mu_i) ** 2) * glcm)))
    sigma_j = float(np.sqrt(np.sum(((j_grid - mu_j) ** 2) * glcm)))
    if sigma_i == 0.0 or sigma_j == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.sum((i_grid - mu_i) * (j_grid - mu_j) * glcm) / (sigma_i * sigma_j))
    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
    }


def _optional_stat(values: list[float], stat: str) -> float | None:
    if not values:
        return None
    if stat == "mean":
        return float(np.mean(values))
    return float(np.std(values))
