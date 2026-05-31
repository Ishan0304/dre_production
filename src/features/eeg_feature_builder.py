"""Handcrafted EEG feature extraction for EDF recordings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EEGFeatureBuildConfig:
    """Configuration for lightweight EEG recording feature extraction."""

    channel_limit: int | None = None
    max_seconds: float | None = 300.0
    epoch_seconds: float = 10.0
    include_frequency_features: bool = True
    include_amplitude_features: bool = True
    include_basic_signal_quality: bool = True
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class EEGRecordingFeatureResult:
    """Feature extraction result for one EEG recording."""

    patient_id: str
    edf_path: str
    feature_values: dict[str, object]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return asdict(self)


class EEGFeatureBuilder:
    """Extract auditable handcrafted features from EDF EEG recordings."""

    def __init__(self, config: EEGFeatureBuildConfig | None = None) -> None:
        self.config = config or EEGFeatureBuildConfig()

    def extract_recording_features(
        self,
        patient_id: str,
        edf_path: str | Path,
    ) -> EEGRecordingFeatureResult:
        """Extract lightweight recording-level EEG features from one EDF file."""

        mne = _import_mne()
        path = Path(edf_path)
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
        if self.config.channel_limit is not None:
            raw.pick(raw.ch_names[: self.config.channel_limit])

        sfreq = float(raw.info["sfreq"])
        duration = float(raw.n_times / sfreq) if sfreq > 0 else 0.0
        stop_seconds = min(duration, self.config.max_seconds) if self.config.max_seconds else duration
        stop_sample = int(stop_seconds * sfreq)
        data = raw.get_data(start=0, stop=stop_sample)
        notes: list[str] = []
        if self.config.notes:
            notes.append(self.config.notes)
        if self.config.max_seconds is not None and duration > self.config.max_seconds:
            notes.append("recording was truncated to configured max_seconds")

        features: dict[str, object] = {
            "patient_id": patient_id,
            "edf_path": str(path),
            "recording_duration_seconds": stop_seconds,
            "sampling_frequency_hz": sfreq,
            "channel_count": int(data.shape[0]),
        }
        if self.config.include_amplitude_features:
            features.update(_amplitude_features(data))
        if self.config.include_basic_signal_quality:
            features.update(_quality_features(data))
        if self.config.include_frequency_features:
            features.update(_frequency_features(data, sfreq))

        return EEGRecordingFeatureResult(
            patient_id=patient_id,
            edf_path=str(path),
            feature_values=features,
            notes=notes,
        )

    def build_recording_feature_table(
        self,
        inventory_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build recording-level feature and error tables from an EDF inventory."""

        feature_rows: list[dict[str, object]] = []
        error_rows: list[dict[str, object]] = []
        if inventory_df.empty:
            return pd.DataFrame(), pd.DataFrame(
                columns=["patient_id", "edf_path", "error_type", "error_message"]
            )

        sorted_inventory = inventory_df.sort_values(["patient_id", "edf_path"]).reset_index(drop=True)
        for _, row in sorted_inventory.iterrows():
            patient_id = str(row["patient_id"])
            edf_path = str(row["edf_path"])
            try:
                result = self.extract_recording_features(patient_id, edf_path)
                feature_rows.append(result.feature_values)
            except Exception as exc:
                error_rows.append(
                    {
                        "patient_id": patient_id,
                        "edf_path": edf_path,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

        feature_df = pd.DataFrame(feature_rows)
        error_df = pd.DataFrame(
            error_rows,
            columns=["patient_id", "edf_path", "error_type", "error_message"],
        )
        return feature_df, error_df


def _import_mne():
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for EEG EDF feature extraction.") from exc
    return mne


def _finite_values(data: np.ndarray) -> np.ndarray:
    return data[np.isfinite(data)]


def _amplitude_features(data: np.ndarray) -> dict[str, float | None]:
    values = _finite_values(data)
    if values.size == 0:
        return {
            "mean_signal_amplitude": None,
            "std_signal_amplitude": None,
            "median_signal_amplitude": None,
            "signal_abs_mean": None,
            "signal_rms": None,
            "signal_line_length_mean": None,
        }
    diffs = np.diff(data, axis=1) if data.shape[1] > 1 else np.array([])
    finite_diffs = _finite_values(diffs)
    return {
        "mean_signal_amplitude": float(np.mean(values)),
        "std_signal_amplitude": float(np.std(values)),
        "median_signal_amplitude": float(np.median(values)),
        "signal_abs_mean": float(np.mean(np.abs(values))),
        "signal_rms": float(np.sqrt(np.mean(values**2))),
        "signal_line_length_mean": (
            float(np.mean(np.abs(finite_diffs))) if finite_diffs.size else None
        ),
    }


def _quality_features(data: np.ndarray) -> dict[str, float]:
    return {
        "finite_sample_fraction": float(np.isfinite(data).mean()) if data.size else 0.0,
        "zero_sample_fraction": float((data == 0).mean()) if data.size else 0.0,
    }


def _frequency_features(data: np.ndarray, sfreq: float) -> dict[str, float | None]:
    if data.size == 0 or sfreq <= 0:
        return _empty_bandpower_features()
    finite_data = np.where(np.isfinite(data), data, 0.0)
    freqs = np.fft.rfftfreq(finite_data.shape[1], d=1.0 / sfreq)
    spectrum = np.abs(np.fft.rfft(finite_data, axis=1)) ** 2
    total_power = float(np.mean(spectrum)) if spectrum.size else 0.0
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 80.0),
    }
    features: dict[str, float | None] = {"spectral_power_mean": total_power}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            features[f"bandpower_{name}_mean"] = None
        else:
            features[f"bandpower_{name}_mean"] = float(np.mean(spectrum[:, mask]))
    return features


def _empty_bandpower_features() -> dict[str, None]:
    features = {"spectral_power_mean": None}
    for name in ("delta", "theta", "alpha", "beta", "gamma"):
        features[f"bandpower_{name}_mean"] = None
    return features
