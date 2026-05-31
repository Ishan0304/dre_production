import numpy as np
import pandas as pd

from features.eeg_feature_builder import (
    EEGFeatureBuilder,
    EEGRecordingFeatureResult,
    _amplitude_features,
    _frequency_features,
    _quality_features,
)


def test_eeg_feature_helpers_compute_basic_signal_features() -> None:
    data = np.array([[0.0, 1.0, -1.0, 0.5], [0.2, 0.1, -0.1, -0.2]])

    amplitude = _amplitude_features(data)
    quality = _quality_features(data)
    frequency = _frequency_features(data, sfreq=4.0)

    assert amplitude["signal_abs_mean"] > 0
    assert amplitude["signal_rms"] > 0
    assert quality["finite_sample_fraction"] == 1.0
    assert "bandpower_delta_mean" in frequency


def test_eeg_recording_feature_table_captures_errors(monkeypatch) -> None:
    builder = EEGFeatureBuilder()

    def fake_extract(patient_id, edf_path):
        if "bad" in str(edf_path):
            raise ValueError("bad recording")
        return EEGRecordingFeatureResult(
            patient_id=patient_id,
            edf_path=str(edf_path),
            feature_values={
                "patient_id": patient_id,
                "edf_path": str(edf_path),
                "feature_a": 1.0,
            },
            notes=[],
        )

    monkeypatch.setattr(builder, "extract_recording_features", fake_extract)
    inventory = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "edf_path": ["good.edf", "bad.edf"],
            "file_name": ["good.edf", "bad.edf"],
        }
    )

    feature_df, error_df = builder.build_recording_feature_table(inventory)

    assert len(feature_df) == 1
    assert len(error_df) == 1
    assert error_df.loc[0, "error_type"] == "ValueError"
