"""
Unit tests for src/data/features.py

Run with:
    pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    rms,
    kurtosis,
    crest_factor,
    peak_to_peak,
    skewness,
    shape_factor,
    extract_time_features,
    extract_freq_features,
    extract_all_single,
    extract_all_features,
    add_rolling_features,
    fit_scaler,
    apply_scaler,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
N_SAMPLES = 20_480
N_FILES   = 30
N_BEARINGS = 4

@pytest.fixture
def sine_signal():
    """Pure 500 Hz sine — known analytical properties."""
    t = np.linspace(0, 1, N_SAMPLES, endpoint=False)
    return np.sin(2 * np.pi * 500 * t).astype(np.float32)

@pytest.fixture
def gaussian_signal():
    return RNG.standard_normal(N_SAMPLES).astype(np.float32)

@pytest.fixture
def zero_signal():
    return np.zeros(N_SAMPLES, dtype=np.float32)

@pytest.fixture
def mock_signals():
    """Synthetic (n_files, n_samples, n_bearings) block."""
    return RNG.standard_normal((N_FILES, N_SAMPLES, N_BEARINGS)).astype(np.float32)

@pytest.fixture
def mock_timestamps():
    return list(pd.date_range("2024-01-01", periods=N_FILES, freq="10min"))

@pytest.fixture
def bearing_names():
    return [f"bearing_{i+1}" for i in range(N_BEARINGS)]


# ── Time-domain feature tests ─────────────────────────────────────────────────

class TestRMS:
    def test_sine_rms(self, sine_signal):
        """RMS of a unit-amplitude sine = 1/√2."""
        assert abs(rms(sine_signal) - 1.0 / np.sqrt(2)) < 1e-3

    def test_constant_signal(self):
        assert abs(rms(np.full(1000, 3.0)) - 3.0) < 1e-6

    def test_zero_signal(self, zero_signal):
        assert rms(zero_signal) == pytest.approx(0.0, abs=1e-9)

    def test_returns_float(self, gaussian_signal):
        assert isinstance(rms(gaussian_signal), float)


class TestKurtosis:
    def test_gaussian_kurtosis_near_zero(self, gaussian_signal):
        """Fisher kurtosis of Gaussian ≈ 0."""
        assert abs(kurtosis(gaussian_signal)) < 0.5

    def test_impulse_has_high_kurtosis(self):
        """A signal with a single sharp spike has high kurtosis."""
        x = np.zeros(N_SAMPLES, dtype=np.float32)
        x[100] = 100.0
        assert kurtosis(x) > 100

    def test_returns_float(self, gaussian_signal):
        assert isinstance(kurtosis(gaussian_signal), float)


class TestCrestFactor:
    def test_sine_crest_factor(self, sine_signal):
        """Crest factor of a sine = √2 ≈ 1.414."""
        assert abs(crest_factor(sine_signal) - np.sqrt(2)) < 0.05

    def test_zero_signal_no_division_error(self, zero_signal):
        result = crest_factor(zero_signal)
        assert np.isfinite(result)

    def test_returns_float(self, gaussian_signal):
        assert isinstance(crest_factor(gaussian_signal), float)


class TestPeakToPeak:
    def test_known_range(self):
        x = np.array([-3.0, 0.0, 5.0], dtype=np.float32)
        assert peak_to_peak(x) == pytest.approx(8.0)

    def test_constant_is_zero(self):
        assert peak_to_peak(np.ones(100, dtype=np.float32)) == pytest.approx(0.0)


class TestSkewness:
    def test_symmetric_signal_near_zero(self, sine_signal):
        assert abs(skewness(sine_signal)) < 0.05

    def test_returns_float(self, gaussian_signal):
        assert isinstance(skewness(gaussian_signal), float)


class TestShapeFactor:
    def test_sine_shape_factor(self, sine_signal):
        """Shape factor of a sine ≈ π/(2√2) ≈ 1.1107."""
        assert abs(shape_factor(sine_signal) - (np.pi / (2 * np.sqrt(2)))) < 0.05

    def test_zero_signal_no_error(self, zero_signal):
        assert np.isfinite(shape_factor(zero_signal))


class TestExtractTimeFeatures:
    def test_keys(self, gaussian_signal):
        feats = extract_time_features(gaussian_signal)
        expected = {"rms", "kurtosis", "crest_factor", "peak_to_peak", "skewness", "shape_factor"}
        assert set(feats.keys()) == expected

    def test_all_finite(self, gaussian_signal):
        feats = extract_time_features(gaussian_signal)
        assert all(np.isfinite(v) for v in feats.values())


# ── Frequency-domain feature tests ───────────────────────────────────────────

class TestExtractFreqFeatures:
    def test_keys(self, gaussian_signal):
        feats = extract_freq_features(gaussian_signal)
        expected = {"band_low", "band_mid", "band_high",
                    "spectral_entropy", "dominant_frequency"}
        assert set(feats.keys()) == expected

    def test_band_energies_sum_leq_one(self, gaussian_signal):
        feats = extract_freq_features(gaussian_signal)
        total = feats["band_low"] + feats["band_mid"] + feats["band_high"]
        assert total <= 1.0 + 1e-5

    def test_band_energies_non_negative(self, gaussian_signal):
        feats = extract_freq_features(gaussian_signal)
        assert feats["band_low"]  >= 0
        assert feats["band_mid"]  >= 0
        assert feats["band_high"] >= 0

    def test_dominant_freq_in_range(self, sine_signal):
        """500 Hz sine should have dominant frequency near 500 Hz."""
        feats = extract_freq_features(sine_signal, fs=20_000)
        assert abs(feats["dominant_frequency"] - 500.0) < 5.0

    def test_spectral_entropy_positive(self, gaussian_signal):
        assert extract_freq_features(gaussian_signal)["spectral_entropy"] > 0


# ── Batch extraction tests ────────────────────────────────────────────────────

class TestExtractAllFeatures:
    def test_output_shape(self, mock_signals, mock_timestamps, bearing_names):
        df = extract_all_features(mock_signals, mock_timestamps, bearing_names)
        n_features_per_bearing = 11   # 6 time + 5 freq
        assert df.shape == (N_FILES, N_BEARINGS * n_features_per_bearing)

    def test_index_is_datetime(self, mock_signals, mock_timestamps, bearing_names):
        df = extract_all_features(mock_signals, mock_timestamps, bearing_names)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_all_finite(self, mock_signals, mock_timestamps, bearing_names):
        df = extract_all_features(mock_signals, mock_timestamps, bearing_names)
        assert df.notna().all().all()

    def test_column_naming(self, mock_signals, mock_timestamps, bearing_names):
        df = extract_all_features(mock_signals, mock_timestamps, bearing_names)
        assert "bearing_1_rms" in df.columns
        assert "bearing_4_kurtosis" in df.columns


# ── Rolling features tests ────────────────────────────────────────────────────

class TestAddRollingFeatures:
    def test_adds_expected_columns(self, mock_signals, mock_timestamps, bearing_names):
        df_base = extract_all_features(mock_signals, mock_timestamps, bearing_names)
        df_roll = add_rolling_features(df_base, base_features=["rms"], windows=[5])
        assert "bearing_1_rms_roll5_mean" in df_roll.columns
        assert "bearing_1_rms_roll5_std"  in df_roll.columns

    def test_shape_increases(self, mock_signals, mock_timestamps, bearing_names):
        df_base = extract_all_features(mock_signals, mock_timestamps, bearing_names)
        df_roll = add_rolling_features(df_base)
        assert df_roll.shape[1] > df_base.shape[1]

    def test_no_nan_after_rolling(self, mock_signals, mock_timestamps, bearing_names):
        df_base = extract_all_features(mock_signals, mock_timestamps, bearing_names)
        df_roll = add_rolling_features(df_base)
        assert not df_roll.isna().any().any()


# ── Scaler tests ──────────────────────────────────────────────────────────────

class TestScaler:
    def test_fit_apply_normalises_train(self):
        X = RNG.standard_normal((100, 8)).astype(np.float32) * 5 + 3
        mu, sigma = fit_scaler(X)
        X_scaled  = apply_scaler(X, mu, sigma)
        assert abs(X_scaled.mean()) < 0.01
        assert abs(X_scaled.std() - 1.0) < 0.01

    def test_zero_std_column_no_inf(self):
        X = np.ones((50, 3), dtype=np.float32)
        mu, sigma = fit_scaler(X)
        X_scaled  = apply_scaler(X, mu, sigma)
        assert np.isfinite(X_scaled).all()

    def test_output_dtype_float32(self):
        X = RNG.standard_normal((50, 4)).astype(np.float32)
        mu, sigma = fit_scaler(X)
        assert apply_scaler(X, mu, sigma).dtype == np.float32
