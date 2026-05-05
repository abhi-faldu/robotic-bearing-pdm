"""
Feature extraction for NASA IMS bearing vibration signals.

Two feature groups are computed per 10-minute snapshot:

  Time-domain   — RMS, kurtosis, crest factor, peak-to-peak, skewness, shape factor
  Frequency-domain — FFT band energies (3 bands), spectral entropy, dominant frequency

Pipeline:
    signals (n_files, 20480, n_bearings)
        → extract_all_features()
        → DataFrame (n_files, n_bearings × 9 raw features)
        → add_rolling_features()
        → DataFrame (n_files, n_bearings × 9 + rolling columns)
        → save to data/processed/features_dataset2.csv
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger(__name__)

SAMPLE_RATE_HZ = 20_000

# Frequency bands (Hz) — chosen to cover sub-harmonic, mid, and high-frequency
# defect signatures typically seen in rolling-element bearing analysis
FREQ_BANDS = {
    "band_low":  (0,    1_000),
    "band_mid":  (1_000, 5_000),
    "band_high": (5_000, 10_000),
}


# ── Time-domain features ──────────────────────────────────────────────────────

def rms(x: np.ndarray) -> float:
    """Root Mean Square — overall vibration energy."""
    return float(np.sqrt(np.mean(x ** 2)))


def kurtosis(x: np.ndarray) -> float:
    """
    Statistical kurtosis (Fisher's definition, zero for Gaussian).
    Rises sharply when bearing defects produce isolated impulsive events.
    Returns 0.0 for constant or near-constant signals.
    """
    return float(np.nan_to_num(stats.kurtosis(x, fisher=True), nan=0.0))


def crest_factor(x: np.ndarray) -> float:
    """
    Peak / RMS — sensitive to early-stage surface pitting before RMS changes.
    Values > 6 are generally considered indicative of bearing distress.
    """
    _rms = rms(x)
    return float(np.max(np.abs(x)) / (_rms + 1e-9))


def peak_to_peak(x: np.ndarray) -> float:
    """Max - Min amplitude — increases with looseness and spalling."""
    return float(np.ptp(x))


def skewness(x: np.ndarray) -> float:
    """Signal asymmetry — changes when bearing geometry is damaged.
    Returns 0.0 for constant or near-constant signals."""
    return float(np.nan_to_num(stats.skew(x), nan=0.0))


def shape_factor(x: np.ndarray) -> float:
    """
    RMS / mean(|x|) — dimensionless ratio, complements crest factor
    for detecting distributed faults vs. localised defects.
    """
    mean_abs = np.mean(np.abs(x))
    return float(rms(x) / (mean_abs + 1e-9))


def extract_time_features(x: np.ndarray) -> dict:
    """Compute all time-domain features for a single 1-D signal."""
    return {
        "rms":          rms(x),
        "kurtosis":     kurtosis(x),
        "crest_factor": crest_factor(x),
        "peak_to_peak": peak_to_peak(x),
        "skewness":     skewness(x),
        "shape_factor": shape_factor(x),
    }


# ── Frequency-domain features ─────────────────────────────────────────────────

def extract_freq_features(
    x: np.ndarray,
    fs: int = SAMPLE_RATE_HZ,
) -> dict:
    """
    Compute frequency-domain features via real FFT.

    Features:
        band_low / band_mid / band_high — energy fraction in each band
        spectral_entropy                — spread of energy across spectrum
        dominant_frequency              — Hz of the peak magnitude bin
    """
    n = len(x)
    freqs = rfftfreq(n, d=1.0 / fs)          # shape: (n//2 + 1,)
    magnitudes = np.abs(rfft(x)) / n          # normalised magnitude spectrum

    # Band energies (fraction of total power)
    total_power = np.sum(magnitudes ** 2) + 1e-9
    band_energies = {}
    for band_name, (f_lo, f_hi) in FREQ_BANDS.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_energies[band_name] = float(np.sum(magnitudes[mask] ** 2) / total_power)

    # Spectral entropy — low when energy concentrates at defect frequencies
    power_norm = (magnitudes ** 2) / total_power
    power_norm = power_norm[power_norm > 0]   # avoid log(0)
    spectral_entropy = float(-np.sum(power_norm * np.log2(power_norm)))

    # Dominant frequency
    dominant_freq = float(freqs[np.argmax(magnitudes)])

    return {
        **band_energies,
        "spectral_entropy":   spectral_entropy,
        "dominant_frequency": dominant_freq,
    }


def extract_all_single(x: np.ndarray, fs: int = SAMPLE_RATE_HZ) -> dict:
    """Full feature set (time + frequency) for a single 1-D signal."""
    return {**extract_time_features(x), **extract_freq_features(x, fs)}


# ── Batch extraction ──────────────────────────────────────────────────────────

def extract_all_features(
    signals: np.ndarray,
    timestamps: list,
    bearing_names: List[str],
    fs: int = SAMPLE_RATE_HZ,
) -> pd.DataFrame:
    """
    Extract time- and frequency-domain features for every snapshot.

    Args:
        signals       : float32 ndarray, shape (n_files, n_samples, n_bearings)
        timestamps    : list of datetime, length n_files
        bearing_names : list of bearing column names, length n_bearings
        fs            : sampling rate in Hz

    Returns:
        DataFrame indexed by timestamp, shape (n_files, n_bearings × 11 features)
        Column names follow the pattern: `{bearing_name}_{feature_name}`
    """
    n_files, _, n_bearings = signals.shape
    logger.info("Extracting features from %d snapshots × %d bearings …",
                n_files, n_bearings)

    records = []
    for i in range(n_files):
        row: dict = {"timestamp": timestamps[i]}
        for col, bname in enumerate(bearing_names):
            feats = extract_all_single(signals[i, :, col], fs)
            row.update({f"{bname}_{k}": v for k, v in feats.items()})
        records.append(row)

    df = pd.DataFrame(records).set_index("timestamp")
    logger.info("Feature matrix shape: %s", df.shape)
    return df.astype(np.float32)


# ── Rolling statistics ────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    base_features: List[str] | None = None,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """
    Append rolling mean and rolling std columns for selected base features.

    Rolling stats help the LSTM detect gradual trends vs. instantaneous spikes.
    A window of 12 ≈ 2 hours; window of 36 ≈ 6 hours (at 10-min intervals).

    Args:
        df            : feature DataFrame with DatetimeIndex
        base_features : list of column name substrings to roll; defaults to
                        ['rms', 'kurtosis', 'crest_factor']
        windows       : list of rolling window sizes (number of snapshots);
                        defaults to [12, 36]

    Returns:
        DataFrame with additional columns `{col}_roll{w}_mean` and `{col}_roll{w}_std`.
        NaN rows at the start (warm-up period) are forward-filled with the first valid value.
    """
    if base_features is None:
        base_features = ["rms", "kurtosis", "crest_factor"]
    if windows is None:
        windows = [12, 36]

    cols_to_roll = [c for c in df.columns
                    if any(c.endswith(f"_{feat}") for feat in base_features)]

    new_cols = {}
    for col in cols_to_roll:
        for w in windows:
            rolled = df[col].rolling(window=w, min_periods=1)
            new_cols[f"{col}_roll{w}_mean"] = rolled.mean()
            new_cols[f"{col}_roll{w}_std"]  = rolled.std().fillna(0)

    result = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info("Added %d rolling columns (windows=%s)", len(new_cols), windows)
    return result.astype(np.float32)


# ── Normalisation ─────────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std from the training (healthy) window.

    Returns:
        mu    : shape (n_features,)
        sigma : shape (n_features,)  — zero-std columns are set to 1 to avoid /0
    """
    mu    = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def apply_scaler(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Z-score normalise using pre-computed training statistics."""
    return ((X - mu) / sigma).astype(np.float32)
