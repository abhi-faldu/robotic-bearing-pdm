"""
NASA IMS Bearing Dataset loader.

Dataset layout on disk (use Dataset 2 — clearest failure progression):
    data/raw/2nd_test/
        2004.02.12.10.32.39   <- space-separated, no header
        2004.02.12.11.02.39
        ...

Each file is one 10-minute vibration snapshot sampled at 20 kHz:
    Dataset 1 → 20480 rows × 8 columns  (2 sensors per bearing, 4 bearings)
    Dataset 2 → 20480 rows × 4 columns  (1 sensor per bearing, 4 bearings)  ← default
    Dataset 3 → 20480 rows × 4 columns

Download from Kaggle: https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
Extract so that 2nd_test/ sits directly under data/raw/.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Dataset metadata ──────────────────────────────────────────────────────────

_DATASET_META = {
    1: {
        "folder": "1st_test",
        "n_channels": 8,
        "bearing_names": [
            "b1_ch1", "b1_ch2",
            "b2_ch1", "b2_ch2",
            "b3_ch1", "b3_ch2",
            "b4_ch1", "b4_ch2",
        ],
    },
    2: {
        "folder": "2nd_test",
        "n_channels": 4,
        "bearing_names": ["bearing_1", "bearing_2", "bearing_3", "bearing_4"],
    },
    3: {
        "folder": "3rd_test",
        "n_channels": 4,
        "bearing_names": ["bearing_1", "bearing_2", "bearing_3", "bearing_4"],
    },
}

SAMPLE_RATE_HZ = 20_000   # 20 kHz — fixed for all IMS datasets
SAMPLES_PER_FILE = 20_480  # 1.024 s of signal per snapshot


# ── File parsing helpers ──────────────────────────────────────────────────────

def _parse_timestamp(filename: str) -> datetime:
    """Parse IMS filename '2004.02.12.10.32.39' → datetime."""
    try:
        return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")
    except ValueError:
        # Some Kaggle re-uploads use underscores — handle gracefully
        cleaned = filename.replace("_", ".").split(".")[: 6]
        return datetime.strptime(".".join(cleaned), "%Y.%m.%d.%H.%M.%S")


def _load_single_file(path: Path, n_channels: int) -> np.ndarray:
    """
    Load one snapshot file → float32 array of shape (SAMPLES_PER_FILE, n_channels).
    Separator is a single space; some Kaggle versions use tabs — both handled.
    """
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape != (SAMPLES_PER_FILE, n_channels):
        raise ValueError(
            f"{path.name}: expected ({SAMPLES_PER_FILE}, {n_channels}), "
            f"got {data.shape}"
        )
    return data


# ── Public API ────────────────────────────────────────────────────────────────

def load_snapshots(
    data_dir: str | Path,
    dataset: int = 2,
    limit: int | None = None,
) -> Tuple[np.ndarray, List[datetime], List[str]]:
    """
    Load all snapshot files from the dataset folder in chronological order.

    Args:
        data_dir: Root of data/raw/ (the folder that contains 2nd_test/, etc.).
        dataset:  IMS dataset number — 1, 2, or 3. Defaults to 2.
        limit:    If set, load only the first `limit` files (useful for quick tests).

    Returns:
        signals    : float32 ndarray, shape (n_files, SAMPLES_PER_FILE, n_channels)
        timestamps : list of datetime, length n_files
        bearing_names: list of column name strings, length n_channels

    Raises:
        FileNotFoundError: if the dataset sub-folder does not exist.
        ValueError:        if any file has an unexpected shape.
    """
    meta = _DATASET_META[dataset]
    folder = Path(data_dir) / meta["folder"]

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset {dataset} folder not found: {folder}\n"
            "Download from Kaggle and extract so that '2nd_test/' sits under data/raw/."
        )

    # Collect and sort files by filename (== chronological order)
    files = sorted(
        [f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")],
        key=lambda f: f.name,
    )
    if not files:
        raise FileNotFoundError(f"No data files found in {folder}")

    if limit is not None:
        files = files[:limit]

    logger.info("Loading %d snapshots from %s …", len(files), folder)

    signals, timestamps = [], []
    for f in files:
        signals.append(_load_single_file(f, meta["n_channels"]))
        timestamps.append(_parse_timestamp(f.name))

    signals_arr = np.stack(signals, axis=0)  # (n_files, 20480, n_channels)
    logger.info("Loaded signals: shape=%s  dtype=%s", signals_arr.shape, signals_arr.dtype)

    return signals_arr, timestamps, meta["bearing_names"]


def load_feature_matrix(
    features_path: str | Path,
) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Load a pre-computed feature matrix saved by features.py.

    The file is a CSV with a 'timestamp' index column and one column per feature.

    Returns:
        X          : float32 ndarray, shape (n_files, n_features)
        timestamps : pd.DatetimeIndex
        feature_names: list of column name strings
    """
    df = pd.read_csv(features_path, index_col="timestamp", parse_dates=True)
    return (
        df.values.astype(np.float32),
        df.index,
        list(df.columns),
    )


def create_windows(
    X: np.ndarray,
    seq_len: int = 50,
    step: int = 1,
) -> np.ndarray:
    """
    Slice a feature matrix into overlapping windows for LSTM input.

    Args:
        X       : 2-D array of shape (n_timesteps, n_features).
        seq_len : Number of consecutive time steps per window. Default 50.
        step    : Stride between window starts. Default 1 (fully overlapping).

    Returns:
        windows : float32 ndarray of shape (n_windows, seq_len, n_features).

    Example:
        X.shape = (984, 8)  →  create_windows(X, seq_len=50, step=1)
        windows.shape = (935, 50, 8)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if seq_len > len(X):
        raise ValueError(f"seq_len ({seq_len}) exceeds number of timesteps ({len(X)})")

    n_windows = (len(X) - seq_len) // step + 1
    windows = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1]))
    # sliding_window_view returns (n_windows, 1, seq_len, n_features) — squeeze axis 1
    windows = windows[::step, 0, :, :]
    return windows.astype(np.float32)


def train_test_split_temporal(
    X: np.ndarray,
    timestamps: List[datetime] | pd.DatetimeIndex,
    train_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Temporal train/test split — NO shuffling.

    The LSTM Autoencoder is trained only on the first `train_frac` of data
    (assumed healthy), and evaluated on the remainder.

    Args:
        X          : feature matrix, shape (n_timesteps, n_features).
        timestamps : matching sequence of timestamps.
        train_frac : fraction of data used for training. Default 0.20.

    Returns:
        X_train, X_test, ts_train, ts_test
    """
    split = int(len(X) * train_frac)
    ts = list(timestamps)
    logger.info(
        "Temporal split — train: %d files (%.0f%%), test: %d files (%.0f%%)",
        split, train_frac * 100,
        len(X) - split, (1 - train_frac) * 100,
    )
    return X[:split], X[split:], ts[:split], ts[split:]
