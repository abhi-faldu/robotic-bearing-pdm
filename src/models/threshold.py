"""
Anomaly threshold calibration for the LSTM Autoencoder.

Strategy: fit a Gaussian to reconstruction errors from the healthy training set,
then set the alert threshold at μ + k·σ (default k=3, ~0.13% false-positive rate
under a true Gaussian).

Persistence: threshold parameters are saved alongside the model checkpoint so the
inference service can load them without rerunning training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.models.lstm_autoencoder import LSTMAutoencoder

logger = logging.getLogger(__name__)


def compute_threshold(
    errors: np.ndarray,
    k: float = 3.0,
) -> tuple[float, float, float]:
    """
    Fit Gaussian to reconstruction errors and return threshold statistics.

    Args:
        errors : 1-D array of per-sample MSE values (training / validation set).
        k      : Number of standard deviations above the mean for the threshold.

    Returns:
        (mu, sigma, threshold)  — all floats.
    """
    mu = float(errors.mean())
    sigma = float(errors.std())
    threshold = mu + k * sigma
    logger.info("Threshold: μ=%.6f  σ=%.6f  t=%.6f (k=%.1f)", mu, sigma, threshold, k)
    return mu, sigma, threshold


def collect_errors(
    model: LSTMAutoencoder,
    windows: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run the model over all windows and collect per-window reconstruction errors.

    Args:
        model      : trained (or partially trained) LSTMAutoencoder.
        windows    : float32 ndarray, shape (n_windows, seq_len, n_features).
        batch_size : inference batch size.
        device     : 'cpu' or 'cuda'.

    Returns:
        errors : float32 ndarray, shape (n_windows,).
    """
    model.eval()
    model.to(device)
    errors = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            err = model.reconstruction_error(batch).cpu().numpy()
            errors.append(err)

    return np.concatenate(errors, axis=0)


def save_threshold(
    mu: float,
    sigma: float,
    threshold: float,
    path: str | Path,
) -> None:
    """Persist threshold parameters as a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"mu": mu, "sigma": sigma, "threshold": threshold}, f, indent=2)
    logger.info("Threshold saved to %s", path)


def load_threshold(path: str | Path) -> tuple[float, float, float]:
    """Load threshold parameters saved by save_threshold()."""
    with open(path) as f:
        data = json.load(f)
    return data["mu"], data["sigma"], data["threshold"]
