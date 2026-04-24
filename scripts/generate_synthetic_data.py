"""
Synthetic NASA IMS Bearing Dataset Generator

Generates a realistic run-to-failure vibration dataset in exactly the same
file format as the real NASA IMS Dataset 2, so the full training pipeline
runs without any Kaggle download.

Usage:
    python scripts/generate_synthetic_data.py

Output:
    data/raw/2nd_test/<timestamp_files>   (default: 984 files, ~2 GB)

Physics simulation:
    - Bearing 1: inner race defect (BPFI) — amplitude modulation + impulsive spikes
    - Bearing 2: outer race defect (BPFO) — growing broadband energy
    - Bearing 3: healthy throughout
    - Bearing 4: healthy throughout

Each file: 20480 rows × 4 columns, space-separated float32 values.
Filenames: '2004.02.12.10.32.39' format (10-minute intervals).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants matching real IMS Dataset 2 ─────────────────────────────────────
FS           = 20_000       # sampling rate (Hz)
N_SAMPLES    = 20_480       # samples per snapshot
N_SNAPSHOTS  = 984          # ~7 days at 10-min intervals
N_BEARINGS   = 4

# Bearing defect frequencies (Hz) for typical ball bearing at ~2000 rpm
BPFI = 296.0   # ball pass frequency inner race
BPFO = 185.0   # ball pass frequency outer race
BSF  = 139.0   # ball spin frequency


def _healthy_signal(
    rng: np.random.Generator,
    n: int,
    fs: int,
    amplitude: float = 0.02,
) -> np.ndarray:
    """White noise + small residual vibration baseline."""
    t = np.linspace(0, n / fs, n, endpoint=False)
    noise   = rng.normal(0, amplitude, n)
    # Shaft harmonics (1×, 2×, 3× running speed at 33 Hz)
    shaft   = 0.003 * np.sin(2 * np.pi * 33 * t)
    shaft  += 0.001 * np.sin(2 * np.pi * 66 * t)
    return (noise + shaft).astype(np.float32)


def _inner_race_signal(
    rng: np.random.Generator,
    n: int,
    fs: int,
    severity: float,       # 0.0 (healthy) → 1.0 (failure)
) -> np.ndarray:
    """
    Inner race defect: impulse train at BPFI modulated by shaft rotation,
    amplitude grows with severity.
    """
    t   = np.linspace(0, n / fs, n, endpoint=False)
    sig = _healthy_signal(rng, n, fs, amplitude=0.02 + 0.05 * severity)

    if severity < 0.05:
        return sig

    # Impulsive spikes at BPFI — amplitude modulated by shaft (1× and 2×)
    impulse_period = int(fs / BPFI)
    modulation     = 1 + 0.5 * np.sin(2 * np.pi * 33 * t)
    for k in range(0, n, impulse_period):
        width = max(1, int(0.0003 * fs))   # 0.3 ms pulse
        end   = min(k + width, n)
        amp   = severity * 0.4 * modulation[k] * rng.uniform(0.7, 1.3)
        sig[k:end] += amp * np.hanning(end - k)

    # Growing carrier at BPFI and harmonics
    sig += severity * 0.05 * np.sin(2 * np.pi * BPFI * t)
    sig += severity * 0.02 * np.sin(2 * np.pi * BPFI * 2 * t)

    return sig.astype(np.float32)


def _outer_race_signal(
    rng: np.random.Generator,
    n: int,
    fs: int,
    severity: float,
) -> np.ndarray:
    """
    Outer race defect: broadband energy increase + BPFO carrier.
    Less impulsive than inner race — more gradual RMS rise.
    """
    t   = np.linspace(0, n / fs, n, endpoint=False)
    sig = _healthy_signal(rng, n, fs, amplitude=0.02 + 0.08 * severity)

    if severity < 0.05:
        return sig

    # BPFO carrier and harmonics
    sig += severity * 0.06 * np.sin(2 * np.pi * BPFO * t)
    sig += severity * 0.03 * np.sin(2 * np.pi * BPFO * 2 * t)
    sig += severity * 0.01 * np.sin(2 * np.pi * BPFO * 3 * t)

    # Broadband energy growth
    sig += rng.normal(0, 0.03 * severity, n)

    return sig.astype(np.float32)


def _severity_curve(
    idx: int,
    total: int,
    onset_frac: float = 0.25,
    acceleration: float = 3.0,
) -> float:
    """
    Sigmoid-like severity curve: healthy → gradual degradation → rapid failure.
    Returns 0.0 at start, 1.0 at end.
    """
    if idx / total < onset_frac:
        return 0.0
    progress = (idx / total - onset_frac) / (1 - onset_frac)
    return float(np.clip(progress ** acceleration, 0.0, 1.0))


def generate_snapshot(
    rng: np.random.Generator,
    idx: int,
    total: int,
) -> np.ndarray:
    """
    Generate one snapshot: 4 bearing signals stacked as (N_SAMPLES, 4).
    """
    sev_b1 = _severity_curve(idx, total, onset_frac=0.30, acceleration=2.5)
    sev_b2 = _severity_curve(idx, total, onset_frac=0.40, acceleration=3.5)

    cols = [
        _inner_race_signal(rng, N_SAMPLES, FS, sev_b1),   # Bearing 1 — BPFI failure
        _outer_race_signal(rng, N_SAMPLES, FS, sev_b2),   # Bearing 2 — BPFO failure
        _healthy_signal(rng, N_SAMPLES, FS, amplitude=0.020),  # Bearing 3 — healthy
        _healthy_signal(rng, N_SAMPLES, FS, amplitude=0.018),  # Bearing 4 — healthy
    ]
    return np.column_stack(cols).astype(np.float32)


def generate_dataset(
    output_dir: Path,
    n_snapshots: int = N_SNAPSHOTS,
    seed: int = 42,
    log_every: int = 100,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rng       = np.random.default_rng(seed)
    start_ts  = datetime(2004, 2, 12, 10, 32, 39)
    interval  = timedelta(minutes=10)

    logger.info(
        "Generating %d synthetic IMS snapshots in %s …", n_snapshots, output_dir
    )

    for i in range(n_snapshots):
        ts       = start_ts + i * interval
        filename = ts.strftime("%Y.%m.%d.%H.%M.%S")
        path     = output_dir / filename

        data = generate_snapshot(rng, i, n_snapshots)
        np.savetxt(path, data, fmt="%.6f", delimiter="\t")

        if (i + 1) % log_every == 0 or i == n_snapshots - 1:
            pct = 100 * (i + 1) / n_snapshots
            logger.info("  %4d / %d  (%.0f%%)  last=%s", i + 1, n_snapshots, pct, filename)

    logger.info("Done. %d files written to %s", n_snapshots, output_dir)
    logger.info("Run training with: python -m src.models.train")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic NASA IMS bearing dataset")
    p.add_argument("--output-dir", default="data/raw/2nd_test",
                   help="Directory to write snapshot files (default: data/raw/2nd_test)")
    p.add_argument("--n-snapshots", type=int, default=N_SNAPSHOTS,
                   help=f"Number of snapshots to generate (default: {N_SNAPSHOTS})")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="Generate only 100 snapshots for a fast smoke test")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    n    = 100 if args.quick else args.n_snapshots
    generate_dataset(
        output_dir  = Path(args.output_dir),
        n_snapshots = n,
        seed        = args.seed,
    )
