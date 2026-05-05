"""
Training script for the LSTM Autoencoder.

Usage:
    python -m src.models.train [--data-dir data/raw] [--epochs 50] [--device cpu]

Outputs (written to models/):
    lstm_autoencoder.pt   — PyTorch state dict
    threshold.json        — anomaly threshold parameters (μ, σ, threshold)
    scaler.npz            — normalisation statistics (mu, sigma)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.features import (
    add_rolling_features,
    apply_scaler,
    extract_all_features,
    fit_scaler,
)
from src.data.loader import create_windows, load_snapshots, train_test_split_temporal
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.threshold import collect_errors, compute_threshold, save_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
SEQ_LEN   = 50
STEP      = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM Autoencoder for bearing PDM")
    p.add_argument("--data-dir",   default="data/raw",   help="Root of raw data folder")
    p.add_argument("--dataset",    type=int, default=2,  help="IMS dataset number (1-3)")
    p.add_argument("--limit",      type=int, default=None, help="Load only first N snapshots")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--n-layers",   type=int, default=2)
    p.add_argument("--train-frac", type=float, default=0.20,
                   help="Fraction of data treated as healthy for training")
    p.add_argument("--k-sigma",    type=float, default=3.0,
                   help="Threshold = μ + k·σ")
    p.add_argument("--device",     default="cpu", help="'cpu' or 'cuda'")
    return p.parse_args(argv)


def build_feature_matrix(args: argparse.Namespace):
    signals, timestamps, bearing_names = load_snapshots(
        args.data_dir, dataset=args.dataset, limit=args.limit
    )
    df = extract_all_features(signals, timestamps, bearing_names)
    df = add_rolling_features(df)
    return df.values.astype(np.float32), list(df.index), list(df.columns)


def train(args: argparse.Namespace) -> None:
    if not (0 < args.train_frac < 1):
        raise SystemExit(f"--train-frac must be in (0, 1), got {args.train_frac}")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    logger.info("Device: %s", device)

    # ── Data preparation ──────────────────────────────────────────────────────
    logger.info("Loading and extracting features …")
    X, timestamps, feature_names = build_feature_matrix(args)

    X_train, X_test, ts_train, ts_test = train_test_split_temporal(
        X, timestamps, train_frac=args.train_frac
    )

    if len(X_train) < SEQ_LEN:
        raise SystemExit(
            f"Train split has only {len(X_train)} rows but SEQ_LEN={SEQ_LEN}. "
            "Increase --train-frac or use more data."
        )

    mu, sigma = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, mu, sigma)
    X_test_scaled  = apply_scaler(X_test,  mu, sigma)

    train_windows = create_windows(X_train_scaled, seq_len=SEQ_LEN, step=STEP)
    if len(train_windows) == 0:
        raise SystemExit(
            f"No training windows formed from {len(X_train_scaled)} rows "
            f"with SEQ_LEN={SEQ_LEN}, STEP={STEP}."
        )
    logger.info("Training windows: %s", train_windows.shape)

    train_tensor  = torch.from_numpy(train_windows)
    train_dataset = TensorDataset(train_tensor)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    n_features = X.shape[1]
    model = LSTMAutoencoder(
        n_features=n_features,
        seq_len=SEQ_LEN,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
    ).to(device)

    logger.info(
        "Model: %d parameters", sum(p.numel() for p in model.parameters())
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_DIR / "lstm_autoencoder.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        epoch_loss /= len(train_windows)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d  loss=%.6f", epoch, args.epochs, epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_path)

    logger.info("Best training loss: %.6f  →  saved to %s", best_loss, best_path)

    # ── Threshold calibration ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device))
    train_errors = collect_errors(model, train_windows, device=device)
    mu_t, sigma_t, threshold = compute_threshold(train_errors, k=args.k_sigma)

    save_threshold(mu_t, sigma_t, threshold, MODEL_DIR / "threshold.json")

    # ── Scaler persistence ────────────────────────────────────────────────────
    np.savez(MODEL_DIR / "scaler.npz", mu=mu, sigma=sigma)
    logger.info("Scaler saved to %s", MODEL_DIR / "scaler.npz")

    # ── Model config persistence ──────────────────────────────────────────────
    model_config = {
        "n_features": n_features,
        "seq_len":    SEQ_LEN,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "n_layers":   args.n_layers,
    }
    config_path = MODEL_DIR / "model_config.json"
    config_path.write_text(json.dumps(model_config, indent=2))
    logger.info("Model config saved to %s", config_path)

    # ── Quick eval on test set ────────────────────────────────────────────────
    if len(X_test_scaled) >= SEQ_LEN:
        test_windows = create_windows(X_test_scaled, seq_len=SEQ_LEN, step=10)
        test_errors  = collect_errors(model, test_windows, device=device)
        anomalies    = (test_errors > threshold).sum()
        logger.info(
            "Test set: %d windows, %d anomalies (%.1f%%)  threshold=%.6f",
            len(test_windows), anomalies,
            100 * anomalies / len(test_windows),
            threshold,
        )

    logger.info("Done.")


if __name__ == "__main__":
    train(parse_args(sys.argv[1:]))
