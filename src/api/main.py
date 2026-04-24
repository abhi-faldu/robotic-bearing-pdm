"""
FastAPI inference service for the LSTM Autoencoder bearing anomaly detector.

Endpoints:
    GET  /health    — liveness check, confirms model is loaded
    POST /predict   — score a single feature window, return anomaly flag + error

Start locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Environment variables (set in .env or docker-compose.yml):
    MODEL_PATH      path to lstm_autoencoder.pt   (default: models/lstm_autoencoder.pt)
    THRESHOLD_PATH  path to threshold.json         (default: models/threshold.json)
    SCALER_PATH     path to scaler.npz             (default: models/scaler.npz)
    DEVICE          'cpu' or 'cuda'                (default: cpu)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.threshold import load_threshold

logger = logging.getLogger("uvicorn.error")

# ── Configuration (from env with sensible defaults) ───────────────────────────

MODEL_PATH     = Path(os.getenv("MODEL_PATH",     "models/lstm_autoencoder.pt"))
THRESHOLD_PATH = Path(os.getenv("THRESHOLD_PATH", "models/threshold.json"))
SCALER_PATH    = Path(os.getenv("SCALER_PATH",    "models/scaler.npz"))
DEVICE         = os.getenv("DEVICE", "cpu")


# ── Application state (populated at startup) ──────────────────────────────────

class _State:
    model: LSTMAutoencoder | None = None
    threshold: float = 0.0
    mu_s: np.ndarray | None = None
    sigma_s: np.ndarray | None = None
    seq_len: int = 50
    n_features: int = 0


state = _State()


# ── Lifespan: load model once at startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, threshold, and scaler into memory on startup."""
    logger.info("Loading model from %s …", MODEL_PATH)

    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not THRESHOLD_PATH.exists():
        logger.error("Threshold file not found: %s", THRESHOLD_PATH)
        raise RuntimeError(f"Threshold file not found: {THRESHOLD_PATH}")

    # Load scaler
    scaler          = np.load(SCALER_PATH)
    state.mu_s      = scaler["mu"].astype(np.float32)
    state.sigma_s   = scaler["sigma"].astype(np.float32)
    state.n_features = len(state.mu_s)

    # Infer seq_len from the saved state dict (first LSTM weight shape)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Build model — we derive dimensions from the checkpoint
    # encoder.lstm.weight_ih_l0 shape: (4*hidden, n_features)
    hidden_dim = checkpoint["encoder.lstm.weight_hh_l0"].shape[1]
    latent_dim = checkpoint["encoder.fc.weight"].shape[0]
    n_layers   = sum(1 for k in checkpoint if k.startswith("encoder.lstm.weight_hh"))

    # seq_len is stored in decoder.seq_len (not a weight, so read from architecture)
    # We default to 50 — can be overridden via env if needed
    state.seq_len = int(os.getenv("SEQ_LEN", "50"))

    state.model = LSTMAutoencoder(
        n_features = state.n_features,
        seq_len    = state.seq_len,
        hidden_dim = hidden_dim,
        latent_dim = latent_dim,
        n_layers   = n_layers,
    )
    state.model.load_state_dict(checkpoint)
    state.model.to(DEVICE)
    state.model.eval()

    _, _, state.threshold = load_threshold(THRESHOLD_PATH)
    logger.info(
        "Model ready — n_features=%d  seq_len=%d  threshold=%.6f",
        state.n_features, state.seq_len, state.threshold,
    )

    yield  # application runs here

    # Teardown (nothing needed for a read-only model)
    logger.info("Shutting down inference service.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Bearing PDM Inference API",
    description = "LSTM Autoencoder anomaly detection for industrial robotic arm bearings.",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health() -> HealthResponse:
    """Liveness check — returns model load status and current threshold."""
    return HealthResponse(
        status       = "ok",
        model_loaded = state.model is not None,
        threshold    = state.threshold,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Score a single feature window.

    The client sends a pre-normalised window of shape [seq_len, n_features].
    The API returns the reconstruction error, threshold, and anomaly flag.

    Raises:
        422  — if the window shape does not match the model's expected input.
        503  — if the model is not loaded.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    window = np.array(request.window, dtype=np.float32)  # (seq_len, n_features)

    # Validate shape
    if window.shape[0] != state.seq_len:
        raise HTTPException(
            status_code=422,
            detail=f"Expected seq_len={state.seq_len} rows, got {window.shape[0]}.",
        )
    if window.shape[1] != state.n_features:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected n_features={state.n_features} columns, "
                f"got {window.shape[1]}."
            ),
        )

    # Inference
    tensor = torch.from_numpy(window).unsqueeze(0).to(DEVICE)  # (1, seq_len, n_features)
    with torch.no_grad():
        error = float(state.model.reconstruction_error(tensor).item())

    is_anomaly    = error > state.threshold
    anomaly_score = error / (state.threshold + 1e-9)

    return PredictResponse(
        reconstruction_error = error,
        threshold            = state.threshold,
        is_anomaly           = is_anomaly,
        anomaly_score        = round(anomaly_score, 4),
    )
