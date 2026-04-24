"""
Pydantic request/response models for the FastAPI inference endpoint.

POST /predict
    Request  : PredictRequest  — a batch of pre-computed feature vectors
    Response : PredictResponse — reconstruction error + anomaly flag per sample
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    """
    A sliding window of feature vectors to score.

    The client sends a 2-D list of shape (seq_len, n_features) — one row per
    10-minute snapshot, already z-score normalised using the training scaler.

    Example (seq_len=3, n_features=2 for brevity):
        {
            "window": [
                [0.12, -0.34],
                [0.15, -0.31],
                [0.18, -0.28]
            ]
        }
    """

    window: List[List[float]] = Field(
        ...,
        description="2-D feature window of shape [seq_len, n_features], z-score normalised.",
        min_length=1,
    )

    @model_validator(mode="after")
    def check_consistent_width(self) -> "PredictRequest":
        """All rows must have the same number of features."""
        widths = {len(row) for row in self.window}
        if len(widths) > 1:
            raise ValueError(
                f"All rows in 'window' must have the same length. "
                f"Got widths: {widths}"
            )
        return self


class PredictResponse(BaseModel):
    """
    Anomaly detection result for a single window.

    Fields:
        reconstruction_error : MSE between the input window and its reconstruction.
        threshold            : The alert threshold (μ+3σ, loaded from threshold.json).
        is_anomaly           : True if reconstruction_error > threshold.
        anomaly_score        : Normalised score = error / threshold. Values > 1 → anomaly.
    """

    reconstruction_error: float = Field(
        ..., description="Mean squared reconstruction error for the input window."
    )
    threshold: float = Field(
        ..., description="Anomaly alert threshold (μ + 3σ from healthy training data)."
    )
    is_anomaly: bool = Field(
        ..., description="True if reconstruction_error exceeds the threshold."
    )
    anomaly_score: float = Field(
        ..., description="Normalised score = error / threshold. >1 means anomaly."
    )


class HealthResponse(BaseModel):
    """Response for the GET /health liveness check."""

    status: str = Field(default="ok")
    model_loaded: bool
    threshold: float
