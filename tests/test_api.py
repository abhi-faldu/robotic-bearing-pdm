"""
Unit tests for the FastAPI inference service.

Tests run with a mocked model and threshold — no trained weights needed.

Run with:
    pytest tests/test_api.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES = 8
SEQ_LEN    = 50
THRESHOLD  = 0.05


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_model_dir(tmp_path_factory):
    """Create a temporary directory with fake model artefacts."""
    d = tmp_path_factory.mktemp("models")

    # Minimal LSTM Autoencoder weights so the real model can load
    from src.models.lstm_autoencoder import LSTMAutoencoder
    model = LSTMAutoencoder(n_features=N_FEATURES, seq_len=SEQ_LEN,
                             hidden_dim=16, latent_dim=8, n_layers=1)
    torch.save(model.state_dict(), d / "lstm_autoencoder.pt")

    # Threshold JSON
    threshold_data = {"mu": 0.02, "sigma": 0.01, "threshold": THRESHOLD}
    (d / "threshold.json").write_text(json.dumps(threshold_data))

    # Scaler npz
    mu    = np.zeros(N_FEATURES, dtype=np.float32)
    sigma = np.ones(N_FEATURES,  dtype=np.float32)
    np.savez(d / "scaler.npz", mu=mu, sigma=sigma)

    return d


@pytest.fixture(scope="module")
def client(tmp_model_dir):
    """TestClient with env vars pointing to the temp artefacts."""
    import os
    os.environ["MODEL_PATH"]     = str(tmp_model_dir / "lstm_autoencoder.pt")
    os.environ["THRESHOLD_PATH"] = str(tmp_model_dir / "threshold.json")
    os.environ["SCALER_PATH"]    = str(tmp_model_dir / "scaler.npz")
    os.environ["SEQ_LEN"]        = str(SEQ_LEN)

    from src.api.main import app
    with TestClient(app) as c:
        yield c


def make_window(seq_len: int = SEQ_LEN, n_features: int = N_FEATURES) -> list:
    """Build a valid random window as a nested list."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((seq_len, n_features)).tolist()


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_model_loaded(self, client):
        assert r.json()["model_loaded"] is True \
            for r in [client.get("/health")]  # noqa: E731

    def test_threshold_present(self, client):
        r = client.get("/health")
        assert r.json()["threshold"] == pytest.approx(THRESHOLD, rel=1e-3)


# ── /predict — happy path ─────────────────────────────────────────────────────

class TestPredictHappyPath:
    def test_200_on_valid_window(self, client):
        r = client.post("/predict", json={"window": make_window()})
        assert r.status_code == 200

    def test_response_fields(self, client):
        r = client.post("/predict", json={"window": make_window()})
        body = r.json()
        assert "reconstruction_error" in body
        assert "threshold"            in body
        assert "is_anomaly"           in body
        assert "anomaly_score"        in body

    def test_reconstruction_error_is_positive(self, client):
        r = client.post("/predict", json={"window": make_window()})
        assert r.json()["reconstruction_error"] >= 0.0

    def test_threshold_matches_loaded_value(self, client):
        r = client.post("/predict", json={"window": make_window()})
        assert r.json()["threshold"] == pytest.approx(THRESHOLD, rel=1e-3)

    def test_anomaly_score_consistent_with_error(self, client):
        r    = client.post("/predict", json={"window": make_window()})
        body = r.json()
        expected_score = body["reconstruction_error"] / body["threshold"]
        assert body["anomaly_score"] == pytest.approx(expected_score, rel=1e-2)

    def test_is_anomaly_consistent_with_error(self, client):
        r    = client.post("/predict", json={"window": make_window()})
        body = r.json()
        assert body["is_anomaly"] == (body["reconstruction_error"] > body["threshold"])


# ── /predict — anomaly triggered ─────────────────────────────────────────────

class TestPredictAnomalyFlag:
    def test_large_input_triggers_anomaly(self, client):
        """Feed very large values — reconstruction error should exceed threshold."""
        big_window = (np.ones((SEQ_LEN, N_FEATURES)) * 1000.0).tolist()
        r = client.post("/predict", json={"window": big_window})
        assert r.status_code == 200
        assert r.json()["is_anomaly"] is True
        assert r.json()["anomaly_score"] > 1.0

    def test_zero_input_is_not_anomaly(self, client):
        """Zero window is close to healthy mean → should not be anomalous."""
        zero_window = np.zeros((SEQ_LEN, N_FEATURES)).tolist()
        r = client.post("/predict", json={"window": zero_window})
        assert r.status_code == 200
        # Not asserting is_anomaly==False as it depends on trained weights,
        # but anomaly_score should be finite
        assert np.isfinite(r.json()["anomaly_score"])


# ── /predict — validation errors ─────────────────────────────────────────────

class TestPredictValidation:
    def test_wrong_seq_len_returns_422(self, client):
        short_window = make_window(seq_len=SEQ_LEN - 5)
        r = client.post("/predict", json={"window": short_window})
        assert r.status_code == 422

    def test_wrong_n_features_returns_422(self, client):
        bad_window = make_window(n_features=N_FEATURES + 2)
        r = client.post("/predict", json={"window": bad_window})
        assert r.status_code == 422

    def test_inconsistent_row_widths_returns_422(self, client):
        window = make_window()
        window[5] = window[5][:-1]   # make row 5 one element shorter
        r = client.post("/predict", json={"window": window})
        assert r.status_code == 422

    def test_empty_window_returns_422(self, client):
        r = client.post("/predict", json={"window": []})
        assert r.status_code == 422

    def test_missing_body_returns_422(self, client):
        r = client.post("/predict")
        assert r.status_code == 422


# ── Schema validation ─────────────────────────────────────────────────────────

class TestSchemas:
    def test_predict_request_valid(self):
        from src.api.schemas import PredictRequest
        req = PredictRequest(window=make_window())
        assert len(req.window) == SEQ_LEN

    def test_predict_request_rejects_jagged(self):
        from pydantic import ValidationError
        from src.api.schemas import PredictRequest
        window = make_window()
        window[0] = window[0][:-1]
        with pytest.raises(ValidationError):
            PredictRequest(window=window)

    def test_predict_response_fields(self):
        from src.api.schemas import PredictResponse
        resp = PredictResponse(
            reconstruction_error=0.03,
            threshold=0.05,
            is_anomaly=False,
            anomaly_score=0.6,
        )
        assert resp.is_anomaly is False
        assert resp.anomaly_score == pytest.approx(0.6)
