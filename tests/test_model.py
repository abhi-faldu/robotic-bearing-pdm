"""
Unit tests for src/models/lstm_autoencoder.py and src/models/threshold.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.threshold import (
    collect_errors,
    compute_threshold,
    load_threshold,
    save_threshold,
)

N_FEATURES = 8
SEQ_LEN    = 50
BATCH      = 16


@pytest.fixture
def model():
    return LSTMAutoencoder(n_features=N_FEATURES, seq_len=SEQ_LEN, hidden_dim=32, latent_dim=16, n_layers=1)


@pytest.fixture
def batch_tensor():
    return torch.randn(BATCH, SEQ_LEN, N_FEATURES)


# ── LSTMAutoencoder ───────────────────────────────────────────────────────────

class TestLSTMAutoencoder:
    def test_output_shape(self, model, batch_tensor):
        out = model(batch_tensor)
        assert out.shape == (BATCH, SEQ_LEN, N_FEATURES)

    def test_reconstruction_error_shape(self, model, batch_tensor):
        errors = model.reconstruction_error(batch_tensor)
        assert errors.shape == (BATCH,)

    def test_reconstruction_error_non_negative(self, model, batch_tensor):
        errors = model.reconstruction_error(batch_tensor)
        assert (errors >= 0).all()

    def test_perfect_reconstruction_gives_zero_error(self):
        """If the model perfectly reconstructs input, MSE should be 0."""
        model = LSTMAutoencoder(N_FEATURES, SEQ_LEN, hidden_dim=32, latent_dim=16, n_layers=1)
        x = torch.zeros(BATCH, SEQ_LEN, N_FEATURES)
        # Override forward to return input unchanged
        model.forward = lambda inp: inp
        errors = model.reconstruction_error(x)
        assert errors.max().item() == pytest.approx(0.0, abs=1e-6)

    def test_gradients_flow(self, model, batch_tensor):
        batch_tensor.requires_grad_(False)
        out = model(batch_tensor)
        loss = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_different_batch_sizes(self, model):
        for bs in [1, 4, 32]:
            x = torch.randn(bs, SEQ_LEN, N_FEATURES)
            out = model(x)
            assert out.shape == (bs, SEQ_LEN, N_FEATURES)


# ── Threshold ─────────────────────────────────────────────────────────────────

class TestComputeThreshold:
    def test_returns_three_floats(self):
        errors = np.random.rand(100).astype(np.float32)
        result = compute_threshold(errors)
        assert len(result) == 3
        mu, sigma, threshold = result
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert isinstance(threshold, float)

    def test_threshold_above_mean(self):
        errors = np.ones(100, dtype=np.float32)
        mu, sigma, threshold = compute_threshold(errors, k=3.0)
        assert threshold >= mu

    def test_k_scaling(self):
        errors = np.random.rand(1000).astype(np.float32)
        _, sigma1, t1 = compute_threshold(errors, k=2.0)
        _, sigma2, t2 = compute_threshold(errors, k=4.0)
        assert t2 > t1

    def test_constant_array(self):
        errors = np.full(50, 0.005, dtype=np.float32)
        mu, sigma, threshold = compute_threshold(errors, k=3.0)
        assert mu == pytest.approx(0.005, abs=1e-6)
        assert sigma == pytest.approx(0.0, abs=1e-6)
        assert threshold == pytest.approx(0.005, abs=1e-6)


class TestCollectErrors:
    def test_shape(self, model):
        windows = np.random.randn(40, SEQ_LEN, N_FEATURES).astype(np.float32)
        errors = collect_errors(model, windows, batch_size=16)
        assert errors.shape == (40,)

    def test_non_negative(self, model):
        windows = np.random.randn(20, SEQ_LEN, N_FEATURES).astype(np.float32)
        errors = collect_errors(model, windows)
        assert (errors >= 0).all()

    def test_model_is_eval_after(self, model):
        windows = np.random.randn(10, SEQ_LEN, N_FEATURES).astype(np.float32)
        collect_errors(model, windows)
        assert not model.training


class TestSaveLoadThreshold:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "threshold.json"
            save_threshold(0.01, 0.002, 0.016, path)
            mu, sigma, threshold = load_threshold(path)
        assert mu       == pytest.approx(0.01,  abs=1e-9)
        assert sigma    == pytest.approx(0.002, abs=1e-9)
        assert threshold == pytest.approx(0.016, abs=1e-9)

    def test_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            save_threshold(1.0, 2.0, 3.0, path)
            with open(path) as f:
                data = json.load(f)
        assert set(data.keys()) == {"mu", "sigma", "threshold"}
