"""
LSTM Autoencoder for unsupervised anomaly detection on bearing vibration features.

Architecture:
    Encoder: 2-layer LSTM  →  bottleneck (last hidden state, dim=32)
    Decoder: 2-layer LSTM  →  linear projection back to n_features

The model is trained to reconstruct sequences of healthy bearing features.
Anomaly score = mean squared reconstruction error per window.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int, n_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)   # h_n: (n_layers, batch, hidden_dim)
        return self.fc(h_n[-1])       # (batch, latent_dim)


class Decoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        latent_dim: int,
        hidden_dim: int,
        n_features: int,
        n_layers: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        h = self.fc(z).unsqueeze(1)               # (batch, 1, hidden_dim)
        h = h.repeat(1, self.seq_len, 1)           # (batch, seq_len, hidden_dim)
        out, _ = self.lstm(h)                      # (batch, seq_len, hidden_dim)
        return self.output_layer(out)              # (batch, seq_len, n_features)


class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM Autoencoder.

    Args:
        n_features  : number of input features per time step
        seq_len     : number of time steps per window (must match training windows)
        hidden_dim  : LSTM hidden units in encoder and decoder
        latent_dim  : bottleneck dimension (compressed representation)
        n_layers    : number of stacked LSTM layers in each of encoder / decoder
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 50,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_layers: int = 2,
    ):
        super().__init__()
        self.encoder = Encoder(n_features, hidden_dim, latent_dim, n_layers)
        self.decoder = Decoder(seq_len, latent_dim, hidden_dim, n_features, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error. Shape: (batch,)"""
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=(1, 2))
