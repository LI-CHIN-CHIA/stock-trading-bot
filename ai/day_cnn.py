"""
1D-CNN for candlestick pattern feature extraction.

Training  : requires PyTorch (host only, `venv/bin/python ai/day_backtest.py`)
Inference : pure numpy forward pass — no torch dependency in Docker
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Architecture constants (must match training)
_C_IN    = 5    # OHLCV
_C1      = 32
_C2      = 32
_EMBED   = 16
_K       = 3
_PADDING = 1
_WIN     = 20


# ── Numpy inference ──────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _conv1d(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    x : (C_in,  L)
    W : (C_out, C_in, K)
    b : (C_out,)
    returns (C_out, L)  [same-length with padding=1, kernel=3]
    """
    K = W.shape[2]
    x_pad = np.pad(x, ((0, 0), (1, 1)))           # (C_in, L+2)
    wins  = np.lib.stride_tricks.sliding_window_view(x_pad, K, axis=1)
    # wins: (C_in, L, K)
    out   = np.einsum("clk,ock->ol", wins, W) + b[:, None]
    return out


class DayCNN:
    """Numpy-based inference wrapper for the trained 1D-CNN."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (5, 20) → returns (16,) pattern embedding"""
        h = _relu(_conv1d(x.astype(np.float32), self.w["c1_W"], self.w["c1_b"]))
        h = _relu(_conv1d(h,                    self.w["c2_W"], self.w["c2_b"]))
        h = h.mean(axis=1)                                 # global avg pool → (32,)
        h = _relu(h @ self.w["fc1_W"].T + self.w["fc1_b"])  # (16,)
        return h

    def batch_forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 5, 20) → (N, 16)"""
        return np.array([self.forward(x) for x in X])

    def save(self, path: Path | str) -> None:
        np.savez_compressed(str(path), **self.w)
        logger.info(f"[DayCNN] weights saved → {path}")

    @classmethod
    def load(cls, path: Path | str) -> "DayCNN":
        data = np.load(str(path))
        weights = {k: data[k] for k in data.files}
        logger.info(f"[DayCNN] weights loaded ← {path}")
        return cls(weights)


# ── PyTorch training ─────────────────────────────────────────────────────────

class _TorchCNN:
    """Internal PyTorch model for training. Not imported at inference time."""

    def _build(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv1d(_C_IN, _C1, _K, padding=_PADDING), nn.ReLU(),   # 0,1
            nn.Conv1d(_C1, _C2, _K, padding=_PADDING),   nn.ReLU(),   # 2,3
            nn.AdaptiveAvgPool1d(1),                                   # 4
            nn.Flatten(),                                              # 5
            nn.Linear(_C2, _EMBED), nn.ReLU(),                        # 6,7  ← embedding
            nn.Linear(_EMBED, 1),                                      # 8    ← regression head
        )

    def train(
        self,
        X: np.ndarray,   # (N, 5, 20)
        y: np.ndarray,   # (N,) regression target
        epochs: int = 40,
        lr: float = 3e-4,
        batch_size: int = 256,
    ) -> DayCNN:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[DayCNN] training on {device}, N={len(X)}, epochs={epochs}")

        net = self._build().to(device)
        Xt  = torch.tensor(X, dtype=torch.float32)
        yt  = torch.tensor(y, dtype=torch.float32)
        ds  = TensorDataset(Xt, yt)
        dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt  = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for ep in range(1, epochs + 1):
            net.train()
            total = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = net(xb).squeeze(1)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            if ep % 10 == 0:
                logger.info(f"  epoch {ep:3d}  mse={total/len(dl):.6f}")

        # Export weights to numpy
        net.cpu().eval()
        sd = net.state_dict()
        weights = {
            "c1_W":  sd["0.weight"].numpy(),
            "c1_b":  sd["0.bias"].numpy(),
            "c2_W":  sd["2.weight"].numpy(),
            "c2_b":  sd["2.bias"].numpy(),
            "fc1_W": sd["6.weight"].numpy(),
            "fc1_b": sd["6.bias"].numpy(),
        }
        return DayCNN(weights)


def train_cnn(X: np.ndarray, y: np.ndarray, **kwargs) -> DayCNN:
    """
    Train the 1D-CNN and return a numpy-only DayCNN instance.
    Requires PyTorch (training only).
    """
    return _TorchCNN().train(X, y, **kwargs)
