"""
Day-trading model: CNN (pattern embedding) + LightGBM (regression).
Predicts max_gain_30min (%) after a bar.
buy_signal = predicted_gain > threshold (default 1.5%).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ai.day_cnn import DayCNN
from ai.day_features import compute_cnn_input, compute_tabular_features

logger = logging.getLogger(__name__)

import os as _os
DEFAULT_THRESHOLD = float(_os.getenv("DAY_MODEL_THRESHOLD", "0.008"))  # default 0.8%
_MODEL_DIR_ENV    = "DATA_DIR"


def _model_dir() -> Path:
    import os
    return Path(os.getenv(_MODEL_DIR_ENV, Path(__file__).parent.parent)) / "models"


class DayModel:
    """Wraps CNN + LightGBM for intraday buy-signal prediction."""

    def __init__(
        self,
        cnn: DayCNN,
        lgbm_booster,                 # lightgbm.Booster or sklearn-like
        feature_names: list[str],
        threshold: float = DEFAULT_THRESHOLD,
        trained_at: str = "",
    ):
        self.cnn           = cnn
        self.lgbm          = lgbm_booster
        self.feature_names = feature_names
        self.threshold     = threshold
        self.trained_at    = trained_at

    # ── Inference ────────────────────────────────────────────────────────────

    def _build_feature_vector(
        self, df: pd.DataFrame, analysis: dict | None
    ) -> np.ndarray | None:
        tab = compute_tabular_features(df, analysis)
        if not tab:
            return None
        cnn_in = compute_cnn_input(df)
        if cnn_in is None:
            cnn_embed = np.zeros(16, dtype=np.float32)
        else:
            cnn_embed = self.cnn.forward(cnn_in)

        tab_vec = np.array([tab.get(f, 0.0) for f in self.feature_names
                            if not f.startswith("cnn_")], dtype=np.float32)
        feat    = np.concatenate([tab_vec, cnn_embed])
        return feat

    def predict_gain(
        self, df: pd.DataFrame, analysis: dict | None = None
    ) -> float:
        """Return predicted max_gain_30min (fraction, e.g. 0.02 = 2%)."""
        feat = self._build_feature_vector(df, analysis)
        if feat is None:
            return 0.0
        try:
            val = float(self.lgbm.predict(feat.reshape(1, -1))[0])
            return val
        except Exception as e:
            logger.debug(f"[DayModel] predict error: {e}")
            return 0.0

    def is_buy_signal(
        self,
        df: pd.DataFrame,
        analysis: dict | None = None,
        threshold: float | None = None,
    ) -> tuple[bool, float]:
        """Return (go_or_no, predicted_gain)."""
        th   = threshold if threshold is not None else self.threshold
        gain = self.predict_gain(df, analysis)
        return gain >= th, gain

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cnn_weights":    self.cnn.w,
            "lgbm":           self.lgbm,
            "feature_names":  self.feature_names,
            "threshold":      self.threshold,
            "trained_at":     self.trained_at,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        # Also keep a symlink-like "latest" copy
        latest = path.parent / "day_model_latest.pkl"
        import shutil
        shutil.copy2(path, latest)
        logger.info(f"[DayModel] saved → {path}")

    @classmethod
    def load(cls, path: Path | str | None = None) -> Optional["DayModel"]:
        if path is None:
            path = _model_dir() / "day_model_latest.pkl"
        path = Path(path)
        if not path.exists():
            logger.info(f"[DayModel] no model found at {path}")
            return None
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            cnn = DayCNN(d["cnn_weights"])
            # 環境變數優先，其次用 pkl 內的值，最後用 DEFAULT_THRESHOLD
            pkl_thresh = d.get("threshold", DEFAULT_THRESHOLD)
            threshold  = float(_os.getenv("DAY_MODEL_THRESHOLD", str(pkl_thresh)))
            return cls(
                cnn           = cnn,
                lgbm_booster  = d["lgbm"],
                feature_names = d["feature_names"],
                threshold     = threshold,
                trained_at    = d.get("trained_at", ""),
            )
        except Exception as e:
            logger.warning(f"[DayModel] load failed: {e}")
            return None

    @classmethod
    def load_latest(cls) -> Optional["DayModel"]:
        return cls.load()
