"""
Trading model wrapper — supports LightGBM, XGBoost, CatBoost, Random Forest, Ensemble.
Switch via MODEL_TYPE env var: lightgbm | xgboost | catboost | random_forest | ensemble
"""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

import os as _os
MODEL_DIR = Path(_os.getenv("DATA_DIR", Path(__file__).parent)) / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TYPE = os.getenv("MODEL_TYPE", "lightgbm").lower()

LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
LABEL_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}
BUY_CLASS_IDX  = 2
SELL_CLASS_IDX = 0
BUY_THRESHOLD  = 0.45


# ── Individual model builders ─────────────────────────────────────────────────

def _build_lgbm(X_train, y_train, X_val, y_val, sample_weight):
    import lightgbm as lgb
    train_set = lgb.Dataset(X_train, label=y_train,
                            feature_name=list(X_train.columns),
                            weight=sample_weight)
    val_set   = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "multiclass", "num_class": 3,
        "num_leaves": 63, "max_depth": 6, "learning_rate": 0.05,
        "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "verbose": -1, "seed": 42,
    }
    booster = lgb.train(
        params, train_set, num_boost_round=400,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    def predict_fn(X):
        return booster.predict(X)
    return booster, predict_fn


def _build_xgb(X_train, y_train, X_val, y_val, sample_weight):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", early_stopping_rounds=50,
        verbosity=0, random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight,
            eval_set=[(X_val, y_val)], verbose=False)
    def predict_fn(X):
        return clf.predict_proba(X)
    return clf, predict_fn


def _build_catboost(X_train, y_train, X_val, y_val, sample_weight):
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.05,
        loss_function="MultiClass", classes_count=3,
        early_stopping_rounds=50, verbose=False,
        random_seed=42,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight,
            eval_set=(X_val, y_val))
    def predict_fn(X):
        return clf.predict_proba(X)
    return clf, predict_fn


def _build_random_forest(X_train, y_train, X_val, y_val, sample_weight):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    def predict_fn(X):
        return clf.predict_proba(X)
    return clf, predict_fn


def _build_double_ensemble(X_train, y_train, X_val, y_val, sample_weight, n_models=5):
    """
    DoubleEnsemble: N 個 LightGBM，各用 80% 隨機特徵子集，
    並以預測難易度動態調整樣本權重（難例加權）。
    受 Microsoft Qlib DoubleEnsemble 啟發。
    """
    import lightgbm as lgb
    n_features  = X_train.shape[1]
    keep        = max(int(n_features * 0.8), 10)
    all_boosters: list = []
    all_masks:   list  = []
    weights = sample_weight.copy()

    for i in range(n_models):
        rng      = np.random.RandomState(42 + i * 7)
        feat_idx = sorted(rng.choice(n_features, size=keep, replace=False).tolist())
        all_masks.append(feat_idx)

        X_tr = X_train.iloc[:, feat_idx]
        X_v  = X_val.iloc[:, feat_idx]
        train_set = lgb.Dataset(X_tr, label=y_train, weight=weights)
        val_set   = lgb.Dataset(X_v,  label=y_val,  reference=train_set)

        params = {
            "objective": "multiclass", "num_class": 3,
            "num_leaves": 63, "max_depth": 6, "learning_rate": 0.05,
            "min_child_samples": 20, "subsample": 0.8,
            "colsample_bytree": 0.8, "verbose": -1, "seed": 42 + i,
        }
        booster = lgb.train(
            params, train_set, num_boost_round=400,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        all_boosters.append(booster)

        # 難例重加權：最大預測機率低 → 樣本難 → 加大權重
        if i < n_models - 1:
            proba      = booster.predict(X_tr)
            confidence = np.max(proba, axis=1)
            difficulty = 1.0 - confidence
            weights    = sample_weight * (0.5 + difficulty)
            weights    = weights / weights.mean()

    def predict_fn(X, boosters=all_boosters, masks=all_masks):
        probas = [b.predict(X.iloc[:, m]) for b, m in zip(boosters, masks)]
        return np.mean(probas, axis=0)

    return all_boosters, predict_fn, all_masks


BUILDERS = {
    "lightgbm":    _build_lgbm,
    "xgboost":     _build_xgb,
    "catboost":    _build_catboost,
    "random_forest": _build_random_forest,
}


# ── TradingModel ──────────────────────────────────────────────────────────────

class TradingModel:
    """
    Multi-backend trading model.
    MODEL_TYPE env var controls which backend is used:
      lightgbm | xgboost | catboost | random_forest | ensemble
    ensemble = soft-vote average of LightGBM + XGBoost + CatBoost.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model_type = MODEL_TYPE
        self._models: list = []          # list of (model_obj, predict_fn)
        self._feature_names: list[str] = []
        self._version: str = ""
        self._de_all_boosters: list = []  # double_ensemble: all N boosters
        self._de_masks: list = []         # double_ensemble: per-booster feature masks

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        common = X.index.intersection(y.index)
        X, y = X.loc[common], y.loc[common]
        if len(X) < 200:
            raise ValueError(f"Insufficient data: {len(X)} rows")

        y_mapped = y.map(LABEL_MAP).fillna(1).astype(int)
        split    = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y_mapped.iloc[:split], y_mapped.iloc[split:]

        self._feature_names = list(X.columns)

        class_counts  = np.bincount(y_train, minlength=3)
        total         = class_counts.sum()
        sample_weight = np.array([total / (3 * max(class_counts[c], 1)) for c in y_train])

        logger.info(f"訓練模型類型: {self.model_type}")

        if self.model_type == "ensemble":
            self._models = []
            for name in ["lightgbm", "xgboost", "catboost"]:
                logger.info(f"  訓練子模型: {name}")
                builder = BUILDERS[name]
                obj, fn = builder(X_train, y_train, X_val, y_val, sample_weight)
                self._models.append((obj, fn))
        elif self.model_type == "double_ensemble":
            logger.info("  訓練 DoubleEnsemble (5× LightGBM + 難例加權)...")
            all_boosters, predict_fn, all_masks = _build_double_ensemble(
                X_train, y_train, X_val, y_val, sample_weight
            )
            self._de_all_boosters = all_boosters
            self._de_masks = all_masks
            self._models = [(all_boosters[0], predict_fn)]
        else:
            builder = BUILDERS.get(self.model_type, _build_lgbm)
            obj, fn = builder(X_train, y_train, X_val, y_val, sample_weight)
            self._models = [(obj, fn)]

        # Evaluate on val set
        val_proba = self._predict_proba(X_val)
        val_pred  = np.argmax(val_proba, axis=1)
        f1     = f1_score(y_val, val_pred, average="macro", zero_division=0)
        buy_f1 = f1_score(y_val, val_pred, labels=[BUY_CLASS_IDX], average="macro", zero_division=0)

        logger.info(f"Training done. Val macro F1={f1:.3f}, Buy F1={buy_f1:.3f}")
        return {
            "val_samples": len(X_val),
            "train_samples": len(X_train),
            "val_macro_f1": round(float(f1), 4),
            "val_buy_f1": round(float(buy_f1), 4),
            "class_distribution": {LABEL_NAMES[i]: int(class_counts[i]) for i in range(3)},
            "model_type": self.model_type,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def _predict_proba(self, X) -> np.ndarray:
        """Average probabilities across all sub-models."""
        if not self._models:
            return np.array([[0.0, 1.0, 0.0]] * len(X))
        probas = [fn(X) for _, fn in self._models]
        return np.mean(probas, axis=0)

    def predict(self, X_row: pd.DataFrame) -> dict:
        if not self._models:
            return {"class": 1, "signal": "HOLD", "buy_proba": 0.0,
                    "sell_proba": 0.0, "hold_proba": 1.0, "trained": False}

        if self._feature_names:
            # 補齊 market_cache 可能缺少的新特徵欄，填 0 而非拋 KeyError
            missing = [c for c in self._feature_names if c not in X_row.columns]
            if missing:
                for c in missing:
                    X_row = X_row.copy()
                    X_row[c] = 0.0
            X = X_row[self._feature_names]
        else:
            X = X_row
        proba = self._predict_proba(X)[0]

        buy_proba  = float(proba[BUY_CLASS_IDX])
        sell_proba = float(proba[SELL_CLASS_IDX])

        if buy_proba >= BUY_THRESHOLD:
            signal = "BUY"
        elif sell_proba >= BUY_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "class": int(np.argmax(proba)),
            "signal": signal,
            "buy_proba":  round(buy_proba, 4),
            "sell_proba": round(sell_proba, 4),
            "hold_proba": round(float(proba[1]), 4),
            "trained": True,
            "model_type": self.model_type,
        }

    def feature_importance(self) -> pd.Series:
        """Return feature importances (averaged for ensemble)."""
        if not self._models:
            return pd.Series(dtype=float)
        all_imp = []
        for obj, _ in self._models:
            try:
                if hasattr(obj, "feature_importance"):        # LightGBM
                    imp = obj.feature_importance(importance_type="gain")
                    all_imp.append(pd.Series(imp, index=self._feature_names))
                elif hasattr(obj, "feature_importances_"):    # sklearn / XGB / CatBoost
                    imp = obj.feature_importances_
                    all_imp.append(pd.Series(imp, index=self._feature_names))
            except Exception:
                pass
        if not all_imp:
            return pd.Series(dtype=float)
        return pd.concat(all_imp, axis=1).mean(axis=1).sort_values(ascending=False)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, version: str = None) -> Path:
        if not self._models:
            raise RuntimeError("No model to save")
        version = version or datetime.now().strftime("%Y%m%d_%H%M")
        self._version = version
        prefix = self.model_type[:4]   # lgbm / xgbo / catb / rand / ense / doub
        path = self.model_dir / f"{prefix}_{version}.pkl"
        payload: dict = {
            "feature_names": self._feature_names,
            "version": version,
            "model_type": self.model_type,
        }
        if self.model_type == "double_ensemble" and self._de_all_boosters:
            payload["de_boosters"] = self._de_all_boosters
            payload["de_masks"]    = self._de_masks
        else:
            payload["models"] = [(obj,) for obj, _ in self._models]
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Model saved: {path}")
        return path

    def load(self, version: str = "latest") -> bool:
        if version == "latest":
            # 先找當前 model_type，找不到就找任何模型
            files = sorted(self.model_dir.glob(f"{self.model_type[:4]}_*.pkl"))
            if not files:
                files = sorted(self.model_dir.glob("*.pkl"))
            if not files:
                return False
            path = files[-1]
        else:
            path = self.model_dir / f"{self.model_type[:4]}_{version}.pkl"
            if not path.exists():
                # fallback: try lgbm prefix
                files = sorted(self.model_dir.glob(f"*_{version}.pkl"))
                if not files:
                    return False
                path = files[0]

        if not path.exists():
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        saved_type = data.get("model_type", "lightgbm")
        # Keep configured model_type (from MODEL_TYPE env) for future training.
        # saved_type is used only for reconstructing the loaded model's predict_fn.
        self._feature_names = data["feature_names"]
        self._version       = data.get("version", "unknown")

        # Rebuild predict_fn for each saved model object
        self._models = []
        import lightgbm as lgb

        if saved_type == "double_ensemble" and "de_boosters" in data:
            all_boosters = data["de_boosters"]
            all_masks    = data["de_masks"]
            self._de_all_boosters = all_boosters
            self._de_masks        = all_masks
            def _de_predict(X, _b=all_boosters, _m=all_masks):
                probas = [b.predict(X.iloc[:, m]) for b, m in zip(_b, _m)]
                return np.mean(probas, axis=0)
            self._models = [(all_boosters[0], _de_predict)]
        else:
            # Support both old format {"booster": ...} and new format {"models": [...]}
            if "models" in data:
                raw_list = data["models"]
            elif "booster" in data:
                raw_list = [(data["booster"],)]
            else:
                raw_list = []

            for (obj,) in raw_list:
                if isinstance(obj, lgb.Booster):
                    fn = obj.predict
                elif hasattr(obj, "predict_proba"):
                    fn = obj.predict_proba
                else:
                    fn = lambda X, o=obj: o.predict_proba(X)
                self._models.append((obj, fn))

        logger.info(f"Model loaded: {path} (saved_type={saved_type}, active_type={self.model_type}, version={self._version})")
        return True

    def is_trained(self) -> bool:
        if self._models:
            return True
        return bool(list(self.model_dir.glob("*.pkl")))

    @property
    def version(self) -> str:
        return self._version
