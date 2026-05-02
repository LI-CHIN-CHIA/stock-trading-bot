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

        X = X_row[self._feature_names] if self._feature_names else X_row
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
        prefix = self.model_type[:4]   # lgbm / xgbo / catb / rand / ense
        path = self.model_dir / f"{prefix}_{version}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "models": [(obj,) for obj, _ in self._models],
                "feature_names": self._feature_names,
                "version": version,
                "model_type": self.model_type,
            }, f)
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
        self.model_type = saved_type
        self._feature_names = data["feature_names"]
        self._version       = data.get("version", "unknown")

        # Rebuild predict_fn for each saved model object
        # Support both old format {"booster": ...} and new format {"models": [...]}
        self._models = []
        import lightgbm as lgb
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

        logger.info(f"Model loaded: {path} (type={self.model_type}, version={self._version})")
        return True

    def is_trained(self) -> bool:
        if self._models:
            return True
        return bool(list(self.model_dir.glob("*.pkl")))

    @property
    def version(self) -> str:
        return self._version
