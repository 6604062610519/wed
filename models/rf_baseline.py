"""
models/rf_baseline.py — Random Forest Baseline

Pixel-wise classification ใช้ 27 features
- Fast to train (sklearn)
- ไม่ capture spatial context (แต่เป็น strong baseline)
- Feature importance interpretability
"""

import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dataset import load_pixel_data
from training.metrics import compute_metrics, find_best_threshold, print_metrics


class RFWildfireModel:
    """
    Random Forest สำหรับ wildfire prediction (pixel-level)

    Parameters
    ----------
    n_estimators : จำนวน trees
    max_depth    : ความลึก tree (None = no limit)
    class_weight : weight สำหรับ class imbalance
    model_type   : "rf" (Random Forest) หรือ "gb" (Gradient Boosting)
    """

    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: Optional[int] = 15,
                 min_samples_leaf: int = 10,
                 class_weight: str = "balanced",
                 model_type: str = "rf",
                 n_jobs: int = -1,
                 seed: int = 42):

        self.name       = f"{model_type.upper()}_Baseline"
        self.model_type = model_type
        self.threshold  = 0.5
        self.feature_names = None

        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                n_jobs=n_jobs,
                random_state=seed,
                oob_score=True,
                verbose=2,
            )
        elif model_type == "gb":
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth or 5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict:
        import time
        print(f"\n{'='*50}")
        print(f"  Training: {self.name}")
        print(f"  Train size: {len(X_train):,} pixels")
        print(f"  Fire rate: {y_train.mean()*100:.2f}%")
        print(f"{'='*50}")

        t0 = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - t0

        print(f"  ✅ Trained in {train_time:.1f}s")
        if hasattr(self.model, "oob_score_"):
            print(f"  OOB Score: {self.model.oob_score_:.4f}")

        if X_val is not None and y_val is not None:
            y_prob_val = self.model.predict_proba(X_val)[:, 1]
            self.threshold = find_best_threshold(y_val, y_prob_val)
            val_metrics = compute_metrics(y_val, y_prob_val,
                                          threshold=self.threshold, prefix="val_")
            print_metrics(val_metrics, f"Validation ({self.name})")
            return {
                "training_time":  train_time,
                "val_f1":         val_metrics["val_f1"],
                "best_threshold": self.threshold,
                "val_metrics":    val_metrics,
            }

        return {"training_time": train_time, "best_threshold": 0.5}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        thr = threshold or self.threshold
        return (self.predict_proba(X) >= thr).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray, prefix: str = "test") -> Dict:
        y_prob = self.predict_proba(X)
        metrics = compute_metrics(y, y_prob, threshold=self.threshold,
                                  prefix=f"{prefix}_")
        print_metrics(metrics, f"{prefix.upper()} ({self.name})")
        return metrics

    def feature_importance(self, feature_names=None, top_k: int = 15) -> Dict:
        if not hasattr(self.model, "feature_importances_"):
            return {}

        names  = feature_names or self.feature_names or \
                 [f"feat_{i}" for i in range(len(self.model.feature_importances_))]
        imps   = self.model.feature_importances_
        ranked = sorted(zip(names, imps), key=lambda x: -x[1])[:top_k]

        print(f"\n  Top {top_k} Feature Importances ({self.name}):")
        for name, imp in ranked:
            bar = "█" * int(imp * 100)
            print(f"    {name:<30s}: {imp:.4f}  {bar}")

        return dict(ranked)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold,
                     "name": self.name}, path)
        print(f"  💾 Saved: {path}")

    @classmethod
    def load(cls, path: str) -> "RFWildfireModel":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model     = data["model"]
        obj.threshold = data.get("threshold", 0.5)
        obj.name      = data.get("name", "RF_Baseline")
        return obj


def train_rf_baseline(data_dir: str,
                      train_months=(1,2,3,4,5,6,7),
                      val_months=(8,9),
                      test_months=(10,11,12),
                      year: int = 2023,
                      n_estimators: int = 200,
                      seed: int = 42) -> Dict:
    print("\n📊 Loading pixel data for RF...")
    X_train, y_train = load_pixel_data(data_dir, list(train_months), years=[year], seed=seed)
    X_val,   y_val   = load_pixel_data(data_dir, list(val_months),   years=[year], seed=seed)
    X_test,  y_test  = load_pixel_data(data_dir, list(test_months),  years=[year], seed=seed)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = RFWildfireModel(n_estimators=n_estimators, seed=seed)
    train_result = model.fit(X_train, y_train, X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test, prefix="test")

    return {"model": model, "train_result": train_result, "test_metrics": test_metrics}


if __name__ == "__main__":
    result = train_rf_baseline("data/processed", n_estimators=50)
    result["model"].feature_importance()
