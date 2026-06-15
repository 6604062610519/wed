"""
training/tuner.py — Hyperparameter Tuning ด้วย Grid Search + Optuna

ทดสอบว่า optuna install อยู่ไหม ถ้าไม่มีใช้ RandomSearch แทน
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Callable, Any, List

# Optional: ใช้ Optuna ถ้ามี
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  ℹ️  optuna ไม่ได้ install — ใช้ Random Search แทน")


# ─────────────────────────────────────────────────────
# Search Spaces
# ─────────────────────────────────────────────────────

def cnn_search_space(trial=None, seed_cfg: dict = None) -> dict:
    """Search space สำหรับ CNN-Patch"""
    if trial is not None and HAS_OPTUNA:
        return {
            "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "base_filters": trial.suggest_categorical("base_filters", [16, 32, 64]),
            "n_levels":     trial.suggest_int("n_levels", 2, 4),
            "dropout":      trial.suggest_float("dropout", 0.0, 0.3),
            "loss":         trial.suggest_categorical("loss", ["focal", "combined"]),
            "focal_alpha":  trial.suggest_float("focal_alpha", 0.6, 0.9),
        }
    # Random fallback
    rng = np.random.default_rng()
    return {
        "lr":           float(10 ** rng.uniform(-4, -2)),
        "weight_decay": float(10 ** rng.uniform(-5, -3)),
        "base_filters": int(rng.choice([16, 32, 64])),
        "n_levels":     int(rng.integers(2, 5)),
        "dropout":      float(rng.uniform(0.0, 0.3)),
        "loss":         str(rng.choice(["focal", "combined"])),
        "focal_alpha":  float(rng.uniform(0.6, 0.9)),
    }


def unet_search_space(trial=None, **kwargs) -> dict:
    """Search space สำหรับ U-Net / ResU-Net"""
    if trial is not None and HAS_OPTUNA:
        return {
            "lr":           trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "base_filters": trial.suggest_categorical("base_filters", [16, 32, 64]),
            "n_levels":     trial.suggest_int("n_levels", 3, 4),
            "dropout":      trial.suggest_float("dropout", 0.0, 0.3),
            "loss":         trial.suggest_categorical("loss", ["focal", "combined", "dice"]),
            "focal_alpha":  trial.suggest_float("focal_alpha", 0.65, 0.90),
        }
    rng = np.random.default_rng()
    return {
        "lr":           float(10 ** rng.uniform(-4.3, -2.3)),
        "weight_decay": float(10 ** rng.uniform(-5, -3)),
        "base_filters": int(rng.choice([16, 32, 64])),
        "n_levels":     int(rng.integers(3, 5)),
        "dropout":      float(rng.uniform(0.0, 0.3)),
        "loss":         str(rng.choice(["focal", "combined", "dice"])),
        "focal_alpha":  float(rng.uniform(0.65, 0.90)),
    }


def convlstm_search_space(trial=None, **kwargs) -> dict:
    """Search space สำหรับ ConvLSTM"""
    if trial is not None and HAS_OPTUNA:
        return {
            "lr":              trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay":    trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64, 128]),
            "n_lstm_layers":   trial.suggest_int("n_lstm_layers", 1, 2),
            "dropout":         trial.suggest_float("dropout", 0.0, 0.3),
            "T":               trial.suggest_categorical("T", [2, 3]),
            "loss":            trial.suggest_categorical("loss", ["focal", "combined"]),
        }
    rng = np.random.default_rng()
    return {
        "lr":              float(10 ** rng.uniform(-4, -2)),
        "weight_decay":    float(10 ** rng.uniform(-5, -3)),
        "hidden_channels": int(rng.choice([32, 64, 128])),
        "n_lstm_layers":   int(rng.integers(1, 3)),
        "dropout":         float(rng.uniform(0.0, 0.3)),
        "T":               int(rng.choice([2, 3])),
        "loss":            str(rng.choice(["focal", "combined"])),
    }


SEARCH_SPACES = {
    "cnn":      cnn_search_space,
    "unet":     unet_search_space,
    "resunet":  unet_search_space,
    "convlstm": convlstm_search_space,
}


# ─────────────────────────────────────────────────────
# Generic tuner
# ─────────────────────────────────────────────────────

class HyperparameterTuner:
    """
    Hyperparameter tuner ที่ใช้ Optuna (ถ้ามี) หรือ Random Search

    Parameters
    ----------
    model_name     : ชื่อ model ("cnn", "unet", "resunet", "convlstm")
    objective_fn   : function(cfg) → val_f1 (float)
    n_trials       : จำนวน HP combinations ที่จะทดลอง
    results_dir    : path สำหรับบันทึก results
    """

    def __init__(self,
                 model_name: str,
                 objective_fn: Callable[[Dict], float],
                 n_trials: int = 10,
                 results_dir: str = "results"):
        self.model_name   = model_name
        self.objective_fn = objective_fn
        self.n_trials     = n_trials
        self.results_dir  = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.search_space_fn = SEARCH_SPACES.get(model_name, cnn_search_space)
        self.all_results: List[Dict] = []

    def run(self) -> Dict:
        """รัน hyperparameter search"""
        print(f"\n{'─'*55}")
        print(f"  HP Tuning: {self.model_name}  ({self.n_trials} trials)")
        print(f"  Method: {'Optuna (TPE)' if HAS_OPTUNA else 'Random Search'}")
        print(f"{'─'*55}")

        if HAS_OPTUNA:
            return self._optuna_search()
        else:
            return self._random_search()

    def _optuna_search(self) -> Dict:
        def objective(trial):
            cfg = self.search_space_fn(trial=trial)
            try:
                val_f1 = self.objective_fn(cfg)
                self.all_results.append({**cfg, "val_f1": val_f1})
                return val_f1
            except Exception as e:
                print(f"    ⚠️  Trial failed: {e}")
                return 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_cfg = study.best_params
        best_val = study.best_value
        print(f"\n  Best val F1: {best_val:.4f}")
        print(f"  Best config: {best_cfg}")
        self._save_results()
        return {"best_cfg": best_cfg, "best_val_f1": best_val,
                "all_results": self.all_results}

    def _random_search(self) -> Dict:
        best_val, best_cfg = -1, None

        for trial_idx in range(self.n_trials):
            cfg = self.search_space_fn(trial=None)
            print(f"\n  Trial {trial_idx+1}/{self.n_trials}: lr={cfg.get('lr',0):.2e} "
                  f"| filters={cfg.get('base_filters','?')} "
                  f"| loss={cfg.get('loss','?')}")

            try:
                val_f1 = self.objective_fn(cfg)
                self.all_results.append({**cfg, "val_f1": val_f1})
                print(f"    → val_f1={val_f1:.4f}")

                if val_f1 > best_val:
                    best_val, best_cfg = val_f1, cfg.copy()
                    print(f"    ✓ New best!")
            except Exception as e:
                print(f"    ⚠️  Failed: {e}")

        print(f"\n  Best val F1: {best_val:.4f}")
        print(f"  Best config: {best_cfg}")
        self._save_results()
        return {"best_cfg": best_cfg, "best_val_f1": best_val,
                "all_results": self.all_results}

    def _save_results(self):
        path = self.results_dir / f"tuning_{self.model_name}.json"
        with open(str(path), "w") as f:
            json.dump({
                "model":      self.model_name,
                "n_trials":   self.n_trials,
                "best":       max(self.all_results, key=lambda x: x.get("val_f1", 0))
                              if self.all_results else {},
                "all_trials": self.all_results,
            }, f, indent=2)
        print(f"  💾 Tuning results saved: {path}")


# ─────────────────────────────────────────────────────
# Default best configs (ใช้ถ้าไม่มีเวลา tune)
# ─────────────────────────────────────────────────────

DEFAULT_CONFIGS = {
    "cnn": {
        "lr": 3e-4, "weight_decay": 1e-4, "base_filters": 32,
        "n_levels": 3, "dropout": 0.1, "loss": "combined", "focal_alpha": 0.75,
    },
    "unet": {
        "lr": 1e-4, "weight_decay": 1e-4, "base_filters": 32,
        "n_levels": 4, "dropout": 0.1, "loss": "combined", "focal_alpha": 0.75,
    },
    "resunet": {
        "lr": 1e-4, "weight_decay": 1e-4, "base_filters": 32,
        "n_levels": 4, "dropout": 0.1, "loss": "combined", "focal_alpha": 0.75,
    },
    "convlstm": {
        "lr": 3e-4, "weight_decay": 1e-4, "hidden_channels": 64,
        "n_lstm_layers": 2, "dropout": 0.1, "T": 3, "loss": "combined",
    },
}
