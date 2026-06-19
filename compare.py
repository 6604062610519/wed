"""
compare.py — Fair Model Comparison Script

รัน wildfire prediction ด้วย 5 models แล้วเปรียบเทียบผลลัพธ์อย่างเป็นธรรม:
  1. Random Forest Baseline
  2. CNN-Patch
  3. U-Net
  4. ResU-Net
  5. ConvLSTM

Usage:
  python3 compare.py --mode quick            # prototype (5 epochs)
  python3 compare.py --mode full             # full (30 epochs + tuning)
  python3 compare.py --models rf cnn unet   # เฉพาะบาง model
  python3 compare.py --no-tune              # ข้าม HP tuning
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data" / "processed"
CKPT_DIR  = BASE_DIR / "checkpoints"
RES_DIR   = BASE_DIR / "results"
CKPT_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

from training.dataset   import WildfireDataset, WildfireSequenceDataset, load_pixel_data
from training.losses    import build_loss
from training.metrics   import compute_metrics, find_best_threshold, print_metrics
from training.trainer   import WildfireTrainer, get_device, build_optimizer, build_scheduler
from training.tuner     import HyperparameterTuner, DEFAULT_CONFIGS
from models.rf_baseline import RFWildfireModel
from models.cnn_patch   import build_cnn
from models.unet        import build_unet
from models.resunet     import build_resunet
from models.convlstm    import build_convlstm
from preprocessing.normalizer import FEATURE_ORDER
from config import YEARS as _CONFIG_YEARS


# ─────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────

TRAIN_MONTHS = list(range(1, 8))   # 1–7
VAL_MONTHS   = [8, 9]
TEST_MONTHS  = [10, 11, 12]
YEARS        = _CONFIG_YEARS       # อ่านจาก config.py (อย่า hardcode)
SEED         = 42

QUICK_CONFIG = dict(n_epochs=5,  n_tune_trials=3,  patch_size=32, batch_size=64, n_estimators=50)
FULL_CONFIG  = dict(n_epochs=30, n_tune_trials=10, patch_size=32, batch_size=64, n_estimators=200)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────

def make_patch_loaders(patch_size=32, batch_size=64, num_workers=0):
    from torch.utils.data import DataLoader
    print("\n📦 Building patch datasets...")
    train_ds = WildfireDataset(str(DATA_DIR), TRAIN_MONTHS, YEARS,
                               patch_size=patch_size, oversample=True,
                               augment=True, seed=SEED)
    val_ds   = WildfireDataset(str(DATA_DIR), VAL_MONTHS,   YEARS,
                               patch_size=patch_size, oversample=False,
                               augment=False, seed=SEED)
    test_ds  = WildfireDataset(str(DATA_DIR), TEST_MONTHS,  YEARS,
                               patch_size=patch_size, oversample=False,
                               augment=False, seed=SEED)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def make_seq_loaders(T=3, patch_size=32, batch_size=32, num_workers=0):
    from torch.utils.data import DataLoader
    print("\n📦 Building sequence datasets (ConvLSTM)...")
    train_ds = WildfireSequenceDataset(str(DATA_DIR), TRAIN_MONTHS, YEARS,
                                        T=T, patch_size=patch_size, seed=SEED)
    val_ds   = WildfireSequenceDataset(str(DATA_DIR), VAL_MONTHS + TRAIN_MONTHS[-T:],
                                        YEARS, T=T, patch_size=patch_size, seed=SEED)
    test_ds  = WildfireSequenceDataset(str(DATA_DIR), TEST_MONTHS + VAL_MONTHS[-T:],
                                        YEARS, T=T, patch_size=patch_size, seed=SEED)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


# ─────────────────────────────────────────────────────
# Model runners
# ─────────────────────────────────────────────────────

def run_rf(cfg: dict) -> dict:
    print("\n" + "═"*55)
    print("  MODEL: Random Forest Baseline")
    print("═"*55)

    X_train, y_train = load_pixel_data(str(DATA_DIR), TRAIN_MONTHS, YEARS, seed=SEED)
    X_val,   y_val   = load_pixel_data(str(DATA_DIR), VAL_MONTHS,   YEARS, seed=SEED)
    X_test,  y_test  = load_pixel_data(str(DATA_DIR), TEST_MONTHS,  YEARS, seed=SEED)

    t0 = time.time()
    model = RFWildfireModel(
        n_estimators=cfg.get("n_estimators", 200),
        max_depth=cfg.get("max_depth", 15),
        seed=SEED
    )
    model.fit(X_train, y_train, X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test, prefix="test")
    model.feature_importance(top_k=10)
    model.save(str(CKPT_DIR / "rf_best.joblib"))

    return {
        "model_name":    "Random_Forest",
        "training_time": time.time() - t0,
        "n_params":      "N/A",
        **test_metrics,
    }


def run_pytorch_model(model_key: str, cfg: dict, loaders: dict,
                      n_epochs: int, device) -> dict:
    print(f"\n{'═'*55}")
    print(f"  MODEL: {model_key.upper()}")
    print(f"{'═'*55}")

    in_channels = len(FEATURE_ORDER)
    assert in_channels > 0, "FEATURE_ORDER is empty!"
    print(f"  🔢 Feature channels: {in_channels} (from FEATURE_ORDER)")

    if model_key == "cnn":
        model = build_cnn(size="medium", in_channels=in_channels)
    elif model_key == "unet":
        model = build_unet(size="small", in_channels=in_channels)
    elif model_key == "resunet":
        model = build_resunet(size="small", in_channels=in_channels)
    elif model_key == "convlstm":
        model = build_convlstm(size="small", in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {model_key}")

    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    loss_fn   = build_loss(cfg.get("loss", "combined"), cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, {**cfg, "n_epochs": n_epochs})

    trainer = WildfireTrainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
        device=device, checkpoint_dir=str(CKPT_DIR), model_name=model_key,
        patience=max(3, n_epochs // 5),
    )

    t0           = time.time()
    trainer.fit(loaders["train"], loaders["val"], n_epochs=n_epochs)
    test_metrics = trainer.evaluate(loaders["test"])
    total_time   = time.time() - t0

    print_metrics(test_metrics, f"TEST — {model_key.upper()}")
    trainer.save_history(str(RES_DIR / f"history_{model_key}.json"))

    return {
        "model_name":    model_key.upper(),
        "training_time": total_time,
        "n_params":      n_params,
        **test_metrics,
    }


def make_objective(model_key: str, loaders: dict, device, n_epochs: int = 3):
    def objective(cfg: dict) -> float:
        try:
            in_channels = len(FEATURE_ORDER)
            if model_key == "cnn":
                model = build_cnn(size="small", in_channels=in_channels)
            elif model_key == "unet":
                model = build_unet(size="tiny", in_channels=in_channels)
            elif model_key == "resunet":
                model = build_resunet(size="small", in_channels=in_channels)
            elif model_key == "convlstm":
                model = build_convlstm(size="small", in_channels=in_channels)
            else:
                return 0.0

            loss_fn   = build_loss(cfg.get("loss", "focal"), cfg)
            optimizer = build_optimizer(model, cfg)

            trainer = WildfireTrainer(
                model=model, loss_fn=loss_fn, optimizer=optimizer,
                device=device, checkpoint_dir=str(CKPT_DIR / "tuning"),
                model_name=f"{model_key}_trial", patience=n_epochs,
            )
            result = trainer.fit(loaders["train"], loaders["val"],
                                  n_epochs=n_epochs, verbose=False)
            return result["best_val_f1"]
        except Exception as e:
            print(f"    Objective error: {e}")
            return 0.0
    return objective


# ─────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────

def print_comparison_table(results: list):
    print(f"\n{'═'*85}")
    print("  MODEL COMPARISON RESULTS (Test Set)")
    print(f"{'═'*85}")

    header = f"  {'Model':<20}"
    for m in ["F1", "Precision", "Recall", "AUC-ROC", "AUC-PR", "IoU", "Time(s)", "Params"]:
        header += f"  {m:>10}"
    print(header)
    print(f"  {'─'*83}")

    best_f1 = max(r.get("test_f1", 0) for r in results)
    for r in sorted(results, key=lambda x: -x.get("test_f1", 0)):
        name  = r.get("model_name", "?")
        f1    = r.get("test_f1", 0)
        pre   = r.get("test_precision", 0)
        rec   = r.get("test_recall", 0)
        auc_r = r.get("test_auc_roc", 0)
        auc_p = r.get("test_auc_pr", 0)
        iou   = r.get("test_iou", 0)
        t     = r.get("training_time", 0)
        n_par = r.get("n_params", "N/A")
        mark  = " ★" if abs(f1 - best_f1) < 0.001 else ""
        n_par_str = f"{n_par:,}" if isinstance(n_par, int) else str(n_par)
        t_str = f"{t:.0f}s"
        print(f"  {name+mark:<20}  {f1:>10.4f}  {pre:>10.4f}  {rec:>10.4f}"
              f"  {auc_r:>10.4f}  {auc_p:>10.4f}  {iou:>10.4f}"
              f"  {t_str:>10}  {n_par_str:>10}")

    print(f"\n  ★ = Best model by F1")


def save_results(results: list):
    path = RES_DIR / "comparison_results.json"
    serializable = []
    for r in results:
        row = {}
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                row[k] = v
            elif isinstance(v, np.floating):
                row[k] = float(v)
            elif isinstance(v, np.integer):
                row[k] = int(v)
            else:
                row[k] = str(v)
        serializable.append(row)
    with open(str(path), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  💾 Results saved: {path}")


def plot_comparison(results: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = [r["model_name"] for r in results]
        f1s    = [r.get("test_f1", 0) for r in results]
        precs  = [r.get("test_precision", 0) for r in results]
        recs   = [r.get("test_recall", 0) for r in results]
        aucs   = [r.get("test_auc_roc", 0) for r in results]
        ious   = [r.get("test_iou", 0) for r in results]

        order  = sorted(range(len(f1s)), key=lambda i: -f1s[i])
        models = [models[i] for i in order]
        f1s    = [f1s[i] for i in order]
        precs  = [precs[i] for i in order]
        recs   = [recs[i] for i in order]
        aucs   = [aucs[i] for i in order]
        ious   = [ious[i] for i in order]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Wildfire Prediction — Model Comparison (Thailand 2023)",
                     fontsize=14, fontweight="bold")

        x = np.arange(len(models))
        w = 0.18
        ax = axes[0]
        ax.bar(x - 2*w, f1s,   w, label="F1",       color="#e74c3c")
        ax.bar(x -   w, precs, w, label="Precision", color="#e67e22")
        ax.bar(x,        recs,  w, label="Recall",    color="#f39c12")
        ax.bar(x +   w, aucs,  w, label="AUC-ROC",  color="#27ae60")
        ax.bar(x + 2*w, ious,  w, label="IoU",       color="#2980b9")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Metric Comparison")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ax2 = axes[1]
        colors = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9", "#8e44ad"]
        bars = ax2.barh(models, f1s, color=colors[:len(models)], alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, f1s):
            ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
        ax2.set_xlim(0, min(max(f1s) + 0.1, 1.0))
        ax2.set_xlabel("F1 Score")
        ax2.set_title("F1 Score Ranking")
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        out_path = str(RES_DIR / "comparison_chart.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Chart saved: {out_path}")
        plt.close()
    except Exception as e:
        print(f"  ⚠️  Plot failed: {e}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Wildfire DL Model Comparison")
    p.add_argument("--mode",    choices=["quick", "full"], default="quick")
    p.add_argument("--models",  nargs="+",
                   choices=["rf", "cnn", "unet", "resunet", "convlstm"],
                   default=["rf", "cnn", "unet", "resunet", "convlstm"])
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--year",    type=int, default=YEARS[-1], help="For printing reference year")
    return p.parse_args()


def main():
    args  = parse_args()
    mode  = args.mode
    run_m = set(args.models)

    cfg_map = QUICK_CONFIG if mode == "quick" else FULL_CONFIG
    n_epochs      = cfg_map["n_epochs"]
    n_tune_trials = 0 if args.no_tune else cfg_map["n_tune_trials"]
    patch_size    = cfg_map["patch_size"]
    batch_size    = cfg_map["batch_size"]

    device = get_device()
    print(f"\n🔥 Wildfire Model Comparison")
    print(f"   Mode   : {mode} ({n_epochs} epochs, {n_tune_trials} tune trials)")
    print(f"   Models : {sorted(run_m)}")
    print(f"   Device : {device}")

    results = []

    if "rf" in run_m:
        results.append(run_rf(DEFAULT_CONFIGS.get("rf", {})))

    need_patch = run_m & {"cnn", "unet", "resunet"}
    patch_loaders = make_patch_loaders(patch_size, batch_size) if need_patch else None

    for key in ["cnn", "unet", "resunet"]:
        if key not in run_m:
            continue
        cfg = DEFAULT_CONFIGS[key].copy()
        if n_tune_trials > 0 and patch_loaders:
            print(f"\n🔍 HP Tuning: {key.upper()} ({n_tune_trials} trials)")
            obj    = make_objective(key, patch_loaders, device, n_epochs=3)
            tuner  = HyperparameterTuner(key, obj, n_trials=n_tune_trials)
            tune_r = tuner.run()
            if tune_r.get("best_cfg"):
                cfg.update(tune_r["best_cfg"])
        results.append(run_pytorch_model(key, cfg, patch_loaders, n_epochs, device))

    if "convlstm" in run_m:
        cfg  = DEFAULT_CONFIGS["convlstm"].copy()
        T    = cfg.get("T", 3)
        loaders = make_seq_loaders(T=T, patch_size=patch_size,
                                    batch_size=max(16, batch_size // 2))
        if n_tune_trials > 0:
            print(f"\n🔍 HP Tuning: ConvLSTM ({n_tune_trials} trials)")
            obj    = make_objective("convlstm", loaders, device, n_epochs=3)
            tuner  = HyperparameterTuner("convlstm", obj, n_trials=n_tune_trials)
            tune_r = tuner.run()
            if tune_r.get("best_cfg"):
                cfg.update(tune_r["best_cfg"])
                T = cfg.get("T", T)
                loaders = make_seq_loaders(T=T, patch_size=patch_size,
                                            batch_size=max(16, batch_size // 2))
        results.append(run_pytorch_model("convlstm", cfg, loaders, n_epochs, device))

    if results:
        print_comparison_table(results)
        save_results(results)
        plot_comparison(results)
        best = max(results, key=lambda r: r.get("test_f1", 0))
        print(f"\n🏆 Best Model: {best['model_name']}  (F1={best.get('test_f1',0):.4f})")

    print("\n🎉 Comparison complete!")
    print(f"   Results: {RES_DIR}")
    print(f"   Checkpoints: {CKPT_DIR}")


if __name__ == "__main__":
    main()
