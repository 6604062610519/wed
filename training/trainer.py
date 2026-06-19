"""
training/trainer.py — Training loop สำหรับ PyTorch models

WildfireTrainer:
  - Training loop พร้อม early stopping
  - Checkpoint: save best model by val F1
  - Optimal threshold search บน val set
  - ใช้ MPS (Apple Silicon) / CUDA / CPU ตามที่มี
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Callable
from tqdm import tqdm

from training.metrics import MetricsTracker, find_best_threshold, print_metrics


# ─────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────

def get_device() -> torch.device:
    """ตรวจหา best device: MPS (M2) > CUDA > CPU"""
    if torch.backends.mps.is_available():
        print("  🍎 Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"  🖥️  Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("  💻 Using CPU")
        return torch.device("cpu")


# ─────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────

class WildfireTrainer:
    """
    Generic trainer สำหรับ PyTorch wildfire models

    Parameters
    ----------
    model        : nn.Module ที่ต้องการ train
    loss_fn      : loss function (returns scalar tensor)
    optimizer    : torch optimizer
    scheduler    : lr scheduler (optional)
    device       : torch.device (auto-detect ถ้าไม่ระบุ)
    checkpoint_dir: path สำหรับบันทึก best model
    model_name   : ชื่อ model (ใช้ใน checkpoint filename)
    patience     : early stopping patience (epochs)
    max_grad_norm: gradient clipping (None = ไม่ clip)
    """

    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler=None,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: str = "checkpoints",
                 model_name: str = "model",
                 patience: int = 7,
                 max_grad_norm: Optional[float] = 1.0):

        self.device         = device or get_device()
        self.model          = model.to(self.device)
        self.loss_fn        = loss_fn
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name     = model_name
        self.patience       = patience
        self.max_grad_norm  = max_grad_norm

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint = self.checkpoint_dir / f"{model_name}_best.pt"

        self.history     = []
        self.best_val_f1 = -1.0
        self.best_threshold = 0.5
        self.no_improve  = 0

    # ─────────────────────────────────────────────────

    def train_epoch(self, loader) -> Dict:
        """รัน 1 epoch training"""
        self.model.train()
        tracker = MetricsTracker()

        for batch_feat, batch_tgt in tqdm(loader, desc="Training", leave=False):
            batch_feat = batch_feat.to(self.device)
            batch_tgt  = batch_tgt.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_feat)

            loss = self.loss_fn(logits, batch_tgt)
            loss.backward()

            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
                tgts  = batch_tgt.cpu().numpy()
                tracker.update(loss.item(), probs, tgts)

        return tracker.compute(threshold=self.best_threshold, prefix="train_")

    @torch.no_grad()
    def eval_epoch(self, loader, prefix: str = "val") -> Dict:
        """รัน 1 epoch evaluation"""
        self.model.eval()
        tracker = MetricsTracker()

        for batch_feat, batch_tgt in tqdm(loader, desc=f"Eval ({prefix})", leave=False):
            batch_feat = batch_feat.to(self.device)
            batch_tgt  = batch_tgt.to(self.device)

            logits = self.model(batch_feat)
            loss   = self.loss_fn(logits, batch_tgt)

            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            tgts  = batch_tgt.cpu().numpy()
            tracker.update(loss.item(), probs, tgts)

        return tracker.compute(prefix=f"{prefix}_")

    # ─────────────────────────────────────────────────

    def fit(self, train_loader, val_loader,
            n_epochs: int = 30,
            verbose: bool = True) -> Dict:
        """
        รัน full training loop

        Returns dict ของ best_val_metrics และ history
        """
        print(f"\n{'═'*55}")
        print(f"  Training: {self.model_name}")
        print(f"  Device  : {self.device}  |  Epochs: {n_epochs}  |  Patience: {self.patience}")
        print(f"{'═'*55}")

        best_val_metrics = {}
        t_start = time.time()

        for epoch in range(1, n_epochs + 1):
            t_ep = time.time()

            train_m = self.train_epoch(train_loader)
            val_m   = self.eval_epoch(val_loader, prefix="val")

            # val_f1 และ best threshold ถูกคำนวณใน MetricsTracker ของ eval_epoch()
            val_f1 = val_m.get("val_f1", 0)

            # LR scheduler
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    try:
                        self.scheduler.step(val_f1)
                    except TypeError:
                        self.scheduler.step()

            # Checkpoint
            if val_f1 > self.best_val_f1:
                self.best_val_f1     = val_f1
                self.best_threshold  = val_m.get("val_threshold", 0.5)
                best_val_metrics     = val_m
                self.no_improve      = 0
                torch.save({
                    "epoch":     epoch,
                    "model_state_dict":    self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_f1":    val_f1,
                    "threshold": self.best_threshold,
                    "history":   self.history,
                }, str(self.best_checkpoint))
            else:
                self.no_improve += 1

            # Log
            ep_time = time.time() - t_ep
            if verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:3d}/{n_epochs} "
                      f"| loss={train_m.get('train_loss',0):.4f} "
                      f"| val_f1={val_f1:.4f} "
                      f"| best={self.best_val_f1:.4f} "
                      f"| lr={lr:.2e} "
                      f"| {ep_time:.1f}s"
                      + (" ✓" if self.no_improve == 0 else f" [{self.no_improve}/{self.patience}]"))

            self.history.append({**train_m, **val_m, "epoch": epoch})

            # Early stopping
            if self.no_improve >= self.patience:
                print(f"\n  ⏹  Early stopping at epoch {epoch} "
                      f"(best val_f1={self.best_val_f1:.4f})")
                break

        total_time = time.time() - t_start
        print(f"\n  ✅ Training complete: {total_time:.1f}s "
              f"| Best val F1={self.best_val_f1:.4f} @ threshold={self.best_threshold:.3f}")

        return {
            "best_val_metrics": best_val_metrics,
            "history":          self.history,
            "best_threshold":   self.best_threshold,
            "best_val_f1":      self.best_val_f1,
            "training_time":    total_time,
        }

    # ─────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader, prefix: str = "test") -> Dict:
        """Evaluate บน test set โดยใช้ best checkpoint + best threshold"""
        if self.best_checkpoint.exists():
            ckpt = torch.load(str(self.best_checkpoint),
                              map_location=self.device,
                              weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            thr = ckpt.get("threshold", self.best_threshold)
            print(f"  📂 Loaded checkpoint (epoch={ckpt['epoch']}, "
                  f"val_f1={ckpt['val_f1']:.4f}, thr={thr:.3f})")
        else:
            thr = self.best_threshold

        self.model.eval()
        tracker = MetricsTracker()

        for batch_feat, batch_tgt in tqdm(loader, desc="Evaluate", leave=False):
            batch_feat = batch_feat.to(self.device)
            batch_tgt  = batch_tgt.to(self.device)

            logits = self.model(batch_feat)
            loss   = self.loss_fn(logits, batch_tgt)

            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            tgts  = batch_tgt.cpu().numpy()
            tracker.update(loss.item(), probs, tgts)

        return tracker.compute(threshold=thr, prefix=f"{prefix}_")

    def save_history(self, path: str):
        """บันทึก training history เป็น JSON"""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


# ─────────────────────────────────────────────────────
# Convenience: build standard optimizer + scheduler
# ─────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """สร้าง optimizer จาก config dict"""
    name = cfg.get("optimizer", "adamw").lower()
    lr   = cfg.get("lr", 1e-3)
    wd   = cfg.get("weight_decay", 1e-4)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict):
    """สร้าง LR scheduler จาก config dict"""
    name = cfg.get("scheduler", "plateau").lower()
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("n_epochs", 30)
        )
    elif name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")
