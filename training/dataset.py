"""
training/dataset.py — WildfireDataset

Patch-based dataset สำหรับ wildfire prediction
- โหลด .npy files (features + target)
- แบ่ง patch ขนาด patch_size×patch_size
- Stratified sampling: oversample fire patches เพื่อแก้ class imbalance (2.3%)
- รองรับ data augmentation (flip, rotate)
- WildfireSequenceDataset: temporal window สำหรับ ConvLSTM
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────
# Patch Extraction
# ─────────────────────────────────────────────────────

def extract_patches(feat: np.ndarray, tgt: np.ndarray,
                    patch_size: int = 32,
                    stride: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    แบ่ง feature/target grid เป็น patches
    feat: (H, W, C)
    tgt:  (H, W)
    returns: feat_patches (N, C, P, P), tgt_patches (N, P, P)
    """
    if stride is None:
        stride = patch_size // 2

    H, W, C = feat.shape
    P = patch_size

    feat_patches, tgt_patches = [], []
    for r in range(0, H - P + 1, stride):
        for c in range(0, W - P + 1, stride):
            fp = feat[r:r+P, c:c+P, :]
            tp = tgt[r:r+P, c:c+P]
            feat_patches.append(fp.transpose(2, 0, 1))  # → (C, P, P)
            tgt_patches.append(tp)

    return np.stack(feat_patches), np.stack(tgt_patches)


def stratified_fire_sample(feat_patches: np.ndarray,
                            tgt_patches: np.ndarray,
                            fire_ratio: float = 0.35,
                            seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample fire patches ให้ได้สัดส่วน fire_ratio
    """
    rng = np.random.default_rng(seed)

    fire_mask   = tgt_patches.sum(axis=(1, 2)) > 0
    no_fire_idx = np.where(~fire_mask)[0]
    fire_idx    = np.where(fire_mask)[0]

    if len(fire_idx) == 0:
        return feat_patches, tgt_patches

    n_fire   = len(fire_idx)
    n_target = int(n_fire / fire_ratio - n_fire)
    n_target = min(n_target, len(no_fire_idx))

    sampled_no_fire = rng.choice(no_fire_idx, n_target, replace=False)
    kept = np.concatenate([fire_idx, sampled_no_fire])
    rng.shuffle(kept)

    return feat_patches[kept], tgt_patches[kept]


# ─────────────────────────────────────────────────────
# Patch Dataset (CNN / U-Net)
# ─────────────────────────────────────────────────────

class WildfireDataset(Dataset):
    """
    Patch-based dataset สำหรับ CNN, U-Net, ResU-Net

    Parameters
    ----------
    data_dir    : path ไปยัง data/processed/
    months      : list ของเดือนที่ใช้ [1, 2, ..., 12]
    years       : list ของปี (default [2021, 2022, 2023])
    patch_size  : ขนาด patch (default 32)
    stride      : stride สำหรับ extraction (None = patch_size//2)
    oversample  : ถ้า True → stratified fire oversampling
    augment     : ถ้า True → random flip/rotate
    seed        : random seed
    """

    def __init__(self, data_dir: str, months: List[int], years: List[int] = [2021, 2022, 2023],
                 patch_size: int = 32, stride: Optional[int] = None,
                 oversample: bool = True, augment: bool = False,
                 seed: int = 42):
        self.patch_size = patch_size
        self.augment    = augment
        self.rng        = np.random.default_rng(seed)

        all_feat, all_tgt = [], []

        for year in years:
            for month in months:
                feat_path = Path(data_dir) / f"features_{year}_{month:02d}.npy"
                tgt_path  = Path(data_dir) / f"target_{year}_{month:02d}.npy"

                if not feat_path.exists() or not tgt_path.exists():
                    print(f"  ⚠️  Missing: {feat_path.name} — skipping")
                    continue

                feat = np.load(str(feat_path)).astype(np.float32)
                tgt  = np.load(str(tgt_path)).astype(np.float32)

                fp, tp = extract_patches(feat, tgt, patch_size, stride)

                if oversample:
                    fp, tp = stratified_fire_sample(fp, tp, seed=seed + month + year)

                all_feat.append(fp)
                all_tgt.append(tp)

        if not all_feat:
            raise ValueError(f"ไม่พบข้อมูลสำหรับ months={months} ใน {data_dir}")

        self.features = np.concatenate(all_feat, axis=0)  # (N, C, P, P)
        self.targets  = np.concatenate(all_tgt,  axis=0)  # (N, P, P)

        n_fire   = (self.targets.sum(axis=(1,2)) > 0).sum()
        fire_pct = 100 * n_fire / len(self.targets)
        print(f"  📦 Dataset months={months}: {len(self.targets):,} patches "
              f"| fire patches: {n_fire:,} ({fire_pct:.1f}%)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.features[idx])
        tgt  = torch.from_numpy(self.targets[idx])

        if self.augment:
            feat, tgt = self._augment(feat, tgt)

        return feat, tgt

    def _augment(self, feat, tgt):
        if self.rng.random() > 0.5:
            feat = torch.flip(feat, dims=[2])
            tgt  = torch.flip(tgt,  dims=[1])
        if self.rng.random() > 0.5:
            feat = torch.flip(feat, dims=[1])
            tgt  = torch.flip(tgt,  dims=[0])
        k = self.rng.integers(0, 4)
        feat = torch.rot90(feat, k=int(k), dims=[1, 2])
        tgt  = torch.rot90(tgt,  k=int(k), dims=[0, 1])
        return feat, tgt


# ─────────────────────────────────────────────────────
# Sequence Dataset (ConvLSTM)
# ─────────────────────────────────────────────────────

class WildfireSequenceDataset(Dataset):
    """
    Temporal window dataset สำหรับ ConvLSTM
    Input:  sequence ของ T เดือน → (T, C, P, P)
    Target: เดือนสุดท้ายใน window → (P, P)
    """

    def __init__(self, data_dir: str, months: List[int],
                 years: List[int] = [2021, 2022, 2023], T: int = 3,
                 patch_size: int = 32, stride: Optional[int] = None,
                 oversample: bool = True, seed: int = 42):
        self.T          = T
        self.patch_size = patch_size
        self.rng        = np.random.default_rng(seed)

        all_seq, all_tgt = [], []

        # ทำทีละปี ป้องกัน window ข้ามปี
        for year in years:
            monthly_feat, monthly_tgt = {}, {}
            for month in months:
                fp = Path(data_dir) / f"features_{year}_{month:02d}.npy"
                tp = Path(data_dir) / f"target_{year}_{month:02d}.npy"
                if fp.exists():
                    monthly_feat[month] = np.load(str(fp)).astype(np.float32)
                    monthly_tgt[month]  = np.load(str(tp)).astype(np.float32)

            sorted_months = sorted(monthly_feat.keys())

            for i in range(T, len(sorted_months)):
                window       = sorted_months[i-T:i]
                target_month = sorted_months[i]

                if not all(m in monthly_feat for m in window + [target_month]):
                    continue

                feats_in_window = [monthly_feat[m] for m in window]
                tgt_grid        = monthly_tgt[target_month]

                seqs, tgts = self._extract_sequence_patches(
                    feats_in_window, tgt_grid, patch_size, stride
                )

                if oversample:
                    seqs, tgts = self._stratified_seq(seqs, tgts, seed=seed+i)

                all_seq.append(seqs)
                all_tgt.append(tgts)

        if not all_seq:
            raise ValueError(f"ไม่เพียงพอสำหรับ T={T} กับ months={months}")

        self.sequences = np.concatenate(all_seq, axis=0)  # (N, T, C, P, P)
        self.targets   = np.concatenate(all_tgt,  axis=0)  # (N, P, P)

        fire_pct = 100*(self.targets.sum(axis=(1,2)) > 0).mean()
        print(f"  📦 Sequence dataset: {len(self.sequences):,} windows "
              f"| fire rate: {fire_pct:.1f}% | T={T}")

    def _extract_sequence_patches(self, feats_list, tgt_grid, patch_size, stride):
        if stride is None:
            stride = patch_size // 2
        H, W, C = feats_list[0].shape
        P = patch_size

        seqs, tgts = [], []
        for r in range(0, H - P + 1, stride):
            for c in range(0, W - P + 1, stride):
                seq = np.stack([
                    f[r:r+P, c:c+P, :].transpose(2, 0, 1)
                    for f in feats_list
                ])  # (T, C, P, P)
                tp = tgt_grid[r:r+P, c:c+P]
                seqs.append(seq)
                tgts.append(tp)

        return np.stack(seqs), np.stack(tgts)

    def _stratified_seq(self, seqs, tgts, seed):
        rng = np.random.default_rng(seed)
        fire_mask  = tgts.sum(axis=(1, 2)) > 0
        fire_idx   = np.where(fire_mask)[0]
        nofire_idx = np.where(~fire_mask)[0]
        if len(fire_idx) == 0:
            return seqs, tgts
        n_nofire = min(len(nofire_idx), len(fire_idx) * 3)
        sampled  = np.concatenate([fire_idx, rng.choice(nofire_idx, n_nofire, replace=False)])
        rng.shuffle(sampled)
        return seqs[sampled], tgts[sampled]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])
        tgt = torch.from_numpy(self.targets[idx])
        return seq, tgt


# ─────────────────────────────────────────────────────
# Pixel Dataset (Random Forest baseline)
# ─────────────────────────────────────────────────────

def load_pixel_data(data_dir: str, months: List[int],
                    years: List[int] = [2021, 2022, 2023],
                    sample_frac: float = 0.1,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    โหลดข้อมูลเป็น pixel-level arrays สำหรับ Random Forest
    returns: X (N, 27), y (N,)
    """
    rng = np.random.default_rng(seed)
    all_X, all_y = [], []

    for year in years:
        for month in months:
            fp = Path(data_dir) / f"features_{year}_{month:02d}.npy"
            tp = Path(data_dir) / f"target_{year}_{month:02d}.npy"
            if not fp.exists():
                continue

            feat = np.load(str(fp)).astype(np.float32)
            tgt  = np.load(str(tp)).astype(np.float32)

            H, W, C = feat.shape
            X = feat.reshape(-1, C)
            y = tgt.reshape(-1)

            fire_idx    = np.where(y > 0)[0]
            no_fire_idx = np.where(y == 0)[0]

            n_fire   = len(fire_idx)
            n_nofire = min(len(no_fire_idx), int(n_fire / 0.15))

            s_nofire = rng.choice(no_fire_idx, n_nofire, replace=False)
            idx = np.concatenate([fire_idx, s_nofire])
            rng.shuffle(idx)

            all_X.append(X[idx])
            all_y.append(y[idx])

    return np.concatenate(all_X), np.concatenate(all_y)


# ─────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────

def make_loaders(data_dir: str, years: List[int] = [2021, 2022, 2023],
                 train_months=(1,2,3,4,5,6,7),
                 val_months=(8,9),
                 test_months=(10,11,12),
                 patch_size: int = 32,
                 batch_size: int = 64,
                 num_workers: int = 0) -> dict:
    train_ds = WildfireDataset(data_dir, list(train_months), years,
                               patch_size=patch_size, oversample=True,
                               augment=True, seed=42)
    val_ds   = WildfireDataset(data_dir, list(val_months),   years,
                               patch_size=patch_size, oversample=False,
                               augment=False, seed=42)
    test_ds  = WildfireDataset(data_dir, list(test_months),  years,
                               patch_size=patch_size, oversample=False,
                               augment=False, seed=42)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    DATA_DIR = "data/processed"
    print("\n=== Testing WildfireDataset ===")
    ds = WildfireDataset(DATA_DIR, months=[1, 2, 3], patch_size=32)
    feat, tgt = ds[0]
    print(f"  patch feature shape: {feat.shape}")
    print(f"  patch target  shape: {tgt.shape}")

    print("\n=== Testing WildfireSequenceDataset ===")
    seq_ds = WildfireSequenceDataset(DATA_DIR, months=[1,2,3,4,5], T=3, patch_size=32)
    seq, tgt = seq_ds[0]
    print(f"  sequence shape: {seq.shape}")

    print("\n=== Testing pixel data for RF ===")
    X, y = load_pixel_data(DATA_DIR, months=[1, 2])
    print(f"  X shape: {X.shape}, fire rate: {y.mean():.3f}")

    print("\n✅ dataset.py OK")
