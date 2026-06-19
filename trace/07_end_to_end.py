"""
trace/07_end_to_end.py
───────────────────────
STEP 7 — Full Synthetic Pipeline End-to-End Test
พิสูจน์ว่า pipeline ทั้งหมดทำงานถูกต้องตั้งแต่ต้นจนจบ:
  Raw dict → Normalize → Stack (H,W,34) → Patch (N,34,32,32) → Model → Loss → Metrics
โดยใช้ synthetic data ทั้งหมด (ไม่ต้องมีข้อมูลจริง)
"""

import sys, os, datetime, time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.normalizer import WildfireNormalizer, FEATURE_ORDER, stack_features
from models.unet    import build_unet
from models.cnn_patch import build_cnn
from training.losses import build_loss
from training.metrics import compute_metrics, find_best_threshold

SECTION = lambda s: print(f"\n{'━'*58}\n  {s}\n{'━'*58}")
PASS = lambda msg: print(f"  ✅ {msg}")
FAIL = lambda msg: print(f"  ❌ {msg}")

print("=" * 58)
print("  END-TO-END SYNTHETIC PIPELINE TEST")
print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 58)

# ─────────────────────────────────────────────────────
# 1. SYNTHETIC RAW DATA (simulates 1 month of Thailand)
# ─────────────────────────────────────────────────────
SECTION("STEP 1 — Synthetic Raw Input Data")
np.random.seed(42)
H, W = 200, 200  # downsized (real: 1659×913)

raw = {
    # Weather (12 features from gee_weather.py)
    "temperature":          np.random.uniform(15, 42,   (H,W)).astype(np.float32),
    "relative_humidity":    np.random.uniform(10, 95,   (H,W)).astype(np.float32),
    "wind_speed":           np.random.exponential(3,    (H,W)).astype(np.float32),
    "wind_u":               np.random.uniform(-15,15,   (H,W)).astype(np.float32),
    "wind_v":               np.random.uniform(-15,15,   (H,W)).astype(np.float32),
    "wind_dir_sin":         np.random.uniform(-1,1,     (H,W)).astype(np.float32),
    "wind_dir_cos":         np.random.uniform(-1,1,     (H,W)).astype(np.float32),
    "precipitation":        np.random.exponential(5,    (H,W)).astype(np.float32),
    "soil_moisture":        np.random.uniform(0,0.5,    (H,W)).astype(np.float32),
    "rain_3d":              np.random.exponential(10,   (H,W)).astype(np.float32),
    "rain_7d":              np.random.exponential(20,   (H,W)).astype(np.float32),
    "rain_14d":             np.random.exponential(40,   (H,W)).astype(np.float32),
    # Satellite (7 features from gee_satellite.py)
    "ndmi_s2":              np.random.uniform(-0.5,0.6, (H,W)).astype(np.float32),
    "ndvi_s2":              np.random.uniform(-0.2,0.9, (H,W)).astype(np.float32),
    "lst_celsius":          np.random.uniform(20,55,    (H,W)).astype(np.float32),
    "lai":                  np.random.exponential(2,    (H,W)).astype(np.float32),
    "burned_binary":        (np.random.rand(H,W)>0.95).astype(np.float32),
    "burned_count_3m":      np.random.randint(0,4,      (H,W)).astype(np.float32),
    "historical_fire_freq": np.random.exponential(3,    (H,W)).astype(np.float32),
    # Terrain (7 features from gee_terrain.py)
    "elevation":            np.random.uniform(0,2600,   (H,W)).astype(np.float32),
    "slope":                np.random.uniform(0,60,     (H,W)).astype(np.float32),
    "aspect":               np.random.uniform(0,360,    (H,W)).astype(np.float32),
    "fire_risk_lc":         np.random.randint(0,6,      (H,W)).astype(np.float32),
    "dist_to_water":        np.random.exponential(5000, (H,W)).astype(np.float32),
    "topo_diversity":       np.random.uniform(0,1,      (H,W)).astype(np.float32),
    # Human (4 features from gee_human.py)
    "pop_density":          np.random.exponential(200,  (H,W)).astype(np.float32),
    "dist_to_road":         np.random.exponential(5000, (H,W)).astype(np.float32),
    "dist_to_settlement":   np.random.exponential(3000, (H,W)).astype(np.float32),
    "night_light":          np.random.exponential(10,   (H,W)).astype(np.float32),
}

# Derived features (computed in main.py step_normalize)
T, RH = raw["temperature"], raw["relative_humidity"]
svp = 0.61078 * np.exp(17.27 * T / (T + 237.3))
raw["vpd"]           = (svp * (1.0 - RH / 100.0)).astype(np.float32)
raw["drought_index"] = (np.maximum(T, 0) / (raw["precipitation"] + 1.0)).astype(np.float32)

# Seasonal (computed in main.py step_normalize)
doy = datetime.date(2023, 3, 15).timetuple().tm_yday   # March = DOY 74
raw["season_sin"] = np.full((H,W), np.sin(2*np.pi*doy/365), dtype=np.float32)
raw["season_cos"] = np.full((H,W), np.cos(2*np.pi*doy/365), dtype=np.float32)

# Target (fire_occurrence) — kept separate
target_raw = (np.random.rand(H, W) > 0.977).astype(np.float32)

print(f"  Raw features : {len(raw)} keys (33 source + 2 derived + 2 seasonal)")
print(f"  Includes aspect (raw degrees) → will be expanded to sin/cos by normalizer")
print(f"  Target shape : {target_raw.shape}  fire_rate={target_raw.mean()*100:.2f}%")
PASS(f"Raw data prepared: {len(raw)} features")

# ─────────────────────────────────────────────────────
# 2. NORMALIZE (simulates train months 1-7, test on month 3)
# ─────────────────────────────────────────────────────
SECTION("STEP 4 — Normalization (fit on train, transform all)")
t0 = time.perf_counter()
norm = WildfireNormalizer()

# Fit on "training" data (simulate 7 months of flat arrays)
train_flat = {k: v.flatten() for k, v in raw.items() if k not in ["season_sin", "season_cos"]}
train_flat["season_sin"] = raw["season_sin"].flatten()
train_flat["season_cos"] = raw["season_cos"].flatten()
norm.fit(train_flat)

# Transform month data (2D arrays, aspect will be circular encoded)
normalized = norm.transform(raw)
t_norm = time.perf_counter() - t0

print(f"  Fit + transform time : {t_norm*1000:.1f}ms")
print(f"  Normalized keys      : {len(normalized)}")
print(f"  (aspect expanded → aspect_sin, aspect_cos  →  +1 key)")
PASS(f"Normalize OK — {len(normalized)} features (includes aspect_sin/cos)")

# ─────────────────────────────────────────────────────
# 3. STACK FEATURES → (H, W, 34)
# ─────────────────────────────────────────────────────
SECTION("STEP 4b — Feature Stacking → (H, W, 34)")
t0 = time.perf_counter()
stacked = stack_features(normalized, FEATURE_ORDER)
t_stack = time.perf_counter() - t0

print(f"  FEATURE_ORDER length : {len(FEATURE_ORDER)}")
print(f"  Stacked shape        : {stacked.shape}")
print(f"  Expected             : ({H}, {W}, {len(FEATURE_ORDER)})")
print(f"  NaN count            : {np.isnan(stacked).sum()}")
print(f"  Stack time           : {t_stack*1000:.2f}ms")
shape_ok = stacked.shape == (H, W, len(FEATURE_ORDER))
nan_ok   = not np.isnan(stacked).any()
PASS(f"Stack shape correct: {shape_ok}") if shape_ok else FAIL(f"Shape wrong: {stacked.shape}")
PASS(f"No NaN in stack: {nan_ok}") if nan_ok else FAIL("NaN found in stack!")

# ─────────────────────────────────────────────────────
# 4. PATCH EXTRACTION → (N, 34, 32, 32)
# ─────────────────────────────────────────────────────
SECTION("STEP 5 — Patch Extraction → (N, C, P, P)")
P, stride = 32, 16
feat_chw = stacked.transpose(2, 0, 1)  # (C, H, W)
C = feat_chw.shape[0]

patches_f, patches_t = [], []
for r in range(0, H - P + 1, stride):
    for c in range(0, W - P + 1, stride):
        patches_f.append(feat_chw[:, r:r+P, c:c+P])
        patches_t.append(target_raw[r:r+P, c:c+P])

pf = np.stack(patches_f)   # (N, C, P, P)
pt = np.stack(patches_t)   # (N, P, P)

fire_in_patch = pt.sum(axis=(1,2)) > 0
n_fire  = fire_in_patch.sum()
n_total = len(pf)

print(f"  Patches extracted    : {n_total}")
print(f"  patch_size={P}, stride={stride}")
print(f"  pf shape             : {pf.shape}  (N, {C}, {P}, {P})")
print(f"  pt shape             : {pt.shape}  (N, {P}, {P})")
print(f"  Fire patches         : {n_fire} ({100*n_fire/n_total:.1f}%)")
PASS(f"Patch shape: {pf.shape}")

# Oversample to 35% fire ratio
fire_idx   = np.where(fire_in_patch)[0]
nofire_idx = np.where(~fire_in_patch)[0]
target_ratio = 0.35
n_nofire_keep = min(int(n_fire / target_ratio * (1 - target_ratio)), len(nofire_idx))
keep = np.concatenate([fire_idx, np.random.choice(nofire_idx, n_nofire_keep, replace=False)])
np.random.shuffle(keep)

pf_bal = pf[keep]
pt_bal = pt[keep]
final_fire_rate = fire_in_patch[keep].mean()
print(f"  After oversample     : {len(keep)} patches ({final_fire_rate*100:.1f}% fire)")
PASS(f"Oversampling OK: {final_fire_rate*100:.1f}% fire patches")

# ─────────────────────────────────────────────────────
# 5. DATALOADER
# ─────────────────────────────────────────────────────
SECTION("STEP 5b — DataLoader Batch")
ds = TensorDataset(
    torch.from_numpy(pf_bal.astype(np.float32)),
    torch.from_numpy(pt_bal.astype(np.float32)),
)
dl = DataLoader(ds, batch_size=16, shuffle=True)
batch_x, batch_y = next(iter(dl))

print(f"  Dataset size         : {len(ds)} samples")
print(f"  batch_x.shape        : {tuple(batch_x.shape)}  (B, C, P, P)")
print(f"  batch_y.shape        : {tuple(batch_y.shape)}  (B, P, P)")
print(f"  batch_x.dtype        : {batch_x.dtype}")
print(f"  batch_y unique       : {batch_y.unique().tolist()}")
shape_b_ok = batch_x.shape[1:] == (C, P, P)
PASS(f"DataLoader batch shape correct: {tuple(batch_x.shape)}") if shape_b_ok else FAIL("Wrong batch shape!")

# ─────────────────────────────────────────────────────
# 6. MODEL FORWARD PASS (all architectures)
# ─────────────────────────────────────────────────────
SECTION("STEP 6 — Model Forward Pass (in_channels=34)")
device = "mps" if torch.backends.mps.is_available() else "cpu"
B = batch_x.shape[0]

models_cfg = [
    ("CNN-Patch",  build_cnn("medium",  in_channels=C)),
    ("U-Net",      build_unet("small",  in_channels=C)),
]

print(f"  Device: {device}")
print(f"  Input: (B={B}, C={C}, P={P}, P={P})")
print()
print(f"  {'Model':<12s} {'Output shape':<20s} {'Params':>12s}  {'OK?'}")
print(f"  {'-'*12} {'-'*20} {'-'*12}  {'-'*5}")

all_model_ok = True
last_logits = None
for name, model in models_cfg:
    model = model.to(device).eval()
    x = batch_x.to(device)
    with torch.no_grad():
        out = model(x)
    ok = out.shape == (B, 1, P, P)
    if not ok:
        all_model_ok = False
    last_logits = out.detach().cpu()
    print(f"  {name:<12s} {str(tuple(out.shape)):<20s} {model.count_params():>12,}  {'✅' if ok else '❌'}")

PASS("All model forward passes OK") if all_model_ok else FAIL("Some models failed!")

# ─────────────────────────────────────────────────────
# 7. LOSS + METRICS
# ─────────────────────────────────────────────────────
SECTION("STEP 6b — Loss & Metrics Verification")
logits = last_logits
y_true = batch_y

# Loss
loss_fn = build_loss("combined")
loss_val = loss_fn(logits, y_true)
print(f"  Combined Loss (random model): {loss_val.item():.4f}")
PASS(f"Loss computes without error: {loss_val.item():.4f}")

# Metrics
probs  = torch.sigmoid(logits.squeeze(1)).numpy()
y_np   = y_true.numpy()
flat_p = probs.flatten()
flat_y = y_np.flatten()

best_t = find_best_threshold(flat_y, flat_p, n_thresholds=20)
m = compute_metrics(flat_y, flat_p, threshold=best_t)

print(f"\n  Best threshold     : {best_t:.3f}")
print(f"  F1 (random model)  : {m.get('f1', 0):.4f}  (expected ~0 for untrained)")
print(f"  AUC-ROC            : {m.get('auc_roc', 0):.4f}  (expected ~0.5 random)")
auc_ok = 0.3 <= m.get('auc_roc', 0) <= 0.7
PASS(f"Metrics compute OK (AUC-ROC≈{m.get('auc_roc',0):.3f})") if auc_ok else FAIL("AUC out of expected range for random model")

# ─────────────────────────────────────────────────────
# 8. FULL PIPELINE SUMMARY
# ─────────────────────────────────────────────────────
SECTION("PIPELINE SUMMARY")
checks = {
    "Raw data prepared (33 features)":           len(raw) >= 33,
    "Normalized (34 features after aspect enc)": len(normalized) >= 34,
    "Stacked shape (H, W, 34)":                  stacked.shape == (H, W, 34),
    "No NaN in stacked array":                   not np.isnan(stacked).any(),
    "Patch shape (N, 34, 32, 32)":               pf.shape[1:] == (34, P, P),
    "Oversampling fire≥30%":                     final_fire_rate >= 0.30,
    "DataLoader batch (B, 34, 32, 32)":          batch_x.shape[1:] == (34, P, P),
    "All models forward OK":                     all_model_ok,
    "Loss computes":                             not torch.isnan(loss_val),
    "Metrics computes":                          "f1" in m,
}

all_pass = True
for check, result in checks.items():
    status = "✅" if result else "❌"
    if not result:
        all_pass = False
    print(f"  {status} {check}")

print()
if all_pass:
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   ✅ ALL CHECKS PASSED — PIPELINE IS READY   ║")
    print("  ║   → พร้อม run กับ real data จาก GEE export   ║")
    print("  ╚══════════════════════════════════════════════╝")
else:
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   ❌ SOME CHECKS FAILED — see above          ║")
    print("  ╚══════════════════════════════════════════════╝")
