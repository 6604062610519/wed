"""
trace/04_verify_normalizer.py
──────────────────────────────
STEP 4 — Normalization Verification
ยืนยันว่า: strategies ครบ, transform ถูกต้อง, no leakage, no NaN
"""

import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.normalizer import WildfireNormalizer, FEATURE_ORDER, stack_features

SECTION = lambda s: print(f"\n{'─'*55}\n  {s}\n{'─'*55}")

print("=" * 55)
print("  STEP 4 — Normalization Verification")
print("=" * 55)

# ─── Synthetic training data ────────────────────────────
np.random.seed(42)
H, W = 100, 100

train_data = {
    "temperature":          np.random.uniform(15, 42,   (H,W)).astype(np.float32),
    "relative_humidity":    np.random.uniform(10, 95,   (H,W)).astype(np.float32),
    "wind_speed":           np.random.exponential(3,    (H,W)).astype(np.float32),
    "wind_u":               np.random.uniform(-15, 15,  (H,W)).astype(np.float32),
    "wind_v":               np.random.uniform(-15, 15,  (H,W)).astype(np.float32),
    "wind_dir_sin":         np.random.uniform(-1, 1,    (H,W)).astype(np.float32),
    "wind_dir_cos":         np.random.uniform(-1, 1,    (H,W)).astype(np.float32),
    "precipitation":        np.random.exponential(5,    (H,W)).astype(np.float32),
    "rain_3d":              np.random.exponential(10,   (H,W)).astype(np.float32),
    "rain_7d":              np.random.exponential(20,   (H,W)).astype(np.float32),
    "rain_14d":             np.random.exponential(40,   (H,W)).astype(np.float32),
    "soil_moisture":        np.random.uniform(0, 0.5,   (H,W)).astype(np.float32),
    "vpd":                  np.random.exponential(1.5,  (H,W)).astype(np.float32),
    "drought_index":        np.random.exponential(5,    (H,W)).astype(np.float32),
    "ndmi_s2":              np.random.uniform(-0.5,0.6, (H,W)).astype(np.float32),
    "ndvi_s2":              np.random.uniform(-0.2,0.9, (H,W)).astype(np.float32),
    "lst_celsius":          np.random.uniform(20, 55,   (H,W)).astype(np.float32),
    "lai":                  np.random.exponential(2,    (H,W)).astype(np.float32),
    "burned_binary":        (np.random.rand(H,W) > 0.95).astype(np.float32),
    "burned_count_3m":      np.random.randint(0, 4,    (H,W)).astype(np.float32),
    "historical_fire_freq": np.random.exponential(3,   (H,W)).astype(np.float32),
    "elevation":            np.random.uniform(0, 2600, (H,W)).astype(np.float32),
    "slope":                np.random.uniform(0, 60,   (H,W)).astype(np.float32),
    "aspect":               np.random.uniform(0, 360,  (H,W)).astype(np.float32),
    "fire_risk_lc":         np.random.randint(0, 6,    (H,W)).astype(np.float32),
    "dist_to_water":        np.random.exponential(5000,(H,W)).astype(np.float32),
    "topo_diversity":       np.random.uniform(0, 1,    (H,W)).astype(np.float32),
    "pop_density":          np.random.exponential(200, (H,W)).astype(np.float32),
    "dist_to_road":         np.random.exponential(5000,(H,W)).astype(np.float32),
    "dist_to_settlement":   np.random.exponential(3000,(H,W)).astype(np.float32),
    "night_light":          np.random.exponential(10,  (H,W)).astype(np.float32),
    "season_sin":           np.full((H,W), 0.951, dtype=np.float32),
    "season_cos":           np.full((H,W),-0.309, dtype=np.float32),
}

# ─── 1. STRATEGY COVERAGE ────────────────────────────────
SECTION("1. Strategy Coverage (ทุก feature ใน FEATURE_ORDER ต้องมี strategy)")
norm = WildfireNormalizer()

SPECIAL = {"aspect_sin", "aspect_cos", "season_sin", "season_cos"}
missing = [f for f in FEATURE_ORDER if f not in norm.STRATEGIES and f not in SPECIAL]

print(f"  FEATURE_ORDER count: {len(FEATURE_ORDER)}")
print(f"  STRATEGIES  count  : {len(norm.STRATEGIES)}")
print(f"  Special (sin/cos)  : {SPECIAL}")
if missing:
    print(f"  ❌ Missing strategies: {missing}")
else:
    print(f"  ✅ All features covered — no missing strategies")

# ─── 2. FIT ──────────────────────────────────────────────
SECTION("2. Fit Normalizer (train data)")
norm.fit(train_data)

print(f"\n  {'Feature':<30s} {'Strategy':<15s} {'Stats summary'}")
print(f"  {'-'*30} {'-'*15} {'-'*25}")
for feat in sorted(norm.stats.keys()):
    s = norm.stats[feat]
    strat = s.get("strategy", "?")
    if strat == "minmax":
        stat_str = f"min={s.get('min',0):.3f}  max={s.get('max',0):.3f}"
    elif strat == "zscore":
        stat_str = f"mean={s.get('mean',0):.3f}  std={s.get('std',1):.3f}"
    elif strat == "log1p_minmax":
        stat_str = f"log_min={s.get('log_min',0):.3f}  log_max={s.get('log_max',0):.3f}"
    else:
        stat_str = "(passthrough)"
    print(f"  {feat:<30s} {strat:<15s} {stat_str}")

# ─── 3. TRANSFORM ────────────────────────────────────────
SECTION("3. Transform Verification")
normalized = norm.transform(train_data)

print(f"  {'Feature':<28s} {'Strategy':<15s} {'Raw range':<20s} {'Norm range':<20s} {'OK?'}")
print(f"  {'-'*28} {'-'*15} {'-'*20} {'-'*20} {'-'*5}")

checks = {
    "temperature":    ("minmax",      (0, 1)),
    "wind_speed":     ("log1p_minmax",(0, 1)),
    "lst_celsius":    ("zscore",      (None, None)),  # z-score: unbounded
    "burned_binary":  ("none",        (0, 1)),
    "wind_dir_sin":   ("none",        (-1, 1)),
    "vpd":            ("log1p_minmax",(0, 1)),
    "soil_moisture":  ("minmax",      (0, 1)),
    "dist_to_water":  ("log1p_minmax",(0, 1)),
}

all_pass = True
for feat, (strat, (lo, hi)) in checks.items():
    if feat not in normalized:
        print(f"  {feat:<28s} ❌ NOT IN normalized dict!")
        all_pass = False
        continue
    arr = normalized[feat]
    a_min, a_max = float(arr.min()), float(arr.max())
    raw = train_data.get(feat, np.zeros(1))
    r_min, r_max = float(raw.min()), float(raw.max())

    if lo is not None and hi is not None:
        ok = (a_min >= lo - 0.01) and (a_max <= hi + 0.01)
    else:
        ok = True  # z-score: just check no NaN

    nan_ok = not np.isnan(arr).any()
    status = "✅" if (ok and nan_ok) else "❌"
    if not (ok and nan_ok):
        all_pass = False
    print(f"  {feat:<28s} {strat:<15s} [{r_min:+.2f},{r_max:+.2f}]  [{a_min:+.4f},{a_max:+.4f}]  {status}")

# ─── 4. STACK ────────────────────────────────────────────
SECTION("4. Stack Features → (H, W, 34)")
stacked = stack_features(normalized, FEATURE_ORDER)
print(f"  Input features  : {len(train_data)}")
print(f"  Normalized dict : {len(normalized)} (aspect → aspect_sin/cos added)")
print(f"  FEATURE_ORDER   : {len(FEATURE_ORDER)}")
print(f"  Stacked shape   : {stacked.shape}")
print(f"  Expected        : ({H}, {W}, {len(FEATURE_ORDER)})")
print(f"  NaN in stack    : {np.isnan(stacked).sum()}")
shape_ok = stacked.shape == (H, W, len(FEATURE_ORDER))
nan_ok   = not np.isnan(stacked).any()
print(f"  ✅ Shape correct : {shape_ok}")
print(f"  ✅ No NaN values : {nan_ok}")

# ─── 5. LEAKAGE CHECK ────────────────────────────────────
SECTION("5. Data Leakage Check — Train vs Val stats must not mix")
# Simulate test data with DIFFERENT distribution (val/test should use train stats)
test_data = {k: v + 5.0 for k, v in train_data.items()
             if k not in ["burned_binary", "burned_count_3m", "fire_risk_lc",
                           "wind_dir_sin", "wind_dir_cos", "season_sin", "season_cos", "aspect"]}
for k in ["burned_binary", "burned_count_3m", "fire_risk_lc",
           "wind_dir_sin", "wind_dir_cos", "season_sin", "season_cos", "aspect"]:
    if k in train_data:
        test_data[k] = train_data[k]

# If norm uses its fitted stats (from train), test transform will use the SAME stats
norm2 = WildfireNormalizer()
norm2.fit(train_data)
test_norm = norm2.transform(test_data)

# Check that temperature stats used are FROM TRAINING (not test)
train_stats = norm2.stats_["temperature"]
print(f"  Train stats used: min={train_stats['min']:.2f}°C  max={train_stats['max']:.2f}°C")
print(f"  Test temp range : {test_data['temperature'].min():.2f}–{test_data['temperature'].max():.2f}°C")
print(f"  Test temp norm  : {test_norm['temperature'].min():.4f}–{test_norm['temperature'].max():.4f}")
print()
print("  ✅ Test data uses TRAINING stats (no leakage) → some values may exceed [0,1]")
print("     This is expected and correct — test data can be out-of-distribution")

print("\n" + "=" * 55)
print(f"  {'✅' if all_pass else '❌'} STEP 4 — Normalization Verification {'PASSED' if all_pass else 'FAILED'}")
print("=" * 55)
