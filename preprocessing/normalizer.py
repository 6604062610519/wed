"""
preprocessing/normalizer.py

Normalization strategies สำหรับทุก feature ใน wildfire prediction pipeline

Strategy ตาม feature type:
  - Min-Max      : Temperature, RH, Slope, NDMI, NDVI
  - Z-Score      : Elevation, LST
  - Log1p+MinMax : Rainfall (skewed), Wind Speed, Distance to Road, Pop Density
  - Circular     : Wind Direction, Aspect (แปลงเป็น sin/cos)
  - Binary       : Burned Area, Fire Occurrence (ไม่ normalize)
  - One-Hot      : Land Cover class

CRITICAL: fit() ต้องทำบน train set เท่านั้น
          แล้วนำ stats ไป transform() ให้ val/test
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NORM_STATS_PATH


# ─────────────────────────────────────────────────────
# Individual normalization functions (stateless)
# ─────────────────────────────────────────────────────

def minmax_normalize(x: np.ndarray,
                     x_min: float, x_max: float,
                     clip: bool = True) -> np.ndarray:
    """Min-Max Normalization → [0, 1]"""
    if x_max == x_min:
        return np.zeros_like(x, dtype=np.float32)
    norm = (x - x_min) / (x_max - x_min)
    if clip:
        norm = np.clip(norm, 0, 1)
    return norm.astype(np.float32)


def zscore_normalize(x: np.ndarray,
                     mean: float, std: float) -> np.ndarray:
    """Z-Score Normalization (Standardization)"""
    if std == 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)


def log1p_minmax_normalize(x: np.ndarray,
                           log_min: float, log_max: float,
                           clip: bool = True) -> np.ndarray:
    """
    Log1p แล้ว Min-Max สำหรับ heavily right-skewed data
    ใช้กับ: Rainfall, Wind Speed, Distance to Road/Settlement, Pop Density
    """
    x_safe = np.where(x < 0, 0, x)  # clamp negative to 0 (rainfall ไม่ติดลบ)
    x_log  = np.log1p(x_safe)
    return minmax_normalize(x_log, log_min, log_max, clip)


def circular_encode(angle_degrees: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    แปลงมุม (°) เป็น (sin, cos) เพื่อป้องกัน discontinuity
    0° ≡ 360°, 1° ≠ 359° ใน regular encoding แต่ใกล้เคียงกันใน sin/cos

    ใช้กับ: Wind Direction (0–360°), Aspect (0–360°)
    Returns: (sin_values, cos_values) — แต่ละอันมี shape เดียวกับ input
    """
    rad    = np.deg2rad(angle_degrees.astype(np.float64))
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)


def seasonal_encode(day_of_year: Union[int, np.ndarray],
                    year_length: int = 365) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode วันในปี (1–365) เป็น (sin, cos) เพื่อให้โมเดลรู้ฤดูกาล
    ใช้กับ: scalar หรือ array ของ day-of-year
    """
    doy = np.asarray(day_of_year, dtype=np.float64)
    sin_doy = np.sin(2 * np.pi * doy / year_length).astype(np.float32)
    cos_doy = np.cos(2 * np.pi * doy / year_length).astype(np.float32)
    return sin_doy, cos_doy


def onehot_landcover(lc: np.ndarray,
                     classes: list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
                     ) -> np.ndarray:
    """
    One-hot encode Land Cover classes
    Input : (H, W) integer array
    Output: (H, W, n_classes) boolean array
    """
    n_classes = len(classes)
    out = np.zeros((*lc.shape, n_classes), dtype=np.float32)
    for i, cls in enumerate(classes):
        out[..., i] = (lc == cls).astype(np.float32)
    return out


# ─────────────────────────────────────────────────────
# WildfireNormalizer — stateful class ที่ fit+transform
# ─────────────────────────────────────────────────────

class WildfireNormalizer:
    """
    Stateful normalizer สำหรับทุก feature ใน wildfire pipeline

    Usage:
        norm = WildfireNormalizer()
        norm.fit(train_data_dict)         # คำนวณ stats จาก train set
        norm.save(path)                   # บันทึก stats
        norm.load(path)                   # โหลด stats ที่บันทึกไว้
        X_norm = norm.transform(data_dict) # apply กับ val/test
    """

    # ─── Feature → strategy mapping ───────────────────
    STRATEGIES = {
        # Meteorological
        "temperature":          "minmax",
        "relative_humidity":    "minmax",
        "wind_speed":           "log1p_minmax",
        "wind_u":               "minmax",       # already bounded ~[-20, 20]
        "wind_v":               "minmax",
        "wind_dir_sin":         "none",         # already [-1, 1]
        "wind_dir_cos":         "none",
        "precipitation":        "log1p_minmax",
        "rain_3d":              "log1p_minmax",
        "rain_7d":              "log1p_minmax",
        "rain_14d":             "log1p_minmax",
        "soil_moisture":        "minmax",       # m³/m³, range [0, 0.5]

        # Derived meteorological features
        "vpd":                  "log1p_minmax", # kPa, right-skewed, ≥ 0
        "drought_index":        "log1p_minmax", # T/P ratio, right-skewed, ≥ 0

        # Satellite
        "ndmi_s2":              "minmax",       # clipped [-1,1] → [0,1]
        "ndvi_s2":              "minmax",
        "ndmi_modis":           "minmax",
        "ndvi_modis":           "minmax",
        "lst_celsius":          "zscore",       # LST spatially variable
        "lai":                  "log1p_minmax", # Leaf Area Index, 0–7, right-skewed
        "burned_binary":        "none",         # binary
        "burned_count_3m":      "minmax",
        "historical_fire_freq": "log1p_minmax", # count (0–120 months), right-skewed

        # Terrain
        "elevation":            "zscore",
        "slope":                "minmax",       # 0–90°
        "aspect":               "none",         # ทำ sin/cos แยกใน circular_encode
        "land_cover":           "none",         # one-hot แยก
        "fire_risk_lc":         "minmax",       # 0–5
        "dist_to_water":        "log1p_minmax", # meters, right-skewed, ≥ 0
        "topo_diversity":       "minmax",       # diversity index, bounded [0, 1]

        # Human factors
        "pop_density":          "log1p_minmax",
        "dist_to_road":         "log1p_minmax",
        "dist_to_settlement":   "log1p_minmax",
        "night_light":          "log1p_minmax",

        # Target
        "fire_occurrence":      "none",         # binary target — ไม่ normalize
    }

    def __init__(self):
        self.stats: Dict[str, dict] = {}
        self._is_fitted = False

    def fit(self, data: Dict[str, np.ndarray],
            percentile_clip: float = 99.9) -> "WildfireNormalizer":
        """
        คำนวณ normalization stats จาก training data
        ใช้ percentile_clip เพื่อป้องกัน outlier ที่รุนแรง

        Args:
            data: dict ของ {feature_name: numpy_array}
            percentile_clip: upper percentile สำหรับ clip outlier (default 99.9%)
        """
        print("🔧 Fitting normalizer on training data...")

        for feat, strategy in self.STRATEGIES.items():
            if feat not in data:
                continue
            arr = data[feat].astype(np.float64)
            valid = arr[~np.isnan(arr)]   # ข้าม NaN

            if len(valid) == 0:
                continue

            # Clip outliers (top percentile_clip)
            p_high = np.percentile(valid, percentile_clip)
            valid  = np.clip(valid, valid.min(), p_high)

            if strategy == "minmax":
                self.stats[feat] = {
                    "strategy": "minmax",
                    "min": float(valid.min()),
                    "max": float(p_high),
                }

            elif strategy == "zscore":
                self.stats[feat] = {
                    "strategy": "zscore",
                    "mean": float(valid.mean()),
                    "std":  float(valid.std()),
                }

            elif strategy == "log1p_minmax":
                log_vals = np.log1p(np.clip(valid, 0, None))
                self.stats[feat] = {
                    "strategy": "log1p_minmax",
                    "log_min": float(log_vals.min()),
                    "log_max": float(log_vals.max()),
                }

            elif strategy == "none":
                self.stats[feat] = {"strategy": "none"}

            print(f"  ✅ {feat:30s} → {strategy}  stats={self.stats[feat]}")

        self._is_fitted = True
        return self

    def transform(self, data: Dict[str, np.ndarray],
                  handle_circular: bool = True,
                  ) -> Dict[str, np.ndarray]:
        """
        Apply normalization ให้ทุก feature ตาม stats ที่ fit ไว้
        Circular features (wind_dir, aspect) จะถูกแปลงเป็น sin/cos อัตโนมัติ

        Returns: dict ของ normalized arrays
                 circular features จะได้ {'feature_sin': ..., 'feature_cos': ...}
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        out = {}

        for feat, arr in data.items():
            arr_f = arr.astype(np.float32)
            strategy_info = self.stats.get(feat, {"strategy": "none"})
            strategy = strategy_info["strategy"]

            if strategy == "minmax":
                out[feat] = minmax_normalize(
                    arr_f,
                    strategy_info["min"],
                    strategy_info["max"]
                )

            elif strategy == "zscore":
                out[feat] = zscore_normalize(
                    arr_f,
                    strategy_info["mean"],
                    strategy_info["std"]
                )

            elif strategy == "log1p_minmax":
                out[feat] = log1p_minmax_normalize(
                    arr_f,
                    strategy_info["log_min"],
                    strategy_info["log_max"]
                )

            elif strategy == "none":
                out[feat] = arr_f

            else:
                out[feat] = arr_f

        # Circular encoding (aspect, wind direction)
        if handle_circular:
            for circ_feat in ["aspect", "wind_direction"]:
                if circ_feat in data:
                    sin_v, cos_v = circular_encode(data[circ_feat])
                    out[f"{circ_feat}_sin"] = sin_v
                    out[f"{circ_feat}_cos"] = cos_v
                    if circ_feat in out:
                        del out[circ_feat]  # ลบ raw angle ออก

        return out

    def fit_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convenience: fit + transform ในขั้นตอนเดียว (train set เท่านั้น)"""
        self.fit(data)
        return self.transform(data)

    def save(self, path: str = NORM_STATS_PATH):
        """บันทึก normalization stats เป็น JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"💾 Normalization stats saved to {path}")

    def load(self, path: str = NORM_STATS_PATH) -> "WildfireNormalizer":
        """โหลด normalization stats จาก JSON"""
        with open(path) as f:
            self.stats = json.load(f)
        self._is_fitted = True
        print(f"📂 Normalization stats loaded from {path}")
        return self


# ─────────────────────────────────────────────────────
# Utility: Build feature stack from normalized dict
# ─────────────────────────────────────────────────────

FEATURE_ORDER = [
    # Weather (dynamic)
    "temperature", "relative_humidity",
    "wind_speed", "wind_u", "wind_v",
    "wind_dir_sin", "wind_dir_cos",
    "precipitation", "rain_3d", "rain_7d", "rain_14d",
    "soil_moisture", "vpd", "drought_index",

    # Satellite (semi-dynamic / annual)
    "ndmi_s2", "ndvi_s2",
    "lst_celsius", "lai",
    "burned_binary", "burned_count_3m",
    "historical_fire_freq",

    # Terrain (static)
    "elevation", "slope",
    "aspect_sin", "aspect_cos",
    "fire_risk_lc", "dist_to_water", "topo_diversity",

    # Human (static/yearly)
    "pop_density", "dist_to_road",
    "dist_to_settlement", "night_light",

    # Seasonal (computed)
    "season_sin", "season_cos",
]


def stack_features(normalized: Dict[str, np.ndarray],
                   order: list = FEATURE_ORDER) -> np.ndarray:
    """
    Stack normalized feature arrays เป็น single numpy array (H, W, C)

    Features ที่ขาดหายไปจะถูก **zero-fill** (ไม่ใช่ skip)
    เพื่อให้ output shape (H, W, C) สม่ำเสมอทุกเดือน
    → ป้องกัน DataLoader crash เมื่อ batch รวม months ที่มี features ต่างกัน

    Args:
        normalized: dict จาก WildfireNormalizer.transform()
        order: ลำดับ channel ที่ต้องการ
    Returns:
        np.ndarray shape (H, W, n_channels)
    """
    # หา reference shape จาก feature ที่มีอยู่
    ref_shape = None
    for feat in order:
        if feat in normalized:
            arr = normalized[feat]
            if arr.ndim == 2:
                ref_shape = arr.shape   # (H, W)
                break
            elif arr.ndim == 3:
                ref_shape = arr.shape[:2]
                break

    if ref_shape is None:
        raise ValueError("No features found in normalized dict to determine shape!")

    channels = []
    missing = []
    for feat in order:
        if feat in normalized:
            arr = normalized[feat]
            if arr.ndim == 2:
                channels.append(arr[..., np.newaxis])
            elif arr.ndim == 1:
                channels.append(arr.reshape(-1, 1))
            else:
                channels.append(arr)
        else:
            # Zero-fill: สร้าง array ว่างขนาดเดียวกับ reference
            zero = np.zeros((*ref_shape, 1), dtype=np.float32)
            channels.append(zero)
            missing.append(feat)

    if missing:
        print(f"  ⚠️  Zero-filled {len(missing)} missing features: {missing}")

    if not channels:
        raise ValueError("No features to stack!")

    return np.concatenate(channels, axis=-1)


# ─────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate dummy data
    H, W = 100, 100
    dummy = {
        "temperature":       np.random.uniform(15, 42, (H, W)).astype(np.float32),
        "relative_humidity": np.random.uniform(10, 95, (H, W)).astype(np.float32),
        "wind_speed":        np.random.exponential(3, (H, W)).astype(np.float32),
        "precipitation":     np.random.exponential(5, (H, W)).astype(np.float32),
        "rain_7d":           np.random.exponential(15, (H, W)).astype(np.float32),
        "ndmi_s2":           np.random.uniform(-0.5, 0.6, (H, W)).astype(np.float32),
        "lst_celsius":       np.random.uniform(20, 55, (H, W)).astype(np.float32),
        "elevation":         np.random.uniform(0, 2600, (H, W)).astype(np.float32),
        "slope":             np.random.uniform(0, 60, (H, W)).astype(np.float32),
        "aspect":            np.random.uniform(0, 360, (H, W)).astype(np.float32),
        "pop_density":       np.random.exponential(200, (H, W)).astype(np.float32),
        "dist_to_road":      np.random.exponential(5000, (H, W)).astype(np.float32),
    }

    norm = WildfireNormalizer()
    normalized = norm.fit_transform(dummy)
    norm.save("/tmp/test_norm_stats.json")

    print("\n📊 Normalized value ranges:")
    for k, v in normalized.items():
        print(f"  {k:30s}: [{v.min():.3f}, {v.max():.3f}]")
