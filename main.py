"""
main.py — Wildfire Prediction Data Pipeline Orchestrator

สั่งรันขั้นตอนต่างๆ ทั้งหมด:
  1. Authenticate GEE
  2. Export static features (Terrain, LandCover, Human Factors)
  3. Export monthly dynamic features (Weather, Satellite) สำหรับทุกเดือน
  4. (หลังดาวน์โหลด) Align rasters → 1km Thailand grid
  5. Normalize ทุก feature
  6. Save processed dataset

Usage:
  # Step 1: Export ข้อมูลไปยัง Google Drive (ต้องมี GEE account)
  python main.py --step export --year 2023

  # Step 2: Align (หลังดาวน์โหลดไฟล์จาก Google Drive มาไว้ใน data/raw/)
  python main.py --step align

  # Step 3: Normalize
  python main.py --step normalize

  # All at once (เฉพาะ align + normalize ถ้ามีข้อมูลแล้ว)
  python main.py --step all_local
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────
# Import project modules
# ─────────────────────────────────────────────────────
from config import (
    YEARS, MONTHS, RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
    NORM_STATS_PATH, get_thailand_geometry,
)

from preprocessing.aligner import (
    align_all_layers, stack_aligned_rasters,
    read_raster, fill_nan, REFERENCE_WIDTH, REFERENCE_HEIGHT,
)
from preprocessing.normalizer import WildfireNormalizer, stack_features, FEATURE_ORDER
from preprocessing.feature_engineer import (
    add_seasonal_features, compute_fwi_series, compute_kbdi_series,
)


# ─────────────────────────────────────────────────────
# STEP 1 — Export from GEE to Google Drive
# ─────────────────────────────────────────────────────

def step_export(year: int, gee_project: str, drive_folder: str):
    """Submit all GEE export tasks."""
    try:
        import ee
    except ImportError:
        print("❌ earthengine-api ไม่ได้ install — รัน: pip install earthengine-api")
        sys.exit(1)

    ee.Authenticate()
    ee.Initialize(project=gee_project)

    from data_collection.gee_terrain import export_static_features
    from data_collection.gee_human   import export_human_features
    from data_collection.gee_weather  import export_monthly_weather
    from data_collection.gee_satellite import export_monthly_satellite

    all_tasks = []

    # Static features (ทำครั้งเดียว)
    print("\n━━━ Static Features ━━━")
    all_tasks += export_static_features(drive_folder)
    all_tasks += export_human_features(year, drive_folder)

    # Dynamic features (รายเดือน)
    print(f"\n━━━ Dynamic Features — {year} ━━━")
    for month in MONTHS:
        print(f"\n  Month {month:02d}:")
        all_tasks += export_monthly_weather(year, month, drive_folder)
        all_tasks += export_monthly_satellite(year, month, drive_folder)

    print(f"\n✅ Submitted {len(all_tasks)} export tasks to Google Drive folder: '{drive_folder}'")
    print("⏳ ตรวจสอบสถานะได้ที่: https://code.earthengine.google.com/tasks")
    print("📥 เมื่อ export เสร็จ ดาวน์โหลดไฟล์จาก Google Drive มาไว้ใน data/raw/")


# ─────────────────────────────────────────────────────
# STEP 2 — Align Rasters
# ─────────────────────────────────────────────────────

def step_align():
    """Align all downloaded GeoTIFFs to common Thailand 1km grid."""
    print("\n━━━ Aligning Rasters ━━━")
    aligned = align_all_layers(
        raw_dir=RAW_DIR,
        interim_dir=INTERIM_DIR,
        categorical_keywords=["land_cover", "burned_binary", "fire_target"],
    )
    print(f"\n✅ Aligned {len(aligned)} raster layers → {INTERIM_DIR}")
    return aligned


# ─────────────────────────────────────────────────────
# STEP 3 — Normalize
# ─────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────
# Band name mapping: file stem → ordered band names in GeoTIFF
# (ต้องตรงกับลำดับ bands ที่ export จาก GEE)
# ─────────────────────────────────────────────────────
FILE_BAND_MAP = {
    "weather": [
        "temperature", "relative_humidity", "wind_u", "wind_v",
        "wind_speed", "wind_dir_sin", "wind_dir_cos",
        "precipitation", "soil_moisture",
        "rain_3d", "rain_7d", "rain_14d",
    ],
    "ndmi_ndvi_s2":    ["ndmi_s2", "ndvi_s2"],
    "ndmi_ndvi_modis": ["ndmi_s2", "ndvi_s2"],  # use same name for normalizer
    "lst":             ["lst_celsius"],
    "lai":             ["lai"],
    "burned":          ["burned_binary", "burned_count_3m"],
    "historical_fire_freq": ["historical_fire_freq"],
    "fire_target":     ["fire_occurrence"],

    "terrain_elev_slope_aspect": ["elevation", "slope", "aspect"],
    "land_cover_esa":  ["land_cover"],
    "fire_risk_lc":    ["fire_risk_lc"],
    "dist_to_water":   ["dist_to_water"],
    "topo_diversity":  ["topo_diversity"],

    "night_light":     ["night_light"],
    "pop_density":     ["pop_density"],
    "dist_to_road":    ["dist_to_road"],
    "dist_to_settlement": ["dist_to_settlement"],
}


def _strip_year_month(key: str) -> str:
    """
    Remove year_month or year-only suffix from file stem.
    'weather_2023_03'        → 'weather'
    'pop_density_2023'       → 'pop_density'
    'night_light_2023'       → 'night_light'
    'terrain_elev_slope_aspect' → unchanged
    """
    import re
    key = re.sub(r"_\d{4}_\d{2}$", "", key)   # _YYYY_MM
    key = re.sub(r"_\d{4}$",       "", key)    # _YYYY
    return key


def _load_as_named_bands(path: str, stem: str,
                          nan_method: str = "mean") -> dict:
    """
    Load a GeoTIFF → dict ของ {band_name: 2D (H,W) array}
    ใช้ FILE_BAND_MAP เพื่อตั้งชื่อ band ตาม file type อัตโนมัติ
    """
    data, _, _ = read_raster(path)            # (n_bands, H, W) หรือ (H, W)
    base_stem  = _strip_year_month(stem)
    band_names = FILE_BAND_MAP.get(base_stem)

    result = {}

    if data.ndim == 2:
        data_filled = fill_nan(data, method=nan_method)
        name = band_names[0] if band_names else base_stem
        result[name] = data_filled

    elif data.ndim == 3:
        n_bands = data.shape[0]
        if band_names and len(band_names) == n_bands:
            for i, name in enumerate(band_names):
                result[name] = fill_nan(data[i], method=nan_method)
        else:
            # Fallback: ตั้งชื่อ b1, b2, ... ถ้า band map ไม่ตรง
            for i in range(n_bands):
                result[f"{base_stem}_b{i+1}"] = fill_nan(data[i], method=nan_method)
    return result


def step_normalize(train_months: list = [1, 2, 3, 4, 5, 6]):
    """
    Load aligned rasters (interim/), compute normalization stats,
    transform all features, save to processed/ as .npy files.

    ใช้ train_months เพื่อ fit normalizer stats (ป้องกัน data leakage)
    """
    import datetime
    print("\n━━━ Loading Aligned Data ━━━")

    interim_path = Path(INTERIM_DIR)
    all_files    = {p.stem: str(p) for p in interim_path.rglob("*.tif")}

    if not all_files:
        print("❌ ไม่พบไฟล์ใน interim/ — กรุณา align ก่อน (--step align)")
        sys.exit(1)

    print(f"Found {len(all_files)} aligned layers")

    # ── แยก static vs annual vs monthly ──
    # monthly: ends with _YYYY_MM
    # annual: ends with _YYYY
    # static: no year or month
    import re
    is_monthly = lambda k: bool(re.search(r"_\d{4}_\d{2}$", k))
    is_annual  = lambda k: bool(re.search(r"_\d{4}$", k))

    monthly_stems = [k for k in all_files if is_monthly(k)]
    annual_stems  = [k for k in all_files if is_annual(k)]
    static_stems  = [k for k in all_files if not is_monthly(k) and not is_annual(k)]

    # โหลด True Static features (band-aware)
    static_data: dict = {}
    for stem in static_stems:
        bands = _load_as_named_bands(all_files[stem], stem, nan_method="median")
        static_data.update(bands)

    # ตรวจสอบ spatial dimensions จาก static data
    ref_array = next(iter(static_data.values()))
    H, W = ref_array.shape[:2]

    print(f"\n📦 Static bands loaded: {sorted(static_data.keys())}")
    print(f"   Grid: H={H}, W={W}")

    # ── สร้าง monthly dataset ──
    train_data_flat: dict = {}

    print("\n━━━ Processing Monthly Data ━━━")
    for year in YEARS:
        # เตรียม Annual Data สำหรับปีนี้
        annual_data = dict(static_data)
        year_annual_stems = [k for k in annual_stems if k.endswith(f"_{year}")]
        for stem in year_annual_stems:
            bands = _load_as_named_bands(all_files[stem], stem, nan_method="median")
            annual_data.update(bands)

        for month in MONTHS:
            print(f"\n  Year {year} | Month {month:02d}")

            # เลือก dynamic files ของเดือนนี้และปีนี้ (ลงท้ายด้วย _YYYY_MM)
            month_stems = [k for k in monthly_stems if k.endswith(f"_{year}_{month:02d}")]

            month_data = dict(annual_data)  # เริ่มด้วย static + annual

            for stem in month_stems:
                bands = _load_as_named_bands(all_files[stem], stem, nan_method="mean")
                month_data.update(bands)

            # Compute VPD & Drought Proxy (KBDI proxy)
            if "temperature" in month_data and "relative_humidity" in month_data:
                T = month_data["temperature"]
                RH = month_data["relative_humidity"]
                svp = 0.61078 * np.exp(17.27 * T / (T + 237.3))
                vpd = svp * (1.0 - RH / 100.0)
                month_data["vpd"] = vpd.astype(np.float32)

            if "temperature" in month_data and "precipitation" in month_data:
                T = month_data["temperature"]
                P = month_data["precipitation"]
                drought = np.maximum(T, 0) / (P + 1.0)
                month_data["drought_index"] = drought.astype(np.float32)

            # Seasonal encoding (broadcast ให้เป็น (H,W) ไม่ใช่ scalar)
            doy = datetime.date(year, month, 15).timetuple().tm_yday
            month_data["season_sin"] = np.full((H, W), np.sin(2*np.pi*doy/365), dtype=np.float32)
            month_data["season_cos"] = np.full((H, W), np.cos(2*np.pi*doy/365), dtype=np.float32)

            # แยก target (fire_occurrence) ออกจาก features
            target = month_data.pop("fire_occurrence", None)

            # Accumulate training data สำหรับ fit normalizer
            if month in train_months:
                for k, v in month_data.items():
                    if k not in train_data_flat:
                        train_data_flat[k] = []
                    arr2d = np.asarray(v)
                    if arr2d.ndim == 2:
                        train_data_flat[k].append(arr2d.flatten())
                    elif arr2d.ndim == 0 or arr2d.ndim == 1:
                        train_data_flat[k].append(arr2d.ravel())

    # Concatenate ทุกเดือน train ของทุกปี
    for k in train_data_flat:
        train_data_flat[k] = np.concatenate(train_data_flat[k])

    print(f"\n  Features loaded for fitting: {sorted(train_data_flat.keys())}")

    # ── Fit normalizer ──
    print("\n━━━ Fitting Normalizer ━━━")
    norm = WildfireNormalizer()
    norm.fit(train_data_flat)
    norm.save(NORM_STATS_PATH)

    # ── Transform & Save ──
    print("\n━━━ Transforming & Saving ━━━")
    for year in YEARS:
        annual_data = dict(static_data)
        year_annual_stems = [k for k in annual_stems if k.endswith(f"_{year}")]
        for stem in year_annual_stems:
            bands = _load_as_named_bands(all_files[stem], stem, nan_method="median")
            annual_data.update(bands)

        for month in MONTHS:
            month_stems = [k for k in monthly_stems if k.endswith(f"_{year}_{month:02d}")]

            month_data = dict(annual_data)
            for stem in month_stems:
                bands = _load_as_named_bands(all_files[stem], stem, nan_method="mean")
                month_data.update(bands)

            if "temperature" in month_data and "relative_humidity" in month_data:
                T = month_data["temperature"]
                RH = month_data["relative_humidity"]
                svp = 0.61078 * np.exp(17.27 * T / (T + 237.3))
                month_data["vpd"] = (svp * (1.0 - RH / 100.0)).astype(np.float32)

            if "temperature" in month_data and "precipitation" in month_data:
                T = month_data["temperature"]
                P = month_data["precipitation"]
                month_data["drought_index"] = (np.maximum(T, 0) / (P + 1.0)).astype(np.float32)

            doy = datetime.date(year, month, 15).timetuple().tm_yday
            month_data["season_sin"] = np.full((H, W), np.sin(2*np.pi*doy/365), dtype=np.float32)
            month_data["season_cos"] = np.full((H, W), np.cos(2*np.pi*doy/365), dtype=np.float32)

            # แยก target
            target = month_data.pop("fire_occurrence", None)

            # Normalize features
            normalized = norm.transform(month_data)

            # Stack → (H, W, C)
            try:
                stacked = stack_features(normalized, FEATURE_ORDER)
            except Exception as e:
                print(f"  ⚠️  Stack failed for {year}_{month:02d}: {e}")
                print(f"       Available: {sorted(normalized.keys())}")
                continue

            # Save features
            out_path = Path(PROCESSED_DIR) / f"features_{year}_{month:02d}.npy"
            np.save(str(out_path), stacked)
            print(f"  💾 features: {out_path}  shape={stacked.shape}")

            # Save target
            if target is not None:
                tgt_path = Path(PROCESSED_DIR) / f"target_{year}_{month:02d}.npy"
                np.save(str(tgt_path), target.astype(np.float32))
                print(f"  🎯 target  : {tgt_path}  shape={target.shape}")

    print(f"\n✅ All months and years processed → {PROCESSED_DIR}")


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wildfire Prediction Data Pipeline"
    )
    parser.add_argument("--step", choices=["export", "align", "normalize", "all_local"],
                        default="all_local",
                        help="Pipeline step to run")
    parser.add_argument("--year", type=int, default=YEARS[-1],
                        help="Year to process (for export step)")
    parser.add_argument("--gee-project", type=str, default="your-gee-project-id",
                        help="Google Earth Engine project ID")
    parser.add_argument("--drive-folder", type=str, default="wildfire_data",
                        help="Google Drive folder name for GEE exports")
    parser.add_argument("--train-months", type=str, default="1,2,3,4,5,6",
                        help="Comma-separated months for fitting normalizer (e.g. '1,2,3')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_months = [int(m) for m in args.train_months.split(",")]

    print("🔥 Wildfire Prediction — Data Pipeline")
    print(f"   Year        : {args.year}")
    print(f"   Step        : {args.step}")
    print(f"   Train months: {train_months}")

    if args.step == "export":
        step_export(args.year, args.gee_project, args.drive_folder)

    elif args.step == "align":
        step_align()

    elif args.step == "normalize":
        step_normalize(train_months)

    elif args.step == "all_local":
        # ถ้ามีข้อมูล raw แล้ว: align + normalize
        step_align()
        step_normalize(train_months)

    print("\n🎉 Pipeline complete!")
