"""
data_collection/gee_export_missing.py
───────────────────────────────────────
Export เฉพาะ features ที่ขาดหายไป:

  1. LAI — MODIS MCD15A3H  (72 files: ปี 2018-2023 × 12 เดือน)
  2. historical_fire_freq  — MODIS MCD64A1 (6 files: ปี 2018-2023)
  3. MODIS NDMI/NDVI fallback — สำหรับเดือนที่ S2 ขาด (11 months)
     - ndmi_ndvi_s2_2018_05 ถึง _10   (6 months)
     - ndmi_ndvi_s2_2020_07 ถึง _09   (3 months)
     - ndmi_ndvi_s2_2022_09            (1 month)
     - ndmi_ndvi_s2_2023_08            (1 month)

รวม: 72 + 6 + 11 = 89 GEE export tasks

Usage:
  python data_collection/gee_export_missing.py --project bnl-wildfire
  python data_collection/gee_export_missing.py --project bnl-wildfire --dry-run
"""

import ee
import argparse
import calendar
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    get_thailand_geometry,
    RESOLUTION_M, TARGET_CRS_WGS84,
    YEARS,
)

# ─────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────
DRIVE_FOLDER = "wildfire_data_chiangmai"

# เดือนที่ S2 ขาดหายไป (จาก data audit 2026-06-18)
MISSING_S2_MONTHS = [
    (2018, 5), (2018, 6), (2018, 7), (2018, 8), (2018, 9), (2018, 10),
    (2020, 7), (2020, 8), (2020, 9),
    (2022, 9),
    (2023, 8),
]


# ─────────────────────────────────────────────────────
# Export helper
# ─────────────────────────────────────────────────────

def _safe_export(image: ee.Image, description: str,
                 region: ee.Geometry,
                 scale: int = RESOLUTION_M,
                 dry_run: bool = False) -> Optional[ee.batch.Task]:
    """Submit a GEE export task, or print what would happen (dry_run)."""
    if dry_run:
        print(f"  [DRY RUN] Would export: {description}")
        return None

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        crs=TARGET_CRS_WGS84,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    try:
        task.start()
        print(f"  ✅ Export started: {description}")
        return task
    except Exception as e:
        print(f"  ⚠️  Export FAILED [{description}]: {e}")
        return None


# ─────────────────────────────────────────────────────
# 1. LAI — MODIS MCD15A3H (ทุกปี, ทุกเดือน)
# ─────────────────────────────────────────────────────

def export_lai_all(region, dry_run: bool = False) -> list:
    """Export LAI for all years and months."""
    print(f"\n{'━'*55}")
    print(f"  Part 1: LAI (MODIS MCD15A3H) — {len(YEARS)} years × 12 months = {len(YEARS)*12} files")
    print(f"{'━'*55}")

    if dry_run:
        for year in YEARS:
            for month in range(1, 13):
                print(f"  [DRY RUN] lai_{year}_{month:02d}")
        return []

    tasks = []
    for year in YEARS:
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end_day = calendar.monthrange(year, month)[1]
            end = f"{year}-{month:02d}-{end_day:02d}"

            lai_col = (
                ee.ImageCollection("MODIS/061/MCD15A3H")
                  .filterDate(start, end)
                  .filterBounds(region)
                  .select("Lai")
            )

            # Use toFloat() to explicitly cast the type, and wrap in a try-except
            # if we wanted to evaluate locally, but here we just build the computation graph.
            # We rename AFTER mean to ensure the output band is exactly "lai"
            lai_monthly = (
                lai_col
                  .select("Lai")
                  .mean()
                  .multiply(0.1)
                  .toFloat()
                  .rename("lai")
                  .unmask(0)
                  .clip(region)
            )

            desc = f"lai_{year}_{month:02d}"
            t = _safe_export(lai_monthly, desc, region, dry_run=dry_run)
            if t:
                tasks.append(t)

    print(f"\n  → {len(tasks)} LAI export tasks submitted")
    return tasks


# ─────────────────────────────────────────────────────
# 2. Historical Fire Frequency (ทุกปี)
# ─────────────────────────────────────────────────────

def export_historical_fire_freq_all(region, dry_run: bool = False) -> list:
    """Export historical fire frequency for each year (10-year lookback)."""
    print(f"\n{'━'*55}")
    print(f"  Part 2: Historical Fire Freq — {len(YEARS)} files (10yr lookback per year)")
    print(f"{'━'*55}")

    if dry_run:
        for year in YEARS:
            print(f"  [DRY RUN] historical_fire_freq_{year}")
        return []

    tasks = []
    for year in YEARS:
        start = f"{year-10}-01-01"
        end   = f"{year-1}-12-31"

        freq = (
            ee.ImageCollection("MODIS/061/MCD64A1")
              .filterDate(start, end)
              .filterBounds(region)
              .select("BurnDate")
              .map(lambda img: img.gt(0).unmask(0))
              .sum()
              .toFloat()
              .rename("historical_fire_freq")
              .clip(region)
        )

        desc = f"historical_fire_freq_{year}"
        t = _safe_export(freq, desc, region, dry_run=dry_run)
        if t:
            tasks.append(t)

    print(f"\n  → {len(tasks)} historical_fire_freq export tasks submitted")
    return tasks


# ─────────────────────────────────────────────────────
# 3. MODIS NDMI/NDVI fallback สำหรับเดือนที่ S2 ขาด
# ─────────────────────────────────────────────────────

def _mask_modis_clouds(image: ee.Image) -> ee.Image:
    """Mask MODIS MOD09A1 clouds using StateQA band."""
    qa = image.select("StateQA")
    cloud_mask = qa.bitwiseAnd(3).neq(0)  # bits 0-1 ≠ 00 = cloudy
    return image.updateMask(cloud_mask.Not())


def export_ndmi_modis_fallback(region, dry_run: bool = False) -> list:
    """
    Export MODIS NDMI/NDVI for months where Sentinel-2 data is missing.
    """
    print(f"\n{'━'*55}")
    print(f"  Part 3: MODIS NDMI/NDVI Fallback — {len(MISSING_S2_MONTHS)} months")
    print(f"{'━'*55}")
    print(f"  Months to fill:")
    for year, month in MISSING_S2_MONTHS:
        print(f"    - {year}-{month:02d}")
    print()

    if dry_run:
        for year, month in MISSING_S2_MONTHS:
            print(f"  [DRY RUN] ndmi_ndvi_s2_{year}_{month:02d}  (MODIS fallback)")
        return []

    tasks = []
    for year, month in MISSING_S2_MONTHS:
        start = f"{year}-{month:02d}-01"
        end_day = calendar.monthrange(year, month)[1]
        end = f"{year}-{month:02d}-{end_day:02d}"

        modis = (
            ee.ImageCollection("MODIS/061/MOD09A1")
              .filterDate(start, end)
              .filterBounds(region)
              .map(_mask_modis_clouds)
        )

        def compute_modis_indices(img: ee.Image) -> ee.Image:
            nir  = img.select("sur_refl_b02").multiply(0.0001)
            swir = img.select("sur_refl_b06").multiply(0.0001)
            red  = img.select("sur_refl_b01").multiply(0.0001)
            ndmi = nir.subtract(swir).divide(nir.add(swir)).rename("ndmi_s2")
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi_s2")
            return ndmi.addBands(ndvi)

        monthly_median = (
            modis
              .map(compute_modis_indices)
              .median()
              .unmask(0)
              .clip(region)
        )

        desc = f"ndmi_ndvi_s2_{year}_{month:02d}"
        t = _safe_export(monthly_median, desc, region, dry_run=dry_run)
        if t:
            tasks.append(t)

    print(f"\n  → {len(tasks)} MODIS fallback export tasks submitted")
    return tasks


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export missing GEE features: LAI, historical_fire_freq, MODIS NDMI fallback"
    )
    parser.add_argument("--project",  type=str, default="bnl-wildfire",
                        help="GEE project ID")
    parser.add_argument("--folder",   type=str, default=DRIVE_FOLDER,
                        help="Google Drive folder name")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print what would be exported without submitting")
    parser.add_argument("--parts",    type=str, default="1,2,3",
                        help="Which parts to run: 1=LAI, 2=hist_fire, 3=MODIS_fallback (default: all)")
    args = parser.parse_args()

    parts = [int(p) for p in args.parts.split(",")]
    dry_run = args.dry_run

    print("=" * 55)
    print("  GEE Missing Features Export")
    print(f"  Project : {args.project}")
    print(f"  Folder  : {args.folder}")
    print(f"  Parts   : {parts}")
    print(f"  Dry run : {dry_run}")
    print("=" * 55)

    if not dry_run:
        ee.Authenticate()
        ee.Initialize(project=args.project)
        region = get_thailand_geometry()
    else:
        print("\n  ⚠️  DRY RUN MODE — no tasks will be submitted\n")
        # stub — not used in dry_run
        region = None
    all_tasks = []

    if 1 in parts:
        all_tasks += export_lai_all(region, dry_run=dry_run)

    if 2 in parts:
        all_tasks += export_historical_fire_freq_all(region, dry_run=dry_run)

    if 3 in parts:
        all_tasks += export_ndmi_modis_fallback(region, dry_run=dry_run)

    print(f"\n{'='*55}")
    print(f"  ✅ Total tasks submitted : {len(all_tasks)}")
    print(f"     Expected              : {len(YEARS)*12 + len(YEARS) + len(MISSING_S2_MONTHS)}")
    if not dry_run:
        print(f"\n  ⏳ Monitor: https://code.earthengine.google.com/tasks")
        print(f"  📥 Download to: data/raw/ เมื่อ export เสร็จ")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
