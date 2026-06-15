"""
reexport_missing.py — Re-export เฉพาะไฟล์ที่ยังขาดใน Google Drive

ไฟล์ที่ขาด (26 ไฟล์):
  Static  (2): terrain_elev_slope_aspect.tif, night_light_2023.tif
  Weather(12): weather_2023_01 → weather_2023_12
  Burned (12): burned_2023_01  → burned_2023_12

Usage:
  python reexport_missing.py --gee-project bnl-wildfire
  python reexport_missing.py --gee-project bnl-wildfire --skip-static   # ถ้า static กำลัง run อยู่
  python reexport_missing.py --gee-project bnl-wildfire --only weather  # เฉพาะ weather
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DRIVE_FOLDER = "wildfire_data"


def parse_args():
    p = argparse.ArgumentParser(description="Re-export missing GEE files")
    p.add_argument("--gee-project", required=True, help="GEE project ID")
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--drive-folder", default=DRIVE_FOLDER)
    p.add_argument("--skip-static", action="store_true",
                   help="Skip static features (terrain, night_light)")
    p.add_argument("--only", choices=["static", "weather", "burned", "all"],
                   default="all", help="Export only this group")
    return p.parse_args()


def main():
    args = parse_args()

    import ee
    ee.Authenticate()
    ee.Initialize(project=args.gee_project)

    from config import MONTHS
    from data_collection.gee_terrain import export_static_features
    from data_collection.gee_human   import export_human_features
    from data_collection.gee_weather  import export_monthly_weather
    from data_collection.gee_satellite import (
        get_burned_area, _safe_export, _export_to_drive,
        get_thailand_geometry, RESOLUTION_M,
    )

    region = get_thailand_geometry()
    all_tasks = []

    print(f"\n🔁 Re-exporting missing files for year {args.year}")
    print(f"   Drive folder: {args.drive_folder}")

    # ─────────────────────────────────────────────────
    # Static: terrain_elev_slope_aspect + night_light
    # ─────────────────────────────────────────────────
    if args.only in ("static", "all") and not args.skip_static:
        print("\n━━━ Static: Terrain ━━━")
        from data_collection.gee_terrain import get_terrain
        terrain = get_terrain(region)
        all_tasks += _safe_export(terrain, "terrain_elev_slope_aspect",
                                  args.drive_folder, region)

        print("\n━━━ Static: Night Light ━━━")
        from data_collection.gee_human import get_night_light, _safe_export as human_safe_export
        ntl = get_night_light(args.year, region)
        all_tasks += human_safe_export(ntl, f"night_light_{args.year}",
                                       args.drive_folder, region)

    # ─────────────────────────────────────────────────
    # Weather: weather_2023_01 → weather_2023_12
    # ─────────────────────────────────────────────────
    if args.only in ("weather", "all"):
        print("\n━━━ Weather (ERA5 Daily — rewrite) ━━━")
        for month in MONTHS:
            tasks = export_monthly_weather(args.year, month, args.drive_folder)
            all_tasks += tasks

    # ─────────────────────────────────────────────────
    # Burned Area: burned_2023_01 → burned_2023_12
    # ─────────────────────────────────────────────────
    if args.only in ("burned", "all"):
        print("\n━━━ Burned Area (MODIS MCD64A1) ━━━")
        for month in MONTHS:
            print(f"\n  Month {month:02d}:")
            burn = get_burned_area(args.year, month, region)
            tasks = _safe_export(burn, f"burned_{args.year}_{month:02d}",
                                 args.drive_folder, region)
            all_tasks += tasks

    print(f"\n✅ Submitted {len(all_tasks)} re-export tasks")
    print("⏳ ตรวจสอบสถานะที่: https://code.earthengine.google.com/tasks")
    print("📥 หลัง export เสร็จ ดาวน์โหลดไฟล์ที่ขาดมาใส่ใน data/raw/")


if __name__ == "__main__":
    main()
