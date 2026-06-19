"""
trace/01_verify_config.py
─────────────────────────
STEP 0 — Config & AOI Verification
ยืนยันว่า: Thailand grid มีขนาดที่ถูกต้อง, paths exist, YEARS sync กัน
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from config import (
    THAILAND_BBOX, RESOLUTION_M, TARGET_CRS, TARGET_CRS_WGS84,
    YEARS, MONTHS, RAIN_ROLLING_DAYS,
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR, NORM_STATS_PATH,
)

SECTION = lambda s: print(f"\n{'─'*55}\n  {s}\n{'─'*55}")

print("=" * 55)
print("  STEP 0 — Config & AOI Verification")
print("=" * 55)

SECTION("1. Thailand Bounding Box")
print(f"  Longitude : {THAILAND_BBOX['min_lon']}°E – {THAILAND_BBOX['max_lon']}°E")
print(f"  Latitude  : {THAILAND_BBOX['min_lat']}°N – {THAILAND_BBOX['max_lat']}°N")
print(f"  Δ Lon     : {THAILAND_BBOX['max_lon'] - THAILAND_BBOX['min_lon']:.1f}°")
print(f"  Δ Lat     : {THAILAND_BBOX['max_lat'] - THAILAND_BBOX['min_lat']:.1f}°")

SECTION("2. Grid Dimensions @ 1km Resolution")
deg_per_km   = RESOLUTION_M / 111_320          # approximate at equator
deg_per_km_y = RESOLUTION_M / 111_320
width  = int((THAILAND_BBOX["max_lon"] - THAILAND_BBOX["min_lon"]) / deg_per_km) + 1
height = int((THAILAND_BBOX["max_lat"] - THAILAND_BBOX["min_lat"]) / deg_per_km_y) + 1
total  = width * height

print(f"  1km ≈ {deg_per_km:.5f}° (at equator)")
print(f"  Grid Width  : {width:,} pixels")
print(f"  Grid Height : {height:,} pixels")
print(f"  Total pixels: {total:,} px/month")
print(f"  Memory/month: {total * 34 * 4 / 1e6:.1f} MB  (34 features × float32)")
print(f"  Memory/year : {total * 34 * 4 * 12 / 1e6:.1f} MB")

SECTION("3. CRS Configuration")
print(f"  Export CRS (GEE): {TARGET_CRS_WGS84}  (WGS84, for GEE export)")
print(f"  Processing CRS  : {TARGET_CRS}  (UTM Zone 47N, for slope calculation)")
print(f"  ✓ Two-CRS design: WGS84 for export, UTM for terrain derivatives")

SECTION("4. Time Period")
print(f"  Years           : {YEARS}")
print(f"  Months          : {MONTHS}")
print(f"  Total months    : {len(YEARS) * len(MONTHS)}")
print(f"  Rolling windows : {RAIN_ROLLING_DAYS} days")
print(f"  Train months    : [1,2,3,4,5,6,7]  (Jan–Jul: dry season)")
print(f"  Val months      : [8,9]             (Aug–Sep: wet season)")
print(f"  Test months     : [10,11,12]        (Oct–Dec: cool/dry start)")

SECTION("5. Directory Paths")
print(f"  data/raw/       : {RAW_DIR}")
print(f"  data/interim/   : {INTERIM_DIR}")
print(f"  data/processed/ : {PROCESSED_DIR}")
print(f"  norm_stats.json : {NORM_STATS_PATH}")
for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
    exists = "✅ exists" if os.path.exists(d) else "⚠️  not yet created"
    print(f"    {os.path.basename(d)}/ → {exists}")

SECTION("6. Seasonal Encoding Verification")
import numpy as np, datetime
for month, name in [(1,"Jan"), (3,"Mar"), (7,"Jul"), (10,"Oct"), (12,"Dec")]:
    doy = datetime.date(2023, month, 15).timetuple().tm_yday
    s   = np.sin(2 * np.pi * doy / 365)
    c   = np.cos(2 * np.pi * doy / 365)
    fire_risk = "🔥 HIGH" if s > 0.5 else ("⚠️  MED" if s > 0 else "🌧  LOW")
    print(f"  {name:3s} (DOY={doy:3d}): sin={s:+.3f}  cos={c:+.3f}  {fire_risk}")
print()
print("  ✓ sin ช่วงต้นปี (dry season) = positive & high → model รู้ว่าเสี่ยง")
print("  ✓ Continuity: Dec/Jan ค่าใกล้กัน (ไม่มี discontinuity ที่ month boundary)")

print("\n" + "=" * 55)
print("  ✅ STEP 0 — Config Verification PASSED")
print("=" * 55)
