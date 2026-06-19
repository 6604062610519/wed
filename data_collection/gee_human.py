"""
data_collection/gee_human.py

ดึงข้อมูลปัจจัยมนุษย์ผ่าน Google Earth Engine และ OpenStreetMap
Features:
  - Population Density (person/km²)   — WorldPop / GPWv4
  - Distance to Road (m)              — OSM via GEE vector / rasterize
  - Distance to Settlement (m)        — OSM / GHSL
  - VIIRS Night-time Light (DNB)      — Proxy ของกิจกรรมมนุษย์

ทุก feature เป็น static หรือ yearly (ไม่เปลี่ยนรายเดือน)
"""

import ee
from typing import Optional

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    get_thailand_geometry, GEE_COLLECTIONS,
    RESOLUTION_M, TARGET_CRS_WGS84, YEARS,
)


def _export_to_drive(image: ee.Image, description: str,
                     folder: str, region: ee.Geometry,
                     scale: int = RESOLUTION_M) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        crs=TARGET_CRS_WGS84,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"  ✅ Export task: {description}")
    return task


# ─────────────────────────────────────────────────────
# Population Density
# ─────────────────────────────────────────────────────

def get_population_density(year: int = 2023,
                           region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    ดึง Population Density จาก WorldPop (100m)
    แนะนำ WorldPop สำหรับ Southeast Asia (มีข้อมูลดีกว่า GPWv4)
    """
    if region is None:
        region = get_thailand_geometry()

    # WorldPop (Global Project) มีข้อมูลถึงปี 2020 เป็นส่วนใหญ่
    target_year = min(year, 2020)

    # ใช้ข้อมูล WorldPop โดยกรองเอาเฉพาะประเทศไทย (country = 'THA')
    pop = ee.ImageCollection("WorldPop/GP/100m/pop") \
            .filter(ee.Filter.eq("country", "THA")) \
            .filterDate(f"{target_year}-01-01", f"{target_year}-12-31") \
            .first() \
            .select("population") \
            .rename("pop_density") \
            .clip(region)

    return pop


# ─────────────────────────────────────────────────────
# Distance to Roads — OSM via GEE
# ─────────────────────────────────────────────────────

def get_distance_to_roads(region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    คำนวณ Road Accessibility proxy จาก 2 แหล่ง (ใช้แหล่งแรกที่สำเร็จ):

    Option A: Oxford MAP Accessibility to Cities 2015 (confirmed ใน GEE)
              → travel_time_to_city (นาที) ซึ่งสัมพันธ์กับระยะทางถนนมาก
              → เปลี่ยนชื่อเป็น dist_to_road เพื่อความสม่ำเสมอ

    Option B: GHSL Built-up proximity (fallback)
              → ระยะทางถึง built-up area เป็น proxy

    หมายเหตุ: GRIP4/regional-roads ถูกลบออกจาก GEE Public Assets แล้ว
              ถ้าต้องการ actual road distance ให้ download OSM .pbf
              แล้ว rasterize ด้วย rasterio แทน (ทำใน step align)
    """
    if region is None:
        region = get_thailand_geometry()

    # ── Option A: Oxford MAP Accessibility to Cities 2015 ──────────────────
    # Dataset: Oxford/MAP/accessibility_to_cities_2015_v1_0
    # band: accessibility — travel time (minutes) to nearest city ≥ 50,000 pop
    # ค่าต่ำ = ใกล้เมือง/ถนนใหญ่ = กิจกรรมมนุษย์สูง = เสี่ยงไฟสูงกว่า
    accessibility = (
        ee.Image("projects/malariaatlasproject/assets/accessibility/accessibility_to_cities/2015_v1_0")
          .select("accessibility")
          .rename("dist_to_road")   # ใช้ชื่อเดิมเพื่อความ compatible
          .unmask(9999)             # NoData (ocean/unconnected) → max value
          .clip(region)
    )
    return accessibility


# ─────────────────────────────────────────────────────
# Distance to Settlements — GHSL / OSM
# ─────────────────────────────────────────────────────

def get_distance_to_settlements(region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    คำนวณระยะทางถึงพื้นที่ชุมชน (เมตร)
    ใช้ Global Human Settlement Layer (GHSL) built-up area
    """
    if region is None:
        region = get_thailand_geometry()

    # GHSL Built-Up Area (GEE: JRC/GHSL/P2023A/GHS_BUILT_S)
    ghsl = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S") \
             .filterDate("2020-01-01", "2020-12-31") \
             .first() \
             .select("built_surface") \
             .rename("built_up") \
             .clip(region)

    # Binary: built-up > 0 = settlement
    settlement_mask = ghsl.gt(0)

    # Distance transform
    dist_settlement = settlement_mask.Not() \
        .cumulativeCost(
            source=settlement_mask,
            maxDistance=100000,  # 100 km max
        ).rename("dist_to_settlement").clip(region)

    return dist_settlement


# ─────────────────────────────────────────────────────
# VIIRS Night-Time Light (DNB) — Human Activity Proxy
# ─────────────────────────────────────────────────────

def get_night_light(year: int = 2023,
                    region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    ดึง VIIRS Night-time Lights (Annual Composite)
    ค่าสูง = กิจกรรมมนุษย์มาก = โอกาสจุดไฟสูงกว่า
    Dataset: NOAA/VIIRS/DNB/ANNUAL_V21
    หมายเหตุ: dataset นี้อาจ lag 1-2 ปี (ปี 2023 อาจยังไม่มีใน GEE)
              ฟังก์ชันนี้จะ auto-fallback ไปปีล่าสุดที่มีข้อมูล
    """
    if region is None:
        region = get_thailand_geometry()

    ntl_collection = ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V21")

    # หา available year โดย fallback ย้อนหลังทีละปีจนกว่าจะเจอข้อมูล
    # (ป้องกัน "input is null" error เมื่อปีที่ขอยังไม่มีใน GEE)
    target_year = year
    ntl = None
    for try_year in range(target_year, target_year - 4, -1):  # ลองย้อนหลัง 3 ปี
        candidate = ntl_collection \
            .filterDate(f"{try_year}-01-01", f"{try_year}-12-31") \
            .first()

        # ตรวจสอบว่ามีข้อมูลจริงหรือเปล่า (getInfo เล็กน้อยเพื่อ validate)
        try:
            info = candidate.getInfo()
            if info is not None and info.get("bands"):
                ntl = candidate \
                    .select("average") \
                    .rename("night_light") \
                    .clip(region)
                if try_year != target_year:
                    print(f"  ℹ️  Night Light: ปี {target_year} ยังไม่มีใน GEE → ใช้ปี {try_year} แทน")
                break
        except Exception:
            continue

    if ntl is None:
        print(f"  ⚠️  Night Light: ไม่พบข้อมูลย้อนหลัง 3 ปี — ใช้ค่า 0 แทน")
        ntl = ee.Image(0).rename("night_light").clip(region)

    return ntl


# ─────────────────────────────────────────────────────
# Export all human factor features
# ─────────────────────────────────────────────────────

def _safe_export(image: ee.Image, description: str,
                 folder: str, region: ee.Geometry,
                 scale: int = RESOLUTION_M) -> list:
    """
    Wrapper รอบ _export_to_drive ที่ catch EEException ได้จริง
    (GEE lazy evaluation → error เกิดที่ task.start() ไม่ใช่ตอนสร้าง image)
    คืนค่า list ของ tasks ที่สำเร็จ ([] ถ้าล้มเหลว)
    """
    try:
        task = _export_to_drive(image, description, folder, region, scale)
        return [task]
    except Exception as e:
        print(f"  ⚠️  Export FAILED [{description}]: {e}")
        return []


def export_human_features(year: int = 2023,
                          drive_folder: str = "wildfire_data_chiangmai") -> list:
    """Export all human factor features to Google Drive."""
    region = get_thailand_geometry()
    tasks  = []

    print("\n👥 Exporting human factor features...")

    # Population Density
    pop = get_population_density(year, region)
    tasks += _safe_export(pop, f"pop_density_{year}", drive_folder, region)

    # Road Accessibility (Oxford MAP)
    dist_road = get_distance_to_roads(region)
    tasks += _safe_export(dist_road, "dist_to_road", drive_folder, region)

    # Distance to Settlements (GHSL)
    dist_settle = get_distance_to_settlements(region)
    tasks += _safe_export(dist_settle, "dist_to_settlement", drive_folder, region)

    # Night-time Light
    ntl = get_night_light(year, region)
    tasks += _safe_export(ntl, f"night_light_{year}", drive_folder, region)

    return tasks


if __name__ == "__main__":
    ee.Authenticate()
    ee.Initialize(project="bnl-wildfire")  # ← เปลี่ยน project ID

    export_human_features(YEARS[-1])
    print("\n✅ Human factor export tasks submitted")
