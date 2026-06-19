"""
data_collection/gee_terrain.py

ดึงข้อมูลภูมิประเทศ (Static — ดึงครั้งเดียว ไม่เปลี่ยนแปลง) และ Land Cover
Features:
  - Elevation (m)          — SRTM 30m → resample 1km
  - Slope (°)              — คำนวณจาก DEM
  - Aspect (°)             — คำนวณจาก DEM (แปลงเป็น sin/cos ใน preprocessing)
  - Land Cover (class)     — ESA WorldCover v200 (10m → resample)
"""

import ee
from typing import Optional

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    get_thailand_geometry, GEE_COLLECTIONS,
    RESOLUTION_M, TARGET_CRS, TARGET_CRS_WGS84,
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
# DEM → Elevation, Slope, Aspect
# ─────────────────────────────────────────────────────

def get_terrain(region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    คำนวณ Elevation, Slope, Aspect จาก SRTM DEM (30m → resample 1km)

    Output bands:
      - elevation  (m)
      - slope      (°)  0–90
      - aspect     (°)  0–360  (sin/cos จะทำใน normalizer.py)
    """
    if region is None:
        region = get_thailand_geometry()

    dem = ee.Image(GEE_COLLECTIONS["SRTM_DEM"]).select("elevation").clip(region)
    # Reproject to metric CRS (UTM) to ensure Slope is calculated correctly (meters/meters instead of meters/degrees)
    dem = dem.reproject(crs=TARGET_CRS, scale=30)

    terrain = ee.Terrain.products(dem)  # returns elevation, slope, aspect, hillshade

    elevation = terrain.select("elevation").rename("elevation")
    slope     = terrain.select("slope").rename("slope")
    aspect    = terrain.select("aspect").rename("aspect")

    # toFloat() เพื่อให้ทุก band มี type เดียวกัน (ป้องกัน "inconsistent types" error)
    return elevation.addBands([slope, aspect]).toFloat()


# ─────────────────────────────────────────────────────
# Land Cover — ESA WorldCover v200
# ─────────────────────────────────────────────────────

# ESA WorldCover class map (สำหรับ reference)
ESA_LANDCOVER_CLASSES = {
    10: "Tree cover",          # ป่าไม้ — เสี่ยงไฟป่าสูง
    20: "Shrubland",           # พุ่มไม้
    30: "Grassland",           # ทุ่งหญ้า
    40: "Cropland",            # เกษตรกรรม — เสี่ยงเผาไร่
    50: "Built-up",            # ชุมชน/เมือง
    60: "Bare/sparse veg.",    # พื้นดินโล่ง
    70: "Snow/Ice",
    80: "Permanent water body",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss/Lichen",
}

def get_land_cover(region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    ดึง ESA WorldCover v200 (2021)
    Output: land_cover — class integer (10, 20, 30, ...)
    ใน preprocessing จะแปลงเป็น one-hot หรือ embedding
    """
    if region is None:
        region = get_thailand_geometry()

    lc = ee.ImageCollection(GEE_COLLECTIONS["ESA_LANDCOVER"]) \
           .first() \
           .select("Map") \
           .rename("land_cover") \
           .clip(region)
    return lc


# ─────────────────────────────────────────────────────
# Fire Risk from Land Cover (derived feature)
# ─────────────────────────────────────────────────────

def get_fire_risk_from_lc(region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    แปลง Land Cover เป็น Fire Risk Score (0–5)
    อิงตาม fuel load และ ignition risk ของแต่ละประเภท
    """
    if region is None:
        region = get_thailand_geometry()

    lc = get_land_cover(region)

    # remap: class → fire_risk_score
    # หมายเหตุ: GEE Python API เวอร์ชันใหม่ต้องใช้ positional args (ไม่ใช่ keyword args)
    fire_risk = lc.remap(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],  # from
        [ 5,  4,  3,  4,  2,  1,  0,  0,  2,  3,   1],  # to
        0                                                  # defaultValue
    ).rename("fire_risk_lc")

    return fire_risk


# ─────────────────────────────────────────────────────
# Distance to Water Bodies
# ─────────────────────────────────────────────────────

def get_dist_to_water(region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    หาระยะห่างจากแหล่งน้ำ (Water Bodies) ที่ใกล้ที่สุด
    ใช้ชุดข้อมูล JRC Global Surface Water (max_extent)
    """
    if region is None:
        region = get_thailand_geometry()
    
    water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").eq(1)
    # หาระยะทาง (pixel) แล้วคูณขนาด pixel เพื่อแปลงเป็นเมตร (โดยประมาณ)
    dist = water.fastDistanceTransform().multiply(ee.Image.pixelArea().sqrt()).rename("dist_to_water").toFloat().clip(region)
    return dist


# ─────────────────────────────────────────────────────
# Topographic Diversity (Proxy for TWI)
# ─────────────────────────────────────────────────────

def get_topo_diversity(region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    ดึงข้อมูลความหลากหลายทางภูมิประเทศ (Topographic Diversity)
    ใช้เป็นตัวแทนของ TWI (Topographic Wetness Index) เพื่อบ่งชี้ความชุ่มชื้นระดับจุลภาค
    """
    if region is None:
        region = get_thailand_geometry()
    
    topo = ee.Image("CSP/ERGo/1_0/Global/SRTM_topoDiversity").select("constant").rename("topo_diversity").toFloat().clip(region)
    return topo


# ─────────────────────────────────────────────────────
# Export all terrain + land cover (static, once)
# ─────────────────────────────────────────────────────

def export_static_features(drive_folder: str = "wildfire_data_chiangmai") -> list:
    """Export all static terrain and land cover features to Google Drive."""
    region = get_thailand_geometry()
    tasks  = []

    print("\n🗻 Exporting static terrain features...")

    # Terrain (Elevation, Slope, Aspect)
    terrain = get_terrain(region)
    tasks.append(_export_to_drive(terrain, "terrain_elev_slope_aspect",
                                  drive_folder, region))

    # Land Cover
    lc = get_land_cover(region)
    tasks.append(_export_to_drive(lc, "land_cover_esa",
                                  drive_folder, region,
                                  scale=RESOLUTION_M))

    # Fire Risk from LC
    fire_risk_lc = get_fire_risk_from_lc(region)
    tasks.append(_export_to_drive(fire_risk_lc, "fire_risk_lc",
                                  drive_folder, region))

    # Distance to Water
    dist_to_water = get_dist_to_water(region)
    tasks.append(_export_to_drive(dist_to_water, "dist_to_water",
                                  drive_folder, region))

    # Topographic Diversity
    topo = get_topo_diversity(region)
    tasks.append(_export_to_drive(topo, "topo_diversity",
                                  drive_folder, region))

    return tasks


if __name__ == "__main__":
    ee.Authenticate()
    ee.Initialize(project="bnl-wildfire")  # ← เปลี่ยน project ID

    export_static_features()
    print("\n✅ Terrain/LandCover export tasks submitted")
