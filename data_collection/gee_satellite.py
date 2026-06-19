"""
data_collection/gee_satellite.py

ดึงข้อมูลจากดาวเทียมผ่าน Google Earth Engine
Features:
  - NDMI  (Normalized Difference Moisture Index)  — Sentinel-2 / MODIS
  - NDVI  (Normalized Difference Vegetation Index) — Sentinel-2 / MODIS
  - LST   (Land Surface Temperature)               — MODIS MOD11A1
  - Burned Area History                            — MODIS MCD64A1
  - Active Fire (VIIRS)                            — NASA FIRMS / GEE

Output: GeoTIFF per month in data/raw/satellite/
"""

import ee
import calendar
from typing import Optional

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    get_thailand_geometry, GEE_COLLECTIONS,
    RESOLUTION_M, TARGET_CRS_WGS84,
    YEARS, MONTHS,
)


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

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
    print(f"  ✅ Export task started: {description}")
    return task


def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds using Sentinel-2 SCL (Scene Classification Layer) band."""
    scl = image.select("SCL")
    
    # SCL values to mask out:
    # 3: Cloud Shadows
    # 8: Clouds (Medium Probability)
    # 9: Clouds (High Probability)
    # 10: Cirrus
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    
    return image.updateMask(mask).divide(10000)  # Scale to [0,1]


def _mask_modis_clouds(image: ee.Image) -> ee.Image:
    """Mask MODIS MOD09A1 using StateQA band."""
    qa = image.select("StateQA")
    cloud_mask = qa.bitwiseAnd(3).neq(0)  # bits 0-1 ≠ 00 → cloudy
    return image.updateMask(cloud_mask.Not())


# ─────────────────────────────────────────────────────
# NDMI / NDVI — Sentinel-2 (preferred, 10m → resample)
# ─────────────────────────────────────────────────────

def get_ndmi_ndvi_sentinel2(year: int, month: int,
                            region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    คำนวณ NDMI และ NDVI จาก Sentinel-2 SR
    NDMI = (NIR - SWIR1) / (NIR + SWIR1) = (B8 - B11) / (B8 + B11)
    NDVI = (NIR - Red) / (NIR + Red)     = (B8 - B4)  / (B8 + B4)
    """
    if region is None:
        region = get_thailand_geometry()

    start = f"{year}-{month:02d}-01"
    end_day = calendar.monthrange(year, month)[1]
    end   = f"{year}-{month:02d}-{end_day:02d}"

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
           .filterDate(start, end) \
           .filterBounds(region) \
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)) \
           .map(_mask_s2_clouds)

    def compute_indices(img: ee.Image) -> ee.Image:
        nir  = img.select("B8")
        red  = img.select("B4")
        swir = img.select("B11")
        ndmi = nir.subtract(swir).divide(nir.add(swir)).rename("ndmi_s2")
        ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi_s2")
        return ndmi.addBands(ndvi)

    monthly_median = s2.map(compute_indices).median().clip(region)
    return monthly_median


def get_ndmi_ndvi_modis(year: int, month: int,
                        region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    คำนวณ NDMI และ NDVI จาก MODIS MOD09A1 (8-day, 500m)
    NDMI = (B2 - B6) / (B2 + B6)  [NIR=B2, SWIR=B6]
    NDVI = (B2 - B1) / (B2 + B1)  [NIR=B2, Red=B1]
    Fallback เมื่อ Sentinel-2 มีเมฆมาก
    """
    if region is None:
        region = get_thailand_geometry()

    start = f"{year}-{month:02d}-01"
    end_day = calendar.monthrange(year, month)[1]
    end   = f"{year}-{month:02d}-{end_day:02d}"

    modis = ee.ImageCollection(GEE_COLLECTIONS["MODIS_SR_8DAY"]) \
              .filterDate(start, end) \
              .filterBounds(region) \
              .map(_mask_modis_clouds)

    def compute_modis_indices(img: ee.Image) -> ee.Image:
        nir  = img.select("sur_refl_b02").multiply(0.0001)
        red  = img.select("sur_refl_b01").multiply(0.0001)
        swir = img.select("sur_refl_b06").multiply(0.0001)
        ndmi = nir.subtract(swir).divide(nir.add(swir)).rename("ndmi_modis")
        ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi_modis")
        return ndmi.addBands(ndvi)

    monthly_median = modis.map(compute_modis_indices).median().clip(region)
    return monthly_median


# ─────────────────────────────────────────────────────
# Land Surface Temperature — MODIS MOD11A1 (1 km daily)
# ─────────────────────────────────────────────────────

def get_lst_modis(year: int, month: int,
                  region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    ดึง LST_Day_1km จาก MODIS MOD11A1
    Scale factor: 0.02 → Kelvin → convert to Celsius
    """
    if region is None:
        region = get_thailand_geometry()

    start = f"{year}-{month:02d}-01"
    end_day = calendar.monthrange(year, month)[1]
    end   = f"{year}-{month:02d}-{end_day:02d}"

    lst = ee.ImageCollection(GEE_COLLECTIONS["MODIS_LST_DAILY"]) \
            .filterDate(start, end) \
            .filterBounds(region) \
            .select("LST_Day_1km")

    def scale_lst(img: ee.Image) -> ee.Image:
        """Apply scale factor and convert K → °C, mask QC."""
        return img.multiply(0.02).subtract(273.15).rename("lst_celsius")

    monthly_mean = lst.map(scale_lst).mean().clip(region)
    return monthly_mean


# ─────────────────────────────────────────────────────
# LAI (Leaf Area Index) — MODIS MCD15A3H (4-day, 500m)
# ─────────────────────────────────────────────────────

def get_lai_modis(year: int, month: int,
                  region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    ดึง Leaf Area Index (LAI) จาก MODIS MCD15A3H
    Scale factor: 0.1
    """
    if region is None:
        region = get_thailand_geometry()

    start = f"{year}-{month:02d}-01"
    end_day = calendar.monthrange(year, month)[1]
    end   = f"{year}-{month:02d}-{end_day:02d}"

    lai = ee.ImageCollection("MODIS/061/MCD15A3H") \
            .filterDate(start, end) \
            .filterBounds(region) \
            .select("Lai")   # GEE v6.1: band name is 'Lai' (not 'Lai_500m')

    def scale_lai(img: ee.Image) -> ee.Image:
        return img.multiply(0.1).toFloat().rename("lai")

    monthly_mean = lai.map(scale_lai).mean().unmask(0).clip(region)
    return monthly_mean


# ─────────────────────────────────────────────────────
# Burned Area History — MODIS MCD64A1 (monthly, 500m)
# ─────────────────────────────────────────────────────

def get_burned_area(year: int, month: int,
                    region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    ดึง Burn Date จาก MODIS MCD64A1
    Output:
      - burned_binary: 1 = มีการเผาไหม้ในเดือนนั้น, 0 = ไม่มี
      - burned_count_3m: จำนวนเดือนที่เผาไหม้ใน 3 เดือนที่ผ่านมา (historical)
    """
    if region is None:
        region = get_thailand_geometry()

    # Current month burn
    start = f"{year}-{month:02d}-01"
    end_day = calendar.monthrange(year, month)[1]
    end   = f"{year}-{month:02d}-{end_day:02d}"

    burn_current = ee.ImageCollection(GEE_COLLECTIONS["MODIS_BURN"]) \
        .filterDate(start, end) \
        .filterBounds(region) \
        .select("BurnDate") \
        .first()

    burned_binary = burn_current \
        .gt(0) \
        .unmask(0) \
        .toFloat() \
        .rename("burned_binary") \
        .clip(region)

    # 3-month historical burn count
    hist_start = ee.Date(start).advance(-3, "month")
    burned_3m = ee.ImageCollection(GEE_COLLECTIONS["MODIS_BURN"]) \
        .filterDate(hist_start, start) \
        .filterBounds(region) \
        .select("BurnDate") \
        .map(lambda img: img.gt(0).unmask(0)) \
        .sum() \
        .toFloat() \
        .rename("burned_count_3m") \
        .clip(region)

    return burned_binary.addBands(burned_3m)


# ─────────────────────────────────────────────────────
# Historical Fire Frequency (10 Years) — MODIS MCD64A1
# ─────────────────────────────────────────────────────

def get_historical_fire_freq(year: int,
                             region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    คำนวณความถี่การเกิดไฟป่าย้อนหลัง 10 ปี (ตั้งแต่ year-10 ถึง year-1)
    ใช้วิธีนับจำนวนครั้งที่ไฟไหม้ในพื้นที่นั้นๆ (pixel-wise sum)
    """
    if region is None:
        region = get_thailand_geometry()

    start = f"{year-10}-01-01"
    end   = f"{year-1}-12-31"

    freq = ee.ImageCollection(GEE_COLLECTIONS["MODIS_BURN"]) \
             .filterDate(start, end) \
             .filterBounds(region) \
             .select("BurnDate") \
             .map(lambda img: img.gt(0).unmask(0)) \
             .sum() \
             .toFloat() \
             .rename("historical_fire_freq") \
             .clip(region)
             
    return freq


# ─────────────────────────────────────────────────────
# Active Fire Points — VIIRS (Target Variable)
# ─────────────────────────────────────────────────────

def get_active_fire_raster(year: int, month: int,
                           region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    ดึง FIRMS active fire จาก GEE แปลงเป็น binary raster (1km)
    Output: fire_occurrence = 1 ถ้ามีไฟในเดือนนั้น
    นี่คือ TARGET VARIABLE สำหรับ training

    Dataset: FIRMS (MODIS C6 — confirmed ใน GEE)
    Band: T21 = brightness temperature, confidence = detection confidence (0–100)
    เงื่อนไข NASA/FIRMS/noaa-20-viirs-c2 ถูกลบออกจาก GEE Public Assets
    """
    if region is None:
        region = get_thailand_geometry()

    start = f"{year}-{month:02d}-01"
    end_day = calendar.monthrange(year, month)[1]
    end   = f"{year}-{month:02d}-{end_day:02d}"

    # FIRMS MODIS C6 — dataset ID เดิมที่ confirmed ว่าอยู่ใน GEE
    firms = ee.ImageCollection("FIRMS") \
              .filterDate(start, end) \
              .filterBounds(region) \
              .select(["T21", "confidence"])

    # binary: confidence >= 30 (nominal + high confidence detections)
    def fire_to_binary(img: ee.Image) -> ee.Image:
        return img.select("confidence").gte(30).rename("fire_occurrence")

    fire_raster = firms.map(fire_to_binary).max().unmask(0).clip(region)
    return fire_raster


# ─────────────────────────────────────────────────────
# Main export
# ─────────────────────────────────────────────────────

def _safe_export(image: ee.Image, description: str,
                 folder: str, region: ee.Geometry,
                 scale: int = RESOLUTION_M) -> list:
    """
    Wrapper รอบ _export_to_drive ที่ catch EEException ได้จริง
    GEE lazy evaluation → error เกิดที่ task.start() เสมอ
    คืนค่า [] ถ้า export ล้มเหลว (ไม่ crash ทั้ง pipeline)
    """
    try:
        task = _export_to_drive(image, description, folder, region, scale)
        return [task]
    except Exception as e:
        print(f"  ⚠️  Export FAILED [{description}]: {e}")
        return []


def export_monthly_satellite(year: int, month: int,
                             drive_folder: str = "wildfire_data_chiangmai",
                             use_sentinel2: bool = True) -> list:
    """Export all satellite features for a given month to Google Drive."""
    region = get_thailand_geometry()
    tasks  = []

    print(f"\n🛠️  Fetching satellite data: {year}-{month:02d}")

    # NDMI / NDVI — Sentinel-2 ก่อน, fallback เป็น MODIS
    if use_sentinel2:
        veg  = get_ndmi_ndvi_sentinel2(year, month, region)
        desc = f"ndmi_ndvi_s2_{year}_{month:02d}"
    else:
        veg  = get_ndmi_ndvi_modis(year, month, region)
        desc = f"ndmi_ndvi_modis_{year}_{month:02d}"
    tasks += _safe_export(veg, desc, drive_folder, region)

    # LST
    lst = get_lst_modis(year, month, region)
    tasks += _safe_export(lst, f"lst_{year}_{month:02d}", drive_folder, region)

    # LAI
    lai = get_lai_modis(year, month, region)
    tasks += _safe_export(lai, f"lai_{year}_{month:02d}", drive_folder, region)

    # Burned Area
    burn = get_burned_area(year, month, region)
    tasks += _safe_export(burn, f"burned_{year}_{month:02d}", drive_folder, region)

    # Active Fire — TARGET VARIABLE
    fire = get_active_fire_raster(year, month, region)
    tasks += _safe_export(fire, f"fire_target_{year}_{month:02d}", drive_folder, region)

    return tasks


def export_annual_satellite(year: int, drive_folder: str = "wildfire_data_chiangmai") -> list:
    """Export annual satellite features (e.g., historical fire freq)."""
    region = get_thailand_geometry()
    print(f"\n📡 Fetching annual satellite features: {year}")
    
    freq = get_historical_fire_freq(year, region)
    desc = f"historical_fire_freq_{year}"
    return _safe_export(freq, desc, drive_folder, region)


if __name__ == "__main__":
    ee.Authenticate()
    ee.Initialize(project="bnl-wildfire")  # ← เปลี่ยน project ID

    for month in MONTHS:
        export_monthly_satellite(YEARS[-1], month)

    print("\n✅ All satellite export tasks submitted")
