"""
data_collection/gee_weather.py

ดึงข้อมูลสภาพอากาศจาก ERA5-Land Daily Aggregated ผ่าน Google Earth Engine
Dataset: ECMWF/ERA5_LAND/DAILY_AGGR (ครอบคลุมถึงปัจจุบัน รวมปี 2023)

Features:
  - Temperature (°C)        → daily mean
  - Relative Humidity (%)   → daily mean (Magnus formula จาก dewpoint)
  - Wind Speed (m/s)        → daily mean magnitude
  - Wind sin/cos            → circular encoding
  - Precipitation (mm)      → daily total
  - Rolling Rain (mm)       → 3d / 7d / 14d accumulation

NOTE:
  ECMWF/ERA5/DAILY        → มีข้อมูลแค่ถึงปี 2020 (เก่า ไม่ใช้)
  ECMWF/ERA5_LAND/DAILY_AGGR → มีถึงปัจจุบัน ✅ (ใช้อันนี้)
"""

import ee
import os
import calendar
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    get_thailand_geometry,
    RESOLUTION_M, TARGET_CRS_WGS84,
    YEARS, MONTHS, RAIN_ROLLING_DAYS,
)

# ✅ ERA5-Land Daily Aggregated — มีข้อมูลครอบคลุมปี 2023
ERA5_DAILY_AGGR_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"

# Band names ใน ERA5_LAND/DAILY_AGGR
ERA5_BANDS = {
    "temp":   "temperature_2m",                # K (daily mean)
    "dewpt":  "dewpoint_temperature_2m",       # K (daily mean)
    "u10":    "u_component_of_wind_10m",       # m/s (daily mean)
    "v10":    "v_component_of_wind_10m",       # m/s (daily mean)
    "precip": "total_precipitation_sum",       # m (daily sum) ← ชื่อต่างจาก ERA5/DAILY
    "soil_w": "volumetric_soil_water_layer_1", # m3/m3 (daily mean soil moisture)
}


# ─────────────────────────────────────────────────────
# Export helper
# ─────────────────────────────────────────────────────

def _safe_export(image: ee.Image, description: str, folder: str,
                 region: ee.Geometry, scale: int = RESOLUTION_M) -> list:
    """GEE lazy eval — errors only occur at task.start(). Catches and logs them."""
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
    try:
        task.start()
        print(f"  ✅ Export task started: {description}")
        return [task]
    except Exception as e:
        print(f"  ⚠️  Export FAILED [{description}]: {e}")
        return []


# ─────────────────────────────────────────────────────
# Per-image processing
# ─────────────────────────────────────────────────────

def _process_era5_image(img: ee.Image) -> ee.Image:
    """
    แปลง ERA5-Land Daily Aggregated image → weather bands พร้อมใช้งาน
    Output bands: temperature, relative_humidity, wind_u, wind_v,
                  wind_speed, wind_dir_sin, wind_dir_cos, precipitation
    """
    PI = 3.14159265358979

    # Temperature K → °C
    T_K = img.select(ERA5_BANDS["temp"])
    T_C = T_K.subtract(273.15).rename("temperature")

    # Relative Humidity จาก dewpoint (Magnus approximation)
    Td_K = img.select(ERA5_BANDS["dewpt"])
    T_C2 = T_K.subtract(273.15)
    Td_C = Td_K.subtract(273.15)
    exp_td = Td_C.multiply(17.625).divide(Td_C.add(243.04)).exp()
    exp_t  = T_C2.multiply(17.625).divide(T_C2.add(243.04)).exp()
    rh = exp_td.divide(exp_t).multiply(100).clamp(0, 100).rename("relative_humidity")

    # Wind
    u10 = img.select(ERA5_BANDS["u10"]).rename("wind_u")
    v10 = img.select(ERA5_BANDS["v10"]).rename("wind_v")
    wind_speed = u10.pow(2).add(v10.pow(2)).sqrt().rename("wind_speed")

    # Wind direction → sin/cos (circular encoding)
    wind_dir_deg = u10.atan2(v10).multiply(180.0 / PI).mod(360)
    wind_sin = wind_dir_deg.multiply(PI / 180.0).sin().rename("wind_dir_sin")
    wind_cos = wind_dir_deg.multiply(PI / 180.0).cos().rename("wind_dir_cos")

    # Precipitation: m → mm (total_precipitation_sum คือ daily sum แล้ว)
    precip = img.select(ERA5_BANDS["precip"]).multiply(1000).rename("precipitation")

    # Soil Moisture (Volumetric Soil Water Layer 1)
    soil_moisture = img.select(ERA5_BANDS["soil_w"]).rename("soil_moisture")

    return T_C.addBands([rh, u10, v10, wind_speed, wind_sin, wind_cos, precip, soil_moisture])


# ─────────────────────────────────────────────────────
# Monthly weather image
# ─────────────────────────────────────────────────────

def get_monthly_weather(year: int, month: int,
                        region: Optional[ee.Geometry] = None) -> ee.Image:
    """
    ดึง ERA5-Land Daily Aggregated แล้ว aggregate เป็น monthly mean + rolling rainfall
    Output bands: temperature, relative_humidity, wind_u, wind_v,
                  wind_speed, wind_dir_sin, wind_dir_cos, precipitation,
                  rain_3d, rain_7d, rain_14d
    """
    if region is None:
        region = get_thailand_geometry()

    start = f"{year}-{month:02d}-01"
    end_day = calendar.monthrange(year, month)[1]
    end   = f"{year}-{month:02d}-{end_day:02d}"

    era5_col = (
        ee.ImageCollection(ERA5_DAILY_AGGR_ID)
          .filterDate(start, end)
          .filterBounds(region)
          .map(_process_era5_image)
    )

    # Monthly mean ของ weather bands
    month_mean = era5_col.mean().clip(region)

    # Rolling accumulated rainfall (เทียบกับ ต้นเดือน ไม่ใช่สิ้นเดือน)
    rolling_bands = []
    for w in RAIN_ROLLING_DAYS:
        roll_start = ee.Date(start).advance(-w, "day")
        rain_w = (
            ee.ImageCollection(ERA5_DAILY_AGGR_ID)
              .filterDate(roll_start, ee.Date(start).advance(1, "day"))
              .filterBounds(region)
              .select(ERA5_BANDS["precip"])
              .sum()
              .multiply(1000)       # m → mm
              .rename(f"rain_{w}d")
              .clip(region)
        )
        rolling_bands.append(rain_w)

    result = month_mean
    for band in rolling_bands:
        result = result.addBands(band)

    return result


# ─────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────

def export_monthly_weather(year: int, month: int,
                           drive_folder: str = "wildfire_data_chiangmai") -> list:
    """Export monthly weather stack to Google Drive."""
    region = get_thailand_geometry()
    print(f"\n📡 Fetching weather: {year}-{month:02d}")

    weather = get_monthly_weather(year, month, region)
    desc = f"weather_{year}_{month:02d}"
    return _safe_export(weather, desc, drive_folder, region)


# ─────────────────────────────────────────────────────
# Standalone
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    ee.Authenticate()
    ee.Initialize(project="bnl-wildfire")  # ← เปลี่ยน project ID

    for month in MONTHS:
        export_monthly_weather(YEARS[-1], month)

    print("\n✅ All weather export tasks submitted")
