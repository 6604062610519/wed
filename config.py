"""
config.py — Global configuration for Wildfire Prediction Data Pipeline

AOI      : Thailand (all country)
Resolution: 1 km
Period   : All 12 months of a configurable year
"""

import ee

# ─────────────────────────────────────────────
# Area of Interest — Thailand bounding box
# ─────────────────────────────────────────────
THAILAND_BBOX = {
    "min_lon": 97.5,
    "max_lon": 105.7,
    "min_lat":  5.6,
    "max_lat": 20.5,
}

def get_thailand_geometry() -> ee.Geometry:
    """Return Thailand bounding box as an Earth Engine geometry."""
    return ee.Geometry.Rectangle([
        THAILAND_BBOX["min_lon"],
        THAILAND_BBOX["min_lat"],
        THAILAND_BBOX["max_lon"],
        THAILAND_BBOX["max_lat"],
    ])

# ─────────────────────────────────────────────
# Spatial resolution
# ─────────────────────────────────────────────
RESOLUTION_M = 1000          # 1 km in metres
TARGET_CRS   = "EPSG:32647"  # WGS84 UTM Zone 47N (covers Thailand perfectly)
TARGET_CRS_WGS84 = "EPSG:4326"

# ─────────────────────────────────────────────
# Time period
# ─────────────────────────────────────────────
YEARS       = [2023]     # Default years for multi-year dataset
MONTHS      = list(range(1, 13))   # [1, 2, ..., 12]  ← all 12 months

# Rolling-window sizes for accumulated metrics
RAIN_ROLLING_DAYS   = [3, 7, 14]   # Accumulated rainfall windows (days)
DROUGHT_ROLLING_DAYS = 60           # For KBDI calculation

# ─────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────
import os
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")   # Aligned rasters (pre-norm)
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # Normalized + stacked

for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# GEE Asset / Collection IDs
# ─────────────────────────────────────────────
GEE_COLLECTIONS = {
    # Weather
    "ERA5_LAND_HOURLY":  "ECMWF/ERA5_LAND/HOURLY",
    "ERA5_DAILY":        "ECMWF/ERA5/DAILY",

    # Satellite
    "MODIS_LST_DAILY":   "MODIS/061/MOD11A1",         # 1 km daily
    "MODIS_SR_8DAY":     "MODIS/061/MOD09A1",          # 500 m 8-day surface reflectance
    "MODIS_BURN":        "MODIS/061/MCD64A1",           # 500 m monthly burned area
    "VIIRS_ACTIVE_FIRE": "FIRMS",                          # MODIS C6 active fire (confirmed in GEE)

    # Terrain (static)
    "SRTM_DEM":          "USGS/SRTMGL1_003",           # 30 m DEM
    "ESA_LANDCOVER":     "ESA/WorldCover/v200",         # 10 m land cover

    # Human factors
    "WORLDPOP":          "WorldPop/GP/100m/pop",        # 100 m population
    "GPW_POP_DENSITY":   "CIESIN/GPWv411/GPW_Population_Density",
}

# ─────────────────────────────────────────────
# Normalization parameters (will be computed
# from training data and saved here as dict)
# ─────────────────────────────────────────────
NORM_STATS_PATH = os.path.join(PROCESSED_DIR, "norm_stats.json")
