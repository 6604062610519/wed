"""
preprocessing/aligner.py

Align และ Resample raster layers ทั้งหมดให้:
  - CRS เดียวกัน  (EPSG:32647 UTM47N หรือ EPSG:4326)
  - Resolution เดียวกัน (1 km)
  - Extent เดียวกัน (Thailand bounding box)
  - NoData → NaN

ใช้ rasterio + numpy (ไม่ต้อง GDAL CLI)
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import THAILAND_BBOX, RESOLUTION_M, TARGET_CRS_WGS84


# ─────────────────────────────────────────────────────
# Reference Grid (1 km over Thailand in WGS84)
# ─────────────────────────────────────────────────────

def get_reference_grid(
    resolution_deg: float = 0.00899,  # ~1 km in degrees at Thailand latitude
    bbox: dict = THAILAND_BBOX,
) -> Tuple[rasterio.transform.Affine, int, int, CRS]:
    """
    คำนวณ reference grid สำหรับ Thailand ที่ 1 km resolution

    Returns:
        (transform, width, height, crs)
    """
    crs = CRS.from_epsg(4326)
    left   = bbox["min_lon"]
    right  = bbox["max_lon"]
    bottom = bbox["min_lat"]
    top    = bbox["max_lat"]

    width  = int((right - left)  / resolution_deg)
    height = int((top - bottom) / resolution_deg)

    transform = from_bounds(left, bottom, right, top, width, height)
    return transform, width, height, crs


REFERENCE_TRANSFORM, REFERENCE_WIDTH, REFERENCE_HEIGHT, REFERENCE_CRS = get_reference_grid()


# ─────────────────────────────────────────────────────
# Core: Reproject and Resample single GeoTIFF
# ─────────────────────────────────────────────────────

def align_raster(src_path: str,
                 dst_path: Optional[str] = None,
                 resampling_method: Resampling = Resampling.bilinear,
                 target_transform: rasterio.transform.Affine = REFERENCE_TRANSFORM,
                 target_width: int = REFERENCE_WIDTH,
                 target_height: int = REFERENCE_HEIGHT,
                 target_crs: CRS = REFERENCE_CRS,
                 nodata_fill: float = np.nan,
                 ) -> np.ndarray:
    """
    Reproject และ Resample GeoTIFF ไปยัง reference grid ของ Thailand

    Args:
        src_path:   Path ของ input GeoTIFF
        dst_path:   Path สำหรับ save output (None = return array เท่านั้น)
        resampling_method: วิธี resample
                   - bilinear: สำหรับ continuous data (T, RH, NDMI)
                   - nearest:  สำหรับ categorical data (Land Cover)
                   - sum:      สำหรับ count data (fire occurrence count)

    Returns:
        numpy array (n_bands, height, width) aligned to reference grid
    """
    with rasterio.open(src_path) as src:
        n_bands = src.count
        src_nodata = src.nodata

        # Compute transform ปลายทาง
        data = np.full(
            (n_bands, target_height, target_width),
            nodata_fill,
            dtype=np.float32
        )

        for band_idx in range(1, n_bands + 1):
            src_data = src.read(band_idx).astype(np.float32)

            # Replace NoData with NaN
            if src_nodata is not None:
                src_data = np.where(src_data == src_nodata, np.nan, src_data)

            out_band = np.full((target_height, target_width), nodata_fill, dtype=np.float32)

            reproject(
                source=src_data,
                destination=out_band,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=resampling_method,
                src_nodata=np.nan,
            )
            data[band_idx - 1] = out_band

    # Save if path provided
    if dst_path is not None:
        _save_aligned(data, dst_path, target_transform, target_crs,
                      band_names=_infer_band_names(src_path, n_bands))

    return data  # (n_bands, H, W)


def _save_aligned(data: np.ndarray,
                  dst_path: str,
                  transform: rasterio.transform.Affine,
                  crs: CRS,
                  band_names: Optional[List[str]] = None):
    """บันทึก aligned raster เป็น GeoTIFF"""
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    n_bands, H, W = data.shape

    with rasterio.open(
        dst_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=n_bands,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="lzw",          # Compression เพื่อลดขนาดไฟล์
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        for i in range(n_bands):
            dst.write(data[i], i + 1)
            if band_names and i < len(band_names):
                dst.update_tags(i + 1, name=band_names[i])

    print(f"  💾 Saved aligned raster: {dst_path}")


def _infer_band_names(src_path: str, n_bands: int) -> List[str]:
    """Infer band names จากชื่อไฟล์"""
    stem = Path(src_path).stem
    if n_bands == 1:
        return [stem]
    return [f"{stem}_b{i+1}" for i in range(n_bands)]


# ─────────────────────────────────────────────────────
# Batch alignment สำหรับทุก layer
# ─────────────────────────────────────────────────────

def align_all_layers(raw_dir: str,
                     interim_dir: str,
                     categorical_keywords: List[str] = ["land_cover", "burned_binary"],
                     ) -> Dict[str, str]:
    """
    Align ทุก GeoTIFF ใน raw_dir → interim_dir

    Args:
        raw_dir:    โฟลเดอร์ที่มี raw GeoTIFFs จาก GEE
        interim_dir: โฟลเดอร์ output (aligned, same grid)
        categorical_keywords: ชื่อ feature ที่ต้องใช้ nearest-neighbor resampling

    Returns:
        dict: {feature_name: aligned_file_path}
    """
    raw_path    = Path(raw_dir)
    interim_path = Path(interim_dir)
    aligned = {}

    tif_files = list(raw_path.glob("**/*.tif")) + list(raw_path.glob("**/*.TIF"))

    for tif in sorted(tif_files):
        stem = tif.stem
        is_categorical = any(kw in stem.lower() for kw in categorical_keywords)
        method = Resampling.nearest if is_categorical else Resampling.bilinear

        out_path = interim_path / tif.relative_to(raw_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n🔄 Aligning: {tif.name}")
        try:
            align_raster(
                src_path=str(tif),
                dst_path=str(out_path),
                resampling_method=method,
            )
            aligned[stem] = str(out_path)
        except Exception as e:
            warnings.warn(f"Failed to align {tif.name}: {e}")

    return aligned


# ─────────────────────────────────────────────────────
# Read aligned raster back to numpy
# ─────────────────────────────────────────────────────

def read_raster(path: str) -> Tuple[np.ndarray, rasterio.transform.Affine, CRS]:
    """
    อ่าน GeoTIFF → numpy array
    Returns: (data, transform, crs)
             data shape: (n_bands, H, W) ถ้า n_bands > 1
                         (H, W) ถ้า n_bands == 1
    """
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        # Replace NoData
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        transform = src.transform
        crs = src.crs

    if data.shape[0] == 1:
        data = data[0]   # squeeze single-band → (H, W)
    return data, transform, crs


def stack_aligned_rasters(paths: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    """
    Stack หลาย aligned rasters เป็น single array

    Args:
        paths: dict {feature_name: file_path}

    Returns:
        (stacked, feature_names)
        stacked shape: (n_features, H, W)
    """
    arrays = []
    names  = []

    for name, path in paths.items():
        data, _, _ = read_raster(path)
        if data.ndim == 2:
            arrays.append(data)
            names.append(name)
        else:
            for i in range(data.shape[0]):
                arrays.append(data[i])
                names.append(f"{name}_b{i+1}")

    return np.stack(arrays, axis=0), names


# ─────────────────────────────────────────────────────
# NaN Imputation
# ─────────────────────────────────────────────────────

def fill_nan(arr: np.ndarray,
             method: str = "mean",
             fill_value: float = 0.0) -> np.ndarray:
    """
    เติม NaN ด้วยวิธีต่างๆ

    Args:
        method: 'mean', 'median', 'zero', 'constant'
    """
    result = arr.copy()
    nan_mask = np.isnan(result)

    if not nan_mask.any():
        return result

    if method == "mean":
        fill = np.nanmean(result)
    elif method == "median":
        fill = np.nanmedian(result)
    elif method == "zero":
        fill = 0.0
    else:
        fill = fill_value

    result[nan_mask] = fill
    print(f"  📝 Filled {nan_mask.sum()} NaN pixels with {method}={fill:.4f}")
    return result


# ─────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Reference Grid (Thailand 1km):")
    print(f"  Width : {REFERENCE_WIDTH}  pixels")
    print(f"  Height: {REFERENCE_HEIGHT} pixels")
    print(f"  CRS   : {REFERENCE_CRS}")
    print(f"  Transform: {REFERENCE_TRANSFORM}")
