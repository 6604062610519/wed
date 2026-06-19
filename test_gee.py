import ee
import sys
import os

sys.path.append('.')
from config import get_thailand_geometry

def main():
    ee.Initialize(project='bnl-wildfire')
    region = get_thailand_geometry()
    
    try:
        # Check MODIS MCD15A3H
        print("Checking MODIS MCD15A3H LAI...")
        lai_col = ee.ImageCollection("MODIS/061/MCD15A3H").filterDate('2018-01-01', '2018-01-31').select("Lai")
        lai_monthly = lai_col.map(lambda img: img.multiply(0.1).toFloat().rename("lai")).mean().unmask(0).clip(region)
        print("LAI bands:", lai_monthly.bandNames().getInfo())
        
        task = ee.batch.Export.image.toDrive(
            image=lai_monthly,
            description="lai_test",
            folder="wildfire_data_chiangmai",
            fileNamePrefix="lai_test",
            region=region,
            scale=1000,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )
        print("Can create task:", task)
        
    except Exception as e:
        print("Error during LAI:", e)

if __name__ == "__main__":
    main()
