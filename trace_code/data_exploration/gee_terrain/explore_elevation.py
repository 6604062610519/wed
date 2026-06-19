import ee

def explore_elevation(project_id="bnl-wildfire"):
    """
    ดึงข้อมูลระดับความสูง (Elevation) จาก JAXA ALOS เพื่อดูค่าดิบ (เมตรจากระดับน้ำทะเล)
    และคำนวณ Slope/Aspect สดๆ เพื่อดูการเปลี่ยนแปลง
    """
    ee.Initialize(project=project_id)
    
    # ยอดดอยอินทนนท์ (สูงที่สุดในไทย) ประมาณ 2,565 เมตร
    point = ee.Geometry.Point([98.487, 18.588])
    
    alos = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").mosaic()
    elevation = alos.select("DSM").rename("elevation")
    # Reproject to UTM (meters) before calculating slope, otherwise slope will be calculated in lat/lon degrees vs meters
    elevation_proj = elevation.reproject(crs="EPSG:32647", scale=30)
    slope = ee.Terrain.slope(elevation_proj).rename("slope")
    
    combined = elevation.addBands(slope)
    
    sample = combined.sampleRegions(
        collection=ee.FeatureCollection([ee.Feature(point.buffer(500))]),
        scale=30 # ความละเอียด ALOS คือ 30 เมตร
    ).getInfo()
    
    print("--- สุ่มค่าพิกเซล ความสูง (Elevation) และ ความลาดชัน (Slope) ดิบ ---")
    print(f"พิกัดทดสอบ: ดอยอินทนนท์")
    for feature in sample['features']:
        props = feature['properties']
        elev = props.get('elevation')
        slp = props.get('slope')
        print(f"Elevation: {elev:.2f} m | Slope: {slp:.2f} degrees")
        
    print("\nสรุป:")
    print("- Elevation มีหน่วยเป็นเมตร (m)")
    print("- Slope ที่คำนวณด้วย ee.Terrain.slope() มีหน่วยเป็นองศา (degrees)")

if __name__ == "__main__":
    explore_elevation(project_id="bnl-wildfire")
