import ee

def explore_chirps_precip(project_id="bnl-wildfire"):
    """
    ดึงข้อมูลปริมาณน้ำฝนรายวัน (CHIRPS Daily) มาดูค่าดิบ (มิลลิเมตร)
    """
    ee.Initialize(project=project_id)
    
    point = ee.Geometry.Point([98.993, 18.793]) # เชียงใหม่
    
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(point) \
        .filterDate("2023-08-01", "2023-08-10") \
        .first()
        
    precip = chirps.select("precipitation")
    
    sample = precip.sampleRegions(
        collection=ee.FeatureCollection([ee.Feature(point.buffer(5000))]),
        scale=5566 # ความละเอียดประมาณ 5 กิโลเมตร
    ).getInfo()
    
    print("--- สุ่มค่าพิกเซล ปริมาณน้ำฝน CHIRPS ดิบ ---")
    unique_values = set()
    for feature in sample['features']:
        val = feature['properties'].get('precipitation')
        if val is not None:
            unique_values.add(val)
            
    for val in sorted(unique_values):
        print(f"Precipitation: {val:.2f} mm/day")
        
    print("\nสรุป:")
    print("- ข้อมูลน้ำฝนจาก CHIRPS มีหน่วยเป็นมิลลิเมตร/วัน (mm/day) อยู่แล้ว สามารถนำไปรวม (Sum) รายเดือนได้เลย")

if __name__ == "__main__":
    explore_chirps_precip(project_id="bnl-wildfire")
