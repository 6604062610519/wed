import ee

def explore_modis_lst(project_id="bnl-wildfire"):
    """
    ดึงข้อมูล LST (อุณหภูมิพื้นผิว) ดิบจาก MODIS (MOD11A2) เพื่อตรวจสอบค่าก่อนนำไปแปลงสเกล
    """
    ee.Initialize(project=project_id)
    
    point = ee.Geometry.Point([98.993, 18.793]) # เชียงใหม่
    
    modis_lst = ee.ImageCollection("MODIS/061/MOD11A2") \
        .filterBounds(point) \
        .filterDate("2023-04-01", "2023-04-30") \
        .first()
        
    lst_day = modis_lst.select("LST_Day_1km")
    
    sample = lst_day.sampleRegions(
        collection=ee.FeatureCollection([ee.Feature(point.buffer(1000))]),
        scale=1000
    ).getInfo()
    
    print("--- สุ่มค่าพิกเซล MODIS LST_Day_1km ดิบ ---")
    unique_values = set()
    for feature in sample['features']:
        val = feature['properties'].get('LST_Day_1km')
        if val is not None:
            unique_values.add(val)
            
    for val in unique_values:
        # สูตรการแปลง: ค่าจริง (Celsius) = (ค่าดิบ * 0.02) - 273.15
        temp_k = val * 0.02
        temp_c = temp_k - 273.15
        print(f"Raw Decimal: {val} | Kelvin: {temp_k:.2f} K | Celsius: {temp_c:.2f} °C")
        
    print("\nสรุป:")
    print("- ค่าดิบที่ได้จาก GEE จะอยู่ในสเกลดิจิตอล (DN)")
    print("- ต้องคูณด้วย 0.02 เพื่อเป็นองศาเคลวิน (Kelvin)")
    print("- และลบด้วย 273.15 เพื่อเป็นองศาเซลเซียส (Celsius)")

if __name__ == "__main__":
    explore_modis_lst(project_id="bnl-wildfire")
