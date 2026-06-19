import ee

def explore_era5_weather(project_id="bnl-wildfire"):
    """
    ดึงข้อมูลสภาพอากาศ (อุณหภูมิอากาศ) จาก ERA5-Land Daily Aggregated
    """
    ee.Initialize(project=project_id)
    
    point = ee.Geometry.Point([98.993, 18.793]) # เชียงใหม่
    
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterBounds(point) \
        .filterDate("2023-04-01", "2023-04-10") \
        .first()
        
    # อุณหภูมิอากาศ 2 เมตรเหนือพื้นดิน (เฉลี่ยรายวัน)
    temp = era5.select("temperature_2m")
    
    sample = temp.sampleRegions(
        collection=ee.FeatureCollection([ee.Feature(point.buffer(5000))]),
        scale=11132 # ความละเอียด ERA5 ประมาณ 11 km
    ).getInfo()
    
    print("--- สุ่มค่าพิกเซล อุณหภูมิอากาศ ERA5 ดิบ ---")
    for feature in sample['features']:
        props = feature['properties']
        t_kelvin = props.get('temperature_2m')
        if t_kelvin is not None:
            t_celsius = t_kelvin - 273.15
            print(f"Temperature (2m): {t_kelvin:.2f} K -> {t_celsius:.2f} °C")
            
    print("\nสรุป:")
    print("- ข้อมูลอุณหภูมิจาก ERA5 เก็บเป็นเคลวิน (Kelvin) ต้องลบ 273.15 เพื่อเป็นเซลเซียสเสมอ")

if __name__ == "__main__":
    explore_era5_weather(project_id="bnl-wildfire")
