import ee

def explore_modis_stateqa(project_id="bnl-wildfire"):
    """
    ดึงข้อมูลค่า StateQA ดิบของ MODIS (MOD09A1) เพื่อตรวจสอบการทำ Cloud Mask
    """
    ee.Initialize(project=project_id)
    
    point = ee.Geometry.Point([98.993, 18.793]) # เชียงใหม่
    
    # ดึงภาพ MODIS Surface Reflectance 8-Day
    modis_img = ee.ImageCollection("MODIS/061/MOD09A1") \
        .filterBounds(point) \
        .filterDate("2023-05-01", "2023-05-31") \
        .first()
        
    qa = modis_img.select("StateQA")
    
    sample = qa.sampleRegions(
        collection=ee.FeatureCollection([ee.Feature(point.buffer(500))]),
        scale=500 
    ).getInfo()
    
    print("--- สุ่มค่าพิกเซล MODIS StateQA ดิบ ---")
    unique_values = set()
    for feature in sample['features']:
        val = feature['properties'].get('StateQA')
        if val is not None:
            unique_values.add(val)
            
    for val in unique_values:
        binary_str = bin(val)
        cloud_state = val & 3 # บิต 0-1
        
        state_desc = "Clear"
        if cloud_state == 1:
            state_desc = "Cloudy"
        elif cloud_state == 2:
            state_desc = "Mixed"
        elif cloud_state == 3:
            state_desc = "Not set, assumed clear"
            
        print(f"Decimal: {val:5d} | Binary: {binary_str:>16s} | "
              f"Bit 0-1 (Cloud State): {cloud_state} -> {state_desc}")
              
    print("\nสรุป:")
    print("- ในฟังก์ชัน _mask_modis_clouds เราใช้ qa.bitwiseAnd(3).neq(0)")
    print("- แปลว่าถ้าบิต 0-1 ไม่เท่ากับ 0 (Clear) จะถือว่าเป็นเมฆทั้งหมด (Cloudy หรือ Mixed)")

if __name__ == "__main__":
    explore_modis_stateqa(project_id="bnl-wildfire")
