import ee

def explore_raw_qa60(project_id="bnl-wildfire"):
    """
    ดึงข้อมูลค่า QA60 ดิบ (Raw data) ของ Sentinel-2 จากจุดตัวอย่างเพื่อดูค่าที่ได้
    ก่อนที่จะทำการ mask cloud
    """
    ee.Initialize(project=project_id)
    
    # กำหนดจุดทดสอบ (เช่น บริเวณเชียงใหม่)
    point = ee.Geometry.Point([98.993, 18.793])
    
    # ค้นหาภาพ Sentinel-2 ในช่วงเวลา 1 เดือน แล้วเอาค่า "สูงสุด" (Max) ของทุกภาพมารวมกัน
    # วิธีนี้การันตีว่าเราจะเจอพิกเซลที่มีเมฆแน่นอน (เพราะวันไหนที่เมฆมา ค่าจะเป็น 1024/2048/3072 ซึ่งมากกว่า 0)
    s2_img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(point) \
        .filterDate("2023-05-01", "2023-05-31") \
        .max()
        
        
    qa60 = s2_img.select("QA60")
    scl = s2_img.select("SCL") # Band SCL คือ Scene Classification ที่แม่นยำกว่า
    
    # ดู Histogram ของทั้ง QA60 และ SCL
    hist_qa60 = qa60.reduceRegion(ee.Reducer.frequencyHistogram(), point.buffer(20000), 60, maxPixels=1e9).getInfo()
    hist_scl = scl.reduceRegion(ee.Reducer.frequencyHistogram(), point.buffer(20000), 60, maxPixels=1e9).getInfo()
    
    qa_counts = hist_qa60.get('QA60', {})
    scl_counts = hist_scl.get('SCL', {})
    
    print("--- 1. สรุปพิกเซล QA60 ดิบ ---")
    for val_str, count in qa_counts.items():
        print(f"QA60 Value: {val_str:>4} | เจอทั้งหมด: {int(count):8d} พิกเซล")
        
    print("\n--- 2. สรุปพิกเซล SCL (Scene Classification) ดิบ ---")
    scl_dict = {
        3: "Cloud Shadows", 4: "Vegetation", 5: "Bare Soils",
        6: "Water", 8: "Clouds (Medium Prob)", 9: "Clouds (High Prob)", 10: "Cirrus"
    }
    for val_str, count in sorted(scl_counts.items(), key=lambda x: int(float(x[0]))):
        val = int(float(val_str))
        desc = scl_dict.get(val, "Other/Clear")
        print(f"SCL Value: {val:>2} ({desc:<20}) | เจอทั้งหมด: {int(count):8d} พิกเซล")
        
    print("\nสรุปความจริงที่โหดร้าย:")
    print("- QA60 ของ Sentinel-2 มักจะมีปัญหา (Bug จากทาง ESA/Google) ที่มักจะให้ค่า 0 ล้วนๆ ทั้งๆ ที่มีเมฆ")
    print("- นักวิจัยส่วนใหญ่จึงเปลี่ยนไปใช้แบนด์ SCL (Scene Classification) ในการทำ Cloud Mask แทนครับ!")
    print("- บิตที่ 10 (ค่า 1024) = เมฆหนา (Opaque clouds)")
    print("- บิตที่ 11 (ค่า 2048) = เมฆเซอรัส (Cirrus clouds)")

if __name__ == "__main__":
    # ใส่ Project ID ของคุณตรงนี้
    explore_raw_qa60(project_id="bnl-wildfire")
