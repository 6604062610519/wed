import ee

def explore_population(project_id="bnl-wildfire"):
    """
    ดึงข้อมูลความหนาแน่นประชากรจาก WorldPop เพื่อดูค่าจำนวนคนต่อพิกเซล
    """
    ee.Initialize(project=project_id)
    
    # ใจกลางเมืองเชียงใหม่ (คนน่าจะเยอะ)
    point = ee.Geometry.Point([98.985, 18.788])
    
    # ข้อมูลปี 2020 (ดึงเฉพาะประเทศไทยผ่าน country code THA)
    pop_img = ee.ImageCollection("WorldPop/GP/100m/pop") \
        .filter(ee.Filter.eq("country", "THA")) \
        .filterDate("2020-01-01", "2020-12-31") \
        .first()
        
    pop = pop_img.select("population")
    
    sample = pop.sampleRegions(
        collection=ee.FeatureCollection([ee.Feature(point.buffer(300))]),
        scale=100 # ความละเอียด 100 เมตร
    ).getInfo()
    
    print("--- สุ่มค่าพิกเซล ความหนาแน่นประชากร (WorldPop) ดิบ ---")
    for feature in sample['features']:
        props = feature['properties']
        p = props.get('population')
        if p is not None:
            print(f"Population: {p:.2f} คน / พิกเซล (100x100m)")
        
    print("\nสรุป:")
    print("- ค่าที่ได้คือประมาณการจำนวนประชากรในพื้นที่ 1 พิกเซล (ขนาด 100x100 เมตร)")

if __name__ == "__main__":
    explore_population(project_id="bnl-wildfire")
