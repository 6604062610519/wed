import ee
ee.Initialize(project="bnl-wildfire")

point = ee.Geometry.Point([98.993, 18.793])
s2_img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(point) \
    .filterDate("2023-05-01", "2023-05-31") \
    .filter(ee.Filter.gt("CLOUDY_PIXEL_PERCENTAGE", 50)) \
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)) \
    .first()

qa60 = s2_img.select("QA60")

hist = qa60.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=s2_img.geometry(),
    scale=500,
    maxPixels=1e9
).getInfo()

print(hist)
