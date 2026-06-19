#!/bin/bash

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
OUT_DIR="/Users/nonghotdog/Projects/BotandLife/web-application/model/wed/trace_output/data_exploration"
mkdir -p "$OUT_DIR"

echo "🚀 กำลังรันสคริปต์ Trace Code ทั้งหมด และเก็บผลลัพธ์ลงในโฟลเดอร์ $OUT_DIR ..."

# 1. Sentinel-2 QA60
echo "1/6 Running explore_qa60.py..."
venv/bin/python3 trace_code/data_exploration/gee_satellite/explore_qa60.py > "$OUT_DIR/explore_qa60.txt" 2>&1
echo "   ✅ บันทึกผลลัพธ์ที่ $OUT_DIR/explore_qa60.txt"

# 2. MODIS StateQA
echo "2/6 Running explore_modis_qa.py..."
venv/bin/python3 trace_code/data_exploration/gee_satellite/explore_modis_qa.py > "$OUT_DIR/explore_modis_qa.txt" 2>&1
echo "   ✅ บันทึกผลลัพธ์ที่ $OUT_DIR/explore_modis_qa.txt"

# 3. MODIS LST
echo "3/6 Running explore_modis_lst.py..."
venv/bin/python3 trace_code/data_exploration/gee_satellite/explore_modis_lst.py > "$OUT_DIR/explore_modis_lst.txt" 2>&1
echo "   ✅ บันทึกผลลัพธ์ที่ $OUT_DIR/explore_modis_lst.txt"

# 4. CHIRPS Precipitation
echo "4/6 Running explore_chirps.py..."
venv/bin/python3 trace_code/data_exploration/gee_satellite/explore_chirps.py > "$OUT_DIR/explore_chirps.txt" 2>&1
echo "   ✅ บันทึกผลลัพธ์ที่ $OUT_DIR/explore_chirps.txt"

# 5. ALOS Elevation (Terrain)
echo "5/6 Running explore_elevation.py..."
venv/bin/python3 trace_code/data_exploration/gee_terrain/explore_elevation.py > "$OUT_DIR/explore_elevation.txt" 2>&1
echo "   ✅ บันทึกผลลัพธ์ที่ $OUT_DIR/explore_elevation.txt"

# 6. WorldPop (Human)
echo "6/6 Running explore_population.py..."
venv/bin/python3 trace_code/data_exploration/gee_human/explore_population.py > "$OUT_DIR/explore_population.txt" 2>&1
echo "   ✅ บันทึกผลลัพธ์ที่ $OUT_DIR/explore_population.txt"

# 7. ERA5 Weather
echo "7/7 Running explore_era5.py..."
venv/bin/python3 trace_code/data_exploration/gee_weather/explore_era5.py > "$OUT_DIR/explore_era5.txt" 2>&1
echo "   ✅ บันทึกผลลัพธ์ที่ $OUT_DIR/explore_era5.txt"

echo ""
echo "🎉 รันเสร็จสิ้น! คุณสามารถเข้าไปดูไฟล์ผลลัพธ์ทั้งหมดได้ในโฟลเดอร์ $OUT_DIR ครับ"
