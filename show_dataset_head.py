import os
import json
import numpy as np
from pathlib import Path

# ==========================================
# ตั้งค่าพาธไฟล์
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

FEAT_PATH = os.path.join(PROCESSED_DIR, "features_2023_01.npy")
TARGET_PATH = os.path.join(PROCESSED_DIR, "target_2023_01.npy")
STATS_PATH = os.path.join(PROCESSED_DIR, "norm_stats.json")

# ==========================================
# รายชื่อ 27 Features ตามลำดับจริงในชุดข้อมูล
# ==========================================
ACTUAL_FEATURES = [
    "temperature", "relative_humidity", "wind_speed", "wind_u", "wind_v",
    "wind_dir_sin", "wind_dir_cos", "precipitation", "rain_3d", "rain_7d", "rain_14d",
    "ndmi_s2", "ndvi_s2", "lst_celsius", "burned_binary", "burned_count_3m",
    "elevation", "slope", "aspect_sin", "aspect_cos", "fire_risk_lc",
    "pop_density", "dist_to_road", "dist_to_settlement", "night_light",
    "season_sin", "season_cos"
]

def inverse_normalize(normalized_dict, stats_dict):
    """ฟังก์ชันแปลงข้อมูลกลับไปเป็นค่าตั้งต้น (Before Normalize)"""
    raw_dict = {}
    
    for feat_name, norm_val in normalized_dict.items():
        # ถ้าเป็น sin/cos ของวงกลม (aspect, wind_dir, season)
        if feat_name.endswith('_sin') or feat_name.endswith('_cos'):
            base_name = feat_name.replace('_sin', '').replace('_cos', '')
            if base_name + '_sin' in normalized_dict and base_name + '_cos' in normalized_dict:
                sin_v = normalized_dict[base_name + '_sin']
                cos_v = normalized_dict[base_name + '_cos']
                # หาค่ามุมองศากลับมา (0-360)
                angle_rad = np.arctan2(sin_v, cos_v)
                angle_deg = (np.degrees(angle_rad) + 360) % 360
                raw_dict[base_name] = angle_deg
            continue
            
        # ดึงวิธีการ Normalize ที่เคยใช้
        feat_stats = stats_dict.get(feat_name, {"strategy": "none"})
        strategy = feat_stats["strategy"]
        
        if strategy == "minmax":
            raw_dict[feat_name] = norm_val * (feat_stats["max"] - feat_stats["min"]) + feat_stats["min"]
        elif strategy == "zscore":
            raw_dict[feat_name] = norm_val * feat_stats["std"] + feat_stats["mean"]
        elif strategy == "log1p_minmax":
            log_val = norm_val * (feat_stats["log_max"] - feat_stats["log_min"]) + feat_stats["log_min"]
            raw_dict[feat_name] = np.expm1(log_val) # exp(x) - 1
        else:
            raw_dict[feat_name] = norm_val
            
    return raw_dict

def main():
    if not os.path.exists(FEAT_PATH):
        print(f"ไม่พบไฟล์ {FEAT_PATH} กรุณาตรวจสอบว่ามีข้อมูล processed แล้ว")
        return

    # โหลดข้อมูล
    print("⏳ กำลังโหลดข้อมูล (อาจใช้เวลาสักครู่เนื่องจากไฟล์ใหญ่)...")
    features = np.load(FEAT_PATH)
    targets = np.load(TARGET_PATH)
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
        
    H, W, C = features.shape
    X = features.reshape(-1, C)
    y = targets.reshape(-1)

    # หาตำแหน่งพิกเซลที่เกิดไฟ (Target=1) และไม่เกิดไฟ (Target=0)
    fire_idx = np.where(y > 0)[0]
    nofire_idx = np.where(y == 0)[0]

    # เลือกตัวอย่างมา 6 แถว (ไม่เกิดไฟ 3 แถว, เกิดไฟ 3 แถว)
    selected_idx = np.concatenate((nofire_idx[:3], fire_idx[:3]))
    X_head_norm = X[selected_idx]
    y_head = y[selected_idx]

    print("\n" + "="*80)
    print("✨ 1. แสดงตัวอย่างข้อมูล 'หลัง Normalize' (สิ่งที่ Model เห็นจริงๆ)")
    print("="*80)
    
    # พิมพ์ Header
    header = f"{'Type':^12} | " + " | ".join([f"{col[:8]:^8}" for col in ACTUAL_FEATURES]) + " || Target"
    print(header)
    print("-" * len(header))
    
    for i, row in enumerate(X_head_norm):
        row_type = "No-Fire" if i < 3 else "Fire"
        row_str = f"{row_type:^12} | " + " | ".join([f"{v:8.4f}" for v in row]) + f" || {y_head[i]:.0f}"
        print(row_str)

    print("\n" + "="*80)
    print("🌱 2. แสดงตัวอย่างข้อมูล 'ก่อน Normalize' (ค่าออริจินัลทางกายภาพ)")
    print("="*80)
    
    # แปลงกลับ (Inverse Transform) ทีละแถว
    raw_rows = []
    for row in X_head_norm:
        # สร้าง Dict เพื่อส่งเข้าฟังก์ชัน
        row_dict = {feat: row[j] for j, feat in enumerate(ACTUAL_FEATURES)}
        raw_dict = inverse_normalize(row_dict, stats)
        raw_rows.append(raw_dict)
        
    # จัดคอลัมน์ใหม่สำหรับ raw data (รวม sin/cos กลับเป็นมุม)
    raw_columns = list(raw_rows[0].keys())
    
    # พิมพ์ Header
    header_raw = f"{'Type':^12} | " + " | ".join([f"{col[:10]:^10}" for col in raw_columns]) + " || Target"
    print(header_raw)
    print("-" * len(header_raw))
    
    for i, row_dict in enumerate(raw_rows):
        row_type = "No-Fire" if i < 3 else "Fire"
        row_str = f"{row_type:^12} | " + " | ".join([f"{row_dict[col]:10.2f}" for col in raw_columns]) + f" || {y_head[i]:.0f}"
        print(row_str)

    # บันทึกข้อมูล Raw ลงไฟล์ CSV
    import csv
    csv_filename = "raw_dataset_sample.csv"
    csv_path = os.path.join(BASE_DIR, csv_filename)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Type'] + raw_columns + ['Target']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, row_dict in enumerate(raw_rows):
            row_to_write = row_dict.copy()
            row_to_write['Type'] = "No-Fire" if i < 3 else "Fire"
            row_to_write['Target'] = int(y_head[i])
            writer.writerow(row_to_write)
    print(f"\n💾 บันทึกตัวอย่างข้อมูลก่อน Normalize (Raw) ลงในไฟล์: {csv_filename} เรียบร้อยแล้ว")

if __name__ == "__main__":
    main()
