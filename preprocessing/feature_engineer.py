"""
preprocessing/feature_engineer.py

คำนวณ derived features ที่สำคัญจาก raw data:
  1. Fire Weather Index (FWI) — Canadian FWI System มาตรฐานโลก
  2. Keetch-Byram Drought Index (KBDI) — ดัชนีแล้งสะสม
  3. Seasonal Encoding (sin/cos ของ day-of-year)
  4. Rolling Statistics (ฝนสะสม, อุณหภูมิเฉลี่ย rolling)

อ้างอิง:
  - FWI: Van Wagner, C.E. (1987) Development and structure of the Canadian Forest Fire Weather Index System
  - KBDI: Keetch & Byram (1968)
"""

import numpy as np
from typing import Dict, Optional, Union


# ─────────────────────────────────────────────────────
# 1. Fire Weather Index (FWI) — Canadian System
# ─────────────────────────────────────────────────────
# FWI ประกอบด้วย sub-indices หลายตัว เราจะคำนวณ
# Fine Fuel Moisture Code (FFMC), Duff Moisture Code (DMC),
# Drought Code (DC) → รวมเป็น Initial Spread Index (ISI)
# และ Build-up Index (BUI) → Fire Weather Index (FWI)

def _ffmc(T: float, H: float, W: float, R: float,
          ffmc_prev: float = 85.0) -> float:
    """
    Fine Fuel Moisture Code (FFMC)
    คำนวณความชื้นของเชื้อเพลิงบาง (ใบไม้แห้ง, กิ่งเล็ก)

    Args:
        T: Temperature (°C)
        H: Relative Humidity (%)
        W: Wind Speed (km/h)
        R: 24-hour Rainfall (mm)
        ffmc_prev: FFMC ของวันก่อนหน้า (default 85)
    """
    mo = 147.2 * (101 - ffmc_prev) / (59.5 + ffmc_prev)

    # Rain correction
    if R > 0.5:
        rf = R - 0.5
        if mo <= 150:
            mr = mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf))
        else:
            mr = mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf)) + \
                 0.0015 * (mo - 150) ** 2 * np.sqrt(rf)
        mo = min(mr, 250)

    # Equilibrium moisture content
    ed = 0.942 * H ** 0.679 + 11 * np.exp((H - 100) / 10) + \
         0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H))
    ew = 0.618 * H ** 0.753 + 10 * np.exp((H - 100) / 10) + \
         0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H))

    if mo > ed:
        kl = 0.424 * (1 - (H / 100) ** 1.7) + 0.0694 * W ** 0.5 * (1 - (H / 100) ** 8)
        kw = kl * 0.581 * np.exp(0.0365 * T)
        m  = ed + (mo - ed) * 10 ** (-kw)
    elif mo < ew:
        kl = 0.424 * (1 - ((100 - H) / 100) ** 1.7) + \
             0.0694 * W ** 0.5 * (1 - ((100 - H) / 100) ** 8)
        kw = kl * 0.581 * np.exp(0.0365 * T)
        m  = ew - (ew - mo) * 10 ** (-kw)
    else:
        m = mo

    return 59.5 * (250 - m) / (147.2 + m)


def _dmc(T: float, H: float, R: float, month: int,
         dmc_prev: float = 6.0) -> float:
    """
    Duff Moisture Code (DMC)
    ความชื้นของชั้นอินทรีย์ในดิน (ลึก ~5 cm)
    """
    # Daylength factor (ปรับตามเส้นขนาน — ใช้ค่า equatorial สำหรับไทย)
    fl = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9,
          12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
    daylength = fl[month - 1]

    # Rain effect
    if R > 1.5:
        re = 0.92 * R - 1.27
        mo = 20 + np.exp(5.6348 - dmc_prev / 43.43)
        if dmc_prev <= 33:
            b = 100 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65:
            b = 14 - 1.3 * np.log(dmc_prev)
        else:
            b = 6.2 * np.log(dmc_prev) - 17.2
        mr = mo + 1000 * re / (48.77 + b * re)
        pr = max(244.72 - 43.43 * np.log(mr - 20), 0)
    else:
        pr = dmc_prev

    # Drying
    if T < -1.1:
        T = -1.1
    k = 1.894 * (T + 1.1) * (100 - H) * daylength * 1e-6
    return pr + 100 * k


def _dc(T: float, R: float, month: int,
        dc_prev: float = 15.0) -> float:
    """
    Drought Code (DC)
    ความแห้งของชั้นอินทรีย์ลึก (~18 cm) — สะสมหลายเดือน
    """
    lf = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8,
           6.4,  5.0,  2.4, 0.4, -1.6, -1.6]
    day_len = lf[month - 1]

    # Rain effect
    if R > 2.8:
        rd = 0.83 * R - 1.27
        qo = 800 * np.exp(-dc_prev / 400)
        qr = qo + 3.937 * rd
        dr = 400 * np.log(800 / qr)
        dc_r = max(dr, 0)
    else:
        dc_r = dc_prev

    # Drying
    v = 0.36 * (T + 2.8) + day_len
    v = max(v, 0)
    return dc_r + 0.5 * v


def compute_fwi_series(temperatures: np.ndarray,
                       humidities: np.ndarray,
                       wind_speeds_kmh: np.ndarray,
                       precipitations: np.ndarray,
                       months: np.ndarray,
                       ffmc0: float = 85.0,
                       dmc0: float = 6.0,
                       dc0: float = 15.0
                       ) -> Dict[str, np.ndarray]:
    """
    คำนวณ FWI time series จาก daily weather data (1D arrays)

    Args:
        temperatures:    (N,) daily mean temp °C
        humidities:      (N,) daily mean RH %
        wind_speeds_kmh: (N,) daily mean wind speed km/h (not m/s!)
        precipitations:  (N,) daily rainfall mm
        months:          (N,) month integer (1–12)

    Returns:
        dict มี keys: ffmc, dmc, dc, isi, bui, fwi
    """
    N = len(temperatures)
    ffmc_arr = np.zeros(N)
    dmc_arr  = np.zeros(N)
    dc_arr   = np.zeros(N)
    isi_arr  = np.zeros(N)
    bui_arr  = np.zeros(N)
    fwi_arr  = np.zeros(N)

    ffmc_prev = ffmc0
    dmc_prev  = dmc0
    dc_prev   = dc0

    for i in range(N):
        T = float(temperatures[i])
        H = float(humidities[i])
        W = float(wind_speeds_kmh[i])
        R = float(precipitations[i])
        m = int(months[i])

        # Sub-indices
        ffmc_i = _ffmc(T, H, W, R, ffmc_prev)
        dmc_i  = _dmc(T, H, R, m, dmc_prev)
        dc_i   = _dc(T, R, m, dc_prev)

        # ISI = f(FFMC, Wind)
        mo  = 147.2 * (101 - ffmc_i) / (59.5 + ffmc_i)
        ff  = np.exp(0.05039 * W)
        fm  = 91.9 * np.exp(-0.1386 * mo) * (1 + mo ** 5.31 / 4.93e7)
        isi_i = 0.208 * ff * fm

        # BUI = f(DMC, DC)
        if dmc_i <= 0.4 * dc_i:
            bui_i = 0.8 * dmc_i * dc_i / (dmc_i + 0.4 * dc_i)
        else:
            bui_i = dmc_i - (1 - 0.8 * dc_i / (dmc_i + 0.4 * dc_i)) * \
                    (0.92 + (0.0114 * dmc_i) ** 1.7)

        # FWI = f(ISI, BUI)
        if bui_i <= 80:
            fd = 0.626 * bui_i ** 0.809 + 2
        else:
            fd = 1000 / (25 + 108.64 * np.exp(-0.023 * bui_i))
        b = 0.1 * isi_i * fd
        fwi_i = np.exp(2.72 * (0.434 * np.log(b)) ** 0.647) if b > 1 else b

        ffmc_arr[i] = ffmc_i
        dmc_arr[i]  = dmc_i
        dc_arr[i]   = dc_i
        isi_arr[i]  = isi_i
        bui_arr[i]  = bui_i
        fwi_arr[i]  = fwi_i

        ffmc_prev = ffmc_i
        dmc_prev  = dmc_i
        dc_prev   = dc_i

    return {
        "ffmc": ffmc_arr.astype(np.float32),
        "dmc":  dmc_arr.astype(np.float32),
        "dc":   dc_arr.astype(np.float32),
        "isi":  isi_arr.astype(np.float32),
        "bui":  bui_arr.astype(np.float32),
        "fwi":  fwi_arr.astype(np.float32),
    }


# ─────────────────────────────────────────────────────
# 2. Keetch-Byram Drought Index (KBDI)
# ─────────────────────────────────────────────────────

def compute_kbdi_series(temperatures: np.ndarray,
                        precipitations: np.ndarray,
                        mean_annual_rain_mm: float = 1400.0,
                        kbdi0: float = 0.0) -> np.ndarray:
    """
    คำนวณ KBDI (0–800) time series

    KBDI:
      0   = ดินอุ้มน้ำเต็มที่ (ไม่มีความเสี่ยงไฟ)
      800 = แห้งแล้งสูงสุด (ความเสี่ยงไฟสูงมาก)

    Args:
        temperatures:        (N,) daily max temp °F (หรือ °C ก็ได้ — ฟังก์ชันแปลงเอง)
        precipitations:      (N,) daily rainfall mm
        mean_annual_rain_mm: ปริมาณฝนเฉลี่ยต่อปีของพื้นที่ (mm)
                             ภาคเหนือไทย ~1200-1600 mm
        kbdi0:               KBDI เริ่มต้น (0 = ดินชุ่มน้ำ)

    Returns:
        kbdi: (N,) float32 array
    """
    N    = len(temperatures)
    kbdi = np.zeros(N, dtype=np.float64)
    q    = kbdi0

    R = mean_annual_rain_mm / 25.4  # mm → inches

    for i in range(N):
        T_C = float(temperatures[i])
        T_F = T_C * 9 / 5 + 32     # °C → °F
        rain_inch = float(precipitations[i]) / 25.4  # mm → inches

        # Net rainfall correction
        if rain_inch > 0.2:
            rain_net = rain_inch - 0.2
        else:
            rain_net = 0

        # Drying factor
        drying = ((800 - q) * (0.968 * np.exp(0.0486 * T_F) - 8.3)) / \
                 (1 + 10.88 * np.exp(-0.0441 * R)) * 0.001

        q = max(q - rain_net * 100 + drying, 0)
        q = min(q, 800)
        kbdi[i] = q

    return kbdi.astype(np.float32)


# ─────────────────────────────────────────────────────
# 3. Seasonal Encoding
# ─────────────────────────────────────────────────────

def add_seasonal_features(data: Dict[str, np.ndarray],
                          day_of_year: Union[int, np.ndarray],
                          year_length: int = 365
                          ) -> Dict[str, np.ndarray]:
    """
    เพิ่ม Seasonal Encoding (sin/cos ของ day-of-year) ลงใน data dict
    """
    doy = np.asarray(day_of_year, dtype=np.float64)
    data["season_sin"] = np.sin(2 * np.pi * doy / year_length).astype(np.float32)
    data["season_cos"] = np.cos(2 * np.pi * doy / year_length).astype(np.float32)
    return data


# ─────────────────────────────────────────────────────
# 4. Rolling Statistics helpers
# ─────────────────────────────────────────────────────

def rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum ด้วย numpy (ไม่ต้อง pandas)"""
    result = np.zeros_like(arr, dtype=np.float32)
    cumsum = np.cumsum(arr, axis=0)
    result[window:] = cumsum[window:] - cumsum[:-window]
    result[:window] = cumsum[:window]  # partial windows at start
    return result


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean"""
    result = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = arr[start:i+1].mean()
    return result


# ─────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 365
    T   = np.random.uniform(20, 40, N).astype(np.float32)
    H   = np.random.uniform(20, 90, N).astype(np.float32)
    W   = np.random.uniform(0, 30, N).astype(np.float32)   # km/h
    R   = np.random.exponential(3, N).astype(np.float32)
    mon = np.array([((i // 30) % 12) + 1 for i in range(N)])

    fwi_results = compute_fwi_series(T, H, W, R, mon)
    kbdi = compute_kbdi_series(T, R)

    print("FWI stats:")
    for k, v in fwi_results.items():
        print(f"  {k}: mean={v.mean():.2f}, max={v.max():.2f}")
    print(f"KBDI: mean={kbdi.mean():.2f}, max={kbdi.max():.2f}")
