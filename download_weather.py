"""
Download historical hourly weather AND air quality data for HCMC from Open-Meteo API.
Free, no API key needed.
"""
import pandas as pd
import requests
import time
import os

# HCMC coordinates
LATITUDE = 10.8231
LONGITUDE = 106.6297

# Date range - extended to 2026 for full coverage
START_DATE = "2018-01-01"
END_DATE = "2026-01-18"

OUTPUT_FILE = r"data\new\hcmc_weather_and_aqi.csv"

# Tách riêng 2 danh sách biến
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover"
]

AQI_VARS = [
    "pm2_5", 
    "pm10", 
    "carbon_monoxide", 
    "nitrogen_dioxide", 
    "sulphur_dioxide", 
    "ozone"
]

def download_data():
    print("Downloading HCMC Weather & AQI data from Open-Meteo...")
    print(f"  Location: {LATITUDE}°N, {LONGITUDE}°E")
    print(f"  Period: {START_DATE} to {END_DATE}")

    all_dfs = []
    for year in range(2018, 2027):
        sd = f"{year}-01-01"
        ed = min(f"{year}-12-31", END_DATE)
        print(f"  Fetching {year}...")

        # 1. Kéo dữ liệu THỜI TIẾT
        w_url = "https://archive-api.open-meteo.com/v1/archive"
        w_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": sd,
            "end_date": ed,
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "Asia/Ho_Chi_Minh",
        }
        w_resp = requests.get(w_url, params=w_params, timeout=60)
        w_resp.raise_for_status()
        df_w = pd.DataFrame(w_resp.json()["hourly"])

        # 2. Kéo dữ liệu CHẤT LƯỢNG KHÔNG KHÍ (PM2.5, PM10...)
        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": sd,
            "end_date": ed,
            "hourly": ",".join(AQI_VARS),
            "timezone": "Asia/Ho_Chi_Minh",
        }
        aq_resp = requests.get(aq_url, params=aq_params, timeout=60)
        aq_resp.raise_for_status()
        df_aq = pd.DataFrame(aq_resp.json()["hourly"])

        # 3. Gộp 2 bảng lại theo cột "time"
        df_merged = pd.merge(df_w, df_aq, on="time", how="outer")
        all_dfs.append(df_merged)
        
        time.sleep(1)  # Tôn trọng API, nghỉ 1s

    # Xử lý cục data tổng
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.rename(columns={"time": "datetime"}, inplace=True)
    df_all["datetime"] = pd.to_datetime(df_all["datetime"])
    df_all = df_all.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    
    # Lưu file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_all.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSaved {len(df_all)} rows to {OUTPUT_FILE}")
    print(f"Columns: {df_all.columns.tolist()}")
    print(f"Date range: {df_all['datetime'].min()} to {df_all['datetime'].max()}")
    print(f"Missing values:\n{df_all.isna().sum()}")

if __name__ == "__main__":
    download_data()