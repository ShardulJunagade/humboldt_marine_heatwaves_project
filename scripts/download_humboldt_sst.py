import os
import time
import requests
import pandas as pd
from io import StringIO
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Bounding Box for Humboldt Current
LAT_MIN, LAT_MAX = -45.0, 5.0
LON_MIN, LON_MAX = -90.0, -70.0

# ERDDAP URL
BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv"

def build_erddap_url(year, lat_min, lat_max, lon_min, lon_max):
    # Time stride = 1 (Daily)
    # Spatial stride = 4 (Every 1 degree to keep download size reasonable but representative)
    time_start = f"{year}-01-01T12:00:00Z"
    time_end = f"{year}-12-31T12:00:00Z"
    
    query = (
        f"?sst[({time_start}):1:({time_end})]"
        f"[(0.0):1:(0.0)]"
        f"[({lat_min}):4:({lat_max})]"
        f"[({lon_min}):4:({lon_max})]"
    )
    return BASE_URL + query

def fetch_year(year, retries=3):
    file_path = os.path.join(RAW_DATA_DIR, f"humboldt_sst_daily_{year}.csv")
    
    if os.path.exists(file_path):
        print(f"Skipping {year}, already downloaded.")
        return
    
    url = build_erddap_url(year, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    print(f"Downloading data for {year}...", end=" ", flush=True)
    
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                # Read CSV, skip the second row which contains unit labels
                df = pd.read_csv(StringIO(resp.text), skiprows=[1])
                
                # Cleanup columns
                df.columns = [c.strip().split(" ")[0] for c in df.columns]
                
                # Keep only time and sst, drop NAs (land pixels)
                if 'time' in df.columns and 'sst' in df.columns:
                    df = df[['time', 'sst']].dropna()
                    
                    if not df.empty:
                        # Convert to datetime and mean spatial aggregation
                        df['time'] = pd.to_datetime(df['time']).dt.date
                        df['sst'] = pd.to_numeric(df['sst'], errors='coerce')
                        
                        daily_mean = df.groupby('time')['sst'].mean().reset_index()
                        daily_mean.to_csv(file_path, index=False)
                        print(f"Success! Saved {len(daily_mean)} days.")
                        return
                    else:
                        print("Empty dataframe after dropping missing values.")
                else:
                    print("Missing expected columns.")
            else:
                print(f"HTTP {resp.status_code} on attempt {attempt+1}")
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
        
        time.sleep(2)
        
    print(f"Failed to fetch data for {year} after {retries} attempts.")


def main():
    print(f"--- Downloading Initial Daily SST for Humboldt Current ({LAT_MIN} to {LAT_MAX} Lat, {LON_MIN} to {LON_MAX} Lon) ---")
    years = range(2021, 2026)
    for y in years:
        fetch_year(y)
        
    print("\nDownload complete. Data saved to:", RAW_DATA_DIR)

if __name__ == "__main__":
    main()
