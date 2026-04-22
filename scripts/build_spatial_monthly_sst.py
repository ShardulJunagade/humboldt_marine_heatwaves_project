import os
from io import StringIO

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

LAT_MIN, LAT_MAX = -45.0, 5.0
LON_MIN, LON_MAX = -90.0, -70.0

BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv"


def fetch_year_spatial(year):
    query = (
        f"?sst[({year}-01-01T12:00:00Z):8:({year}-12-31T12:00:00Z)]"
        "[(0.0):1:(0.0)]"
        f"[({LAT_MIN}):4:({LAT_MAX})]"
        f"[({LON_MIN}):4:({LON_MAX})]"
    )
    r = requests.get(BASE_URL + query, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), skiprows=[1])
    df.columns = [c.strip().split(" ")[0] for c in df.columns]
    keep_cols = [c for c in ["time", "latitude", "longitude", "sst"] if c in df.columns]
    df = df[keep_cols].dropna()
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"])
    df["sst"] = pd.to_numeric(df["sst"], errors="coerce")
    df = df.dropna()
    return df


def main():
    out_file = os.path.join(PROC_DIR, "spatial_monthly_sst.csv")
    rows = []

    for year in range(1982, 2026):
        print(f"Fetching spatial SST for {year}...")
        try:
            ydf = fetch_year_spatial(year)
        except Exception as e:
            print(f"  Failed {year}: {e}")
            continue
        if ydf.empty:
            continue

        ydf["month"] = ydf["time"].dt.to_period("M").dt.to_timestamp()
        mdf = (
            ydf.groupby(["month", "latitude", "longitude"])['sst']
            .mean()
            .reset_index()
            .rename(columns={"month": "date"})
        )
        rows.append(mdf)

    if not rows:
        print("No spatial SST data retrieved.")
        return

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_file, index=False)
    print(f"Saved {len(out)} rows to {out_file}")


if __name__ == "__main__":
    main()
