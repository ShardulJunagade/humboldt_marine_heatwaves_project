import os
from io import StringIO

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# Humboldt domain
LAT_MIN, LAT_MAX = -45.0, 5.0
LON_MIN, LON_MAX = -90.0, -70.0


def fetch_oni_monthly():
    """Fetch NOAA CPC ONI table and convert to long monthly format."""
    url = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
    # Fallback text source used by NOAA PSL tools.
    txt_url = "https://psl.noaa.gov/data/correlation/oni.data"

    try:
        r = requests.get(txt_url, timeout=45)
        r.raise_for_status()
    except Exception:
        # If PSL endpoint is unavailable, keep an empty file and continue pipeline.
        pd.DataFrame(columns=["year", "month", "oni", "date"]).to_csv(
            os.path.join(PROC_DIR, "oni_monthly.csv"), index=False
        )
        return pd.DataFrame(columns=["year", "month", "oni", "date"])

    rows = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line or line.startswith("ANOM") or line.startswith("YR"):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        year = int(parts[0])
        vals = parts[1:13]
        if any(v in {"-99.99", "99.99"} for v in vals):
            continue
        for month, v in enumerate(vals, start=1):
            rows.append({"year": year, "month": month, "oni": float(v)})

    oni = pd.DataFrame(rows)
    oni["date"] = pd.to_datetime(dict(year=oni["year"], month=oni["month"], day=1))
    oni = oni.sort_values("date").reset_index(drop=True)
    oni.to_csv(os.path.join(PROC_DIR, "oni_monthly.csv"), index=False)
    return oni


def _read_erddap_csv(url):
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), skiprows=[1])


def fetch_wind_monthly():
    """Fetch monthly wind fields and aggregate a Humboldt regional mean proxy."""
    base = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdlasFnWPr_LonPM180.csv"
    query = (
        "?u_mean[(1982-01-16T12:00:00Z):1:(2025-12-16T12:00:00Z)]"
        f"[({LAT_MIN}):2:({LAT_MAX})]"
        f"[({LON_MIN}):2:({LON_MAX})],"
        "v_mean[(1982-01-16T12:00:00Z):1:(2025-12-16T12:00:00Z)]"
        f"[({LAT_MIN}):2:({LAT_MAX})]"
        f"[({LON_MIN}):2:({LON_MAX})]"
    )

    try:
        df = _read_erddap_csv(base + "".join(query))
    except Exception:
        empty = pd.DataFrame(columns=["date", "u_mean", "v_mean", "wind_speed"])
        empty.to_csv(os.path.join(PROC_DIR, "wind_monthly.csv"), index=False)
        return empty

    df.columns = [c.strip().split(" ")[0] for c in df.columns]
    keep = [c for c in ["time", "u_mean", "v_mean"] if c in df.columns]
    df = df[keep].dropna()
    if df.empty:
        empty = pd.DataFrame(columns=["date", "u_mean", "v_mean", "wind_speed"])
        empty.to_csv(os.path.join(PROC_DIR, "wind_monthly.csv"), index=False)
        return empty

    df["time"] = pd.to_datetime(df["time"])
    for col in ["u_mean", "v_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    monthly = (
        df.assign(date=df["time"].dt.to_period("M").dt.to_timestamp())
        .groupby("date")[["u_mean", "v_mean"]]
        .mean()
        .reset_index()
    )
    monthly["wind_speed"] = np.sqrt(monthly["u_mean"] ** 2 + monthly["v_mean"] ** 2)
    monthly.to_csv(os.path.join(PROC_DIR, "wind_monthly.csv"), index=False)
    return monthly


def fetch_chlorophyll_monthly():
    """Fetch chlorophyll from CoastWatch VIIRS gap-filled product (2018+)."""
    base = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNnoaaSNPPnoaa20chlaGapfilledDaily.csv"
    query = (
        "?chlor_a[(2018-05-30T12:00:00Z):7:(2025-12-31T12:00:00Z)]"
        "[(0.0):1:(0.0)]"
        f"[({LAT_MIN}):24:({LAT_MAX})]"
        f"[({LON_MIN}):24:({LON_MAX})]"
    )

    try:
        df = _read_erddap_csv(base + query)
    except Exception:
        empty = pd.DataFrame(columns=["date", "chlor_a"])
        empty.to_csv(os.path.join(PROC_DIR, "chlorophyll_monthly.csv"), index=False)
        return empty

    df.columns = [c.strip().split(" ")[0] for c in df.columns]
    if not {"time", "chlor_a"}.issubset(df.columns):
        empty = pd.DataFrame(columns=["date", "chlor_a"])
        empty.to_csv(os.path.join(PROC_DIR, "chlorophyll_monthly.csv"), index=False)
        return empty

    df = df[["time", "chlor_a"]].dropna()
    if df.empty:
        empty = pd.DataFrame(columns=["date", "chlor_a"])
        empty.to_csv(os.path.join(PROC_DIR, "chlorophyll_monthly.csv"), index=False)
        return empty

    df["time"] = pd.to_datetime(df["time"])
    df["chlor_a"] = pd.to_numeric(df["chlor_a"], errors="coerce")
    df = df.dropna()

    monthly = (
        df.assign(date=df["time"].dt.to_period("M").dt.to_timestamp())
        .groupby("date")["chlor_a"]
        .mean()
        .reset_index()
    )
    monthly.to_csv(os.path.join(PROC_DIR, "chlorophyll_monthly.csv"), index=False)
    return monthly


def build_driver_table():
    oni = fetch_oni_monthly()
    wind = fetch_wind_monthly()
    chl = fetch_chlorophyll_monthly()

    # Start with ONI timeline where available.
    if oni.empty:
        drivers = pd.DataFrame(columns=["date", "oni", "u_mean", "v_mean", "wind_speed", "chlor_a"])
    else:
        drivers = oni[["date", "oni"]].copy()
        if not wind.empty:
            drivers = drivers.merge(wind, on="date", how="left")
        else:
            drivers[["u_mean", "v_mean", "wind_speed"]] = np.nan

        if not chl.empty:
            drivers = drivers.merge(chl, on="date", how="left")
        else:
            drivers["chlor_a"] = np.nan

    drivers.to_csv(os.path.join(PROC_DIR, "drivers_monthly.csv"), index=False)
    return drivers


def main():
    print("Fetching external drivers: ONI, wind, chlorophyll...")
    drivers = build_driver_table()
    print(f"Driver table rows: {len(drivers)}")
    print("Saved: oni_monthly.csv, wind_monthly.csv, chlorophyll_monthly.csv, drivers_monthly.csv")


if __name__ == "__main__":
    main()
