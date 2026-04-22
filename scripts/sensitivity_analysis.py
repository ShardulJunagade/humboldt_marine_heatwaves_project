import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def circular_window(center_doy, half_width=5):
    return [((center_doy - 1 + shift) % 366) + 1 for shift in range(-half_width, half_width + 1)]


def build_threshold(df, q=0.9):
    baseline = df[(df["time"].dt.year >= 1982) & (df["time"].dt.year <= 2011)].copy()
    stats = []
    for doy in range(1, 367):
        wdays = circular_window(doy, 5)
        vals = baseline.loc[baseline["doy"].isin(wdays), "sst"]
        stats.append({
            "doy": doy,
            "seas": vals.mean(),
            "thresh": vals.quantile(q),
        })
    clim = pd.DataFrame(stats)
    clim["seas"] = clim["seas"].rolling(31, center=True, min_periods=1).mean()
    clim["thresh"] = clim["thresh"].rolling(31, center=True, min_periods=1).mean()
    return clim


def detect_count(df, clim, min_duration=5):
    tmp = df.merge(clim, on="doy", how="left")
    tmp["is_hw"] = tmp["sst"] > tmp["thresh"]
    lengths = []
    run = 0
    for v in tmp["is_hw"].values:
        if v:
            run += 1
        else:
            if run > 0:
                lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)

    valid = [x for x in lengths if x >= min_duration]
    return len(valid), int(np.sum(valid))


def main():
    daily = pd.read_csv(os.path.join(PROC_DIR, "daily_sst_with_clim.csv"), parse_dates=["time"])
    daily["doy"] = daily["time"].dt.dayofyear

    settings = [(0.9, 5), (0.9, 7), (0.95, 5), (0.95, 7)]
    rows = []
    for q, d in settings:
        clim = build_threshold(daily[["time", "doy", "sst"]].copy(), q=q)
        n_events, total_days = detect_count(daily[["time", "doy", "sst"]].copy(), clim, min_duration=d)
        rows.append({
            "percentile_threshold": q,
            "min_duration_days": d,
            "event_count": n_events,
            "total_mhw_days": total_days,
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PROC_DIR, "sensitivity_detection_summary.csv"), index=False)

    labels = [f"q={r['percentile_threshold']}, d={int(r['min_duration_days'])}" for _, r in out.iterrows()]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, out["event_count"], color="#2a9d8f")
    plt.title("Detection Sensitivity: Event Count by Threshold/Duration")
    plt.ylabel("Detected Events")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "sensitivity_event_count.png"), dpi=300)
    plt.close()

    print("Saved sensitivity_detection_summary.csv and sensitivity_event_count.png")


if __name__ == "__main__":
    main()
