import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def classify_enso(oni):
    if pd.isna(oni):
        return "Unknown"
    if oni >= 0.5:
        return "El Nino"
    if oni <= -0.5:
        return "La Nina"
    return "Neutral"


def main():
    daily = pd.read_csv(os.path.join(PROC_DIR, "daily_sst_with_clim.csv"), parse_dates=["time"])
    events = pd.read_csv(os.path.join(PROC_DIR, "mhw_events.csv"), parse_dates=["start_date", "end_date"])
    drivers = pd.read_csv(os.path.join(PROC_DIR, "drivers_monthly.csv"), parse_dates=["date"])

    if drivers.empty or "oni" not in drivers.columns:
        print("drivers_monthly.csv missing ONI data. Run fetch_external_drivers.py first.")
        return

    daily["date"] = daily["time"].dt.to_period("M").dt.to_timestamp()
    daily["mhw_day"] = daily["is_heatwave"].astype(int)
    monthly = (
        daily.groupby("date")
        .agg(
            mhw_days=("mhw_day", "sum"),
            mean_anomaly=("anomaly", "mean"),
            max_anomaly=("anomaly", "max")
        )
        .reset_index()
    )

    merged = monthly.merge(drivers[[c for c in drivers.columns if c in ["date", "oni", "wind_speed", "chlor_a"]]], on="date", how="inner")
    merged = merged.dropna(subset=["oni"])

    if merged.empty:
        print("No overlapping monthly ONI and SST series after merge.")
        return

    # Lag correlation between ONI and MHW days.
    lags = range(-12, 13)
    corrs = []
    for lag in lags:
        corr = merged["oni"].corr(merged["mhw_days"].shift(lag))
        corrs.append(corr)

    corr_df = pd.DataFrame({"lag_months": list(lags), "corr": corrs})
    peak_row = corr_df.iloc[corr_df["corr"].abs().idxmax()]

    plt.figure(figsize=(10, 5))
    plt.plot(corr_df["lag_months"], corr_df["corr"], marker="o", color="#1273de")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.scatter([peak_row["lag_months"]], [peak_row["corr"]], color="red", zorder=3, label=f"Peak: lag={int(peak_row['lag_months'])}, r={peak_row['corr']:.2f}")
    plt.title("ONI vs Humboldt MHW Days: Lag Correlation")
    plt.xlabel("Lag (months, positive means ONI leads)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "oni_mhw_lag_correlation.png"), dpi=300)
    plt.close()

    merged["enso_phase"] = merged["oni"].apply(classify_enso)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=merged, x="enso_phase", y="mhw_days", order=["La Nina", "Neutral", "El Nino"], palette="Set2")
    plt.title("Monthly MHW Days by ENSO Phase")
    plt.xlabel("ENSO Phase (ONI)")
    plt.ylabel("MHW Days per Month")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mhw_days_by_enso_phase.png"), dpi=300)
    plt.close()

    events["start_month"] = events["start_date"].dt.to_period("M").dt.to_timestamp()
    ephase = events.merge(merged[["date", "enso_phase"]], left_on="start_month", right_on="date", how="left")
    summary = (
        ephase.groupby("enso_phase")
        .agg(
            event_count=("start_date", "count"),
            mean_duration=("duration", "mean"),
            mean_max_intensity=("max_intensity", "mean")
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(PROC_DIR, "enso_event_summary.csv"), index=False)

    corr_df.to_csv(os.path.join(PROC_DIR, "oni_mhw_lag_correlation.csv"), index=False)
    merged.to_csv(os.path.join(PROC_DIR, "monthly_sst_drivers_merged.csv"), index=False)

    print("Saved ENSO coupling outputs and figures.")


if __name__ == "__main__":
    main()
