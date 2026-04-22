import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    daily = pd.read_csv(os.path.join(PROC_DIR, "daily_sst_with_clim.csv"), parse_dates=["time"])
    drivers = pd.read_csv(os.path.join(PROC_DIR, "drivers_monthly.csv"), parse_dates=["date"])

    daily["date"] = daily["time"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        daily.groupby("date")
        .agg(mhw_days=("is_heatwave", "sum"), mean_anomaly=("anomaly", "mean"))
        .reset_index()
    )
    monthly["event_month"] = (monthly["mhw_days"] >= 5).astype(int)

    merged = monthly.merge(drivers, on="date", how="left")
    merged.to_csv(os.path.join(PROC_DIR, "monthly_driver_composite_table.csv"), index=False)

    plot_vars = [v for v in ["oni", "wind_speed", "chlor_a"] if v in merged.columns]
    melted = merged.melt(id_vars=["event_month"], value_vars=plot_vars, var_name="driver", value_name="value").dropna()

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=melted,
        x="driver",
        y="value",
        hue="event_month",
        palette={0: "#8ecae6", 1: "#fb8500"},
    )
    plt.title("External Driver Distributions: Event vs Non-event Months")
    plt.xlabel("Driver")
    plt.ylabel("Value")
    plt.legend(title="Event Month", labels=["No", "Yes"])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "driver_event_composites.png"), dpi=300)
    plt.close()

    print("Saved monthly_driver_composite_table.csv and driver_event_composites.png")


if __name__ == "__main__":
    main()
