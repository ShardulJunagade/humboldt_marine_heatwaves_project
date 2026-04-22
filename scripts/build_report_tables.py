import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def main():
    events = pd.read_csv(os.path.join(PROC_DIR, "mhw_events.csv"), parse_dates=["start_date", "end_date"])
    annual = pd.read_csv(os.path.join(PROC_DIR, "annual_summary.csv"))
    enso = pd.read_csv(os.path.join(PROC_DIR, "enso_event_summary.csv")) if os.path.exists(os.path.join(PROC_DIR, "enso_event_summary.csv")) else pd.DataFrame()

    top = events.sort_values("max_intensity", ascending=False).head(10).copy()
    top["start_date"] = top["start_date"].dt.strftime("%Y-%m-%d")
    top["end_date"] = top["end_date"].dt.strftime("%Y-%m-%d")
    top.to_csv(os.path.join(PROC_DIR, "table_top10_events.csv"), index=False)

    decade = annual.copy()
    decade["decade"] = (decade["year"] // 10) * 10
    decade_summary = (
        decade.groupby("decade")
        .agg(
            mean_events=("event_count", "mean"),
            mean_duration_days=("total_duration", "mean"),
            mean_mhw_days=("mhw_days", "mean"),
            max_event_intensity=("max_event_intensity", "max"),
        )
        .reset_index()
    )
    decade_summary.to_csv(os.path.join(PROC_DIR, "table_decadal_summary.csv"), index=False)

    if not enso.empty:
        enso.to_csv(os.path.join(PROC_DIR, "table_enso_phase_summary.csv"), index=False)

    print("Saved report table csv files.")


if __name__ == "__main__":
    main()
