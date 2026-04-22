import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, theilslopes
from statsmodels.tsa.seasonal import STL

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def mk_theilsen(series):
    x = np.arange(len(series))
    y = series.values.astype(float)
    tau, pval = kendalltau(x, y)
    slope, intercept, low, high = theilslopes(y, x, 0.95)
    return {
        "tau": tau,
        "p_value": pval,
        "theil_sen_slope_per_step": slope,
        "slope_ci_low": low,
        "slope_ci_high": high,
    }


def main():
    annual = pd.read_csv(os.path.join(PROC_DIR, "annual_summary.csv"))
    annual = annual.sort_values("year").reset_index(drop=True)

    # Fill any missing years explicitly.
    all_years = pd.DataFrame({"year": np.arange(int(annual["year"].min()), int(annual["year"].max()) + 1)})
    annual = all_years.merge(annual, on="year", how="left").fillna(0)

    # STL needs evenly spaced index.
    freq_series = pd.Series(annual["event_count"].values, index=pd.period_range(str(int(annual["year"].min())), str(int(annual["year"].max())), freq="Y"))
    dur_series = pd.Series(annual["total_duration"].values, index=freq_series.index)

    stl_freq = STL(freq_series, period=7, robust=True).fit()
    stl_dur = STL(dur_series, period=7, robust=True).fit()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(annual["year"], freq_series.values, color="#1f77b4", label="Observed")
    axes[0].plot(annual["year"], stl_freq.trend.values, color="red", label="STL Trend")
    axes[0].set_title("STL Decomposition: Annual MHW Event Frequency")
    axes[0].set_ylabel("Events")
    axes[0].legend()

    axes[1].plot(annual["year"], dur_series.values, color="#2ca02c", label="Observed")
    axes[1].plot(annual["year"], stl_dur.trend.values, color="red", label="STL Trend")
    axes[1].set_title("STL Decomposition: Annual Total MHW Duration")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Days")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "stl_decomposition_frequency_duration.png"), dpi=300)
    plt.close()

    mk_freq = mk_theilsen(annual["event_count"])
    mk_dur = mk_theilsen(annual["total_duration"])

    out = pd.DataFrame([
        {"metric": "event_count", **mk_freq},
        {"metric": "total_duration", **mk_dur},
    ])
    out.to_csv(os.path.join(PROC_DIR, "trend_significance_mk_theilsen.csv"), index=False)

    print("Saved STL and Mann-Kendall/Theil-Sen outputs.")


if __name__ == "__main__":
    main()
