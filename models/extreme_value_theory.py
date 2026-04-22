import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("Loading parsed Marine Heatwave Events...")
events = pd.read_csv(os.path.join(PROC_DIR, "mhw_events.csv"), parse_dates=['start_date', 'end_date'])

if len(events) < 10:
    print("Not enough events to fit distributions reliably.")
    exit()

print("Fitting Generalized Extreme Value (GEV) distribution to Max Intensity...")
# Extracting highest intensities for each year (Block Maxima)
events['year'] = events['start_date'].dt.year
annual_max = events.groupby('year')['max_intensity'].max().dropna()

# Fit GEV parameters: shape (c), location (loc), scale (scale)
shape, loc, scale = gev.fit(annual_max.values)

x = np.linspace(annual_max.min() - 0.5, annual_max.max() + 1.0, 100)
pdf = gev.pdf(x, shape, loc, scale)

plt.figure(figsize=(8, 6))
plt.hist(annual_max, bins=15, density=True, alpha=0.5, color='orange', label='Observed Annual Maximums')
plt.plot(x, pdf, 'r-', lw=2, label=f'Fitted GEV PDF\n(Shape={shape:.2f})')
plt.title('Extreme Value Theory on MHW Max Intensity (1982-2025)', fontsize=14)
plt.xlabel('Maximum Intensity Anomaly (°C)')
plt.ylabel('Density Probability')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "evt_fit_mhw_intensity.png"), dpi=300)
plt.close()

# Evaluate Return Period (1 in 50 years, 1 in 100 years)
# For GEV, return level expected = isf(1/T)
rp_10 = gev.isf(1/10, shape, loc, scale)
rp_50 = gev.isf(1/50, shape, loc, scale)
rp_100 = gev.isf(1/100, shape, loc, scale)

print("\n--- Return Period Projections ---")
print(f"1-in-10 Year MHW Intensity: +{rp_10:.2f}°C Anomaly")
print(f"1-in-50 Year MHW Intensity: +{rp_50:.2f}°C Anomaly")
print(f"1-in-100 Year MHW Intensity: +{rp_100:.2f}°C Anomaly")
print("---------------------------------\n")

print("Statistical EVT models generated and rendered to figures/")
