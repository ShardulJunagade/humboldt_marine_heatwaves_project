import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

events = pd.read_csv(os.path.join(PROC_DIR, "mhw_events.csv"), parse_dates=['start_date', 'end_date'])
events['year'] = events['start_date'].dt.year

# Categories based on Hobday proxies (we use raw threshold multiples essentially if we don't have the 90p diff saved,
# but we will just bin raw intensity since Humboldt has specific baselines).
# Moderate: < 1.0, Strong: 1.0 to 1.5, Severe: 1.5 to 2.0, Extreme: > 2.0
def categorize(intensity):
    if intensity < 1.0: return 'Moderate'
    elif intensity < 1.5: return 'Strong'
    elif intensity < 2.0: return 'Severe'
    return 'Extreme'

events['category'] = events['max_intensity'].apply(categorize)
category_order = ['Moderate', 'Strong', 'Severe', 'Extreme']
color_map = {'Moderate': '#ffc107', 'Strong': '#fd7e14', 'Severe': '#dc3545', 'Extreme': '#6f42c1'}

cat_counts = events.groupby(['year', 'category']).size().unstack(fill_value=0)
for cat in category_order:
    if cat not in cat_counts.columns:
        cat_counts[cat] = 0

cat_counts = cat_counts[category_order]
ax = cat_counts.plot(kind='bar', stacked=True, figsize=(14, 6), color=[color_map[c] for c in category_order], width=0.8)
plt.title('Annual Marine Heatwave Events by Severity Category', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Events', fontsize=12)
plt.legend(title='Category (Hobday Proxy)', title_fontsize='11')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "mhw_severity_annual.png"), dpi=300)
plt.close()

# Trend Analysis 
# Frequency
freq_data = events.groupby('year').size().reset_index(name='count')
years = np.arange(1982, 2026)
all_years = pd.DataFrame({'year': years})
freq_data = all_years.merge(freq_data, on='year', how='left').fillna(0)

X = sm.add_constant(freq_data['year'])
model_freq = sm.OLS(freq_data['count'], X).fit()

# Total Annual Duration
dur_data = events.groupby('year')['duration'].sum().reset_index(name='total_duration')
dur_data = all_years.merge(dur_data, on='year', how='left').fillna(0)

X_dur = sm.add_constant(dur_data['year'])
model_dur = sm.OLS(dur_data['total_duration'], X_dur).fit()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Frequency
ax1.scatter(freq_data['year'], freq_data['count'], color='#007bff', label='Events per year')
ax1.plot(freq_data['year'], model_freq.predict(X), color='red', lw=2, label=f"Trend (p={model_freq.pvalues.iloc[1]:.3f})")
ax1.set_title('MHW Frequency Over Time', fontsize=14)
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Events')
ax1.legend()

# Plot Duration
ax2.scatter(dur_data['year'], dur_data['total_duration'], color='#28a745', label='Days per year')
ax2.plot(dur_data['year'], model_dur.predict(X_dur), color='red', lw=2, label=f"Trend (p={model_dur.pvalues.iloc[1]:.3f})")
ax2.set_title('Cumulative MHW Duration Over Time', fontsize=14)
ax2.set_xlabel('Year')
ax2.set_ylabel('Total Days')
ax2.legend()

fig.suptitle('Long-term Decadal Trends in Marine Heatwaves (Linear Regression)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "mhw_long_term_trends.png"), dpi=300)
plt.close()

print("Statistics and Category plots mapped out.")
