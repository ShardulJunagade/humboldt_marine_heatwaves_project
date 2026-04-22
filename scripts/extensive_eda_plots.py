import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

print("Loading processed daily SST data...")
df = pd.read_csv(os.path.join(PROC_DIR, "daily_sst_with_clim.csv"), parse_dates=['time'])
events = pd.read_csv(os.path.join(PROC_DIR, "mhw_events.csv"), parse_dates=['start_date', 'end_date'])

# 1. SST Distribution Shift (Before 2000 vs After 2000)
plt.figure(figsize=(10, 6))
df['period'] = np.where(df['time'].dt.year < 2004, '1982-2003', '2004-2025')
sns.kdeplot(data=df, x='sst', hue='period', fill=True, common_norm=False, palette='coolwarm')
plt.title('Humboldt Current Daily SST Distribution Shift', fontsize=14)
plt.xlabel('Sea Surface Temperature (°C)')
plt.ylabel('Density')
plt.savefig(os.path.join(FIG_DIR, "sst_distribution_shift.png"), dpi=300)
plt.close()

# 2. Monthly Heatwave Burden (Days per month)
print("Evaluating Seasonal MHW Calendar Burden...")
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['is_mhw_day'] = df['sst'] > df['thresh']

burden = df.groupby(['year', 'month'])['is_mhw_day'].sum().reset_index()
burden_pivot = burden.pivot(index='year', columns='month', values='is_mhw_day').fillna(0)

plt.figure(figsize=(12, 8))
sns.heatmap(burden_pivot, cmap='YlOrRd', cbar_kws={'label': 'MHW Days'}, linewidths=.5)
plt.title('Monthly Marine Heatwave Burden (1982-2025)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Year')
plt.savefig(os.path.join(FIG_DIR, "monthly_mhw_burden.png"), dpi=300)
plt.close()

# 3. Anomaly Probability Matrix (Markov proxy)
print("Computing Daily Transition Probabilities...")
df['state'] = np.where(df['is_mhw_day'], 'MHW', 'Normal')
df['next_state'] = df['state'].shift(-1)
transitions = pd.crosstab(df['state'], df['next_state'], normalize='index') * 100

plt.figure(figsize=(6, 5))
sns.heatmap(transitions, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Transition Probability (%)'})
plt.title('Daily State Transition Probabilities', fontsize=14)
plt.savefig(os.path.join(FIG_DIR, "markov_transitions.png"), dpi=300)
plt.close()

print("Extensive EDA Plots successfully rendered to figures/")
