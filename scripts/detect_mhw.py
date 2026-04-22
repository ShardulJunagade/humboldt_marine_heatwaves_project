import os
import glob
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(PROC_DIR, exist_ok=True)


def circular_doy_window(center_doy, half_width=5):
    """Return a circular +/- window over 366-day calendar."""
    return [((center_doy - 1 + shift) % 366) + 1 for shift in range(-half_width, half_width + 1)]

def main():
    print("Aggregating daily SST files...")
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not files:
        print("No raw data files found.")
        return
        
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df.sort_values("time").reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])
    df['doy'] = df['time'].dt.dayofyear

    print("Computing 30-year climatological baseline (1982-2011) and 90th percentile thresholds...")
    # Baseline period as recommended by WMO and Hobday
    baseline = df[(df['time'].dt.year >= 1982) & (df['time'].dt.year <= 2011)].copy()

    # Calculate 11-day rolling window 90th percentile for each DOY
    doy_stats = []
    for doy in range(1, 367): # DOY 1 to 366 (leap years)
        # Find days within the +/- 5 day window (handling wrap-around at end/start of year)
        window_days = circular_doy_window(doy, half_width=5)
        window_data = baseline[baseline['doy'].isin(window_days)]['sst']
        
        if len(window_data) > 0:
            mean_sst = window_data.mean()
            p90_sst = window_data.quantile(0.90)
            doy_stats.append({'doy': doy, 'seas': mean_sst, 'thresh': p90_sst})

    clim = pd.DataFrame(doy_stats)
    
    # Smooth the climatology and threshold (Hobday uses a 31-day moving average)
    clim['seas'] = clim['seas'].rolling(window=31, min_periods=1, center=True).mean()
    clim['thresh'] = clim['thresh'].rolling(window=31, min_periods=1, center=True).mean()

    clim.to_csv(os.path.join(PROC_DIR, "climatology.csv"), index=False)

    # Merge climatology back to full dataset
    print("Detecting Marine Heatwaves (duration >= 5 days)...")
    df = df.merge(clim, on='doy', how='left')
    df['anomaly'] = df['sst'] - df['seas']
    df['is_heatwave'] = df['sst'] > df['thresh']

    mhw_events = []
    current_event = []
    
    for i, row in df.iterrows():
        if row['is_heatwave']:
            current_event.append(row)
        else:
            if len(current_event) >= 5:
                event_df = pd.DataFrame(current_event)
                mhw_events.append({
                    'start_date': event_df['time'].min(),
                    'end_date': event_df['time'].max(),
                    'duration': len(event_df),
                    'max_intensity': event_df['anomaly'].max(),
                    'mean_intensity': event_df['anomaly'].mean(),
                    'cumulative_intensity': event_df['anomaly'].sum(),
                    'peak_sst': event_df['sst'].max()
                })
            current_event = []
            
    # Catch any event ongoing at the end of the series
    if len(current_event) >= 5:
        event_df = pd.DataFrame(current_event)
        mhw_events.append({
            'start_date': event_df['time'].min(),
            'end_date': event_df['time'].max(),
            'duration': len(event_df),
            'max_intensity': event_df['anomaly'].max(),
            'mean_intensity': event_df['anomaly'].mean(),
            'cumulative_intensity': event_df['anomaly'].sum(),
            'peak_sst': event_df['sst'].max()
        })

    events_df = pd.DataFrame(mhw_events)
    print(f"Detected {len(events_df)} distinct Marine Heatwaves.")

    annual_summary = (
        df.assign(year=df['time'].dt.year)
        .groupby('year')
        .agg(
            mhw_days=('is_heatwave', 'sum'),
            mean_sst=('sst', 'mean'),
            mean_anomaly=('anomaly', 'mean'),
            max_anomaly=('anomaly', 'max')
        )
        .reset_index()
    )

    event_annual = (
        events_df.assign(year=events_df['start_date'].dt.year)
        .groupby('year')
        .agg(
            event_count=('start_date', 'count'),
            total_duration=('duration', 'sum'),
            max_event_intensity=('max_intensity', 'max'),
            mean_event_duration=('duration', 'mean')
        )
        .reset_index()
    )
    annual_summary = annual_summary.merge(event_annual, on='year', how='left').fillna(0)
    
    events_df.to_csv(os.path.join(PROC_DIR, "mhw_events.csv"), index=False)
    annual_summary.to_csv(os.path.join(PROC_DIR, "annual_summary.csv"), index=False)
    # Save full processed daily table for downstream analyses.
    df[['time', 'sst', 'seas', 'thresh', 'anomaly', 'is_heatwave']].to_csv(os.path.join(PROC_DIR, "daily_sst_with_clim.csv"), index=False)
    
    print("Preprocessing and MHW detection complete. Outputs saved to data/processed/")

if __name__ == "__main__":
    main()