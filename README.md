# Coastal Marine Heatwaves in the Humboldt Current

A comprehensive research project evaluating marine heatwaves (MHWs) along the Humboldt Current (5°N to 45°S, 70°W to 90°W). This project uses high-resolution daily SST data via ERDDAP to apply threshold-based (Hobday et al., 2016) detection methods and conduct detailed spatial and temporal analysis of heatwave burden, persistence, and impacts.

## Overview

Unlike other regions (like the Arabian Sea / Bay of Bengal), the Humboldt Current is an Eastern Boundary Current characterized by intense coastal upwelling and massive influence from the El Niño-Southern Oscillation (ENSO) system. This project investigates both foundational characteristics of historical events and their coupling with large-scale upwelling processes.

## Project Structure

```
humboldt_marine_heatwaves_project/
├── data/
│   ├── raw/                 # Raw daily SST extractions (via ERDDAP)
│   └── processed/           # Processed datasets and heatwave event catalogs
├── notebooks/               # Jupyter notebooks for advanced EDA, PCA, and event trends
├── scripts/
│   └── download_humboldt_sst.py  # Data extraction script
├── figures/                 # Diagnostic and final report plots
├── report/                  # LaTeX source code for the final report
└── logs/                    # Execution logs
```

## Setup Instructions

### 1. Prerequisites

Ensure you have Python 3.9+ installed. It's recommended to run everything in a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install standard data science dependencies:

```bash
pip install pandas numpy requests xarray matplotlib seaborn scipy
```

### 2. Download Data

Run the data extraction script. This script pulls daily resolution Sea Surface Temperature (SST) data (using NOAA OISST v2.1) directly via ERDDAP. The retrieval is optimized by using a spatial stride of 4 (1 degree resolution) to be mindful of download sizes while capturing the structure of heatwaves effectively.

```bash
# Run from the project root:
python humboldt_marine_heatwaves_project/scripts/download_humboldt_sst.py
```

*Note: This will download historical data year-by-year from 1982 to 2025. It caches automatically, so interrupted downloads can be resumed directly.*

### 3. Next Steps (In Progress)

Once you verify the raw data, we will:
1. Aggregate the downloaded data to build 30-year climatologies.
2. Formally detect instances crossing the 90th percentile threshold.
3. Conduct EOF/PCA decomposition of the anomaly structures across the coast.
