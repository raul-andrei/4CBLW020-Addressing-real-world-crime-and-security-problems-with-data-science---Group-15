# Addressing Real-World Crime and Security Problems with Data Science
**Group 15 — 4CBLW020**

---

## Research Question

> How can identifying and targeting brokerage crimes improve the efficiency of police resource allocation while maintaining fairness and ethical accountability?

---

## Project Overview

This project uses historical crime data from data.police.uk to forecast violent crime demand across England and Wales. The core idea is that certain "broker crimes" (such as robbery, theft from the person, and possession of weapons) act as early warning signals for future violence. We quantify this through betweenness centrality scores derived from a crime co-occurrence network, and use these scores as extra regressors in a Prophet forecasting model.

---

## Repository Structure

```
├── code/
│   ├── data_code/              # Raw data download and cleaning scripts
│   ├── models_london_tests/    # London-level model experiments
│   ├── models_final/           # Final UK-scale model pipeline
│   ├── network_analysis/       # Crime co-occurrence network and brokerage score generation
│   └── dashboard/              # Dash dashboard for interactive exploration
│
├── data/
│   ├── cleaned_data/
│   │   ├── by_force_cleaned/   # One parquet file per police force
│   │   └── panels/             # Aggregated panels
│   ├── global_brokerage.csv    # Betweenness centrality scores per crime type
│   └── lsoa_ward_mapping.csv   # ONS LSOA 2011 to Ward 2021 lookup
│
├── output/
│   ├── results/                # Model comparison CSVs and per-LSOA/ward results
│   └── pictures/               # Figures and visualisations
│
├── requirements.txt
└── README.md
```

---

## Data Sources

| Dataset | Source | Description |
|---|---|---|
| Street-level crime data | [data.police.uk](https://data.police.uk/data/) | All police-recorded crimes in England and Wales, Dec 2013 onwards |
| LSOA to Ward mapping | [ONS Geoportal](https://geoportal.statistics.gov.uk) | LSOA 2011 to Electoral Ward 2021 best-fit lookup |
| Brokerage scores | Network analysis (`global_brokerage.csv`) | Current flow betweenness centrality per crime type |

**Note on raw data:** The raw crime CSVs are not included in this repository due to file size. Download all available months from data.police.uk and place them in `data/uk_raw_data/` before running the pipeline.

---

## Environment Setup

**Requirements:** Python 3.13+

```bash
pip install -r requirements.txt
```

---

## How to Reproduce the Analysis

### Step 1 — Download the raw data

Download all street-level crime CSVs from [data.police.uk/data](https://data.police.uk/data/) and place them in:
data/uk_raw_data/

### Step 2 — Clean and split by force

```bash
python src/data_code/clean_by_force.py
```

Output: `data/cleaned_data/by_force_cleaned/*.parquet`

### Step 3 — Build the UK panel

```bash
python src/models_final/build_panel_uk.py
```

Output: `data/cleaned_data/panels/uk_panel.parquet`

### Step 4 — Run the models

> **[to be completed once the UK model run is finalised]**

### Step 5 — Run the dashboard

> **[to be completed once the dashboard is finalised]**

---

## Results

> **[to be added once the final UK model run is complete]**

---

## Network Analysis

> **[to be completed by Nikola]**