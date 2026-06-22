# Addressing Real-World Crime and Security Problems with Data Science
**Group 15 — 4CBLW020**

---

## Research Question

> How can identifying and targeting brokerage crimes improve the efficiency of police resource allocation while maintaining fairness and ethical accountability?

---

## Project Overview

This project uses street-level crime data from [data.police.uk](https://data.police.uk/data/) (England & Wales, 2017–2026, ~49M records) to forecast violence demand and support police resource allocation. The core idea: certain **"broker" crime types** (e.g. robbery, theft from the person, possession of weapons) sit between communities of crime in a co-occurrence network and act as early signals for **Violence and sexual offences (V&SO)**, our forecast target.

The pipeline has four parts:

1. **EDA** — descriptive charts over the full dataset.
2. **Brokerage network analysis** — a crime co-occurrence network whose betweenness centrality quantifies how much each crime type "brokers". This produces a per-ward `avg_betweenness` score.
3. **Forecasting** — a Prophet model per ward forecasting next-month V&SO, with the lagged brokerage activity as an extra regressor, evaluated against a plain baseline. (A regression analysis separately tests the brokerage→violence link.)
4. **Dashboard** — an interactive Dash app (the brokerage risk map, the per-force brokerage network, the V&SO forecast map, and a proportional resource-allocation view).

---

## Repository Structure (`src/`)

```
src/
├── data_code/                  # Data cleaning pipeline + EDA
│   ├── crimes_db.py            #   builds data/crimes.db from cleaned per-force parquets
│   ├── merge_data.py, combine_forces.py, remove_duplicates.py,
│   │   split_by_force.py, final_london_clean.py, ...   # raw → cleaned pipeline
│   └── eda.py                  #   the 7 EDA graphs (reads crimes.db via DuckDB)
│
├── network_analysis/           # Crime co-occurrence network & brokerage
│   ├── preparation.py          #   DB access + co-occurrence (lift) matrix
│   ├── analysis.py             #   per-crime brokerage metrics (betweenness, constraint, ...)
│   ├── sensitivity_analysis.py #   graph construction, primary + sensitivity brokerage analysis
│   ├── scores.py               #   per-ward avg_betweenness over time
│   ├── network_visualization.py#   the presentation network figure (PNG)
│   ├── results_analysis.py     #   figures from the sensitivity sweeps
│   └── regressionAnalysis.py   #   GLM: does brokerage predict next-month violence?
│
├── new_models_test/            # Forecasting pipeline (the report's models)
│   ├── build_panel.py          #   builds the ward-month panel (target + regressors)
│   ├── run_forecast.py         #   backtest: Prophet baseline vs brokerage-augmented
│   ├── production_models.py    #   train once + save + predict next month (feeds dashboard)
│   └── compare_models.py       #   paired comparison of the two Prophet variants
│
├── models_london_tests/        # Earlier London-scale Prophet experiments (exploratory)
│
└── dashboard/                  # Dash app
    ├── artifacts.py            #   precomputes the heavy artifacts the app loads
    ├── app.py                  #   the dashboard server
    └── assets/                 #   style.css + precomputed artifacts
```

All scripts are run as **modules from the repository root** (e.g. `python -m src.data_code.eda`) so the `src.*` imports resolve.

---

## Prerequisites

- **Python 3.12 or 3.13** (the stack uses pandas 3.0 / numpy 2.4).
- The **data files** in `data/` (see [Data](#2-data)). They are large and **not** committed to the repo (`data/` is git-ignored).

---

## 1. Environment setup

Create and activate a virtual environment, then install the dependencies.

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Once the venv is active, the plain `python` command points at it. On Windows you can also skip activation and call the interpreter directly: `venv\Scripts\python.exe -m ...`.

`prophet` pulls in `cmdstanpy`, which compiles a small Stan model on first use — the first Prophet run may pause briefly while that happens.

---

## 2. Data

Everything reads from `data/`. Most of these files are committed to the repo. The
only exception is the **crime database**, which is too large for GitHub (357 MB,
over the 100 MB file limit), so it is shared separately.

| File | In repo? | Used by | What it is |
|---|---|---|---|
| `data/crimes.db` | **No — download (see below)** | EDA, brokerage, regression, models, dashboard | DuckDB database, one `crimes` table (~49M rows: `lsoa_code, crime_type, year, month_num, longitude, latitude, last_outcome, reported_by`) |
| `data/wards_dec2021_uk_bgc_4326.geojson` | Yes | dashboard | Ward boundaries (WGS84) for the maps |
| `data/lsoa_ward_mapping.csv` | Yes | brokerage, dashboard | ONS LSOA-2011 → Ward-2021 lookup |
| `data/global_brokerage.csv` | Yes | models, dashboard | Per-crime-type brokerage scores |
| `data/ward_brokerage_scores.parquet` | Yes | regression | Per-ward brokerage score per month |

**Get `crimes.db`:** download it from the team OneDrive or DropBox and place it at `data/crimes.db`:

> **OneDrive link (only people with a TUE email can access):** `https://tuenl-my.sharepoint.com/:u:/g/personal/j_f_i_m_couwenberg_student_tue_nl/IQArarnHKYPSR4APJCLOR-abAdbCRH1AD0jJVkZwfXe0UzI?e=t7crf5`

> **DropBox link:** `https://www.dropbox.com/scl/fi/amso52hrcfywm64i4wjn9/crimes.db?rlkey=g1iq2soca0own6ccit4zta1s0&st=33vgb94r&dl=1`

That single file is all you need to run everything below.

<details>
<summary>Alternative: rebuild <code>crimes.db</code> from the raw data</summary>

The raw cleaning pipeline lives in `src/data_code/` (download all months from
data.police.uk, merge, de-duplicate, and split into one cleaned parquet per force
under `data/cleaned_data/by_force_cleaned/`). The database is then assembled from
those parquets:

```bash
python -m src.data_code.crimes_db
```

This reads `data/cleaned_data/by_force_cleaned/*.parquet` and writes `data/crimes.db`.
</details>

---

## 3. Running the analyses

Run all commands from the repo root with the venv active.

### 3.1 Exploratory data analysis

```bash
python -m src.data_code.eda
```
Aggregates the full 49M-row table inside DuckDB and writes 7 charts + CSVs to `src/output/` (crime-type counts, monthly trend, outcomes by type, seasonality heatmap, annual trend, top LSOAs, a national hexbin density map, and crime mix in the top LSOAs). Runs in a few seconds.

### 3.2 Brokerage network analysis

The brokerage metrics feed two places: the **dashboard** (Brokerage Network page + risk map) and the **forecasting models** (the `avg_betweenness` regressor). To regenerate the standalone brokerage outputs:

```bash
# Per-crime brokerage table (primary specification: lift, presence threshold 3, kNN k=3)
python -m src.network_analysis.sensitivity_analysis      # -> global_brokerage.csv

# The presentation network figure (V&SO node highlighted in red)
python -m src.network_analysis.network_visualization     # -> network2.png

# Per-ward avg_betweenness over time (used by the models & regression; prints a summary)
python -m src.network_analysis.scores
```

The robustness sweeps that back the appendix figures are heavier: run `run_sensitivity_per_crime()` / `run_brokerage_per_force()` in `sensitivity_analysis.py` to produce the `sensitivity_results_*.csv` files, then `python -m src.network_analysis.results_analysis` to turn them into figures.

### 3.3 Regression analysis

Tests whether per-ward brokerage predicts next-month violence (statsmodels GLM):

```bash
python -m src.network_analysis.regressionAnalysis
```
This prints the model summary. It reads `data/ward_brokerage_scores.parquet` (the per-ward brokerage-score table, committed to the repo) and `data/crimes.db`. To regenerate that table from scratch - This is optional, only if it gives you an error that "ward_brokerage_scores" doesn't exist:
```bash
python -c "from src.network_analysis.scores import calculate_average_betweenness as f; f(score_where_sql='year BETWEEN 2017 AND 2026').to_parquet('data/ward_brokerage_scores.parquet')"
```
(The `regression.ipynb` notebook at the repo root contains the same analysis in interactive form.)

### 3.4 Forecasting models (Prophet)

The forecasting models use the full DuckDB table and fit one Prophet model per ward, so these runs are **heavy (minutes, not seconds)**.

**Backtest / evaluation** — holds out a test period and compares the plain Prophet **baseline** against the **brokerage-augmented** variant:
```bash
python -m src.new_models_test.run_forecast
```
Writes `predictions.csv`, per-ward errors, and a metric summary to `src/new_models_test/output/`.

**Train + forecast next month** (this is what the dashboard's forecast map consumes) - This is used to RETRAIN the model, NOT needed to run the dashboard:
```bash
python -m src.new_models_test.production_models           # train on all history + save
python -m src.new_models_test.production_models predict    # forecast the next month
```
Trained models are saved under `src/new_models_test/output/models/prophet/`.

Optionally, after a backtest you can run a paired comparison of the two Prophet variants (baseline vs brokerage-augmented) on the identical ward-months:
```bash
python -m src.new_models_test.compare_models
```

### 3.5 Dashboard

The dashboard loads **precomputed artifacts** that are committed to the repo
(`src/dashboard/assets/`: `brokerage_networks.json`, `ward_snapshot.parquet`,
`forecast_snapshot.parquet`, `ward_force_mapping.parquet`), so with `data/crimes.db`
in place you can just launch it:

```bash
python -m src.dashboard.app
```
Then open **http://127.0.0.1:8050**. Use the sun/moon toggle in the sidebar for light/dark mode.

**Rebuilding the artifacts** (only needed after a data refresh). The forecast map's
artifact is produced from a trained Prophet model, so train first, then rebuild:

```bash
python -m src.new_models_test.production_models     # train + save the Prophet model
python -m src.dashboard.artifacts                   # regenerate all dashboard artifacts
```

---

## Outputs at a glance

| Step | Command | Output location |
|---|---|---|
| EDA | `python -m src.data_code.eda` | `src/output/` |
| Brokerage table | `python -m src.network_analysis.sensitivity_analysis` | `src/network_analysis/global_brokerage.csv` |
| Network figure | `python -m src.network_analysis.network_visualization` | `network2.png` (repo root) |
| Sensitivity figures | `python -m src.network_analysis.results_analysis` | `src/network_analysis/figures/` |
| Regression | `python -m src.network_analysis.regressionAnalysis` | printed GLM summary |
| Model backtest | `python -m src.new_models_test.run_forecast` | `src/new_models_test/output/` |
| Trained models | `python -m src.new_models_test.production_models` | `src/new_models_test/output/models/prophet/` |
| Dashboard | `python -m src.dashboard.app` | http://127.0.0.1:8050 |
