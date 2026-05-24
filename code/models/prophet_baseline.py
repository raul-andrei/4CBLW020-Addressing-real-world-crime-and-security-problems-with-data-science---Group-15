import duckdb
import pandas as pd
from prophet import Prophet
import os
import json

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL_PATH = os.path.join(BASE, "data", "cleaned_data", "panels", "london_panel.parquet")
OUTPUT_PATH = os.path.join(BASE, "output", "results", "prophet_baseline_results.csv")

# Settings 
TARGET_CRIME = "Violence and sexual offences"
TRAIN_END = "2024-12-01"
TEST_START = "2025-01-01"
N_LSOAS = 300 

# Load data
print("Loading panel...")
con = duckdb.connect()
df = con.execute(f"""
    SELECT lsoa_code, month, crime_count
    FROM read_parquet('{PANEL_PATH}')
    WHERE crime_type = '{TARGET_CRIME}'
    ORDER BY lsoa_code, month
""").df()
con.close()

print(f"Loaded {len(df):,} rows for '{TARGET_CRIME}'")

# Get list of LSOAs
lsoas = df["lsoa_code"].unique()
if N_LSOAS:
    lsoas = lsoas[:N_LSOAS]
print(f"Running Prophet on {len(lsoas)} LSOAs...")

# Run Prophet per LSOA
results = []

for i, lsoa in enumerate(lsoas):
    lsoa_df = df[df["lsoa_code"] == lsoa].copy()
    lsoa_df = lsoa_df.rename(columns={"month": "ds", "crime_count": "y"})
    lsoa_df["ds"] = pd.to_datetime(lsoa_df["ds"])

    # Split train/test
    train = lsoa_df[lsoa_df["ds"] <= TRAIN_END]
    test = lsoa_df[lsoa_df["ds"] >= TEST_START]

    if len(train) < 24 or len(test) == 0:
        continue  # skip LSOAs with not enough data

    # Fit model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="additive")
    model.fit(train)

    # Predict
    future = model.make_future_dataframe(periods=len(test), freq="MS")
    forecast = model.predict(future)

    # Compare predictions vs actual
    pred = forecast[forecast["ds"] >= TEST_START][["ds", "yhat"]].reset_index(drop=True)
    actual = test[["ds", "y"]].reset_index(drop=True)
    merged = pred.merge(actual, on="ds")

    mae = (merged["yhat"] - merged["y"]).abs().mean()
    rmse = ((merged["yhat"] - merged["y"]) ** 2).mean() ** 0.5

    results.append({
        "lsoa_code": lsoa,
        "mae":       round(mae, 3),
        "rmse":      round(rmse, 3),
        "n_test":    len(merged)
    })

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(lsoas)} done...")

# Save results 
results_df = pd.DataFrame(results)
output_path = os.path.join(BASE, "output", "results", "prophet_baseline_results.csv")
results_df.to_csv(output_path, index=False)

# Summary
results_df = pd.DataFrame(results)
print(f"\nDone! {len(results_df)} LSOAs processed")
print(f"Average MAE:  {results_df['mae'].mean():.2f}")
print(f"Average RMSE: {results_df['rmse'].mean():.2f}")
print(f"Results saved to {OUTPUT_PATH}")

# Save summary
summary_path = os.path.join(BASE, "output", "results", "model_comparison.csv")
summary = pd.DataFrame([{
    "model": "baseline",
    "avg_mae": round(results_df["mae"].mean(), 3),
    "avg_rmse": round(results_df["rmse"].mean(), 3),
    "n_lsoas": len(results_df)
}])
summary.to_csv(summary_path, index=False)
print(f"Summary saved to {summary_path}")