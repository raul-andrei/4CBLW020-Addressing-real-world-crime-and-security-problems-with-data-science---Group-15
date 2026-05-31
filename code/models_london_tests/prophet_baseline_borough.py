import duckdb
import pandas as pd
from prophet import Prophet
import os

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PANEL_PATH = os.path.join(BASE, "data", "cleaned_data", "panels", "london_panel_borough.parquet")
OUTPUT_PATH = os.path.join(BASE, "output", "results", "prophet_baseline_borough_results.csv")

# Settings
TARGET_CRIME = "Violence and sexual offences"
TRAIN_END = "2024-12-01"
TEST_START = "2025-01-01"

# Load data
print("Loading panel...")
con = duckdb.connect()

df = con.execute(f"""
    SELECT borough, month, crime_count
    FROM read_parquet('{PANEL_PATH}')
    WHERE crime_type = '{TARGET_CRIME}'
    ORDER BY borough, month
""").df()

con.close()

print(f"Loaded {len(df):,} rows for '{TARGET_CRIME}'")

# Get list of boroughs
boroughs = df["borough"].unique()
print(f"Running Prophet on {len(boroughs)} boroughs...")

# Run Prophet per borough
results = []

for i, borough in enumerate(boroughs):
    borough_df = df[df["borough"] == borough].copy()
    borough_df = borough_df.rename(columns={"month": "ds", "crime_count": "y"})
    borough_df["ds"] = pd.to_datetime(borough_df["ds"])

    # Split train/test
    train = borough_df[borough_df["ds"] <= TRAIN_END]
    test = borough_df[borough_df["ds"] >= TEST_START]

    if len(train) < 24 or len(test) == 0:
        continue

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
    mape = ((merged["yhat"] - merged["y"]).abs() / merged["y"].replace(0, float("nan"))).mean() * 100

    results.append({
        "borough":  borough,
        "mae":      round(mae, 3),
        "rmse":     round(rmse, 3),
        "mape":     round(mape, 3),
        "n_test":   len(merged)
    })

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(boroughs)} done...")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

# Summary
print(f"\nDone! {len(results_df)} boroughs processed")
print(f"Average MAE:  {results_df['mae'].mean():.2f}")
print(f"Average RMSE: {results_df['rmse'].mean():.2f}")
print(f"Average MAPE: {results_df['mape'].mean():.2f}%")
print(f"Results saved to {OUTPUT_PATH}")

# Save summary
summary_path = os.path.join(BASE, "output", "results", "model_comparison_borough.csv")
summary = pd.DataFrame([{
    "model":      "baseline",
    "avg_mae":    round(results_df["mae"].mean(), 3),
    "avg_rmse":   round(results_df["rmse"].mean(), 3),
    "avg_mape":   round(results_df["mape"].mean(), 3),
    "n_boroughs": len(results_df)
}])
summary.to_csv(summary_path, index=False)
print(f"Summary saved to {summary_path}")