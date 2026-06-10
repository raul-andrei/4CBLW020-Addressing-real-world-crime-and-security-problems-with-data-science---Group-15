import duckdb
import pandas as pd
from prophet import Prophet
import os


# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL_PATH = os.path.join(BASE, "data", "cleaned_data", "panels", "london_panel_borough.parquet")
OUTPUT_PATH = os.path.join(BASE, "output", "results", "prophet_brokerage_borough_results.csv")

# Settings
TARGET_CRIME = "Violence and sexual offences"
TRAIN_END = "2024-12-01"
TEST_START = "2025-01-01"

# Brokerage scores
BROKERAGE_SCORES = {
    "Theft from the person": (12.2 + 10.5) / 2,
    "Possession of weapons": (12.0 + 10.5) / 2,
    "Robbery": (11.5 + 12.2) / 2,
    "Bicycle theft": (10.5 + 4.8) / 2,
    "Criminal damage and arson": (8.0 + 5.8) / 2,
    "Drugs": (7.3 + 7.5) / 2,
    "Public order": (7.1 + 7.3) / 2,
    "Anti-social behaviour": (6.9 + 2.1) / 2,
    "Other theft": (6.1 + 6.7) / 2,
    "Shoplifting": (5.8 + 9.8) / 2,
    "Vehicle crime": (5.4 + 6.4) / 2,
    "Burglary": (5.3 + 4.8) / 2,
}

# Load full panel
print("Loading panel...")
con = duckdb.connect()
df = con.execute(f"""
    SELECT borough, month, crime_type, crime_count
    FROM read_parquet('{PANEL_PATH}')
    ORDER BY borough, month, crime_type
""").df()
con.close()

df["month"] = pd.to_datetime(df["month"])
print(f"Loaded {len(df):,} rows")

# Build brokerage activity feature per borough per month
print("Building brokerage activity feature...")
broker_crimes = list(BROKERAGE_SCORES.keys())
broker_df = df[df["crime_type"].isin(broker_crimes)].copy()
broker_df["weighted"] = broker_df.apply(
    lambda r: r["crime_count"] * BROKERAGE_SCORES[r["crime_type"]], axis=1
)
brokerage_activity = (
    broker_df.groupby(["borough", "month"])["weighted"]
    .sum()
    .reset_index()
    .rename(columns={"weighted": "brokerage_activity"})
)

# Normalise to 0-1
max_val = brokerage_activity["brokerage_activity"].max()
brokerage_activity["brokerage_activity"] = brokerage_activity["brokerage_activity"] / max_val

# Load violence data
violence_df = df[df["crime_type"] == TARGET_CRIME][["borough", "month", "crime_count"]].copy()

# Merge violence and brokerage activity
merged_df = violence_df.merge(brokerage_activity, on=["borough", "month"], how="left")
merged_df["brokerage_activity"] = merged_df["brokerage_activity"].fillna(0)

# Run Prophet per borough
boroughs = merged_df["borough"].unique()
print(f"Running Prophet on {len(boroughs)} boroughs...")
results = []

for i, borough in enumerate(boroughs):
    borough_df = merged_df[merged_df["borough"] == borough].copy()
    borough_df = borough_df.rename(columns={"month": "ds", "crime_count": "y"})

    train = borough_df[borough_df["ds"] <= TRAIN_END]
    test = borough_df[borough_df["ds"] >= TEST_START]

    if len(train) < 24 or len(test) == 0:
        continue

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="additive")
    model.add_regressor("brokerage_activity")
    model.fit(train)

    future = test[["ds", "brokerage_activity"]].copy()
    future = pd.concat([train[["ds", "brokerage_activity"]], future], ignore_index=True)
    forecast = model.predict(future)

    pred = forecast[forecast["ds"] >= TEST_START][["ds", "yhat"]].reset_index(drop=True)
    actual = test[["ds", "y"]].reset_index(drop=True)
    comp = pred.merge(actual, on="ds")

    mae = (comp["yhat"] - comp["y"]).abs().mean()
    rmse = ((comp["yhat"] - comp["y"]) ** 2).mean() ** 0.5
    mape = ((comp["yhat"] - comp["y"]).abs() / comp["y"].replace(0, float("nan"))).mean() * 100

    results.append({
        "borough": borough,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 3),
        "n_test": len(comp)
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

if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    existing = existing[existing["model"] != "brokerage"]
    new_row = pd.DataFrame([{
        "model": "brokerage",
        "avg_mae": round(results_df["mae"].mean(), 3),
        "avg_rmse": round(results_df["rmse"].mean(), 3),
        "avg_mape": round(results_df["mape"].mean(), 3),
        "n_boroughs": len(results_df)
    }])
    summary = pd.concat([existing, new_row], ignore_index=True)
else:
    summary = pd.DataFrame([{
        "model": "brokerage",
        "avg_mae": round(results_df["mae"].mean(), 3),
        "avg_rmse": round(results_df["rmse"].mean(), 3),
        "avg_mape": round(results_df["mape"].mean(), 3),
        "n_boroughs": len(results_df)
    }])

summary.to_csv(summary_path, index=False)
print(f"Summary saved to {summary_path}")