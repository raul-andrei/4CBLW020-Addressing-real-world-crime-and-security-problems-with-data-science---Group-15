import duckdb
import pandas as pd
from prophet import Prophet
import os

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL_PATH = os.path.join(BASE, "data", "cleaned_data", "panels", "london_panel.parquet")
OUTPUT_PATH = os.path.join(BASE, "output", "results", "prophet_brokerage_results.csv")

# Settings
TARGET_CRIME = "Violence and sexual offences"
TRAIN_END = "2024-12-01"
TEST_START = "2025-01-01"
N_LSOAS = 300

# Brokerage scores combined
# BROKERAGE_SCORES = {
#     "Theft from the person": (12.2 + 10.5) / 2,
#     "Possession of weapons": (12.0 + 10.5) / 2,
#     "Robbery": (11.5 + 12.2) / 2,
#     "Bicycle theft": (10.5 + 4.8) / 2,
#     "Criminal damage and arson": (8.0 + 5.8) / 2,
#     "Violence and sexual offences": (7.2 + 6.1) / 2,
#     "Drugs": (7.3 + 7.5) / 2,
#     "Public order": (7.1 + 7.3)  / 2,
#     "Anti-social behaviour": (6.9 + 2.1) / 2,
#     "Other theft": (6.1 + 6.7) / 2,
#     "Shoplifting": (5.8 + 9.8) / 2,
#     "Vehicle crime": (5.4 + 6.4) / 2, 
#     "Burglary": (5.3 + 4.8) / 2,
# }

# BROKERAGE SCORES Co-occurrence
BROKERAGE_SCORES = {
    "Theft from the person": 8.638 - 1.917 + 1,
    "Possession of weapons": 8.638 - 2.167 + 1,
    "Robbery": 8.638 - 2.417 + 1,
    "Bicycle theft": 8.638 - 3.472 + 1,
    "Criminal damage and arson": 8.638 - 5.972 + 1,
    "Drugs": 8.638 - 6.694 + 1,
    "Violence and sexual offences": 8.638 - 6.694 + 1,
    "Public order": 8.638 - 6.833 + 1,
    "Anti-social behaviour": 8.638 - 7.056 + 1,
    "Other theft": 8.638 - 7.889 + 1,
    "Shoplifting": 8.638 - 8.139 + 1,
    "Vehicle crime": 8.638 - 8.500 + 1,
    "Burglary": 8.638 - 8.638 + 1,
}

# BROKERAGE SCORES Spearman
# BROKERAGE_SCORES = {
#     "Robbery": 11.944 - 1.944 + 1,
#     "Theft from the person": 11.944 - 3.472 + 1,
#     "Possession of weapons": 11.944 - 3.556 + 1,
#     "Shoplifting": 11.944 - 4.250 + 1,
#     "Drugs": 11.944 - 6.444 + 1,
#     "Public order": 11.944 - 6.583 + 1,
#     "Other theft": 11.944 - 7.167 + 1,
#     "Vehicle crime": 11.944 - 7.472 + 1,
#     "Violence and sexual offences": 11.944 - 7.917 + 1,
#     "Criminal damage and arson": 11.944 - 8.194 + 1,
#     "Bicycle theft": 11.944 - 9.194 + 1,
#     "Burglary": 11.944 - 9.278 + 1,
#     "Anti-social behaviour": 11.944 - 11.944 + 1,
# }

# Load full panel
print("Loading panel...")
con = duckdb.connect()
df = con.execute(f"""
    SELECT lsoa_code, month, crime_type, crime_count
    FROM read_parquet('{PANEL_PATH}')
    ORDER BY lsoa_code, month, crime_type
""").df()
con.close()

df["month"] = pd.to_datetime(df["month"])
print(f"Loaded {len(df):,} rows")

# Build brokerage activity feature per LSOA per month
print("Building brokerage activity feature...")

broker_crimes = list(BROKERAGE_SCORES.keys())
broker_df = df[df["crime_type"].isin(broker_crimes)].copy()
broker_df["weighted"] = broker_df.apply(
    lambda r: r["crime_count"] * BROKERAGE_SCORES[r["crime_type"]], axis=1
)
brokerage_activity = (
    broker_df.groupby(["lsoa_code", "month"])["weighted"]
    .sum()
    .reset_index()
    .rename(columns={"weighted": "brokerage_activity"})
)

# Normalise to 0-1
max_val = brokerage_activity["brokerage_activity"].max()
brokerage_activity["brokerage_activity"] = brokerage_activity["brokerage_activity"] / max_val

# Load violence data
violence_df = df[df["crime_type"] == TARGET_CRIME][["lsoa_code", "month", "crime_count"]].copy()

# Merge violence and brokerage activity
merged_df = violence_df.merge(brokerage_activity, on=["lsoa_code", "month"], how="left")
merged_df["brokerage_activity"] = merged_df["brokerage_activity"].fillna(0)

# Run Prophet per LSOA
lsoas = merged_df["lsoa_code"].unique()
if N_LSOAS:
    lsoas = lsoas[:N_LSOAS]

print(f"Running Prophet on {len(lsoas)} LSOAs...")
results = []

for i, lsoa in enumerate(lsoas):
    lsoa_df = merged_df[merged_df["lsoa_code"] == lsoa].copy()
    lsoa_df = lsoa_df.rename(columns={"month": "ds", "crime_count": "y"})

    train = lsoa_df[lsoa_df["ds"] <= TRAIN_END]
    test = lsoa_df[lsoa_df["ds"] >= TEST_START]

    if len(train) < 24 or len(test) == 0:
        continue

    # Fit model with brokerage activity as extra regressor
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="additive")
    model.add_regressor("brokerage_activity")
    model.fit(train)

    # Predict
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
        "lsoa_code": lsoa,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 3),
        "n_test": len(comp)
    })

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(lsoas)} done...")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nDone! {len(results_df)} LSOAs processed")
print(f"Average MAE: {results_df['mae'].mean():.2f}")
print(f"Average RMSE: {results_df['rmse'].mean():.2f}")
print(f"Average MAPE: {results_df['mape'].mean():.2f}%")
print(f"Results saved to {OUTPUT_PATH}")

# Save summary
summary_path = os.path.join(BASE, "output", "results", "model_comparison.csv")

new_row = pd.DataFrame([{
    "model": "brokerage",
    "avg_mae": round(results_df["mae"].mean(), 3),
    "avg_rmse": round(results_df["rmse"].mean(), 3),
    "avg_mape": round(results_df["mape"].mean(), 3),
    "n_lsoas": len(results_df)
}])

if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    existing = existing[existing["model"] != "brokerage"]  # remove old brokerage row
    summary = pd.concat([existing, new_row], ignore_index=True)
else:
    summary = new_row

summary.to_csv(summary_path, index=False)
print(f"Summary saved to {summary_path}")