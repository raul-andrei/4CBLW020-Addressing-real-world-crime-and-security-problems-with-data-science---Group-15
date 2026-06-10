import duckdb
import numpy as np
import pandas as pd
from prophet import Prophet
import os

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL_PATH = os.path.join(BASE, "data", "cleaned_data", "panels", "london_panel.parquet")
BROKERAGE_SCORES_PATH = os.path.join(BASE, "data", "global_brokerage.csv")
OUTPUT_PATH = os.path.join(BASE, "output", "results", "prophet_brokerage_new_scores_results.csv")

# Settings
TARGET_CRIME = "Violence and sexual offences"
TRAIN_END = "2024-12-01"
TEST_START = "2025-01-01"
N_LSOAS = 300


# Step 1 - Load and log-normalize brokerage scores from Nikola's file
print("Loading brokerage scores...")
bc_df = pd.read_csv(BROKERAGE_SCORES_PATH)
bc_scores = dict(zip(bc_df["crime_type"], bc_df["current_flow_betweenness"]))

# Log-normalize: log(1 + score) then divide by max
bc_normalized = {crime: np.log1p(score) for crime, score in bc_scores.items()}
max_log = max(bc_normalized.values())
bc_normalized = {crime: score / max_log for crime, score in bc_normalized.items()}

print("Log-normalized brokerage scores:")
for crime, score in sorted(bc_normalized.items(), key=lambda x: x[1], reverse=True):
    print(f"  {crime}: {score:.4f}")


# Step 2 - Load full panel
print("\nLoading panel...")
con = duckdb.connect()
df = con.execute(f"""
    SELECT lsoa_code, month, crime_type, crime_count
    FROM read_parquet('{PANEL_PATH}')
    ORDER BY lsoa_code, month, crime_type
""").df()
con.close()

df["month"] = pd.to_datetime(df["month"])
print(f"Loaded {len(df):,} rows")


# Step 3 - Compute proportion-based brokerage activity per LSOA per month
# For each LSOA and month:
#   - compute share of each crime type (count / total crimes that month)
#   - multiply share by log-normalized brokerage score
#   - sum across all crime types -> one score per LSOA per month
print("\nBuilding proportion-based brokerage activity feature...")

# Only keep crime types that have a brokerage score
scored_crimes = set(bc_normalized.keys())
broker_df = df[df["crime_type"].isin(scored_crimes)].copy()

missing = set(df["crime_type"].unique()) - scored_crimes
if missing:
    print(f"Crime types with no brokerage score (excluded): {missing}")

# Total crimes per LSOA per month (across all scored crime types)
totals = broker_df.groupby(["lsoa_code", "month"])["crime_count"].sum().reset_index(name="total")
broker_df = broker_df.merge(totals, on=["lsoa_code", "month"])

# Share of each crime type
broker_df["share"] = broker_df["crime_count"] / broker_df["total"]

# Weighted share by brokerage score
broker_df["brokerage_score"] = broker_df["crime_type"].map(bc_normalized)
broker_df["weighted"] = broker_df["share"] * broker_df["brokerage_score"]

# Sum weighted shares -> avg_betweenness per LSOA per month
brokerage_activity = (
    broker_df.groupby(["lsoa_code", "month"])["weighted"]
    .sum()
    .reset_index()
    .rename(columns={"weighted": "brokerage_activity"})
)

print(f"Brokerage activity range: [{brokerage_activity['brokerage_activity'].min():.4f}, {brokerage_activity['brokerage_activity'].max():.4f}]")


# Step 4 - Load violence data and merge with brokerage activity
violence_df = df[df["crime_type"] == TARGET_CRIME][["lsoa_code", "month", "crime_count"]].copy()

merged_df = violence_df.merge(brokerage_activity, on=["lsoa_code", "month"], how="left")
merged_df["brokerage_activity"] = merged_df["brokerage_activity"].fillna(0)


# Step 5 - Run Prophet per LSOA
lsoas = merged_df["lsoa_code"].unique()
if N_LSOAS:
    lsoas = lsoas[:N_LSOAS]

print(f"\nRunning Prophet on {len(lsoas)} LSOAs...")
results = []

for i, lsoa in enumerate(lsoas):
    lsoa_df = merged_df[merged_df["lsoa_code"] == lsoa].copy()
    lsoa_df = lsoa_df.rename(columns={"month": "ds", "crime_count": "y"})

    train = lsoa_df[lsoa_df["ds"] <= TRAIN_END]
    test = lsoa_df[lsoa_df["ds"] >= TEST_START]

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
        "lsoa_code": lsoa,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 3),
        "n_test": len(comp)
    })

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(lsoas)} done...")


# Step 6 - Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nDone! {len(results_df)} LSOAs processed")
print(f"Average MAE:  {results_df['mae'].mean():.3f}")
print(f"Average RMSE: {results_df['rmse'].mean():.3f}")
print(f"Average MAPE: {results_df['mape'].mean():.3f}%")
print(f"Results saved to {OUTPUT_PATH}")

# Save to model comparison summary
summary_path = os.path.join(BASE, "output", "results", "model_comparison.csv")

new_row = pd.DataFrame([{
    "model": "brokerage_new_scores",
    "avg_mae": round(results_df["mae"].mean(), 3),
    "avg_rmse": round(results_df["rmse"].mean(), 3),
    "avg_mape": round(results_df["mape"].mean(), 3),
    "n_lsoas": len(results_df)
}])

if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    existing = existing[existing["model"] != "brokerage_new_scores"]
    summary = pd.concat([existing, new_row], ignore_index=True)
else:
    summary = new_row

summary.to_csv(summary_path, index=False)
print(f"Summary saved to {summary_path}")