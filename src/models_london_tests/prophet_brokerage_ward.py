import duckdb
import numpy as np
import pandas as pd
from prophet import Prophet
import os

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL_PATH = os.path.join(BASE, "data", "cleaned_data", "panels", "london_panel.parquet")
WARD_MAPPING_PATH = os.path.join(BASE, "data", "lsoa_ward_mapping.csv")
BROKERAGE_SCORES_PATH = os.path.join(BASE, "data", "global_brokerage.csv")
OUTPUT_PATH = os.path.join(BASE, "output", "results", "prophet_brokerage_ward_results.csv")

# Settings
TARGET_CRIME = "Violence and sexual offences"
TRAIN_END = "2024-12-01"
TEST_START = "2025-01-01"


# Step 1 - Load and log-normalize brokerage scores
print("Loading brokerage scores...")
bc_df = pd.read_csv(BROKERAGE_SCORES_PATH)
bc_scores = dict(zip(bc_df["crime_type"], bc_df["current_flow_betweenness"]))

bc_normalized = {crime: np.log1p(score) for crime, score in bc_scores.items()}
max_log = max(bc_normalized.values())
bc_normalized = {crime: score / max_log for crime, score in bc_normalized.items()}

print("Log-normalized brokerage scores:")
for crime, score in sorted(bc_normalized.items(), key=lambda x: x[1], reverse=True):
    print(f"  {crime}: {score:.4f}")


# Step 2 - Load full panel (all crime types)
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


# Step 3 - Map LSOAs to wards
print("Mapping LSOAs to wards...")
ward_mapping = pd.read_csv(WARD_MAPPING_PATH, usecols=["LSOA11CD", "WD21CD", "LAD21CD"], encoding="utf-8-sig")
ward_mapping = ward_mapping[ward_mapping["LAD21CD"].str.startswith("E09")]

df = df.merge(ward_mapping, left_on="lsoa_code", right_on="LSOA11CD", how="left")

missing = df["WD21CD"].isna().sum()
if missing > 0:
    print(f"Warning: {missing} rows could not be mapped to a ward and will be dropped")
    df = df.dropna(subset=["WD21CD"])

# Aggregate crime counts from LSOA to ward level
ward_df = df.groupby(["WD21CD", "month", "crime_type"])["crime_count"].sum().reset_index()
print(f"Aggregated to {ward_df['WD21CD'].nunique()} wards")


# Step 4 - Compute proportion-based brokerage activity per ward per month
print("\nBuilding proportion-based brokerage activity feature...")

scored_crimes = set(bc_normalized.keys())
broker_df = ward_df[ward_df["crime_type"].isin(scored_crimes)].copy()

missing_crimes = set(ward_df["crime_type"].unique()) - scored_crimes
if missing_crimes:
    print(f"Crime types with no brokerage score (excluded): {missing_crimes}")

# Total crimes per ward per month
totals = broker_df.groupby(["WD21CD", "month"])["crime_count"].sum().reset_index(name="total")
broker_df = broker_df.merge(totals, on=["WD21CD", "month"])

# Share of each crime type
broker_df["share"] = broker_df["crime_count"] / broker_df["total"]

# Weighted share by brokerage score
broker_df["brokerage_score"] = broker_df["crime_type"].map(bc_normalized)
broker_df["weighted"] = broker_df["share"] * broker_df["brokerage_score"]

# Sum weighted shares -> one brokerage activity score per ward per month
brokerage_activity = (
    broker_df.groupby(["WD21CD", "month"])["weighted"]
    .sum()
    .reset_index()
    .rename(columns={"weighted": "brokerage_activity"})
)

print(f"Brokerage activity range: [{brokerage_activity['brokerage_activity'].min():.4f}, {brokerage_activity['brokerage_activity'].max():.4f}]")


# Step 5 - Load violence at ward level and merge with brokerage activity
violence_ward = ward_df[ward_df["crime_type"] == TARGET_CRIME][["WD21CD", "month", "crime_count"]].copy()

merged_df = violence_ward.merge(brokerage_activity, on=["WD21CD", "month"], how="left")
merged_df["brokerage_activity"] = merged_df["brokerage_activity"].fillna(0)


# Step 6 - Run Prophet per ward
wards = merged_df["WD21CD"].unique()
print(f"\nRunning Prophet on {len(wards)} wards...")
results = []

for i, ward in enumerate(wards):
    w_df = merged_df[merged_df["WD21CD"] == ward].copy()
    w_df = w_df.rename(columns={"month": "ds", "crime_count": "y"})

    train = w_df[w_df["ds"] <= TRAIN_END]
    test = w_df[w_df["ds"] >= TEST_START]

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
        "ward_code": ward,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 3),
        "n_test": len(comp)
    })

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(wards)} done...")


# Step 7 - Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nDone! {len(results_df)} wards processed")
print(f"Average MAE:  {results_df['mae'].mean():.3f}")
print(f"Average RMSE: {results_df['rmse'].mean():.3f}")
print(f"Average MAPE: {results_df['mape'].mean():.3f}%")
print(f"Results saved to {OUTPUT_PATH}")

# Save to summary
summary_path = os.path.join(BASE, "output", "results", "model_comparison_ward.csv")

new_row = pd.DataFrame([{
    "model": "brokerage_new_scores",
    "avg_mae": round(results_df["mae"].mean(), 3),
    "avg_rmse": round(results_df["rmse"].mean(), 3),
    "avg_mape": round(results_df["mape"].mean(), 3),
    "n_wards": len(results_df)
}])

if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    existing = existing[existing["model"] != "brokerage_new_scores"]
    summary = pd.concat([existing, new_row], ignore_index=True)
else:
    summary = new_row

summary.to_csv(summary_path, index=False)
print(f"Summary saved to {summary_path}")