import duckdb
import pandas as pd
from prophet import Prophet
import os

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL_PATH = os.path.join(BASE, "data", "cleaned_data", "panels", "london_panel.parquet")
WARD_MAPPING_PATH = os.path.join(BASE, "data", "lsoa_ward_mapping.csv")
OUTPUT_PATH = os.path.join(BASE, "output", "results", "prophet_brokerage_ward_cooc_results.csv")

# Settings
TARGET_CRIME = "Violence and sexual offences"
TRAIN_END = "2024-12-01"
TEST_START = "2025-01-01"

# Co-occurrence brokerage scores (max_rank - rank + 1)
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


# Step 1 - Load full panel (all crime types)
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


# Step 2 - Map LSOAs to wards (London only)
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


# Step 3 - Build brokerage activity feature per ward per month
# Same approach as LSOA model: weighted sum of raw counts × fixed brokerage scores
print("\nBuilding brokerage activity feature...")

broker_crimes = list(BROKERAGE_SCORES.keys())
broker_df = ward_df[ward_df["crime_type"].isin(broker_crimes)].copy()
broker_df["weighted"] = broker_df.apply(
    lambda r: r["crime_count"] * BROKERAGE_SCORES[r["crime_type"]], axis=1
)
brokerage_activity = (
    broker_df.groupby(["WD21CD", "month"])["weighted"]
    .sum()
    .reset_index()
    .rename(columns={"weighted": "brokerage_activity"})
)

# Normalise to 0-1
max_val = brokerage_activity["brokerage_activity"].max()
brokerage_activity["brokerage_activity"] = brokerage_activity["brokerage_activity"] / max_val

print(f"Brokerage activity range: [{brokerage_activity['brokerage_activity'].min():.4f}, {brokerage_activity['brokerage_activity'].max():.4f}]")


# Step 4 - Load violence at ward level and merge with brokerage activity
violence_ward = ward_df[ward_df["crime_type"] == TARGET_CRIME][["WD21CD", "month", "crime_count"]].copy()

merged_df = violence_ward.merge(brokerage_activity, on=["WD21CD", "month"], how="left")
merged_df["brokerage_activity"] = merged_df["brokerage_activity"].fillna(0)


# Step 5 - Run Prophet per ward
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


# Step 6 - Save results
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
    "model": "brokerage_cooc",
    "avg_mae": round(results_df["mae"].mean(), 3),
    "avg_rmse": round(results_df["rmse"].mean(), 3),
    "avg_mape": round(results_df["mape"].mean(), 3),
    "n_wards": len(results_df)
}])

if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    existing = existing[existing["model"] != "brokerage_cooc"]
    summary = pd.concat([existing, new_row], ignore_index=True)
else:
    summary = new_row

summary.to_csv(summary_path, index=False)
print(f"Summary saved to {summary_path}")