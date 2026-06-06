"""
Pooled LightGBM (Poisson) forecast of Violence and sexual offences (V&SO).

One single model trained across ALL wards together (pooled), so it can borrow
strength between wards — unlike the per-ward Prophet models. Counts target, so
objective='poisson' (log link => predictions are non-negative by construction; no
clipping needed).

Features (deliberately NO autoregressive V&SO lag — by design):
    WD21CD          : ward identity, CATEGORICAL -> lets the pooled model learn a
                      per-ward level (the between-ward signal). Replaces what a
                      per-ward model's intercept would do.
    month_of_year   : 1-12, CATEGORICAL -> seasonality (LightGBM does NOT infer
                      seasonality; as a category, Dec & Jan can both be "high"
                      without a false Jan<...<Dec ordering).
    time_index      : the integer `period` (year*12 + month-1), NUMERIC -> lets
                      the trees track slow trend.
    <regressor>     : the chosen brokerage feature, already lag-1 (known one month
                      ahead). Default = rank_activity_safe_lag1.

Protocol:
    Time-ordered split at SPLIT_DATE (no shuffling). Fit once on train, predict the
    whole test period. Because the regressor is lagged, test predictions are a
    genuine one-step-ahead forecast (no leakage). Same SPLIT_DATE / ward sample as
    run_forecast.py so the numbers are comparable to the Prophet results.

Reuses the existing panel + the exact lagged columns from build_panel (no re-lagging).

Run from repo root:
    venv\\Scripts\\python.exe -m code.new_models_test.lgbm_forecast
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb

# NB: import only from build_panel (NOT run_forecast) so we don't inherit its
# OMP_NUM_THREADS=1 setting, which would throttle LightGBM's internal threading.
from code.new_models_test.build_panel import (
    TARGET, build_ward_panel, add_date_columns, select_random_wards,
    add_lagged_regressors, lag_name,
)

# ============================ CONFIG (edit me) ===============================
N_WARDS = 1000               # random sample size (int) or "all"; matches run_forecast
RANDOM_SEED = 42             # same seed as run_forecast -> same wards, comparable
SPLIT_DATE = "2024-01-01"    # last TRAIN month (inclusive); test = the month after, onward
SCORE_START_YEAR = 2017      # rank weights fit on [START .. SPLIT month]

# The single brokerage regressor to add. Options that exist in the panel:
#   lag_name("rank_activity_safe")  -> rank-weighted activity (recommended)
#   lag_name("avg_betweenness")     -> share-weighted continuous betweenness
#   [lag_name(b) for b in BROKERS]  -> the 3 raw broker counts (use a list)
REGRESSOR = lag_name("rank_activity_safe")

# Feature lists (top-level so they're easy to tweak).
NUMERIC_FEATURES = ["time_index"]
CATEGORICAL_FEATURES = ["WD21CD", "month_of_year"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [REGRESSOR]

# Modest, anti-overfit LightGBM settings for a smallish monthly panel.
LGBM_PARAMS = dict(
    objective="poisson",        # counts target; predictions >= 0
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,      # high -> guards against per-ward overfit (WD21CD is high-cardinality)
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=-1,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PRED_PATH = OUTPUT_DIR / "lgbm_predictions.csv"
PER_WARD_PATH = OUTPUT_DIR / "lgbm_per_ward_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "lgbm_summary_metrics.csv"

# rank weights fit on the TRAINING window only (exact month cutoff -> no leakage).
_SPLIT = pd.Timestamp(SPLIT_DATE)
_SPLIT_PERIOD = _SPLIT.year * 12 + (_SPLIT.month - 1)
SCORE_WHERE_SQL = (
    f"year >= {SCORE_START_YEAR} AND (year * 12 + (month_num - 1)) <= {_SPLIT_PERIOD}"
)
# ============================================================================


def _metrics(df: pd.DataFrame, keys) -> pd.DataFrame:
    """MAE / RMSE / WMAPE grouped by `keys`.

    WMAPE (volume-weighted MAPE) = sum|err| / sum(actual). Unlike a plain per-row
    MAPE it isn't dominated by low-count wards (where dividing by a tiny actual
    blows the percentage up), so it's the honest percentage metric for counts.
    """
    g = df.groupby(keys)
    out = g.agg(mae=("abs_err", "mean"),
                rmse=("sq_err", "mean"),
                abs_err_sum=("abs_err", "sum"),
                y_sum=("y_true", "sum"),
                n=("abs_err", "size"))
    out["rmse"] = np.sqrt(out["rmse"])
    out["wmape"] = out["abs_err_sum"] / out["y_sum"].replace(0, np.nan) * 100
    return out.drop(columns=["abs_err_sum", "y_sum"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(SPLIT_DATE)
    reg_cols = REGRESSOR if isinstance(REGRESSOR, list) else [REGRESSOR]

    print(f"Config: N_WARDS={N_WARDS}  SPLIT_DATE={SPLIT_DATE}  regressor={reg_cols}")
    print(f"Rank weights fit on: {SCORE_WHERE_SQL}")

    # --- build the panel (reuses existing, already-within-ward-lagged columns) -
    print("Loading ~49M-row crime table and building panel (a few minutes)...", flush=True)
    panel = build_ward_panel(score_where_sql=SCORE_WHERE_SQL)
    panel = add_date_columns(panel)
    panel, selected = select_random_wards(panel, N_WARDS, RANDOM_SEED)
    panel = add_lagged_regressors(panel)   # drops the first month per ward (NaN lags)

    # --- calendar features + categoricals ------------------------------------
    panel["month_of_year"] = panel["month"]
    panel["time_index"] = panel["period"]

    # NB: WD21CD is already in FEATURES (it's a categorical feature), so don't
    # list it again here or `data` ends up with a duplicate column.
    data = panel[["ds", TARGET] + FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        data[c] = data[c].astype("category")

    # --- confirm no NaNs in the feature matrix (constraint) ------------------
    print("\nNaNs per column (features + target):")
    print(data[FEATURES + [TARGET]].isna().sum().to_string())
    before = len(data)
    data = data.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    if len(data) != before:
        print(f"Dropped {before - len(data)} rows with NaNs (first-month-per-ward lags).")

    # --- time-ordered split (NO shuffle) -------------------------------------
    train = data[data["ds"] <= split]
    test = data[data["ds"] > split]
    print(f"\nWards: {data['WD21CD'].nunique()} | train rows: {len(train):,} | test rows: {len(test):,}")
    if test.empty:
        print("No test rows — nothing to score.")
        return

    # --- fit one pooled Poisson model ----------------------------------------
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[FEATURES], train[TARGET], categorical_feature=CATEGORICAL_FEATURES)

    # Poisson => predictions are already non-negative.
    preds = test[["WD21CD", "ds", TARGET]].rename(columns={TARGET: "y_true"}).reset_index(drop=True)
    preds["y_pred"] = model.predict(test[FEATURES])

    # --- metrics (same definitions as run_forecast, for comparability) -------
    err = preds["y_pred"] - preds["y_true"]
    preds["abs_err"] = err.abs()
    preds["sq_err"] = err ** 2

    preds[["WD21CD", "ds", "y_true", "y_pred"]].to_csv(PRED_PATH, index=False)

    overall = {                                              # pooled across all rows
        "mae": preds["abs_err"].mean(),
        "rmse": float(np.sqrt(preds["sq_err"].mean())),
        # WMAPE = sum|err| / sum(actual): volume-weighted, robust to low-count wards
        "wmape": preds["abs_err"].sum() / preds["y_true"].sum() * 100,
        "n": len(preds),
    }
    per_ward = _metrics(preds, "WD21CD").reset_index()
    per_ward.to_csv(PER_WARD_PATH, index=False)

    summary = pd.DataFrame({
        "model": ["lgbm_poisson"],
        "mae": [overall["mae"]],
        "rmse": [overall["rmse"]],
        "wmape": [overall["wmape"]],
        "n": [int(overall["n"])],
        "mae_ward_mean": [per_ward["mae"].mean()],
        "rmse_ward_mean": [per_ward["rmse"].mean()],
        "wmape_ward_mean": [per_ward["wmape"].mean()],
        "mae_ward_med": [per_ward["mae"].median()],
        "rmse_ward_med": [per_ward["rmse"].median()],
        "wmape_ward_med": [per_ward["wmape"].median()],
        "n_wards": [per_ward["WD21CD"].nunique()],
    })
    summary.to_csv(SUMMARY_PATH, index=False)

    # --- report --------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"POOLED LightGBM (Poisson)  target = {TARGET}, counts")
    print("=" * 70)
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(summary.to_string(index=False))

    imp = pd.DataFrame({
        "feature": model.booster_.feature_name(),
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    print("\nFeature importance (gain):")
    with pd.option_context("display.float_format", lambda v: f"{v:,.0f}"):
        print(imp.to_string(index=False))

    print(f"\nSaved:\n  {PRED_PATH}\n  {PER_WARD_PATH}\n  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
