"""
Forecast Violence and sexual offences (V&SO) per ward with Prophet, and test
whether broker-crime activity (lagged) improves on a plain baseline.

Three model variants per ward (all COUNTS, never shares):
    baseline : Prophet, trend + yearly seasonality only
    score    : baseline + avg_betweenness_lag1            (brokerage activity)
    brokers  : baseline + Robbery_lag1, Theft from the person_lag1,
                          Possession of weapons_lag1

Evaluation: one-step-ahead on the test period (Jan 2023+).
    FAST_MODE = True  -> fit once on the training window, predict the whole test
                         period in one go (regressors still supplied). Fast demo.
    FAST_MODE = False -> expanding-window refit: for each test month t, fit on
                         everything up to t-1 and predict t. Rigorous, slower.

Run from the repo root:
    venv\\Scripts\\python.exe -m code.new_models_test.run_forecast
    (or just run this file directly; it bootstraps sys.path.)
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

# --- make `from code...` imports work whether run as a module or a script ----
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# --- silence Prophet / cmdstanpy / stan console flooding (must precede import) -
for _name in ("prophet", "cmdstanpy", "stan", "numexpr"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)
logging.getLogger("cmdstanpy").disabled = True
warnings.simplefilter("ignore")

from prophet import Prophet  # noqa: E402

try:
    from tqdm import tqdm  # progress bar over wards
except ImportError:  # optional dependency; fall back to a no-op wrapper
    def tqdm(it, **kwargs):
        return it

from code.new_models_test.build_panel import (  # noqa: E402
    TARGET,
    BROKERS,
    build_ward_panel,
    add_date_columns,
    select_top_wards,
    add_lagged_regressors,
    validate_columns,
    lag_name,
)

# ============================ CONFIG (edit me) ===============================
N_WARDS = 100                  # int, or "all"
SPLIT_DATE = "2024-01-01"     # last TRAIN month (inclusive); test = the month after, onward
FAST_MODE = False              # True = fit-once demo; False = expanding-window refit
SCORE_START_YEAR = 2017       # brokerage weights are fit on [START .. SPLIT year]
MIN_TRAIN_MONTHS = 24         # skip wards with fewer training months than this

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PRED_PATH = OUTPUT_DIR / "predictions.csv"
PER_WARD_PATH = OUTPUT_DIR / "per_ward_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary_metrics.csv"

# avg_betweenness AND rank_safe weights are fit on the TRAINING window only.
# Use an exact MONTH cutoff (not just the split year): if SPLIT_DATE falls
# mid-year, the later months of that year are in the TEST set and must not leak
# into the weights. period = year*12 + (month_num-1), matching scores.py.
_SPLIT = pd.Timestamp(SPLIT_DATE)
_SPLIT_PERIOD = _SPLIT.year * 12 + (_SPLIT.month - 1)
SCORE_WHERE_SQL = (
    f"year >= {SCORE_START_YEAR} AND (year * 12 + (month_num - 1)) <= {_SPLIT_PERIOD}"
)

# variant_name -> list of regressor columns (already lag-1, already known at t)
#   score    : your continuous current-flow-betweenness, proportion-weighted
#   brokers  : the 3 raw broker counts (Prophet learns the weights)
#   rank_his : Raul's rank-weighted activity, his ORIGINAL all-data weights (leaky)
#   rank_safe: same construction, weights recomputed on the train window (no leak)
VARIANTS: dict[str, list[str]] = {
    "baseline": [],
    "score": [lag_name("avg_betweenness")],
    "brokers": [lag_name(b) for b in BROKERS],
    "rank_his": [lag_name("rank_activity")],
    "rank_safe": [lag_name("rank_activity_safe")],
}
# ============================================================================


def make_model() -> Prophet:
    """A plain monthly Prophet: trend + yearly seasonality, nothing else."""
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )


def _fit(train: pd.DataFrame, regs: list[str]) -> Prophet:
    model = make_model()
    for r in regs:
        model.add_regressor(r)
    model.fit(train[["ds", "y"] + regs])
    return model


def predict_fast(train: pd.DataFrame, test: pd.DataFrame, regs: list[str]) -> pd.DataFrame:
    """Fit once on train, predict the whole test period in one shot."""
    model = _fit(train, regs)
    forecast = model.predict(test[["ds"] + regs])
    out = test[["ds", "y"]].reset_index(drop=True).copy()
    # Counts can't be negative -> clip; keeps every variant on the same footing.
    out["y_pred"] = forecast["yhat"].clip(lower=0).to_numpy()
    return out


def predict_rolling(wdf: pd.DataFrame, split: pd.Timestamp, regs: list[str]) -> pd.DataFrame:
    """Expanding-window one-step-ahead: refit per test month on data up to t-1."""
    test = wdf[wdf["ds"] > split].sort_values("ds")
    rows = []
    for _, r in test.iterrows():
        t = r["ds"]
        train_t = wdf[wdf["ds"] < t]
        if len(train_t) < MIN_TRAIN_MONTHS:
            continue
        model = _fit(train_t, regs)
        fut = pd.DataFrame({"ds": [t], **{reg: [r[reg]] for reg in regs}})
        yhat = float(model.predict(fut)["yhat"].iloc[0])
        rows.append({"ds": t, "y": r["y"], "y_pred": max(0.0, yhat)})
    return pd.DataFrame(rows)


def evaluate() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(SPLIT_DATE)

    print(f"Config: N_WARDS={N_WARDS}  SPLIT_DATE={SPLIT_DATE}  FAST_MODE={FAST_MODE}", flush=True)
    print(f"Brokerage weights fit on: {SCORE_WHERE_SQL}", flush=True)

    # --- build panel ---------------------------------------------------------
    print("Loading ~49M-row crime table from DuckDB and aggregating "
          "(slowest step, no bar — give it a few minutes)...", flush=True)
    panel = build_ward_panel(score_where_sql=SCORE_WHERE_SQL)
    validate_columns(panel)
    panel = add_date_columns(panel)
    panel, selected = select_top_wards(panel, N_WARDS)
    print(f"Selected {len(selected)} wards by total {TARGET!r}.")
    panel = add_lagged_regressors(panel)

    reg_cols = sorted({c for regs in VARIANTS.values() for c in regs})
    keep = ["WD21CD", "ds", TARGET] + reg_cols

    # --- per-ward modelling --------------------------------------------------
    all_preds: list[pd.DataFrame] = []
    skipped: list[str] = []
    wards = panel["WD21CD"].unique()

    for ward in tqdm(wards, desc="modelling wards", unit="ward"):
        wdf = (
            panel.loc[panel["WD21CD"] == ward, keep]
            .rename(columns={TARGET: "y"})
            .sort_values("ds")
            .reset_index(drop=True)
        )
        train = wdf[wdf["ds"] <= split]
        test = wdf[wdf["ds"] > split]

        if len(train) < MIN_TRAIN_MONTHS or len(test) == 0:
            skipped.append(ward)
            continue

        for variant, regs in VARIANTS.items():
            if FAST_MODE:
                pred = predict_fast(train, test, regs)
            else:
                pred = predict_rolling(wdf, split, regs)
            if pred.empty:
                continue
            pred = pred.assign(WD21CD=ward, variant=variant)
            all_preds.append(pred)

    if skipped:
        print(f"\nSkipped {len(skipped)} ward(s) with < {MIN_TRAIN_MONTHS} "
              f"training months or no test data: {skipped}")

    if not all_preds:
        print("No predictions produced — nothing to score.")
        return

    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds[["WD21CD", "ds", "variant", "y", "y_pred"]].rename(columns={"y": "y_true"})
    preds.to_csv(PRED_PATH, index=False)

    # --- metrics (vectorized; robust across pandas versions) -----------------
    err = preds["y_pred"] - preds["y_true"]
    # MAPE per the london-test convention: |yhat - y| / y, with y == 0 dropped
    # (else division by zero); averaged then x100. Months with y == 0 are
    # excluded from MAPE only, not from MAE/RMSE.
    ape = err.abs() / preds["y_true"].replace(0, np.nan)
    preds_m = preds.assign(abs_err=err.abs(), sq_err=err ** 2, ape=ape)

    def _agg(keys) -> pd.DataFrame:
        g = preds_m.groupby(keys).agg(
            mae=("abs_err", "mean"),
            rmse=("sq_err", "mean"),
            mape=("ape", "mean"),
            n=("abs_err", "size"),
        )
        g["rmse"] = np.sqrt(g["rmse"])
        g["mape"] = g["mape"] * 100  # report as a percentage
        return g

    # Pooled across all (ward, month) test points.
    overall = _agg("variant").reindex(list(VARIANTS))  # baseline/score/brokers order

    # Per ward x variant.
    per_ward = _agg(["WD21CD", "variant"]).reset_index()
    per_ward.to_csv(PER_WARD_PATH, index=False)

    # Mean of per-ward errors (equal weight per ward, not per observation).
    per_ward_mean = (
        per_ward.groupby("variant")[["mae", "rmse", "mape"]]
        .mean()
        .reindex(list(VARIANTS))
        .rename(columns={"mae": "mae_ward_mean",
                         "rmse": "rmse_ward_mean",
                         "mape": "mape_ward_mean"})
    )
    # Median across wards: robust to the handful of wards whose error explodes
    # (small counts / trend breaks), which drag the *mean* MAPE way up.
    per_ward_median = (
        per_ward.groupby("variant")[["mae", "rmse", "mape"]]
        .median()
        .reindex(list(VARIANTS))
        .rename(columns={"mae": "mae_ward_med",
                         "rmse": "rmse_ward_med",
                         "mape": "mape_ward_med"})
    )

    summary = overall.join(per_ward_mean).join(per_ward_median)
    summary["n_wards"] = preds.groupby("variant")["WD21CD"].nunique().reindex(list(VARIANTS))
    summary.to_csv(SUMMARY_PATH)

    # --- report --------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS  (target = {TARGET}, counts)")
    print("=" * 70)
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(summary.to_string())

    base_mae = overall.loc["baseline", "mae"]
    base_rmse = overall.loc["baseline", "rmse"]
    base_mape = overall.loc["baseline", "mape"]
    print("\nPooled change vs baseline (negative = better):")
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        d_mae = (overall.loc[variant, "mae"] - base_mae) / base_mae * 100
        d_rmse = (overall.loc[variant, "rmse"] - base_rmse) / base_rmse * 100
        d_mape = (overall.loc[variant, "mape"] - base_mape) / base_mape * 100
        print(f"  {variant:8s}  MAE {d_mae:+.2f}%   RMSE {d_rmse:+.2f}%   MAPE {d_mape:+.2f}%")

    print(f"\nSaved:\n  {PRED_PATH}\n  {PER_WARD_PATH}\n  {SUMMARY_PATH}")


if __name__ == "__main__":
    evaluate()
