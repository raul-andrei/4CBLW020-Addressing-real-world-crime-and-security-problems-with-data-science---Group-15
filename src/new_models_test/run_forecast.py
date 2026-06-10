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
    venv\\Scripts\\python.exe -m src.new_models_test.run_forecast
    (or just run this file directly; it bootstraps sys.path.)
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

# --- make `from src...` imports work whether run as a module or a script ----
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- keep each parallel worker single-threaded so joblib's ward-level parallelism
#     doesn't oversubscribe the CPU (BLAS/OpenMP spinning up threads inside every
#     Prophet fit). MUST be set before numpy is imported.
import os
for _thr in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thr, "1")

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

from src.new_models_test.build_panel import (  # noqa: E402
    TARGET,
    build_ward_panel,
    add_date_columns,
    select_random_wards,
    add_lagged_regressors,
    validate_columns,
    lag_name,
)

# ============================ CONFIG (edit me) ===============================
N_WARDS = 1000                # random sample size (int), or "all"
RANDOM_SEED = 42             # makes the random ward sample reproducible
SPLIT_DATE = "2024-01-01"     # last TRAIN month (inclusive); test = the month after, onward
FAST_MODE = False              # True = fit-once demo; False = expanding-window refit
N_JOBS = -1                   # ward-loop parallelism: 1 = serial (debug), -1 = all cores
SCORE_START_YEAR = 2017       # brokerage weights are fit on [START .. SPLIT year]
MIN_TRAIN_MONTHS = 24         # skip wards with fewer training months than this

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PRED_PATH = OUTPUT_DIR / "predictions.csv"
PER_WARD_PATH = OUTPUT_DIR / "per_ward_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary_metrics.csv"

# rank_safe weights are fit on the TRAINING window only.
# Use an exact MONTH cutoff (not just the split year): if SPLIT_DATE falls
# mid-year, the later months of that year are in the TEST set and must not leak
# into the weights. period = year*12 + (month_num-1), matching scores.py.
_SPLIT = pd.Timestamp(SPLIT_DATE)
_SPLIT_PERIOD = _SPLIT.year * 12 + (_SPLIT.month - 1)
SCORE_WHERE_SQL = (
    f"year >= {SCORE_START_YEAR} AND (year * 12 + (month_num - 1)) <= {_SPLIT_PERIOD}"
)

# variant_name -> list of regressor columns (already lag-1, already known at t)
#   baseline  : Prophet, trend + yearly seasonality only
#   rank_safe : rank-weighted activity, weights recomputed on the train window (no leak)
VARIANTS: dict[str, list[str]] = {
    "baseline": [],
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
        # We only ever use yhat (the point forecast), never the intervals, so skip
        # the posterior-predictive simulation. yhat is unchanged; predict is faster.
        uncertainty_samples=0,
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


def _silence_cmdstanpy() -> None:
    """Force-silence cmdstanpy in THIS process.

    The import-time suppression at the top of the module doesn't stick inside
    joblib (loky) workers: cmdstanpy adds its StreamHandler when prophet is
    imported — after our suppression runs — and the `disabled` flag gets reset in
    the spawned process. Clearing the handlers here (once cmdstanpy is fully
    loaded) reliably stops the per-fit "Chain [1] ... processing" flood.
    """
    lg = logging.getLogger("cmdstanpy")
    lg.handlers.clear()
    lg.setLevel(logging.CRITICAL)
    lg.disabled = True
    lg.propagate = False


def _model_one_ward(ward: str, wdf: pd.DataFrame, split: pd.Timestamp):
    """Fit every variant for one ward; return (pred_frames, skipped_ward).

    Top-level (picklable) so joblib workers can call it. `skipped_ward` is the
    ward src when it has too little data, otherwise None. `wdf` already has the
    target renamed to 'y' and is sorted by ds.
    """
    _silence_cmdstanpy()  # idempotent; needed because loky workers re-enable it
    train = wdf[wdf["ds"] <= split]
    test = wdf[wdf["ds"] > split]
    if len(train) < MIN_TRAIN_MONTHS or len(test) == 0:
        return [], ward
    preds = []
    for variant, regs in VARIANTS.items():
        pred = predict_fast(train, test, regs) if FAST_MODE else predict_rolling(wdf, split, regs)
        if not pred.empty:
            preds.append(pred.assign(WD21CD=ward, variant=variant))
    return preds, None


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
    panel, selected = select_random_wards(panel, N_WARDS, RANDOM_SEED)
    print(f"Selected {len(selected)} wards at random (seed={RANDOM_SEED}).")
    panel = add_lagged_regressors(panel)

    reg_cols = sorted({c for regs in VARIANTS.values() for c in regs})
    keep = ["WD21CD", "ds", TARGET] + reg_cols

    # --- split into independent per-ward frames (one model job each) ----------
    ward_frames = [
        (ward, g.rename(columns={TARGET: "y"}).sort_values("ds").reset_index(drop=True))
        for ward, g in panel[keep].groupby("WD21CD", sort=False)
    ]

    # --- model each ward (serial if N_JOBS==1 for easy debugging, else joblib) -
    if N_JOBS == 1:
        results = [_model_one_ward(ward, wdf, split)
                   for ward, wdf in tqdm(ward_frames, desc="modelling wards", unit="ward")]
    else:
        from joblib import Parallel, delayed
        print(f"Modelling {len(ward_frames)} wards in parallel (n_jobs={N_JOBS})...", flush=True)
        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(_model_one_ward)(ward, wdf, split) for ward, wdf in ward_frames
        )

    all_preds: list[pd.DataFrame] = []
    skipped: list[str] = []
    for preds_list, skipped_ward in results:
        if skipped_ward is not None:
            skipped.append(skipped_ward)
        all_preds.extend(preds_list)

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
    # WMAPE (volume-weighted MAPE) = sum|err| / sum(actual). Robust to the many
    # low-count wards that make a plain per-row MAPE explode. MAE/RMSE unchanged.
    err = preds["y_pred"] - preds["y_true"]
    preds_m = preds.assign(abs_err=err.abs(), sq_err=err ** 2)

    def _agg(keys) -> pd.DataFrame:
        g = preds_m.groupby(keys).agg(
            mae=("abs_err", "mean"),
            rmse=("sq_err", "mean"),
            abs_err_sum=("abs_err", "sum"),
            y_sum=("y_true", "sum"),
            n=("abs_err", "size"),
        )
        g["rmse"] = np.sqrt(g["rmse"])
        g["wmape"] = g["abs_err_sum"] / g["y_sum"].replace(0, np.nan) * 100
        return g.drop(columns=["abs_err_sum", "y_sum"])

    # Pooled across all (ward, month) test points.
    overall = _agg("variant").reindex(list(VARIANTS))  # baseline/score/brokers order

    # Per ward x variant.
    per_ward = _agg(["WD21CD", "variant"]).reset_index()
    per_ward.to_csv(PER_WARD_PATH, index=False)

    # Mean of per-ward errors (equal weight per ward, not per observation).
    per_ward_mean = (
        per_ward.groupby("variant")[["mae", "rmse", "wmape"]]
        .mean()
        .reindex(list(VARIANTS))
        .rename(columns={"mae": "mae_ward_mean",
                         "rmse": "rmse_ward_mean",
                         "wmape": "wmape_ward_mean"})
    )
    # Median across wards: robust to the handful of wards whose error explodes
    # (small counts / trend breaks), which drag the *mean* error up.
    per_ward_median = (
        per_ward.groupby("variant")[["mae", "rmse", "wmape"]]
        .median()
        .reindex(list(VARIANTS))
        .rename(columns={"mae": "mae_ward_med",
                         "rmse": "rmse_ward_med",
                         "wmape": "wmape_ward_med"})
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
    base_wmape = overall.loc["baseline", "wmape"]
    print("\nPooled change vs baseline (negative = better):")
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        d_mae = (overall.loc[variant, "mae"] - base_mae) / base_mae * 100
        d_rmse = (overall.loc[variant, "rmse"] - base_rmse) / base_rmse * 100
        d_wmape = (overall.loc[variant, "wmape"] - base_wmape) / base_wmape * 100
        print(f"  {variant:8s}  MAE {d_mae:+.2f}%   RMSE {d_rmse:+.2f}%   WMAPE {d_wmape:+.2f}%")

    print(f"\nSaved:\n  {PRED_PATH}\n  {PER_WARD_PATH}\n  {SUMMARY_PATH}")


if __name__ == "__main__":
    evaluate()
