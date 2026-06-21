"""
Compare the two Prophet variants (baseline vs brokerage-augmented) on the same wards.

Reads output/predictions.csv and inner-joins the variants on (WD21CD, ds) so the
metrics use the same ward-months for both. (run_forecast's own summary uses each
variant's full ward set, which can differ because Prophet skips wards with
< MIN_TRAIN_MONTHS, so numbers may differ slightly.)

Reports per model: pooled MAE / RMSE / WMAPE (and % change vs baseline), per-ward
MAE / WMAPE mean and median, pairwise per-ward win-rate, and a Wilcoxon test on
per-ward MAE.

Run run_forecast.py first, then from the repo root:
    python -m src.new_models_test.compare_models
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PROPHET_PRED = OUTPUT_DIR / "predictions.csv"
SUMMARY_OUT = OUTPUT_DIR / "model_comparison_summary.csv"
PERWARD_OUT = OUTPUT_DIR / "model_comparison_per_ward_mae.csv"

BASELINE_MODEL = "prophet_baseline"          # reference for the "% change" columns
# display order; extras appended after
PREFERRED_ORDER = ["prophet_baseline", "prophet_rank_safe"]


def load_predictions() -> pd.DataFrame:
    """Stack all model predictions into one long frame: WD21CD, ds, y_true, y_pred, model."""
    frames = []
    if PROPHET_PRED.exists():
        p = pd.read_csv(PROPHET_PRED, parse_dates=["ds"])
        p["model"] = "prophet_" + p["variant"].astype(str)
        frames.append(p[["WD21CD", "ds", "y_true", "y_pred", "model"]])
        print(f"Loaded {PROPHET_PRED.name}: {p['variant'].nunique()} variant(s), {len(p):,} rows")
    if not frames:
        raise SystemExit("No prediction files found; run run_forecast.py first.")
    return pd.concat(frames, ignore_index=True)


def _per_ward_mae_wmape(wide: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-ward MAE and WMAPE matrices (index=WD21CD, columns=models)."""
    wd = wide.reset_index()
    mae, wmape = {}, {}
    for m in models:
        ae = (wd[m] - wd["y_true"]).abs()
        g = pd.DataFrame({"WD21CD": wd["WD21CD"], "ae": ae, "y": wd["y_true"]})
        agg = g.groupby("WD21CD").agg(ae_mean=("ae", "mean"),
                                      ae_sum=("ae", "sum"),
                                      y_sum=("y", "sum"))
        mae[m] = agg["ae_mean"]
        wmape[m] = agg["ae_sum"] / agg["y_sum"].replace(0, np.nan) * 100
    return pd.DataFrame(mae), pd.DataFrame(wmape)


def main() -> None:
    long = load_predictions()

    present = set(long["model"].unique())
    models = [m for m in PREFERRED_ORDER if m in present] + \
             [m for m in long["model"].unique() if m not in PREFERRED_ORDER]

    # --- paired wide table: one y_pred column per model + the shared y_true ---
    wide = long.pivot_table(index=["WD21CD", "ds"], columns="model", values="y_pred")
    wide["y_true"] = long.groupby(["WD21CD", "ds"])["y_true"].first()
    n_before = len(wide)
    wide = wide.dropna(subset=models + ["y_true"])   # keep only ward-months common to ALL models
    print(f"\nModels: {models}")
    print(f"Paired ward-months: {len(wide):,} (dropped {n_before - len(wide):,} not common to all)")
    print(f"Common wards: {wide.reset_index()['WD21CD'].nunique()}")
    if wide.empty:
        raise SystemExit("No overlapping ward-months found.")

    # --- pooled metrics on the identical cells --------------------------------
    y = wide["y_true"]
    rows = []
    for m in models:
        err = wide[m] - y
        ae = err.abs()
        rows.append({
            "model": m,
            "mae": ae.mean(),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "wmape": ae.sum() / y.sum() * 100,
        })
    summary = pd.DataFrame(rows).set_index("model")

    # --- per-ward mean/median -------------------------------------------------
    pw_mae, pw_wmape = _per_ward_mae_wmape(wide, models)
    summary["mae_ward_mean"] = pw_mae.mean()
    summary["mae_ward_med"] = pw_mae.median()
    summary["wmape_ward_mean"] = pw_wmape.mean()
    summary["wmape_ward_med"] = pw_wmape.median()
    summary["n_cells"] = len(wide)
    summary["n_wards"] = pw_mae.notna().sum()

    # --- % change vs baseline -------------------------------------------------
    if BASELINE_MODEL in summary.index:
        for met in ["mae", "rmse", "wmape"]:
            base = summary.loc[BASELINE_MODEL, met]
            summary[f"{met}_vs_base%"] = (summary[met] - base) / base * 100

    summary = summary.reindex(models)
    summary.to_csv(SUMMARY_OUT)
    pw_mae.to_csv(PERWARD_OUT)

    # --- report ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("POOLED METRICS (identical ward-months for every model)")
    print("=" * 78)
    cols = ["mae", "rmse", "wmape", "mae_ward_med", "wmape_ward_med"]
    if "mae_vs_base%" in summary.columns:
        cols += ["mae_vs_base%", "wmape_vs_base%"]
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(summary[cols].to_string())

    # --- pairwise win-rate + Wilcoxon on per-ward MAE -------------------------
    print("\n" + "=" * 78)
    print("PAIRWISE per-ward comparison (MAE; lower = better)")
    print("=" * 78)
    for a, b in itertools.combinations(models, 2):
        sub = pw_mae[[a, b]].dropna()
        d = sub[a] - sub[b]                       # negative => a better
        a_win = (d < 0).mean() * 100
        b_win = (d > 0).mean() * 100
        try:
            _, pval = wilcoxon(sub[a], sub[b])
            psig = f"p={pval:.2e}"
        except ValueError:
            psig = "p=n/a"
        print(f"{a:18s} vs {b:18s}: {a.split('_')[-1]} better in {a_win:4.0f}% of wards, "
              f"{b.split('_')[-1]} in {b_win:4.0f}%  | median dMAE={d.median():+.3f}  "
              f"| Wilcoxon {psig} (n={len(sub)})")

    print(f"\nSaved:\n  {SUMMARY_OUT}\n  {PERWARD_OUT}")


if __name__ == "__main__":
    main()
