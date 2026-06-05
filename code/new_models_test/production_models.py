"""
Production models: train ONCE, save, then load & forecast next month without
ever refitting.

Why this is separate from run_forecast.py
-----------------------------------------
`run_forecast.py` is the *backtest*: it holds out a test period and (to keep the
evaluation honest) fits the brokerage weights only on the training window, and it
refits per test month. None of that applies once you actually deploy:

  * There is no held-out test set when you forecast the genuine future, so we use
    ALL available history — for both the rank weights and the Prophet models.
    (Using less would just throw away signal; the train-only window only ever
    existed to prevent test leakage during evaluation.)
  * We fit ONE model per ward per variant on everything up to the latest month,
    serialise it, and reload it later to predict — no retraining each month.

Only the `rank_safe` model is trained for production (the baseline exists purely
for the backtest comparison in run_forecast.py, so it isn't persisted here).

The rank_safe regressor
-----------------------
`rank_safe` uses the rank-weighted broker activity, lagged one month. Here we use
the RAW weighted sum (count x weight, NOT normalised to [0,1]). Prophet
standardises regressors internally, so the model and its predictions are
identical to the normalised version — but raw means there is no normalisation
constant to store and reproduce at serve time. We just save the per-crime weights.

Files written under output/models/
----------------------------------
  rank_safe_models.json.gz  {WD21CD: prophet_json}
  last_regressors.csv       WD21CD, rank_raw_last  (each ward's latest activity =
                            the lag-1 regressor needed to predict the next month)
  manifest.json             weights, regressor name, cutoff/next month, config

Usage
-----
  Train + save:   venv\\Scripts\\python.exe -m code.new_models_test.production_models
  Predict next:   venv\\Scripts\\python.exe -m code.new_models_test.production_models predict
                  (or import predict_next_month())
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from prophet.serialize import model_to_json, model_from_json

# Reuse the canonical panel builders, the Prophet config, and run-wide settings.
from code.new_models_test.build_panel import (
    TARGET, build_ward_panel, add_date_columns, select_top_wards,
    compute_rank_weights_safe, lag_name,
)
from code.new_models_test.run_forecast import (
    N_WARDS, MIN_TRAIN_MONTHS, OUTPUT_DIR, make_model, tqdm,
)

MODELS_DIR = OUTPUT_DIR / "models"

# Production uses ALL available history (no leakage concern for true future prediction).
PROD_WEIGHTS_SQL = "year BETWEEN 2017 AND 2026"
REG_NAME = "rank_lag1"                 # the (raw) lagged rank-activity regressor
PROD_VARIANTS = {"rank_safe": [REG_NAME]}


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def _save_variant(name: str, models: dict) -> None:
    """Save {ward: Prophet} as a single gzipped JSON of Prophet model strings."""
    blob = {ward: model_to_json(m) for ward, m in models.items()}
    with gzip.open(MODELS_DIR / f"{name}_models.json.gz", "wt", encoding="utf-8") as f:
        json.dump(blob, f)


def _load_variant(name: str) -> dict:
    """Load {ward: Prophet} from the gzipped JSON written by _save_variant."""
    with gzip.open(MODELS_DIR / f"{name}_models.json.gz", "rt", encoding="utf-8") as f:
        blob = json.load(f)
    return {ward: model_from_json(s) for ward, s in blob.items()}


# --------------------------------------------------------------------------- #
# Build the training frame (panel + raw lagged rank activity)
# --------------------------------------------------------------------------- #
def _build_training_frame() -> tuple[pd.DataFrame, dict]:
    """Return (panel, weights) ready for fitting.

    panel columns of interest: WD21CD, ds, period, y (=V&SO count),
    rank_raw (raw count x weight activity), rank_lag1 (rank_raw shifted +1 month
    within ward — the regressor known one month ahead).
    """
    weights = compute_rank_weights_safe(PROD_WEIGHTS_SQL)

    panel = build_ward_panel(score_where_sql=PROD_WEIGHTS_SQL)
    panel = add_date_columns(panel)
    panel, _ = select_top_wards(panel, N_WARDS)

    # Raw rank-weighted broker activity (sum over all crime types of count*weight).
    # Computed BEFORE renaming the target, because the weights include V&SO too.
    panel["rank_raw"] = sum(
        panel[c] * w for c, w in weights.items() if c in panel.columns
    )
    panel = panel.rename(columns={TARGET: "y"})

    # Lag the regressor within ward (row t carries month t-1's activity).
    panel = panel.sort_values(["WD21CD", "period"]).reset_index(drop=True)
    panel[REG_NAME] = panel.groupby("WD21CD")["rank_raw"].shift(1)
    return panel, weights


# --------------------------------------------------------------------------- #
# Train + save
# --------------------------------------------------------------------------- #
def train_and_save() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    panel, weights = _build_training_frame()

    cutoff = panel["ds"].max()                       # train on everything up to here
    next_ds = (cutoff + pd.DateOffset(months=1)).normalize()
    print(f"Training on data up to {cutoff.date()}; next forecast month = {next_ds.date()}",
          flush=True)

    models = {v: {} for v in PROD_VARIANTS}
    last_reg = {}        # ward -> latest rank_raw (the regressor for `next_ds`)
    skipped = []

    for ward, w in tqdm(panel.groupby("WD21CD"), desc="training wards", unit="ward"):
        w = w.sort_values("ds")
        if len(w) < MIN_TRAIN_MONTHS:
            skipped.append(ward)
            continue

        # rank_safe: needs the lagged regressor (drops the first month per ward)
        wr = w.dropna(subset=[REG_NAME])
        if len(wr) >= MIN_TRAIN_MONTHS:
            mr = make_model()
            mr.add_regressor(REG_NAME)
            mr.fit(wr[["ds", "y", REG_NAME]])
            models["rank_safe"][ward] = mr

        # latest activity -> the lag-1 regressor value for predicting `next_ds`
        last_reg[ward] = float(w["rank_raw"].iloc[-1])

    # --- persist ---------------------------------------------------------
    for name, md in models.items():
        _save_variant(name, md)
    pd.DataFrame(
        [{"WD21CD": k, "rank_raw_last": v} for k, v in last_reg.items()]
    ).to_csv(MODELS_DIR / "last_regressors.csv", index=False)
    manifest = {
        "weights": weights,
        "reg_name": REG_NAME,
        "weights_sql": PROD_WEIGHTS_SQL,
        "cutoff": str(cutoff.date()),
        "next_ds": str(next_ds.date()),
        "variants": PROD_VARIANTS,
        "n_models": {k: len(v) for k, v in models.items()},
    }
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved {manifest['n_models']} models to {MODELS_DIR}")
    if skipped:
        print(f"Skipped {len(skipped)} ward(s) with < {MIN_TRAIN_MONTHS} months.")


# --------------------------------------------------------------------------- #
# Load + predict next month (no DB, no refit)
# --------------------------------------------------------------------------- #
def predict_next_month() -> pd.DataFrame:
    """Forecast the month immediately after training, using only saved artifacts.

    For the immediate next month this needs NO database access and NO refitting:
    each ward's lag-1 regressor is its latest observed activity, saved at train
    time. (To roll further forward once a genuinely new month of crime is
    observed, recompute that month's activity as sum(new_counts * weights) — the
    weights are in manifest.json — and feed it as REG_NAME; the models are reused
    as-is. Refit periodically, e.g. quarterly, so the trend stays current.)
    """
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
    reg_name = manifest["reg_name"]
    next_ds = pd.Timestamp(manifest["next_ds"])

    last_reg = pd.read_csv(MODELS_DIR / "last_regressors.csv")
    reg_map = dict(zip(last_reg["WD21CD"], last_reg["rank_raw_last"]))

    rows = []
    for ward, m in _load_variant("rank_safe").items():
        fut = pd.DataFrame({"ds": [next_ds], reg_name: [reg_map.get(ward)]})
        yhat = float(m.predict(fut)["yhat"].iloc[0])
        rows.append({"WD21CD": ward, "ds": next_ds, "variant": "rank_safe",
                     "y_pred": max(0.0, yhat)})

    out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "forecast_next.csv"
    out.to_csv(out_path, index=False)
    print(f"Forecast for {next_ds.date()}: {len(out)} rows -> {out_path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_next_month()
    else:
        train_and_save()
