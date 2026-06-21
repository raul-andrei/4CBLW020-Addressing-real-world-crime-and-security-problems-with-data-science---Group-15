"""
Train one Prophet model per ward on all history, save them, then load and forecast
the next month without refitting.

Unlike run_forecast.py (the backtest), this fits on all available data: there is
no held-out test set when predicting the real future. The regressor is the raw
rank-weighted broker activity, lagged one month.

Files written under output/models/prophet/:
  rank_safe_models.json.gz   {WD21CD: prophet_json}
  last_regressors.csv        WD21CD, rank_raw_last (each ward's latest activity)
  manifest.json              weights, cutoff/next month, config

Run from the repo root:
  python -m src.new_models_test.production_models           # train + save
  python -m src.new_models_test.production_models predict   # forecast next month
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

# panel builders + shared run settings
from src.new_models_test.build_panel import (
    TARGET, build_ward_panel, add_date_columns, compute_rank_weights_safe,
)
from src.new_models_test.run_forecast import (
    MIN_TRAIN_MONTHS, OUTPUT_DIR, make_model, tqdm,
)

MODELS_DIR = OUTPUT_DIR / "models" / "prophet"

# all available history
PROD_WEIGHTS_SQL = "year BETWEEN 2017 AND 2026"
REG_NAME = "rank_lag1"                 # lagged rank-activity regressor


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

    Panel columns: WD21CD, ds, period, y (V&SO count), rank_raw (count*weight
    activity), rank_lag1 (rank_raw lagged one month within ward).
    """
    weights = compute_rank_weights_safe(PROD_WEIGHTS_SQL)

    # all wards, all months
    panel = build_ward_panel(score_where_sql=PROD_WEIGHTS_SQL)
    panel = add_date_columns(panel)

    # Raw rank-weighted broker activity (sum over all crime types of count*weight).
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
def _train_prophet(panel: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """One Prophet model per ward (rank_safe regressor). Saves the gzipped JSON."""
    models, skipped = {}, []
    for ward, w in tqdm(panel.groupby("WD21CD"), desc="training prophet wards", unit="ward"):
        wr = w[w["ds"] <= cutoff].dropna(subset=[REG_NAME])
        if len(wr) < MIN_TRAIN_MONTHS:
            skipped.append(ward)
            continue
        m = make_model()
        m.add_regressor(REG_NAME)
        m.fit(wr[["ds", "y", REG_NAME]])
        models[ward] = m
    _save_variant("rank_safe", models)
    if skipped:
        print(f"Skipped {len(skipped)} ward(s) with < {MIN_TRAIN_MONTHS} months.")
    return {"rank_safe": len(models)}


def train_and_save() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    panel, weights = _build_training_frame()

    cutoff = panel["ds"].max()                       # train on everything up to here
    next_ds = (cutoff + pd.DateOffset(months=1)).normalize()
    print(f"Training Prophet up to {cutoff.date()}; next forecast month = "
          f"{next_ds.date()}", flush=True)

    # Each ward's latest activity = the lag-1 regressor for next_ds.
    last_reg = panel.sort_values("ds").groupby("WD21CD")["rank_raw"].last()

    n_models = _train_prophet(panel, cutoff)

    # --- saved artifacts -------------------------------------------------
    pd.DataFrame({"WD21CD": last_reg.index, "rank_raw_last": last_reg.to_numpy()}) \
        .to_csv(MODELS_DIR / "last_regressors.csv", index=False)
    manifest = {
        "model": "prophet",
        "weights": weights,
        "reg_name": REG_NAME,
        "weights_sql": PROD_WEIGHTS_SQL,
        "cutoff": str(cutoff.date()),
        "next_ds": str(next_ds.date()),
        "features": [REG_NAME],
        "n_models": n_models,
    }
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved prophet -> {MODELS_DIR}  ({n_models})")


# --------------------------------------------------------------------------- #
# Load + predict next month
# --------------------------------------------------------------------------- #
def _predict_prophet(next_ds: pd.Timestamp, last_reg: pd.DataFrame) -> pd.DataFrame:
    """Predict next month per ward from the saved per-ward Prophet models."""
    reg_map = dict(zip(last_reg["WD21CD"], last_reg["rank_raw_last"]))
    rows = []
    for ward, m in _load_variant("rank_safe").items():
        fut = pd.DataFrame({"ds": [next_ds], REG_NAME: [reg_map.get(ward)]})
        yhat = float(m.predict(fut)["yhat"].iloc[0])
        rows.append({"WD21CD": ward, "ds": next_ds, "y_pred": max(0.0, yhat)})
    return pd.DataFrame(rows)


def predict_next_month() -> pd.DataFrame:
    """Forecast the month after training from the saved models (no DB, no refit).

    Each ward's lag-1 regressor is its latest observed activity, saved at train time.
    """
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
    next_ds = pd.Timestamp(manifest["next_ds"])
    last_reg = pd.read_csv(MODELS_DIR / "last_regressors.csv")

    out = _predict_prophet(next_ds, last_reg)

    out_path = OUTPUT_DIR / "forecast_next_prophet.csv"
    out.to_csv(out_path, index=False)
    print(f"Forecast for {next_ds.date()} (prophet): {len(out)} rows -> {out_path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_next_month()
    else:
        train_and_save()
