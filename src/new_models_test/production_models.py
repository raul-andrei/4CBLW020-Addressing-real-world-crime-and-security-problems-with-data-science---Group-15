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

Choosing the model (MODEL toggle)
---------------------------------
Set MODEL = "prophet" or "lgbm" at the top and run. Each writes to its OWN folder
(output/models/<MODEL>/), so you can train and keep BOTH (just run once per value).
  * prophet : one Prophet model PER WARD (rank_safe regressor), as before.
  * lgbm    : one POOLED Poisson LightGBM across all wards (features: WD21CD,
              month_of_year, time_index, and the same lagged rank activity).
LightGBM trains in seconds once the panel is built, so keeping both is cheap.

The rank activity regressor (both models)
-----------------------------------------
Uses the RAW rank-weighted broker activity, lagged one month (count x weight, NOT
normalised). Both Prophet (standardises regressors internally) and LightGBM (trees
are scale-free) are invariant to the [0,1] normalisation, so raw means there is no
normalisation constant to reproduce at serve time — we just save the weights.

Files written under output/models/<MODEL>/
-------------------------------------------
  prophet:  rank_safe_models.json.gz   {WD21CD: prophet_json}
  lgbm:     lgbm_model.pkl  +  lgbm_categories.json  (training category levels)
  both:     last_regressors.csv  (WD21CD, rank_raw_last = each ward's latest
                                   activity = the lag-1 regressor for next month)
            manifest.json        (model, weights, cutoff/next month, config)

Usage
-----
  Train + save:   venv\\Scripts\\python.exe -m src.new_models_test.production_models
  Predict next:   venv\\Scripts\\python.exe -m src.new_models_test.production_models predict
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

import numpy as np
import pandas as pd
from prophet.serialize import model_to_json, model_from_json

# Reuse the canonical panel builders, the Prophet config, and run-wide settings.
from src.new_models_test.build_panel import (
    TARGET, build_ward_panel, add_date_columns, compute_rank_weights_safe,
)
from src.new_models_test.run_forecast import (
    MIN_TRAIN_MONTHS, OUTPUT_DIR, make_model, tqdm,
)

# Which model to train & persist this run. Flip and re-run to save the other;
# each writes to its own folder so both coexist.
MODEL = "lgbm"                         # "prophet" or "lgbm"

MODELS_DIR = OUTPUT_DIR / "models" / MODEL

# Production uses ALL available history (no leakage concern for true future prediction).
PROD_WEIGHTS_SQL = "year BETWEEN 2017 AND 2026"
REG_NAME = "rank_lag1"                 # the (raw) lagged rank-activity regressor (both models)

# LightGBM settings (only used when MODEL == "lgbm"); pooled Poisson, as in lgbm_forecast.
LGBM_FEATURES = ["WD21CD", "month_of_year", "time_index", REG_NAME]
LGBM_CATEGORICAL = ["WD21CD", "month_of_year"]
LGBM_PARAMS = dict(
    objective="poisson", n_estimators=400, learning_rate=0.05, num_leaves=31,
    min_child_samples=100, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
    random_state=42, n_jobs=-1, verbose=-1,
)


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
    month_of_year, time_index (calendar features for LightGBM), rank_raw (raw
    count x weight activity), rank_lag1 (rank_raw shifted +1 month within ward —
    the regressor known one month ahead).
    """
    weights = compute_rank_weights_safe(PROD_WEIGHTS_SQL)

    # Production trains on ALL wards and ALL time periods (no selection, no split).
    panel = build_ward_panel(score_where_sql=PROD_WEIGHTS_SQL)
    panel = add_date_columns(panel)

    # Raw rank-weighted broker activity (sum over all crime types of count*weight).
    # Computed BEFORE renaming the target, because the weights include V&SO too.
    panel["rank_raw"] = sum(
        panel[c] * w for c, w in weights.items() if c in panel.columns
    )
    # Calendar features (used by the LightGBM model; Prophet ignores them).
    panel["month_of_year"] = panel["month"]
    panel["time_index"] = panel["period"]
    panel = panel.rename(columns={TARGET: "y"})

    # Lag the regressor within ward (row t carries month t-1's activity).
    panel = panel.sort_values(["WD21CD", "period"]).reset_index(drop=True)
    panel[REG_NAME] = panel.groupby("WD21CD")["rank_raw"].shift(1)
    return panel, weights


# --------------------------------------------------------------------------- #
# Train + save  (branches on MODEL)
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


def _train_lgbm(panel: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """One pooled Poisson LightGBM across all wards. Pickles the model + the
    training category levels (so the serve-time frame is encoded identically)."""
    import pickle
    import lightgbm as lgb

    train = panel[panel["ds"] <= cutoff].dropna(subset=LGBM_FEATURES + ["y"]).copy()
    for c in LGBM_CATEGORICAL:
        train[c] = train[c].astype("category")

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[LGBM_FEATURES], train["y"], categorical_feature=LGBM_CATEGORICAL)

    with open(MODELS_DIR / "lgbm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    # Save the exact training categories (native types) so predict can rebuild the
    # same codes via pd.Categorical(..., categories=saved).
    cats = {c: train[c].cat.categories.tolist() for c in LGBM_CATEGORICAL}
    (MODELS_DIR / "lgbm_categories.json").write_text(json.dumps(cats))
    return {"lgbm_poisson": 1, "train_rows": int(len(train))}


def train_and_save() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    panel, weights = _build_training_frame()

    cutoff = panel["ds"].max()                       # train on everything up to here
    next_ds = (cutoff + pd.DateOffset(months=1)).normalize()
    print(f"MODEL={MODEL} | training up to {cutoff.date()}; next forecast month = "
          f"{next_ds.date()}", flush=True)

    # Shared by both models: each ward's latest activity = the lag-1 regressor for next_ds.
    last_reg = panel.sort_values("ds").groupby("WD21CD")["rank_raw"].last()

    if MODEL == "prophet":
        n_models = _train_prophet(panel, cutoff)
    elif MODEL == "lgbm":
        n_models = _train_lgbm(panel, cutoff)
    else:
        raise SystemExit(f"Unknown MODEL={MODEL!r}; set it to 'prophet' or 'lgbm'.")

    # --- shared artifacts ------------------------------------------------
    pd.DataFrame({"WD21CD": last_reg.index, "rank_raw_last": last_reg.to_numpy()}) \
        .to_csv(MODELS_DIR / "last_regressors.csv", index=False)
    manifest = {
        "model": MODEL,
        "weights": weights,
        "reg_name": REG_NAME,
        "weights_sql": PROD_WEIGHTS_SQL,
        "cutoff": str(cutoff.date()),
        "next_ds": str(next_ds.date()),
        "features": LGBM_FEATURES if MODEL == "lgbm" else [REG_NAME],
        "n_models": n_models,
    }
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved {MODEL} -> {MODELS_DIR}  ({n_models})")


# --------------------------------------------------------------------------- #
# Load + predict next month (no DB, no refit)
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


def _predict_lgbm(next_ds: pd.Timestamp, last_reg: pd.DataFrame) -> pd.DataFrame:
    """Predict next month for all wards from the saved pooled LightGBM model."""
    import pickle
    with open(MODELS_DIR / "lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    cats = json.loads((MODELS_DIR / "lgbm_categories.json").read_text())

    feat = pd.DataFrame({
        "WD21CD": last_reg["WD21CD"].to_numpy(),
        "month_of_year": int(next_ds.month),
        "time_index": int(next_ds.year * 12 + (next_ds.month - 1)),
        REG_NAME: last_reg["rank_raw_last"].to_numpy(),
    })
    # Encode categoricals with the SAME category levels/order used at training.
    feat["WD21CD"] = pd.Categorical(feat["WD21CD"], categories=cats["WD21CD"])
    feat["month_of_year"] = pd.Categorical(feat["month_of_year"], categories=cats["month_of_year"])

    yhat = np.clip(model.predict(feat[LGBM_FEATURES]), 0, None)  # Poisson is >=0 anyway
    return pd.DataFrame({"WD21CD": last_reg["WD21CD"], "ds": next_ds, "y_pred": yhat})


def predict_next_month() -> pd.DataFrame:
    """Forecast the month immediately after training, using only saved artifacts.

    For the immediate next month this needs NO database access and NO refitting:
    each ward's lag-1 regressor is its latest observed activity, saved at train
    time. (To roll further forward once a genuinely new month of crime is
    observed, recompute that month's activity as sum(new_counts * weights) — the
    weights are in manifest.json — and feed it as the regressor; models are reused
    as-is. Refit periodically, e.g. quarterly, so the trend stays current.)

    Reads from output/models/<MODEL>/, where MODEL is whatever is set at the top.
    """
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
    next_ds = pd.Timestamp(manifest["next_ds"])
    last_reg = pd.read_csv(MODELS_DIR / "last_regressors.csv")

    if manifest["model"] == "prophet":
        out = _predict_prophet(next_ds, last_reg)
    else:
        out = _predict_lgbm(next_ds, last_reg)

    out_path = OUTPUT_DIR / f"forecast_next_{manifest['model']}.csv"
    out.to_csv(out_path, index=False)
    print(f"Forecast for {next_ds.date()} ({manifest['model']}): {len(out)} rows -> {out_path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_next_month()
    else:
        train_and_save()
