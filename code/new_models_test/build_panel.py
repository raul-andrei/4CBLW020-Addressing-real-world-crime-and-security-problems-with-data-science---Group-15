"""
Build the ward x month panel used by the V&SO forecasting experiment.

What this module produces (one row per ward-month):
    WD21CD, period, ds, year, month,
    <one column per crime type holding the monthly count>,
    avg_betweenness,                     (proportion-weighted brokerage activity)
    <broker>_lag1 ... , avg_betweenness_lag1   (regressors, shifted +1 month *within ward*)

Everything here is COUNTS (never shares). The only proportion-based quantity is
`avg_betweenness`, which is the existing brokerage-activity regressor; it is then
lagged like the rest.

Reused, unchanged, from the existing pipeline (code/network_analysis/scores.py):
    get_crime_data, aggregate_lsoas_to_wards, calculate_average_betweenness,
and through calculate_average_betweenness, run_primary_brokerage_analysis from
sensitivity_analysis (so the brokerage weights can be fit on the training window
only -> no leakage).
"""

from __future__ import annotations

import pandas as pd

from code.network_analysis.scores import (
    get_crime_data,
    aggregate_lsoas_to_wards,
    calculate_average_betweenness,
)

TARGET = "Violence and sexual offences"
BROKERS = ["Robbery", "Theft from the person", "Possession of weapons"]

# Rank-based brokerage weights, COPIED VERBATIM from a colleague's model
# (code/models_london_tests/prophet_brokerage_ward_cooc.py) so the `rank` variant
# is an identical weighting. Values are (max_rank - rank + 1) on the averaged
# co-occurrence ranks; higher = more central / more "broker-like".
RANK_BROKERAGE_SCORES = {
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

# Source columns that get shifted +1 month within each ward.
#   rank_activity      -> colleague's ORIGINAL hard-coded weights (all-data -> leaky)
#   rank_activity_safe -> leakage-safe rank weights recomputed on the train window
LAG_SOURCE_COLS = BROKERS + ["avg_betweenness", "rank_activity", "rank_activity_safe"]


def lag_name(col: str) -> str:
    return f"{col}_lag1"


def _weighted_activity(panel: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted SUM OF RAW COUNTS over crime types (count x weight), normalised
    to [0, 1]. Like the colleague's code, every crime type is weighted, not just
    the 3 brokers. The [0, 1] scaling is cosmetic (Prophet standardises
    regressors internally) and kept only to mirror his construction."""
    cols = [c for c in weights if c in panel.columns]
    weighted = sum(panel[c] * weights[c] for c in cols)
    max_val = weighted.max()
    return weighted / max_val if max_val else weighted


def compute_rank_weights_safe(score_where_sql: str | None) -> dict[str, float]:
    """Leakage-safe rank weights: rerun the brokerage analysis on the SAME
    (training) window used for `score`, average the rank across the 4 brokerage
    metrics, then apply the colleague's (max_rank - rank + 1) formula.

    Rank direction: betweenness / current-flow-betweenness / eigenvector ->
    higher = more broker-like (ascending=False); constraint -> lower = more
    broker-like (ascending=True).

    NOTE: this RECONSTRUCTS his recipe. His exact hard-coded numbers also appear
    to average across forces/configs, which a single run can't reproduce; this
    reconstruction matches his top-4 brokers and correlates ~0.88 (Spearman) with
    his ranks, but is computed only on data up to the split -> no leakage.
    """
    # Lazy import (pulls igraph/leidenalg) so importing this module stays cheap.
    from code.network_analysis.preparation import connect
    from code.network_analysis.sensitivity_analysis import run_primary_brokerage_analysis

    where = score_where_sql or "year BETWEEN 2017 AND 2026"
    con = connect()
    df = run_primary_brokerage_analysis(where_sql=where, con=con)
    con.close()
    if df is None or df.empty:
        raise RuntimeError(f"no brokerage scores for rank weights (where={where!r})")

    d = df.set_index("crime_type")
    avg_rank = pd.DataFrame({
        "betweenness": d["betweenness"].rank(ascending=False),
        "cfb":         d["current_flow_betweenness"].rank(ascending=False),
        "eigenvector": d["eigenvector"].rank(ascending=False),
        "constraint":  d["constraint"].rank(ascending=True),
    }).mean(axis=1)
    weights = (avg_rank.max() - avg_rank + 1)
    return weights.to_dict()


def add_rank_activity(panel: pd.DataFrame) -> pd.DataFrame:
    """`rank_activity`: colleague's ORIGINAL hard-coded weights (derived from
    all 2017-2026 data -> contains test-period leakage). Kept as a labelled
    reference variant so leaky-vs-safe can be compared directly.

    Deliberate deviation from his script (same as the safe version): the column
    is lag-1'd downstream and evaluated one-step-ahead, so it sits on the same
    footing as baseline/score/brokers and the comparison isolates the weighting,
    not the evaluation protocol."""
    missing = [c for c in RANK_BROKERAGE_SCORES if c not in panel.columns]
    if missing:
        raise KeyError(f"rank activity: crime columns missing from panel: {missing}")
    panel = panel.copy()
    panel["rank_activity"] = _weighted_activity(panel, RANK_BROKERAGE_SCORES)
    return panel


def add_rank_activity_safe(panel: pd.DataFrame, score_where_sql: str | None) -> pd.DataFrame:
    """`rank_activity_safe`: same construction as `rank_activity` but with
    leakage-safe weights recomputed on the training window only."""
    weights = compute_rank_weights_safe(score_where_sql)
    panel = panel.copy()
    panel["rank_activity_safe"] = _weighted_activity(panel, weights)
    return panel


def build_ward_panel(score_where_sql: str | None) -> pd.DataFrame:
    """Return the wide ward-month panel with `avg_betweenness` merged in.

    score_where_sql is forwarded to calculate_average_betweenness so the broker
    weights are fit only on the chosen (training) window. The ~49M-row crime
    table is pulled from DuckDB exactly once and reused for both the panel and
    the brokerage-activity regressor.
    """
    # --- one DB pull, then aggregate LSOA -> ward (existing helpers) ----------
    ward_long = aggregate_lsoas_to_wards(get_crime_data())  # WD21CD, period, crime_type, count

    # --- pivot to one row per (ward, period), one column per crime type -------
    panel = ward_long.pivot_table(
        index=["WD21CD", "period"],
        columns="crime_type",
        values="count",
        aggfunc="sum",   # combos are unique; sum just keeps the count as-is
        fill_value=0,
    ).reset_index()
    panel.columns.name = None  # drop the leftover 'crime_type' axis name

    # --- merge the avg_betweenness regressor (reuses the same aggregation) ----
    # Pass ward_agg=ward_long so calculate_average_betweenness does not re-read
    # the 49M-row table; score_where_sql keeps the weights leakage-safe.
    bet = calculate_average_betweenness(score_where_sql=score_where_sql, ward_agg=ward_long)
    panel = panel.merge(bet, on=["WD21CD", "period"], how="left")
    # Every panel ward-period comes from the same aggregation, so a miss only
    # happens if a ward-period had no scored crime types -> 0 brokerage activity.
    panel["avg_betweenness"] = panel["avg_betweenness"].fillna(0)

    # --- rank-weighted activity features (his original + leakage-safe) --------
    panel = add_rank_activity(panel)                          # all-data weights (leaky)
    panel = add_rank_activity_safe(panel, score_where_sql)    # train-window weights (safe)

    return panel


def add_date_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Recover year/month from `period` and build a first-of-month `ds`."""
    panel = panel.copy()
    panel["year"] = panel["period"] // 12
    panel["month"] = panel["period"] % 12 + 1
    panel["ds"] = pd.to_datetime(
        dict(year=panel["year"], month=panel["month"], day=1)
    )
    return panel


def select_top_wards(panel: pd.DataFrame, n_wards) -> tuple[pd.DataFrame, list[str]]:
    """Keep the `n_wards` wards with the highest total V&SO (or 'all')."""
    totals = panel.groupby("WD21CD")[TARGET].sum().sort_values(ascending=False)
    if isinstance(n_wards, str) and n_wards.lower() == "all":
        selected = totals.index.tolist()
    else:
        selected = totals.head(int(n_wards)).index.tolist()
    return panel[panel["WD21CD"].isin(selected)].copy(), selected


def add_lagged_regressors(panel: pd.DataFrame) -> pd.DataFrame:
    """Shift the regressor columns +1 month *within each ward*.

    Row t then carries month t-1's values, so when forecasting month t the
    regressors are already known. The first month per ward becomes NaN after the
    shift and is dropped.
    """
    panel = panel.sort_values(["WD21CD", "period"]).reset_index(drop=True)
    for col in LAG_SOURCE_COLS:
        panel[lag_name(col)] = panel.groupby("WD21CD")[col].shift(1)
    lagged = [lag_name(c) for c in LAG_SOURCE_COLS]
    panel = panel.dropna(subset=lagged).reset_index(drop=True)
    return panel


def validate_columns(panel: pd.DataFrame) -> None:
    """Confirm the exact target/broker column names exist before lagging."""
    print("\nPanel columns:", panel.columns.tolist())
    required = [TARGET] + BROKERS + ["avg_betweenness"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(
            f"Expected columns not found in panel: {missing}\n"
            f"Available: {panel.columns.tolist()}"
        )
