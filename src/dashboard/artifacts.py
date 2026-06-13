"""Build the heavy dashboard artifacts once, so the app just loads them.

For now: the brokerage-crime network figure (crime types = nodes). The graph
construction + brokerage metrics are reused from
``src.network_analysis.network_visualization.build_brokerage_graph`` (single
source of truth); here we only do the plotly rendering in the dashboard's dark
theme.

Encoding (option 3, channels swapped so brokers pop):
    node size   = brokerage centrality (current-flow betweenness) -> brokers pop
    node colour = how common the crime type is (total volume, log scale)
    edge width  = co-occurrence weight (no hover on edges)
"""

import json
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.network_analysis.preparation import connect
from src.network_analysis.scores import calculate_average_betweenness
from src.network_analysis.network_visualization import build_brokerage_graph
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"

DASHBOARD_ASSETS = ROOT / "src" / "dashboard" / "assets"

# --- dashboard palette (matches src/dashboard/app.py) -----------------------
BG_CARD = "#111520"
BORDER = "#1e2640"
ACCENT = "#4f7cff"
TEXT_PRI = "#e8eaf2"
TEXT_SEC = "#7a82a0"

# Colour now encodes VOLUME (size encodes brokerage). Deliberately a cool blue
# sequential -- NOT the dashboard's red "risk" scale -- so colour reads clearly
# as "how much" and isn't mistaken for brokerage/risk (that's the size channel).
VOLUME_SCALE = [
    [0.0, "#2c3656"],
    [0.5, ACCENT],
    [1.0, "#8fb4ff"],
]


def make_network():
    """Return the brokerage-crime network as a dark-themed plotly Figure."""
    G, metrics, _mean_ranks, pos = build_brokerage_graph()
    nodes = list(G.nodes())

    # --- crime-type volumes -> node size (option 3) -------------------------
    con = connect()
    vol = con.execute(
        "SELECT crime_type, COUNT(*) AS n FROM crimes GROUP BY crime_type"
    ).df()
    con.close()
    volume = dict(zip(vol["crime_type"], vol["n"]))

    # --- edges: one muted line per edge, width ~ weight, no hover -----------
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    edge_traces = []
    for (u, v), w in zip(G.edges(), weights):
        wn = w / max_w
        edge_traces.append(go.Scatter(
            x=[pos[u][0], pos[v][0]],
            y=[pos[u][1], pos[v][1]],
            mode="lines",
            line=dict(width=1 + 5 * wn ** 0.7,
                      color=f"rgba(122,130,160,{0.20 + 0.45 * wn:.3f})"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # --- node size = brokerage centrality (NaN off-LCC -> least central) ----
    # Linear min-max (not log/sqrt) so the few brokers visibly pop; centrality
    # has no extreme outlier the way volume does, so no compression is wanted.
    cfb = metrics["current_flow_betweenness"].reindex(nodes)
    cfb_vals = cfb.fillna(cfb.min()).to_numpy()
    span = cfb_vals.max() - cfb_vals.min()
    sizes = (18 + (cfb_vals - cfb_vals.min()) / span * (56 - 18)) if span else np.full(len(nodes), 30.0)

    # --- node colour = volume on a LOG scale (Violence is ~16M vs ~0.4M for the
    # brokers; without log it pegs the scale and flattens everyone else) ------
    vols = np.array([volume.get(n, 0) for n in nodes], dtype=float)
    log_vol = np.log10(np.maximum(vols, 1.0))
    # human-readable colourbar ticks at round volumes within the data range
    tick_v = np.array([0.5e6, 1e6, 2e6, 5e6, 10e6])
    tick_v = tick_v[(tick_v >= vols.min()) & (tick_v <= vols.max())]
    tickvals = np.log10(tick_v)
    ticktext = [f"{v / 1e6:g}M" for v in tick_v]

    customdata = np.column_stack([
        [volume.get(n, 0) for n in nodes],
        metrics["betweenness"].reindex(nodes).to_numpy(),
        cfb.to_numpy(),
        metrics["constraint"].reindex(nodes).to_numpy(),
        metrics["degree"].reindex(nodes).to_numpy(),
    ])

    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        text=nodes,
        textposition="bottom center",
        textfont=dict(color=TEXT_PRI, size=10, family="DM Mono, monospace"),
        marker=dict(
            size=sizes,
            color=log_vol,
            colorscale=VOLUME_SCALE,
            line=dict(width=1.2, color=BORDER),
            showscale=True,
            colorbar=dict(
                title=dict(text="Crime<br>volume", font=dict(color=TEXT_SEC, size=11)),
                thickness=10,
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(color=TEXT_SEC),
            ),
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Volume: %{customdata[0]:,.0f} crimes<br>"
            "Current-flow betweenness: %{customdata[2]:.3f}<br>"
            "Betweenness: %{customdata[1]:.3f}<br>"
            "Constraint: %{customdata[3]:.3f}<br>"
            "Degree: %{customdata[4]:.0f}<extra></extra>"
        ),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        hoverlabel=dict(bgcolor=BG_CARD, bordercolor=BORDER,
                        font=dict(color=TEXT_PRI, size=11)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    # equal aspect so nodes stay circular and the layout isn't stretched
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.write_json(DASHBOARD_ASSETS / "brokerage_network.json")  # for reuse in the dashboard without rebuilding
    return fig


def _ward_names() -> pd.DataFrame:
    """ward_code -> ward_name from the reprojected ward geojson (data/)."""
    with open(DATA / "wards_dec2021_uk_bgc_4326.geojson", encoding="utf-8") as f:
        gj = json.load(f)
    return pd.DataFrame([
        {"ward_code": p["WD21CD"], "ward_name": p.get("WD21NM") or p["WD21CD"]}
        for p in (ft["properties"] for ft in gj["features"])
        if p.get("WD21CD")
    ])


def build_ward_snapshot():
    """Precompute the per-ward brokerage snapshot the dashboard map reads.

    Moved verbatim from app.make_ward_data so the dashboard no longer runs the
    49M-row pipeline at boot. Trailing-3-month mean avg_betweenness per ward,
    percentile -> brokerage_score, with risk/action/units/crimes still faked off
    that score (the alternative colouring is a later discussion). Writes a tiny
    (~8k-row) parquet to the dashboard assets.
    """
    np.random.seed(42)
    random.seed(42)

    crime_pool = [
        "Robbery",
        "Drug Offences",
        "Anti-social Behaviour",
        "Bicycle Theft",
        "Shoplifting",
        "Burglary",
        "Public Order",
        "Violence",
    ]

    avg_betweenness_df = calculate_average_betweenness(
        score_where_sql="YEAR BETWEEN 2017 AND 2026"
    )

    max_period = avg_betweenness_df["period"].max()
    recent = avg_betweenness_df[avg_betweenness_df["period"] >= max_period - 2]
    ward_score = (
        recent.groupby("WD21CD")["avg_betweenness"]
        .mean()
        .reset_index()
    )

    ward_boundary_df = _ward_names()

    df = ward_boundary_df.merge(
        ward_score, left_on="ward_code", right_on="WD21CD", how="inner"
    )
    df = df.drop(columns="WD21CD")

    df["brokerage_score"] = (df["avg_betweenness"].rank(pct=True) * 100).round(2)

    def _score_to_risk(score):
        if score >= 80:
            return "High"
        elif score >= 60:
            return "Medium"
        else:
            return "Low"

    def _score_to_action(score):
        if score >= 80:
            return "+2 patrol units · Focused evening patrol"
        elif score >= 60:
            return "+1 patrol unit · Increased monitoring"
        else:
            return "Routine patrol"

    def _score_to_units(score):
        if score >= 80:
            return 6
        elif score >= 60:
            return 4
        else:
            return 2

    df["risk_level"] = df["brokerage_score"].apply(_score_to_risk)
    df["suggested_action"] = df["brokerage_score"].apply(_score_to_action)
    df["recommended_units"] = df["brokerage_score"].apply(_score_to_units)

    df["brokerage_crimes"] = [
        ", ".join(random.sample(crime_pool, 3)) for _ in range(len(df))
    ]
    df["predicted_risk"] = df["risk_level"].map({
        "High": "High probability of violent escalation in next 14 days",
        "Medium": "Moderate escalation risk in next 14 days",
        "Low": "Low short-term escalation risk",
    })

    df.to_parquet(DASHBOARD_ASSETS / "ward_snapshot.parquet")
    return df


def _last_actual_vso(cutoff_period: int) -> pd.DataFrame:
    """Actual V&SO count per ward in the model's cutoff month (the month right
    before the forecast).

    Pulled from the SAME source the model's panel builds its target `y` from --
    aggregate_lsoas_to_wards(get_crime_data()) filtered to TARGET -- so the number
    equals the model's last observed `y` exactly, no definitional drift. We skip
    the full build_ward_panel (which also runs the heavy brokerage/igraph step we
    don't need here) and just take the V&SO counts at the cutoff period.
    """
    from src.network_analysis.scores import get_crime_data, aggregate_lsoas_to_wards
    from src.new_models_test.build_panel import TARGET

    ward_long = aggregate_lsoas_to_wards(get_crime_data())  # WD21CD, period, crime_type, count
    last = ward_long[
        (ward_long["crime_type"] == TARGET) & (ward_long["period"] == cutoff_period)
    ]
    return (
        last.groupby("WD21CD")["count"].sum()
        .rename("vso_last_month").reset_index()
        .rename(columns={"WD21CD": "ward_code"})
    )


def build_forecast_snapshot(use_prophet: bool = True):
    """Precompute the next-month V&SO forecast per ward -> parquet the map reads.

    Reuses production_models.predict_next_month, which loads the saved models +
    last_regressors and predicts the month AFTER training -- no DB, no refit.
    `use_prophet` selects which pre-trained model to read:
        True  -> output/models/prophet  (one Prophet per ward)
        False -> output/models/lgbm     (pooled Poisson LightGBM)
    The chosen model must already be trained. Point forecast only --
    the saved predict path returns yhat with no interval.

    Also attaches the cutoff month's ACTUAL V&SO count (vso_last_month) and the
    absolute change (forecast_vso_change = forecast_vso - vso_last_month) so the
    map can show forecasted movement, not just level. The baseline is pinned to
    next_ds - 1 month, so it stays the model's true cutoff month even if the DB
    later gains a newer month.
    """
    import src.new_models_test.production_models as pm

    model = "prophet" if use_prophet else "lgbm"
    models_dir = pm.OUTPUT_DIR / "models" / model
    if not (models_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"No saved {model!r} model under {models_dir}. Train it first: set "
            f"MODEL = {model!r} at the top of production_models.py and run "
            "`python -m src.new_models_test.production_models`."
        )

    # Point production_models at the chosen model folder, then reuse its exact
    # load+predict path so the serve logic stays in one place.
    pm.MODELS_DIR = models_dir
    forecast = pm.predict_next_month()                  # WD21CD, ds, y_pred
    next_ds = pd.Timestamp(forecast["ds"].iloc[0])

    # Cutoff month = the month before the forecast (period = year*12 + month-1).
    next_period = next_ds.year * 12 + (next_ds.month - 1)
    last_actual = _last_actual_vso(next_period - 1)

    df = _ward_names().merge(
        forecast.rename(columns={"WD21CD": "ward_code", "y_pred": "forecast_vso"}),
        on="ward_code", how="inner",
    )
    df = df.drop(columns="ds").merge(last_actual, on="ward_code", how="left")
    # A ward with no row in the cutoff month had zero V&SO that month.
    df["vso_last_month"] = df["vso_last_month"].fillna(0)
    df["forecast_vso_change"] = df["forecast_vso"] - df["vso_last_month"]
    
    df["model"] = model
    df["forecast_month"] = next_ds.date().isoformat()

    df.to_parquet(DASHBOARD_ASSETS / "forecast_snapshot.parquet")
    ch = df["forecast_vso_change"]
    print(f"forecast_snapshot.parquet: {len(df)} wards | {model} | {next_ds.date()} "
          f"| change [{ch.min():.1f}, {ch.max():.1f}]")


def build_ward_force_mapping():
    """Precompute a ward -> police force mapping for the forecast map page."""
    con = connect()
    sql = "SELECT DISTINCT lsoa_code, reported_by, Count(*) AS n FROM crimes GROUP BY lsoa_code, reported_by"
    forces = con.execute(sql).df()
    con.close()

    lsoa_ward_mapping = pd.read_csv(DATA / "lsoa_ward_mapping.csv", sep=';', usecols=['LSOA11CD', 'WD21CD'])  # lsoa_code -> ward_code
    merged = forces.merge(lsoa_ward_mapping, left_on='lsoa_code', right_on='LSOA11CD') 

    ward_forces = (
        merged.groupby(['WD21CD', 'reported_by'])['n'].sum().reset_index()
        .sort_values('n', ascending=False)
        .drop_duplicates('WD21CD')
        .rename(columns={'reported_by': 'police_force', 'WD21CD' : 'ward_code'})
        [['ward_code', 'police_force']]
    )

    ward_forces.to_parquet(DASHBOARD_ASSETS / "ward_force_mapping.parquet")


if __name__ == "__main__":
    make_network()
    build_ward_snapshot()
    build_forecast_snapshot(use_prophet=True)
    build_ward_force_mapping()
    print("artifacts built")
