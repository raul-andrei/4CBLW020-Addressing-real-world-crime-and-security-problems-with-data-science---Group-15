"""Build the heavy dashboard artifacts once, so the app just loads them.

For now: the brokerage-crime network figure (crime types = nodes). The graph
construction + brokerage metrics are reused from
``src.network_analysis.network_visualization.build_brokerage_graph`` (single
source of truth); here we only do the plotly rendering in the dashboard's dark
theme.

One network is built per police force (plus an "All forces" global network) by
filtering the co-occurrence pipeline with a per-force ``reported_by`` SQL filter.
They're all serialised into a single ``brokerage_networks.json`` that the
dashboard loads at boot and switches between via a force dropdown.

Encoding (option 3, channels swapped so brokers pop):
    node size   = brokerage centrality (current-flow betweenness) -> brokers pop
    node colour = how common the crime type is (total volume, log scale)
    edge width  = co-occurrence weight (no hover on edges)
    target node = Violence and sexual offences drawn red (our forecast target)
"""

import json
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from src.network_analysis.preparation import connect
from src.network_analysis.scores import calculate_average_betweenness
from src.network_analysis.network_visualization import build_brokerage_graph
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"

DASHBOARD_ASSETS = ROOT / "src" / "dashboard" / "assets"

# --- dashboard palettes (one per theme; matches src/dashboard/app.py THEMES) ---
# Colour encodes VOLUME (size encodes brokerage). The volume scale is a cool blue
# sequential -- NOT the red "risk" scale -- so colour reads as "how much" and isn't
# mistaken for brokerage/risk (that's the size channel). Each theme's scale is
# tuned to read on that theme's card background (dark navy card vs white card).
NETWORK_THEMES = {
    "dark": {
        "bg_card": "#111520",
        "node_line": "#1e2640",
        "text_pri": "#e8eaf2",
        "text_sec": "#7a82a0",
        "edge_rgb": "122,130,160",
        "target_colour": "#ff4f4f",
        "volume_scale": [[0.0, "#2c3656"], [0.5, "#4f7cff"], [1.0, "#8fb4ff"]],
    },
    "light": {
        "bg_card": "#ffffff",
        "node_line": "#9aa6bd",
        "text_pri": "#16233f",
        "text_sec": "#5b6678",
        "edge_rgb": "110,124,150",
        "target_colour": "#e23b3b",
        "volume_scale": [[0.0, "#bcd2f5"], [0.5, "#2f6fed"], [1.0, "#15356f"]],
    },
}

# The forecast target. Its node is drawn solid red (off the volume scale) so it
# reads instantly as "what we're predicting", regardless of its volume/brokerage.
TARGET_CRIME = "Violence and sexual offences"

# Force key for the global (unfiltered) network in the saved artifact / dropdown.
ALL_FORCES = "All forces"


def _human_count(v: float) -> str:
    """Compact volume label: 1_500 -> '1.5k', 2_000_000 -> '2M'."""
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _volume_colorbar_ticks(vols):
    """Round '1/2/5 x 10^k' colourbar ticks within the volume range.

    Returns (tickvals_in_log10, ticktext). Adaptive to the slice size so the bar
    stays labelled for both the Met (millions) and small forces (thousands).
    """
    vols = np.asarray(vols, dtype=float)
    vols = vols[vols > 0]
    if vols.size == 0:
        return [], []
    vmin, vmax = vols.min(), vols.max()
    candidates = [m * 10 ** k for k in range(0, 9) for m in (1, 2, 5)]
    ticks = [c for c in candidates if vmin <= c <= vmax]
    if not ticks:
        ticks = [vmax]
    return list(np.log10(ticks)), [_human_count(c) for c in ticks]


def _crime_volumes(con, where_sql=None):
    """crime_type -> total count for a slice (None where_sql = all data)."""
    extra = f"WHERE {where_sql}" if where_sql else ""
    vol = con.execute(
        f"SELECT crime_type, COUNT(*) AS n FROM crimes {extra} GROUP BY crime_type"
    ).df()
    return dict(zip(vol["crime_type"], vol["n"]))


def _network_figure(G, metrics, pos, volume, theme="dark"):
    """Render one brokerage graph as a themed plotly Figure.

    Size = current-flow betweenness (brokerage), colour = crime volume (log), with
    the target crime split into its own solid-red trace so it pops as the target.
    """
    pal = NETWORK_THEMES.get(theme, NETWORK_THEMES["dark"])
    nodes = list(G.nodes())

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
                      color=f"rgba({pal['edge_rgb']},{0.20 + 0.45 * wn:.3f})"),
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

    customdata = np.column_stack([
        [volume.get(n, 0) for n in nodes],
        metrics["betweenness"].reindex(nodes).to_numpy(),
        cfb.to_numpy(),
        metrics["constraint"].reindex(nodes).to_numpy(),
        metrics["degree"].reindex(nodes).to_numpy(),
    ])

    hovertemplate = (
        "<b>%{text}</b><br>"
        "Volume: %{customdata[0]:,.0f} crimes<br>"
        "Current-flow betweenness: %{customdata[2]:.3f}<br>"
        "Betweenness: %{customdata[1]:.3f}<br>"
        "Constraint: %{customdata[3]:.3f}<br>"
        "Degree: %{customdata[4]:.0f}<extra></extra>"
    )

    # Split the target out so it can be solid red while the rest stay on the
    # volume colour scale. Boolean masks keep every per-node array aligned.
    nodes_arr = np.array(nodes)
    xs = np.array([pos[n][0] for n in nodes])
    ys = np.array([pos[n][1] for n in nodes])
    is_target = nodes_arr == TARGET_CRIME
    is_other = ~is_target

    # Colourbar reflects the non-target nodes (the target is off-scale in red).
    tickvals, ticktext = _volume_colorbar_ticks(vols[is_other])

    node_trace = go.Scatter(
        x=xs[is_other],
        y=ys[is_other],
        mode="markers+text",
        text=nodes_arr[is_other],
        textposition="bottom center",
        textfont=dict(color=pal["text_pri"], size=10, family="DM Mono, monospace"),
        marker=dict(
            size=sizes[is_other],
            color=log_vol[is_other],
            colorscale=pal["volume_scale"],
            line=dict(width=1.2, color=pal["node_line"]),
            showscale=True,
            colorbar=dict(
                title=dict(text="Crime<br>volume", font=dict(color=pal["text_sec"], size=11)),
                thickness=10,
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(color=pal["text_sec"]),
            ),
        ),
        customdata=customdata[is_other],
        hovertemplate=hovertemplate,
        showlegend=False,
    )

    data = edge_traces + [node_trace]

    if is_target.any():
        target_trace = go.Scatter(
            x=xs[is_target],
            y=ys[is_target],
            mode="markers+text",
            text=nodes_arr[is_target],
            textposition="bottom center",
            textfont=dict(color=pal["target_colour"], size=10, family="DM Mono, monospace"),
            marker=dict(
                size=sizes[is_target],
                color=pal["target_colour"],
                line=dict(width=1.2, color=pal["node_line"]),
                showscale=False,
            ),
            customdata=customdata[is_target],
            hovertemplate=hovertemplate,
            showlegend=False,
        )
        data.append(target_trace)

    fig = go.Figure(data=data)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        hoverlabel=dict(bgcolor=pal["bg_card"], bordercolor=pal["node_line"],
                        font=dict(color=pal["text_pri"], size=11)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    # equal aspect so nodes stay circular and the layout isn't stretched
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _build_force_figures(con, where_sql=None):
    """Build the brokerage graph once for a slice, then render it in every theme.

    The graph/metrics/layout are theme-independent, so the heavy step runs once;
    only the cheap plotly rendering repeats per theme. Returns {theme -> Figure}.
    """
    G, metrics, _mean_ranks, pos = build_brokerage_graph(where_sql=where_sql, con=con)
    volume = _crime_volumes(con, where_sql)
    return {theme: _network_figure(G, metrics, pos, volume, theme=theme)
            for theme in NETWORK_THEMES}


def make_network():
    """Build one brokerage network per police force (+ a global "All forces" one),
    in every theme, and serialise them to ``brokerage_networks.json``.

    Returns {theme -> {force -> Figure}}; the saved file is {theme -> {force ->
    plotly JSON}}, which the dashboard indexes by the active theme and force.
    """
    con = connect()
    forces = con.execute("""
        SELECT DISTINCT reported_by FROM crimes
        WHERE reported_by IS NOT NULL
        ORDER BY reported_by
    """).df()["reported_by"].tolist()

    # figures_by_force: {force -> {theme -> Figure}}
    figures_by_force = {ALL_FORCES: _build_force_figures(con, where_sql=None)}
    for force in forces:
        force_sql = force.replace("'", "''")  # SQL-escape single quotes
        figures_by_force[force] = _build_force_figures(con, where_sql=f"reported_by = '{force_sql}'")
    con.close()

    # Pivot to {theme -> {force -> json}} (the shape the dashboard loads).
    figures = {
        theme: {force: per_theme[theme] for force, per_theme in figures_by_force.items()}
        for theme in NETWORK_THEMES
    }
    combined = {
        theme: {force: pio.to_json(fig) for force, fig in by_force.items()}
        for theme, by_force in figures.items()
    }
    out = DASHBOARD_ASSETS / "brokerage_networks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(combined, f)
    print(f"brokerage_networks.json: {len(NETWORK_THEMES)} themes x "
          f"({ALL_FORCES} + {len(forces)} forces) networks")
    return figures


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
