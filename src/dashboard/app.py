import dash
from dash import Dash, html, dcc, Input, Output, State
from duckdb import connect
import plotly.express as px
import plotly
import pandas as pd
import numpy as np
import json ## for the UK Map
from pathlib import Path
from duckdb import connect

app = Dash(__name__, suppress_callback_exceptions=True)

# ── Theming ──
# Two palettes. The CSS chrome is themed via CSS variables in style.css (a
# light-theme class on <body>); Plotly figures set their colours in Python, so
# each figure builder takes a theme and reads from here. Kept in sync with style.css.
THEMES = {
    "dark": {
        "bg": "#0a0d14", "bg_card": "#111520",
        "border": "#1e2640", "border_bright": "#2a3560", "grid": "#1e2640",
        "accent": "#4f7cff", "accent_dim": "#2a3f80",
        "text_pri": "#e8eaf2", "text_sec": "#7a82a0", "text_mute": "#3d4460",
        "warning": "#ffb84f", "success": "#4fff9f", "danger": "#ff4f4f",
        "map_style": "carto-darkmatter",
        "covid_fill": "rgba(255,184,79,0.06)",
        # sequential risk heat + diverging (negative<->positive) map scales
        "seq_scale": [[0.0, "#f5f5f5"], [0.35, "#ffd166"], [0.65, "#ff8c42"], [1.0, "#ff4f4f"]],
        "div_scale": [[0.0, "#4f7cff"], [0.5, "#1b2236"], [1.0, "#ff6b6b"]],
        "area_palette": px.colors.qualitative.Light24,
    },
    "light": {
        "bg": "#eef2f9", "bg_card": "#ffffff",
        "border": "#dbe3f0", "border_bright": "#b9c6de", "grid": "#e6ebf4",
        "accent": "#2f6fed", "accent_dim": "#aac3f7",
        "text_pri": "#16233f", "text_sec": "#5b6678", "text_mute": "#98a2b5",
        "warning": "#ed7d2b", "success": "#1f9d57", "danger": "#e23b3b",
        "map_style": "carto-positron",
        "covid_fill": "rgba(237,125,43,0.13)",
        "seq_scale": [[0.0, "#dbe6ff"], [0.35, "#ffd166"], [0.65, "#f7861f"], [1.0, "#d62828"]],
        "div_scale": [[0.0, "#2f6fed"], [0.5, "#eef2f9"], [1.0, "#d62828"]],
        "area_palette": px.colors.qualitative.Dark24,
    },
}


def palette(theme):
    return THEMES.get(theme, THEMES["dark"])


def plotly_layout(theme="dark"):
    """Shared Plotly layout (axes/legend/hover) in the given theme."""
    p = palette(theme)
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color=p["text_sec"], size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, zeroline=False, color=p["text_mute"],
                   tickfont=dict(size=10), linecolor=p["border"]),
        yaxis=dict(showgrid=True, gridcolor=p["grid"], zeroline=False,
                   color=p["text_mute"], tickfont=dict(size=10), linecolor=p["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=p["text_sec"])),
        hoverlabel=dict(bgcolor=p["bg_card"], bordercolor=p["border"],
                        font=dict(color=p["text_pri"], size=11)),
    )

## Making the UK Map (Loading fake data)
import random
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DASHBOARD_ASSETS = BASE_DIR / "src" / "dashboard" / "assets"
#LSOA_PATH = BASE_DIR / "data" / "lsoa_boundaries.geojson"
WARD_PATH = BASE_DIR / "data" / "wards_dec2021_uk_bgc_4326.geojson"

#with open(LSOA_PATH, "r", encoding="utf-8") as f:
#    LSOA_GEOJSON = json.load(f)

with open(WARD_PATH, "r", encoding="utf-8") as f:
    WARD_GEOJSON = json.load(f)


# Brokerage networks, one per force (+ "All forces") per theme, precomputed by
# artifacts.py and stored as {theme -> {force -> plotly JSON}}. Parsed once here.
with open(DASHBOARD_ASSETS / "brokerage_networks.json", encoding="utf-8") as f:
    _BROKERAGE_RAW = json.load(f)
BROKERAGE_NETWORKS = {
    theme: {force: plotly.io.from_json(s) for force, s in forces.items()}
    for theme, forces in _BROKERAGE_RAW.items()
}
BROKERAGE_FORCES = list(BROKERAGE_NETWORKS["dark"].keys())  # "All forces" first (build order)
FORECASTS = pd.read_parquet(DASHBOARD_ASSETS / "forecast_snapshot.parquet")
WARD_FORCE_MAPPING = pd.read_parquet(DASHBOARD_ASSETS / "ward_force_mapping.parquet")

def extract_lsoa_base(geojson):
    rows = []

    for feature in geojson["features"]:
        props = feature["properties"]

        # adjust these names if your GeoJSON uses different property names
        code = props.get("LSOA21CD") or props.get("LSOA11CD")
        name = props.get("LSOA21NM") or props.get("LSOA11NM") or code

        if code:
            rows.append({
                "ward_code": code,
                "ward_name": name
            })

    return pd.DataFrame(rows)

def extract_ward_base(geojson):
    rows = []

    for feature in geojson["features"]:
        props = feature["properties"]

        code = props.get("WD21CD")
        name = props.get("WD21NM") or code

        if code:
            rows.append({
                "ward_code": code,
                "ward_name": name
            })

    return pd.DataFrame(rows)

def ward_boundaries_by_force(mapping_df, ward_geojson):
    force_wards = {}
    for _, row in mapping_df.iterrows():
        force = row["police_force"]
        ward = row["ward_code"]
        if force not in force_wards:
            force_wards[force] = set()
        force_wards[force].add(ward)

    force_boundaries = {}
    for force, wards in force_wards.items():
        boundaries = []
        for feature in ward_geojson["features"]:
            if feature["properties"].get("WD21CD") in wards:
                boundaries.append(feature)
        # plotly's geojson= expects a FeatureCollection dict, not a bare feature list
        force_boundaries[force] = {"type": "FeatureCollection", "features": boundaries}

    return force_boundaries

# Precompute the per-force ward subsets once at boot (not per render). Each value
# is a FeatureCollection; the forecast map embeds only the selected force's wards.
FORCE_WARD_BOUNDARIES = ward_boundaries_by_force(WARD_FORCE_MAPPING, WARD_GEOJSON)
EMPTY_FEATURE_COLLECTION = {"type": "FeatureCollection", "features": []}

def make_fake_lsoa_data(geojson):
    df = extract_ward_base(geojson)

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
        "Violence"
    ]

    df["brokerage_score"] = np.random.randint(20, 96, size=len(df))

    def score_to_risk(score):
        if score >= 80:
            return "High"
        elif score >= 60:
            return "Medium"
        else:
            return "Low"

    def score_to_action(score):
        if score >= 80:
            return "+2 patrol units · Focused evening patrol"
        elif score >= 60:
            return "+1 patrol unit · Increased monitoring"
        else:
            return "Routine patrol"

    def score_to_units(score):
        if score >= 80:
            return 6
        elif score >= 60:
            return 4
        else:
            return 2

    df["risk_level"] = df["brokerage_score"].apply(score_to_risk)
    df["suggested_action"] = df["brokerage_score"].apply(score_to_action)
    df["recommended_units"] = df["brokerage_score"].apply(score_to_units)
    df["brokerage_crimes"] = [
        ", ".join(random.sample(crime_pool, 3)) for _ in range(len(df))
    ]
    df["predicted_risk"] = df["risk_level"].map({
        "High": "High probability of violent escalation in next 14 days",
        "Medium": "Moderate escalation risk in next 14 days",
        "Low": "Low short-term escalation risk"
    })

    return df

# Per-ward brokerage snapshot, precomputed by artifacts.py (build_ward_snapshot).
# Regenerate after a data refresh.
WARD_SNAPSHOT = DASHBOARD_ASSETS / "ward_snapshot.parquet"
if not WARD_SNAPSHOT.exists():
    raise FileNotFoundError(
        f"{WARD_SNAPSHOT} missing -- run: python -m src.dashboard.artifacts"
    )
WARD_DF = pd.read_parquet(WARD_SNAPSHOT)

# per-ward lookups for the "Selected Ward" click panel
WARD_BY_CODE = WARD_DF.set_index("ward_code")
WARD_FORCE_BY_CODE = dict(zip(WARD_FORCE_MAPPING["ward_code"], WARD_FORCE_MAPPING["police_force"]))
FORECAST_BY_WARD = FORECASTS.set_index("ward_code")
## Making actual map - lsoa based map, keeping it here just for a moment,
## Delete later

def make_ward_map(theme="dark"):
    p = palette(theme)
    fig = px.choropleth_map(
        WARD_DF,
        geojson=WARD_GEOJSON,
        locations="ward_code",
        featureidkey="properties.WD21CD",   # change if needed
        color="brokerage_score",
        hover_name="ward_name",
        hover_data={
            "ward_code": True,
            "brokerage_score": True,
            "risk_level": True,
            "brokerage_crimes": False,
            "predicted_risk": False,
            "suggested_action": False,
            "recommended_units": False
        },
        labels={
            "ward_code": "Ward code",
            "brokerage_score": "Brokerage score",
            "risk_level": "Assesed Risk Level",
            "predicted_risk": "Assesed short-term escalation risk",
            "suggested_action": "Suggested action",
            "recommended_units": "Recommended units to deploy",
        },
        color_continuous_scale=p["seq_scale"],
        map_style=p["map_style"],
        zoom=5.1,
        center={"lat": 54.5, "lon": -2.5},
        opacity=0.65
    )

    fig.update_traces(
        marker_line_width=0.3,
        marker_line_color=p["border"],
        # Only the real fields; the click handler looks the rest up by ward_code.
        customdata=np.stack([
            WARD_DF["ward_code"],
            WARD_DF["ward_name"],
            WARD_DF["brokerage_score"],
            WARD_DF["risk_level"],
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Ward code: %{customdata[0]}<br>"
            "Brokerage score: %{customdata[2]:.0f} / 100<br>"
            "Risk level: %{customdata[3]}<extra></extra>"
        ),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title=dict(text="Brokerage score", font=dict(color=p["text_sec"])),
            thickness=10,
            tickfont=dict(color=p["text_sec"])
        )
    )

    return fig

# Sample data (replace with real data later)
CRIME_TYPES = [
    "Violence and sexual offences", "Anti-social behaviour", "Criminal damage and arson",
    "Other theft", "Vehicle crime", "Burglary", "Shoplifting", "Drugs",
    "Public order", "Robbery", "Theft from the person", "Bicycle theft",
    "Possession of weapons", "Other crime"
]

def get_forces():
    conn = connect(BASE_DIR / "data" / "crimes.db")
    sql = "SELECT DISTINCT reported_by FROM crimes"
    df = conn.execute(sql).df()
    return df['reported_by'].tolist()

FORCES = get_forces()
FORCES.append("All forces")
FORCES[0], FORCES[-1] = FORCES[-1], FORCES[0]  # Move "All forces" to the front

BROKERS = [
    ("Theft from the person", 0.280),
    ("Possession of weapons", 0.261),
    ("Robbery", 0.245),
    ("Bicycle theft", 0.198),
    ("Criminal damage", 0.167),
    ("Drugs", 0.152),
    ("Violence", 0.141),
    ("Public order", 0.138),
]

def make_time_series(force):
    conn = connect(BASE_DIR / "data" / "crimes.db")
    if force == "All forces":
        sql = f""" SELECT year, month_num, COUNT(*) as crime_count
                FROM crimes
                GROUP BY year, month_num
                ORDER BY year, month_num """
    else:
        sql = f""" SELECT year, month_num, COUNT(*) as crime_count
                FROM crimes
                WHERE reported_by = '{force}'
                GROUP BY year, month_num
                ORDER BY year, month_num """
        
    df = conn.execute(sql).df()
    conn.close()

    df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month_num'], day=1))
    df = df.drop(columns=['year', 'month_num'])
    return df

    
def make_crime_distribution(force):
    conn = connect(BASE_DIR / "data" / "crimes.db")
    if force == "All forces":
        sql = """
            SELECT crime_type, COUNT(*) AS crime_count
            FROM crimes
            GROUP BY crime_type
            ORDER BY crime_count DESC
        """
    else:
        sql = f"""
            SELECT crime_type, COUNT(*) AS crime_count
            FROM crimes
            WHERE reported_by = '{force}'
            GROUP BY crime_type
            ORDER BY crime_count DESC
        """
    df = conn.execute(sql).df()
    df['crime_count'] = df['crime_count'].sort_values(ascending=False)
    return df


def make_crime_mix_over_time(force):
    """Monthly count per crime type -> share-over-time (100% stacked area)."""
    conn = connect(BASE_DIR / "data" / "crimes.db")
    if force == "All forces":
        sql = """ SELECT year, month_num, crime_type, COUNT(*) AS crime_count
                FROM crimes
                GROUP BY year, month_num, crime_type
                ORDER BY year, month_num """
    else:
        sql = f""" SELECT year, month_num, crime_type, COUNT(*) AS crime_count
                FROM crimes
                WHERE reported_by = '{force}'
                GROUP BY year, month_num, crime_type
                ORDER BY year, month_num """

    df = conn.execute(sql).df()
    df = df.sort_values(by="crime_type", ascending=False)
    conn.close()

    df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month_num'], day=1))
    df = df.drop(columns=['year', 'month_num'])
    return df


# Groups the 27 raw last_outcome values into readable buckets. The data-gap values
# (e.g. "Court result unavailable") go to Unknown, as does anything unmapped.
OUTCOME_BUCKETS = {
    "Investigation complete; no suspect identified":       "No suspect identified",
    "Unable to prosecute suspect":                         "Unable to prosecute",
    "Under investigation":                                 "Under investigation",
    "Awaiting court outcome":                              "Charged / court",
    "Offender sent to prison":                             "Charged / court",
    "Offender given community sentence":                   "Charged / court",
    "Defendant found not guilty":                          "Charged / court",
    "Offender fined":                                      "Charged / court",
    "Offender given suspended prison sentence":            "Charged / court",
    "Offender given conditional discharge":                "Charged / court",
    "Suspect charged as part of another case":             "Charged / court",
    "Court case unable to proceed":                        "Charged / court",
    "Defendant sent to Crown Court":                       "Charged / court",
    "Offender ordered to pay compensation":                "Charged / court",
    "Offender deprived of property":                       "Charged / court",
    "Offender given absolute discharge":                   "Charged / court",
    "Local resolution":                                    "Out-of-court resolution",
    "Offender given a caution":                            "Out-of-court resolution",
    "Offender given a drugs possession warning":           "Out-of-court resolution",
    "Offender given penalty notice":                       "Out-of-court resolution",
    "Offender otherwise dealt with":                       "Out-of-court resolution",
    "Further investigation is not in the public interest": "No further action",
    "Formal action is not in the public interest":         "No further action",
    "Further action is not in the public interest":        "No further action",
    "Action to be taken by another organisation":          "No further action",
    "None":                                                "Unknown / not recorded",
    "Status update unavailable":                           "Unknown / not recorded",
    "Court result unavailable":                            "Unknown / not recorded",
}


def make_outcome_breakdown(force):
    """Share of crimes ending in each outcome bucket (the case clear-up view)."""
    conn = connect(BASE_DIR / "data" / "crimes.db")
    if force == "All forces":
        sql = """ SELECT last_outcome, COUNT(*) AS n
                FROM crimes
                GROUP BY last_outcome """
    else:
        sql = f""" SELECT last_outcome, COUNT(*) AS n
                FROM crimes
                WHERE reported_by = '{force}'
                GROUP BY last_outcome """

    df = conn.execute(sql).df()
    conn.close()

    df['bucket'] = df['last_outcome'].map(OUTCOME_BUCKETS).fillna("Unknown / not recorded")
    out = df.groupby('bucket', as_index=False)['n'].sum()
    out['share'] = out['n'] / out['n'].sum() * 100
    return out


# Sidebar
def sidebar():
    return html.Div(className="sidebar", children=[
        html.Div(className="sidebar-logo", children=[
            "POLICE", html.Span("ANALYTICS")
        ]),
        html.Div(className="sidebar-label", children="Navigation"),
        html.Div(id="sidebar-nav"),
        html.Button(id="theme-toggle", className="theme-toggle", n_clicks=0,
                    children="☀  Light mode"),
        html.Div(className="sidebar-footer", children=[
            html.Div("Data: data.police.uk"),
            html.Div("All outputs are decision-support tools only. Final allocation decisions remain with police officers.",
                     className="sidebar-disclaimer")

        ])
    ])

# Overview page
# Overview page
def overview_page(theme="dark"):

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Overview", className="page-tag"),
            html.Div("Brokerage Crime Intelligence System", className="page-title"),
            html.Div("Turning brokerage crime patterns into actionable resource allocation decisions",
                     className="page-subtitle"),
        ]),

        # Map + selected LSOA details row
        html.Div(className="chart-grid", children=[
            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Brokerage Risk Map", className="card-title"),
                        html.Div("UK wards coloured by brokerage-risk score", className="card-subtitle"),
                    ]),
                ]),
                dcc.Graph(
                    id="lsoa-map",
                    figure=make_ward_map(theme),
                    config={"displayModeBar": False},
                    style={"height": "520px"}
                ),
            ]),

            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Selected Ward", className="card-title"),
                        html.Div("Click an area on the map for details", className="card-subtitle"),
                    ])
                ]),
                html.Div(
                    id="lsoa-details",
                    # no colour set, so it inherits the themed body text colour
                    style={
                        "padding": "16px",
                        "fontFamily": "DM Mono, monospace",
                        "fontSize": "14px"
                    },
                    children="Click a ward to view brokerage details."
                )
            ]),
        ]),

        # Leaderboard row
        html.Div(className="card", style={"marginTop": "24px"}, children=[
            html.Div(className="card-header", children=[
                html.Div([
                    html.Div("Highest Risk wards", className="card-title"),
                    html.Div("Top areas ranked by brokerage-risk score", className="card-subtitle"),
                ]),
            ]),
            make_ward_leaderboard(10)
        ])
    ])
    


# Force Explorer page
def brokerage_network_page(theme="dark"):
    default_force = BROKERAGE_FORCES[0]  # "All forces"
    return html.Div(className="network-graph-placeholder", children=[
        html.Div(
            children=[
                html.Div(
                    className="card",
                    children=[
                        html.Div(className="page-header",
                                children=[
                                    html.Div("Brokerage Network", className="page-title"),
                                    html.Div("Size encodes brokerage centrality · colour encodes crime volume · target (Violence and sexual offences) in red",
                                             className="page-subtitle"),
                                ]),
                        html.Div(className="force-selector", children=[
                            html.Label("Select Police Force"),
                            dcc.Dropdown(
                                className="policeforce-dropdown",
                                id="network-force-select",
                                options=[{"label": f, "value": f} for f in BROKERAGE_FORCES],
                                value=default_force,
                                clearable=False,
                            ),
                        ]),
                        dcc.Graph(
                            id='network-graph',
                            figure=BROKERAGE_NETWORKS[theme][default_force],
                            config={"displayModeBar": False}
                        )
                    ]
                )
            ]
        )
    ])


def _merge_forcast_with_mapping(forecast_df, mapping_df):
    """Merge the forecast dataframe with the ward-force mapping to get police force info."""
    merged = forecast_df.merge(mapping_df, on="ward_code", how="inner")
    return merged

def _force_map_view(geojson, ward_codes):
    """Compute a (center, zoom) that frames the given wards, derived from the geojson bbox."""
    codes = set(ward_codes)
    bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]  # min_lon, min_lat, max_lon, max_lat

    def walk(coords):
        if isinstance(coords[0], (float, int)):
            lon, lat = coords[0], coords[1]
            bounds[0] = min(bounds[0], lon); bounds[1] = min(bounds[1], lat)
            bounds[2] = max(bounds[2], lon); bounds[3] = max(bounds[3], lat)
        else:
            for c in coords:
                walk(c)

    for ft in geojson["features"]:
        if ft["properties"].get("WD21CD") in codes:
            walk(ft["geometry"]["coordinates"])

    if bounds[0] == float("inf"):  # no matching wards: fall back to UK-wide view
        return {"lat": 54.5, "lon": -2.5}, 5.1

    min_lon, min_lat, max_lon, max_lat = bounds
    center = {"lon": (min_lon + max_lon) / 2, "lat": (min_lat + max_lat) / 2}
    span = max(max_lon - min_lon, max_lat - min_lat)
    # calibrated so the UK-wide span (~10 deg) maps to zoom ~5.1, then clamped
    zoom = 8.4 - np.log2(span) if span > 0 else 9.0
    zoom = float(np.clip(zoom - 0.3, 4.0, 11.0))
    return center, zoom

def forecast_per_force_map(police_force: str = "Metropolitan Police Service", theme="dark"):
    p = palette(theme)
    df = _merge_forcast_with_mapping(FORECASTS, WARD_FORCE_MAPPING)
    df = df[df["police_force"] == police_force]

    force_specific = FORCE_WARD_BOUNDARIES.get(police_force, EMPTY_FEATURE_COLLECTION)

    vso_abs_max = max(abs(df["forecast_vso_change"].min()), df["forecast_vso_change"].max())
    center, zoom = _force_map_view(force_specific, df["ward_code"])

    fig = px.choropleth_map(
        data_frame=df,
        geojson=force_specific,
        locations="ward_code",
        featureidkey="properties.WD21CD",
        color="forecast_vso_change",
        hover_name="ward_name",
        hover_data={
            "ward_code": True,
            "forecast_vso_change": ":.1f",
            "forecast_vso": ":.1f",
            "forecast_month": True,
        },
        labels={
            "ward_code": "Ward code",
            "forecast_vso_change": "Forecast change",
            "forecast_vso": "Forecast VSO",
            "forecast_month": "Forecast month",
        },
        color_continuous_scale=p["div_scale"],
        color_continuous_midpoint=0,
        range_color=(-vso_abs_max, vso_abs_max),
        map_style=p["map_style"],
        zoom=zoom,
        center=center,
        opacity=0.65
    )
    fig.update_traces(
        marker_line_width=0.3,
        marker_line_color=p["border"],
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        # key uirevision on the force so switching forces recenters the map
        # (a constant uirevision would keep the old camera on the new figure)
        uirevision=police_force,
        coloraxis_colorbar=dict(
            title=dict(text="Forecast change", font=dict(color=p["text_sec"])),
            thickness=10,
            tickfont=dict(color=p["text_sec"])
        )
    )

    return fig


## Forecast page
def forecast_page(theme="dark"):

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Forecast", className="page-tag"),
            html.Div("Violence & Sexual Offences Forecast", className="page-title"),
            html.Div("Forecasted change in violence & sexual offences for the coming month, ward by ward",
                     className="page-subtitle"),
        ]),
        html.Div(className="force-selector", children=[
            html.Label("Select Police Force"),
            dcc.Dropdown(
                className="policeforce-dropdown",
                id="police-force-select",
                options=[{"label": f, "value": f} for f in sorted(_merge_forcast_with_mapping(FORECASTS, WARD_FORCE_MAPPING)["police_force"].unique())],
                value="Metropolitan Police Service",
                clearable=False,
            ),
        ]),
        html.Div(className="card", children=[
            html.Div(className="card-header", children=[
                html.Div([
                    html.Div("Next-Month Forecast Map", className="card-title"),
                    html.Div("Ward-level change vs the latest month · blue = falling, red = rising",
                             className="card-subtitle"),
                ])
            ]),
            dcc.Graph(
                id="forecast-map",
                figure=forecast_per_force_map(theme=theme),
                config={"displayModeBar": False},
                style={"height": "70vh"},
            ),
        ]),
    ],
    className="forecast-main")


## Resource allocation page
def compute_allocation(police_force, min_pct=0.3, max_pct=20.0):
    df = _merge_forcast_with_mapping(FORECASTS, WARD_FORCE_MAPPING)
    df = df[df["police_force"] == police_force].copy()

    total = df["forecast_vso"].sum()
    if total == 0:
        df["allocation_pct"] = 0.0
        return df

    raw_pct = df["forecast_vso"] / total * 100
    eligible = raw_pct >= min_pct

    eligible_total = df.loc[eligible, "forecast_vso"].sum()
    if eligible_total == 0:
        df["allocation_pct"] = 0.0
        return df

    df["allocation_pct"] = 0.0
    df.loc[eligible, "allocation_pct"] = (df.loc[eligible, "forecast_vso"] / eligible_total * 100).round(2)
    df["allocation_pct"] = df["allocation_pct"].clip(upper=max_pct)

    return df


def allocation_map(police_force="Metropolitan Police Service", theme="dark"):
    p = palette(theme)
    df = compute_allocation(police_force)
    force_specific = FORCE_WARD_BOUNDARIES.get(police_force, EMPTY_FEATURE_COLLECTION)
    center, zoom = _force_map_view(force_specific, df["ward_code"])

    fig = px.choropleth_map(
        data_frame=df,
        geojson=force_specific,
        locations="ward_code",
        featureidkey="properties.WD21CD",
        color="allocation_pct",
        hover_name="ward_name",
        hover_data={
            "ward_code": True,
            "allocation_pct": ":.2f",
        },
        labels={
            "ward_code": "Ward code",
            "allocation_pct": "Allocation (%)",
        },
        color_continuous_scale=p["seq_scale"],
        map_style=p["map_style"],
        zoom=zoom,
        center=center,
        opacity=0.65
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color=p["border"])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision=police_force,
        coloraxis_colorbar=dict(
            title=dict(text="Allocation (%)", font=dict(color=p["text_sec"])),
            thickness=10,
            tickfont=dict(color=p["text_sec"])
        )
    )
    return fig


def allocation_leaderboard(police_force="Metropolitan Police Service", n=10):
    df = compute_allocation(police_force)
    top = df[df["allocation_pct"] > 0].sort_values("allocation_pct", ascending=False).head(n)
    max_alloc = top["allocation_pct"].max() if len(top) > 0 else 1.0

    return html.Div(className="broker-list", children=[
        html.Div(className="broker-item", children=[
            html.Div(f"#{i+1}", className="broker-rank"),
            html.Div(row["ward_name"], className="broker-name"),
            html.Div(className="broker-bar-wrap", children=[
                html.Div(className="broker-bar", style={"width": f"{row['allocation_pct'] / max_alloc * 100}%"})
            ]),
            html.Div(f"{row['allocation_pct']:.2f}%", className="broker-score"),
        ])
        for i, (_, row) in enumerate(top.iterrows())
    ])


def allocation_page(theme="dark"):
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Allocation", className="page-tag"),
            html.Div("Resource Allocation", className="page-title"),
            html.Div("Proportional task force deployment based on forecasted violence risk",
                     className="page-subtitle"),
        ]),
        html.Div(className="force-selector", children=[
            html.Label("Select Police Force"),
            dcc.Dropdown(
                className="policeforce-dropdown",
                id="allocation-force-select",
                options=[{"label": f, "value": f} for f in sorted(WARD_FORCE_MAPPING["police_force"].unique())],
                value="Metropolitan Police Service",
                clearable=False,
            ),
        ]),
        html.Div(className="card", children=[
            html.Div(className="card-header", children=[
                html.Div([
                    html.Div("Task Force Deployment Map", className="card-title"),
                    html.Div("Wards coloured by % of task force allocated", className="card-subtitle"),
                ])
            ]),
            dcc.Graph(
                id="allocation-map",
                figure=allocation_map(theme=theme),
                config={"displayModeBar": False},
                style={"height": "520px"}
            ),
        ]),
        html.Div(className="card", style={"marginTop": "24px"}, children=[
            html.Div(className="card-header", children=[
                html.Div([
                    html.Div("Top Allocated Wards", className="card-title"),
                    html.Div("Wards receiving the highest share of the task force", className="card-subtitle"),
                ])
            ]),
            html.Div(id="allocation-leaderboard", children=allocation_leaderboard())
        ])
    ])

## General trends page
def general_trends_page(theme="dark"):
    # Figures here are filled by update_explorer (which reads the theme); the page
    # only needs the themed chrome, handled by CSS.
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Trends", className="page-tag"),
            html.Div("General Trends", className="page-title"),
            html.Div("Explore general crime trends by police force", className="page-subtitle"),
        ]),

        html.Div(className="force-selector", children=[
            html.Label("Select Police Force"),
            dcc.Dropdown(
                className="policeforce-dropdown",
                id="force-dropdown",
                options=[{"label": f, "value": f} for f in FORCES],
                value="All forces",
                clearable=False,
            ),
        ]),

        html.Div(className="chart-grid", children=[
            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Crime Over Time", className="card-title"),
                        html.Div("Monthly totals · shaded COVID period", className="card-subtitle"),
                    ]),
                ]),
                dcc.Graph(
                    id="time-series-chart",
                    config={"displayModeBar": False},
                    style={"height": "280px"}
                ),
            ]),

            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Crime Type Breakdown", className="card-title"),
                        html.Div("Distribution across categories", className="card-subtitle"),
                    ]),
                ]),
                dcc.Graph(
                    id="crime-dist-chart",
                    config={"displayModeBar": False},
                    style={"height": "280px"}
                ),
            ]),
        ]),

        html.Div(className="chart-grid", children=[
            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Crime Mix Over Time", className="card-title"),
                        html.Div("Share of each crime type per month", className="card-subtitle"),
                    ]),
                ]),
                dcc.Graph(
                    id="crime-mix-chart",
                    config={"displayModeBar": False},
                    style={"height": "320px"}
                ),
            ]),

            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Case Outcomes", className="card-title"),
                        html.Div("What happens to reported crimes", className="card-subtitle"),
                    ]),
                ]),
                dcc.Graph(
                    id="outcome-chart",
                    config={"displayModeBar": False},
                    style={"height": "320px"}
                ),
            ]),
        ]),
    ])


## Leaderboard function (Areas with highest brokerage scores)
def make_ward_leaderboard(n=10):
    top_wards = WARD_DF.sort_values("brokerage_score", ascending=False).head(n)

    return html.Div(className="broker-list", children=[
        html.Div(className="broker-item", children=[
            html.Div(f"#{i+1}", className="broker-rank"),
            html.Div(row["ward_name"], className="broker-name"),
            html.Div(className="broker-bar-wrap", children=[
                html.Div(
                    className="broker-bar",
                    style={"width": f"{row['brokerage_score']}%"}
                )
            ]),
            html.Div(str(round(row["brokerage_score"],2)), className="broker-score"),
        ])
        for i, (_, row) in enumerate(top_wards.iterrows())
    ])

# Placeholder pages
def placeholder_page(title, tag, message):
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div(tag, className="page-tag"),
            html.Div(title, className="page-title"),
        ]),
        html.Div(className="card", children=[
            html.Div(className="coming-soon", children=[
                html.Div("◎", className="coming-soon-icon"),
                html.Div(message, className="coming-soon-text"),
            ])
        ])
    ])

# App layout
app.layout = html.Div(className="dashboard-wrapper", children=[
    dcc.Location(id="url"),
    # theme persists across reloads; a clientside callback mirrors it onto <body>
    # for the CSS, and render_page reads it to rebuild the figures
    dcc.Store(id="theme-store", storage_type="local", data="dark"),
    html.Div(id="theme-dummy", style={"display": "none"}),
    sidebar(),
    html.Div(className="main-content", children=[
        # spinner for the server round-trip on page/force changes
        dcc.Loading(
            id="page-loading",
            delay_show=150,          # don't flash on fast (non-map) pages
            overlay_style={"visibility": "visible", "opacity": 0.35},
            custom_spinner=html.Div(className="page-loader", children=[
                html.Div(className="page-loader-spinner"),
                html.Div("Loading map, please wait…", className="page-loader-text"),
            ]),
            children=html.Div(id="page-content"),
        ),
    ]),
])

# Routing. Also re-fires on theme change so the page rebuilds in the new theme.
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("theme-store", "data"),
)
def render_page(path, theme):
    theme = theme or "dark"
    if path == "/brokerage-network":
        return brokerage_network_page(theme)
    elif path == "/forecast":
        return forecast_page(theme)
    elif path == "/general-trends":
        return general_trends_page(theme)
    elif path == "/allocation":
        return allocation_page(theme)

    return overview_page(theme)

# Force Explorer callbacks
@app.callback(
    Output("time-series-chart", "figure"),
    Output("crime-dist-chart",  "figure"),
    Output("crime-mix-chart",   "figure"),
    Output("outcome-chart",     "figure"),
    Input("force-dropdown", "value"),
    State("theme-store", "data"),
)
def update_explorer(force, theme):
    p = palette(theme)
    layout = plotly_layout(theme)
    df_ts = make_time_series(force)

    # Time series
    ts_fig = px.line(df_ts, x="date", y="crime_count")

    ts_fig.update_xaxes(
    dtick="M12",          # one tick per year (use "M3" for quarterly, etc.)
    tickformat="%Y",      # label as "2017"  (or "%b %Y" -> "Feb 2017")
    )

    ts_fig.update_traces(line_color=p["accent"], hovertemplate="%{x|%b %Y}: %{y:,}")

    ts_fig.add_vrect(x0="2020-03-01", x1="2021-07-01",
                     fillcolor=p["covid_fill"],
                     layer="below", line_width=0,
                     annotation_text="COVID", annotation_position="top left",
                     annotation_font=dict(size=9, color=p["warning"]))

    # Crime distribution
    df = make_crime_distribution(force)

    dist_fig = px.bar(
        df, x="crime_count", y="crime_type", orientation="h",
        opacity=0.7, color_discrete_sequence=[p["accent"]],
    )
    dist_fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} crimes<extra></extra>")

    ts_fig.update_layout(**layout, height=280)

    dist_fig.update_layout(**layout, height=280)
    dist_fig.update_xaxes(title_text=None, showgrid=True, gridcolor=p["grid"])
    dist_fig.update_yaxes(title_text=None, tickfont=dict(size=9),
                          gridcolor="rgba(0,0,0,0)",
                          categoryorder="total ascending")  # largest bar at top

    # Crime mix over time (100% stacked area -> share per month)
    df_mix = make_crime_mix_over_time(force)
    mix_fig = px.area(
        df_mix, x="date", y="crime_count", color="crime_type",
        groupnorm="fraction",
        color_discrete_sequence=p["area_palette"],
    )
    mix_fig.update_traces(
        line=dict(width=0),
        hovertemplate="%{x|%b %Y}<br>%{fullData.name}: %{y:.1%}<extra></extra>",
    )
    mix_fig.update_layout(**layout, height=320)
    mix_fig.update_xaxes(title_text=None, dtick="M12", tickformat="%Y")
    mix_fig.update_yaxes(title_text=None, range=[0, 1], tickformat=".0%",
                         showgrid=False)
    mix_fig.update_layout(legend=dict(
        orientation="h", x=0, y=-0.18, font=dict(size=9, color=p["text_sec"]),
        bgcolor="rgba(0,0,0,0)", title_text=None,
    ))

    # case outcomes: share per outcome bucket
    df_out = make_outcome_breakdown(force)
    out_fig = px.bar(
        df_out, x="share", y="bucket", orientation="h",
        opacity=0.7, color_discrete_sequence=[p["accent"]],
    )
    out_fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:.1f}% of crimes<extra></extra>")
    out_fig.update_layout(**layout, height=320)
    out_fig.update_xaxes(title_text=None, ticksuffix="%", showgrid=True, gridcolor=p["grid"])
    out_fig.update_yaxes(title_text=None, tickfont=dict(size=9),
                         gridcolor="rgba(0,0,0,0)",
                         categoryorder="total ascending")  # largest share at top

    return ts_fig, dist_fig, mix_fig, out_fig

@app.callback(
    Output("sidebar-nav", "children"),
    Input("url", "pathname")
)
def update_nav(path):
    pages = [
        ("/", "Overview"),
        ("/brokerage-network", "Brokerage Network"),
        ("/forecast", "Forecast"),
        ("/allocation", "Resource Allocation"),
        ("/general-trends", "General Trends")
    ]
    return [
        dcc.Link(
            className="nav-item active" if path == href else "nav-item",
            href=href,
            children=[html.Div(className="nav-dot"), label]
        )
        for href, label in pages
    ]

@app.callback(
    Output("lsoa-details", "children"),
    Input("lsoa-map", "clickData")
)
def update_lsoa_details(clickData):
    if not clickData:
        return html.Div([
            html.Div("No ward selected", style={"fontWeight": "bold", "marginBottom": "10px"}),
            html.Div("Click a ward on the map to see its brokerage score and "
                     "next-month violence forecast."),
        ])

    ward_code = clickData["points"][0]["customdata"][0]
    if ward_code not in WARD_BY_CODE.index:
        return html.Div("No data for this ward.")

    ward = WARD_BY_CODE.loc[ward_code]
    ward_name = ward["ward_name"]
    score = float(ward["brokerage_score"])
    risk_level = ward["risk_level"]
    force = WARD_FORCE_BY_CODE.get(ward_code, "—")

    # Brokerage rank across all wards (brokerage_score is already a percentile).
    total = len(WARD_DF)
    rank = int((WARD_DF["brokerage_score"] > score).sum()) + 1
    top_pct = rank / total * 100

    def row(label, value, value_style=None):
        return html.Div(style={"marginBottom": "8px"}, children=[
            html.Span(f"{label}  ", style={"opacity": 0.6}),
            html.Span(value, style={**{"fontWeight": 600}, **(value_style or {})}),
        ])

    def section(title):
        return html.Div(title.upper(), style={
            "opacity": 0.5, "fontSize": "11px", "letterSpacing": "1.5px",
            "marginTop": "18px", "marginBottom": "8px",
        })

    children = [
        html.Div(ward_name, style={"fontSize": "18px", "fontWeight": "bold", "marginBottom": "4px"}),
        html.Div(f"{ward_code} · {force}", style={"opacity": 0.6, "fontSize": "12px"}),

        section("Brokerage risk"),
        row("Score:", f"{score:.0f} / 100"),
        row("Risk band:", risk_level),
        row("Rank:", f"#{rank:,} of {total:,} wards (top {top_pct:.0f}%)"),
    ]

    if ward_code in FORECAST_BY_WARD.index:
        fc = FORECAST_BY_WARD.loc[ward_code]
        change = float(fc["forecast_vso_change"])
        month_label = pd.Timestamp(fc["forecast_month"]).strftime("%b %Y")
        if change > 0:
            arrow, sign, colour = "↑", "+", "#e23b3b"   # more violence forecast
        elif change < 0:
            arrow, sign, colour = "↓", "", "#1f9d57"    # less violence forecast
        else:
            arrow, sign, colour = "→", "", None
        children += [
            section(f"Violence & sexual offences forecast · {month_label}"),
            row("Expected:", f"{float(fc['forecast_vso']):.0f} crimes"),
            row("Last month:", f"{float(fc['vso_last_month']):.0f} crimes"),
            row("Change:", f"{arrow} {sign}{change:.0f} vs last month",
                value_style={"color": colour} if colour else None),
        ]

    return html.Div(children)


@app.callback(
    Output("forecast-map", "figure"),
    Input("police-force-select", "value"),
    State("theme-store", "data"),
    prevent_initial_call=True
)
def update_forecast_map(police_force, theme):
    return forecast_per_force_map(police_force, theme=theme)

@app.callback(
    Output("allocation-map", "figure"),
    Output("allocation-leaderboard", "children"),
    Input("allocation-force-select", "value"),
    State("theme-store", "data"),
    prevent_initial_call=True
)
def update_allocation(police_force, theme):
    return allocation_map(police_force, theme=theme), allocation_leaderboard(police_force)

@app.callback(
    Output("network-graph", "figure"),
    Input("network-force-select", "value"),
    State("theme-store", "data"),
    prevent_initial_call=True
)
def update_network(force, theme):
    # Precomputed per force + theme; just swap the already-built figure.
    nets = BROKERAGE_NETWORKS.get(theme, BROKERAGE_NETWORKS["dark"])
    return nets.get(force, nets[BROKERAGE_FORCES[0]])


# ── Theme toggle ──────────────────────────────────────────────────────────────
@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(n_clicks, current):
    return "light" if (current or "dark") == "dark" else "dark"


@app.callback(
    Output("theme-toggle", "children"),
    Input("theme-store", "data"),
)
def theme_toggle_label(theme):
    # label shows the mode you'd switch to
    return "🌙  Dark mode" if theme == "light" else "☀  Light mode"


# mirror the theme onto <body> so the CSS restyles (incl. the portaled dropdown menus)
app.clientside_callback(
    """
    function(theme) {
        document.body.classList.toggle('light-theme', theme === 'light');
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-dummy", "children"),
    Input("theme-store", "data"),
)

if __name__ == "__main__":
    app.run(debug=True)