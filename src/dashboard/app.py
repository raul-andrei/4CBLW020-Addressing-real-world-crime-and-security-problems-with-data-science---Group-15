import dash
from dash import Dash, html, dcc, Input, Output
from duckdb import connect
import plotly.express as px
import plotly
import pandas as pd
import numpy as np
import json ## for the UK Map 
from pathlib import Path
from duckdb import connect

app = Dash(__name__, suppress_callback_exceptions=True)

BG = "#0a0d14"
BG_CARD = "#111520"
BORDER = "#1e2640"
ACCENT = "#4f7cff"
TEXT_PRI = "#e8eaf2"
TEXT_SEC = "#7a82a0"
TEXT_MUTE = "#3d4460"
WARNING = "#ffb84f"
SUCCESS = "#4fff9f"

## Making the UK Map (Loading fake data)
import random
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DASHBOARD_ASSETS = BASE_DIR / "src" / "dashboard" / "assets"
LSOA_PATH = BASE_DIR / "data" / "lsoa_boundaries.geojson"
WARD_PATH = BASE_DIR / "data" / "wards_dec2021_uk_bgc_4326.geojson"

with open(LSOA_PATH, "r", encoding="utf-8") as f:
    LSOA_GEOJSON = json.load(f)

with open(WARD_PATH, "r", encoding="utf-8") as f:
    WARD_GEOJSON = json.load(f)


BROKERAGE_NETWORK = plotly.io.read_json(DASHBOARD_ASSETS / "brokerage_network.json")
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

LSOA_DF = make_fake_lsoa_data(LSOA_GEOJSON)

# Per-ward brokerage snapshot, precomputed once by
# src/dashboard/artifacts.py (build_ward_snapshot) so the dashboard doesn't run
# the 49M-row pipeline at boot. Regenerate after a data refresh.
WARD_SNAPSHOT = DASHBOARD_ASSETS / "ward_snapshot.parquet"
if not WARD_SNAPSHOT.exists():
    raise FileNotFoundError(
        f"{WARD_SNAPSHOT} missing -- run: python -m src.dashboard.artifacts"
    )
WARD_DF = pd.read_parquet(WARD_SNAPSHOT)
## Making actual map - lsoa based map, keeping it here just for a moment,
## Delete later
def make_lsoa_map():
    fig = px.choropleth_map(
        LSOA_DF,
        geojson=LSOA_GEOJSON,
        locations="ward_code",
        featureidkey="properties.WD21CD",
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
        color_continuous_scale=[
            [0.0, "#f5f5f5"],
            [0.35, "#ffd166"],
            [0.65, "#ff8c42"],
            [1.0, "#ff4f4f"]
        ],
        map_style="carto-darkmatter",
        zoom=5.1,
        center={"lat": 54.5, "lon": -2.5},
        opacity=0.65
    )

    fig.update_traces(
        marker_line_width=0.3,
        marker_line_color=BORDER,
        customdata=np.stack([
            LSOA_DF["ward_code"],
            LSOA_DF["ward_name"],
            LSOA_DF["brokerage_score"],
            LSOA_DF["risk_level"],
            LSOA_DF["brokerage_crimes"],
            LSOA_DF["predicted_risk"],
            LSOA_DF["suggested_action"],
            LSOA_DF["recommended_units"]
        ], axis=-1)
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Brokerage score",
            thickness=10,
            tickfont=dict(color=TEXT_SEC)
        )
    )

    return fig

def make_ward_map():
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
        color_continuous_scale=[
            [0.0, "#f5f5f5"],
            [0.35, "#ffd166"],
            [0.65, "#ff8c42"],
            [1.0, "#ff4f4f"]
        ],
        map_style="carto-darkmatter",
        zoom=5.1,
        center={"lat": 54.5, "lon": -2.5},
        opacity=0.65
    )

    fig.update_traces(
        marker_line_width=0.3,
        marker_line_color=BORDER,
        customdata=np.stack([
            WARD_DF["ward_code"],
            WARD_DF["ward_name"],
            WARD_DF["brokerage_score"],
            WARD_DF["risk_level"],
            WARD_DF["brokerage_crimes"],
            WARD_DF["predicted_risk"],
            WARD_DF["suggested_action"],
            WARD_DF["recommended_units"]
        ], axis=-1)
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Brokerage score",
            thickness=10,
            tickfont=dict(color=TEXT_SEC)
        )
    )

    return fig

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color=TEXT_SEC, size=11),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(showgrid=False, zeroline=False, color=TEXT_MUTE,
               tickfont=dict(size=10), linecolor=BORDER),
    yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False,
               color=TEXT_MUTE, tickfont=dict(size=10), linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=TEXT_SEC)),
    hoverlabel=dict(bgcolor=BG_CARD, bordercolor=BORDER,
                    font=dict(color=TEXT_PRI, size=11)),
)

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


# Sidebar
def sidebar():
    return html.Div(className="sidebar", children=[
        html.Div(className="sidebar-logo", children=[
            "POLICE", html.Span("ANALYTICS")
        ]),
        html.Div(className="sidebar-label", children="Navigation"),
        html.Div(id="sidebar-nav"),
        html.Div(className="sidebar-footer", children=[
            html.Div("Group 15 · MD-CBL 2025–26"),
            html.Div("Data: data.police.uk"),
        ])
    ])

# Overview page
# Overview page
def overview_page():

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
                    html.Div("Sample data", className="card-badge warning"),
                ]),
                dcc.Graph(
                    id="lsoa-map",
                    figure=make_ward_map(),
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
                    style={
                        "padding": "16px",
                        "color": TEXT_PRI,
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
                    html.Div("Highest Risk wards", className="card-title"),
                    html.Div("Top areas ranked by brokerage-risk score", className="card-subtitle"),
                ]),
                html.Div("Sample data", className="card-badge warning"),
            ]),
            make_ward_leaderboard(10)
        ])
    ])
    


# Force Explorer page
def brokerage_network_page():
    return html.Div(className="network-graph-placeholder", children=[
        html.Div(
            children=[
                html.Div(
                    className="card",
                    children=[
                        html.Div(className="page-header", 
                                children=[
                                    html.Div("Brokerage Network", className="page-title"),
                                    html.Div("Size encodes crime volume, color encodes brokerage", className="page-subtitle"),
                                ]),  
                        dcc.Graph(
                        id = 'network-graph',
                        figure=BROKERAGE_NETWORK,
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
    # Calibrated so the UK-wide span (~10 deg) maps to zoom ~5.1; clamp + small padding.
    zoom = 8.4 - np.log2(span) if span > 0 else 9.0
    zoom = float(np.clip(zoom - 0.3, 4.0, 11.0))
    return center, zoom

def forecast_per_force_map(police_force: str = "Metropolitan Police Service"):
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
        color_continuous_midpoint=0,
        range_color=(-vso_abs_max, vso_abs_max),
        map_style="carto-darkmatter",
        zoom=zoom,
        center=center,
        opacity=0.65
    )
    fig.update_traces(
        marker_line_width=0.3,
        marker_line_color=BORDER,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        # Tie the UI-state revision to the force so switching forces is treated as
        # a new view and Plotly actually applies the computed center/zoom. (With a
        # constant uirevision, Plotly preserves the current camera on figure
        # replacement and ignores the new center/zoom -- so the map wouldn't recenter.)
        uirevision=police_force,
        coloraxis_colorbar=dict(
            title="Forecast change",
            thickness=10,
            tickfont=dict(color=TEXT_SEC)
        )
    )

    return fig


## Forecast page
def forecast_page():

    return html.Div([
        html.Div(className="card", children=[
            dcc.Graph(
                id="forecast-map",
                figure=forecast_per_force_map(),
                config={"displayModeBar": False},
                )
        ]),
        html.Div(className="card", children=[   
            dcc.Dropdown(
                className="policeforce-dropdown",
                id="police-force-select",
                options=[{"label": f, "value": f} for f in sorted(_merge_forcast_with_mapping(FORECASTS, WARD_FORCE_MAPPING)["police_force"].unique())],
                value="Metropolitan Police Service",
                clearable=False,
            )
        ])
    ],
    className="forecast-main") #empty for now

## General trends page 
def general_trends_page():
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Trends", className="page-tag"),
            html.Div("General Trends", className="page-title"),
            html.Div("Explore general crime trends by police force", className="page-subtitle"),
        ]),

        html.Div(className="force-selector", children=[
            html.Label("Select Police Force"),
            dcc.Dropdown(
                id="force-dropdown",
                options=[{"label": f, "value": f} for f in FORCES],
                value="All forces",
                clearable=False,
                style={
                    "background": BG_CARD,
                    "border": f"1px solid {BORDER}",
                    "color": TEXT_PRI,
                    "borderRadius": "8px"
                }
            ),
        ]),

        html.Div(className="chart-grid", children=[
            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Crime Over Time", className="card-title"),
                        html.Div("Monthly totals · shaded COVID period", className="card-subtitle"),
                    ]),
                    html.Div("Sample data", className="card-badge warning"),
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
                    html.Div("Sample data", className="card-badge warning"),
                ]),
                dcc.Graph(
                    id="crime-dist-chart",
                    config={"displayModeBar": False},
                    style={"height": "280px"}
                ),
            ]),
        ]),
    ])


## Leaderboard function (Areas with highest brokerage scores)
def make_lsoa_leaderboard(n=10):
    top_lsoas = LSOA_DF.sort_values("brokerage_score", ascending=False).head(n)

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
            html.Div(str(row["brokerage_score"]), className="broker-score"),
        ])
        for i, (_, row) in enumerate(top_lsoas.iterrows())
    ])
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
            html.Div(str(row["brokerage_score"]), className="broker-score"),
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
    sidebar(),
    html.Div(className="main-content", children=[
        # Spinner covers the server round-trip (build + transfer) for whichever
        # callback writes into page-content / the maps -- page navigation and
        # force changes both flow through here.
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

# Routing
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(path):
    if path == "/brokerage-network":
        return brokerage_network_page()
    elif path == "/forecast":
        return forecast_page()
    elif path == "/general-trends":
        return general_trends_page()
    elif path == "/allocation":
        return placeholder_page("Resource Allocation", "Allocation",
                                "Coming soon · awaiting forecast results")
    return overview_page()

# Force Explorer callbacks
@app.callback(
    Output("time-series-chart", "figure"),
    Output("crime-dist-chart",  "figure"),
    Input("force-dropdown", "value"),
)
def update_explorer(force):
    df_ts = make_time_series(force)

    # Time series
    ts_fig = px.line(df_ts, x="date", y="crime_count")

    ts_fig.update_xaxes(
    dtick="M12",          # one tick per year (use "M3" for quarterly, etc.)
    tickformat="%Y",      # label as "2017"  (or "%b %Y" -> "Feb 2017")
    )

    ts_fig.update_traces(hovertemplate="%{x|%b %Y}: %{y:,}")

    ts_fig.add_vrect(x0="2020-03-01", x1="2021-07-01",
                     fillcolor="rgba(255,184,79,0.06)",
                     layer="below", line_width=0,
                     annotation_text="COVID", annotation_position="top left",
                     annotation_font=dict(size=9, color=WARNING))

    # Crime distribution
    df = make_crime_distribution(force)

    dist_fig = px.bar(
        df, x="crime_count", y="crime_type", orientation="h",
        opacity=0.7, color_discrete_sequence=[ACCENT],
    )
    dist_fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} crimes<extra></extra>")

    ts_fig.update_layout(**PLOTLY_LAYOUT, height=280)

    dist_fig.update_layout(**PLOTLY_LAYOUT, height=280)
    dist_fig.update_xaxes(title_text=None, showgrid=True, gridcolor=BORDER)
    dist_fig.update_yaxes(title_text=None, tickfont=dict(size=9),
                          gridcolor="rgba(0,0,0,0)",
                          categoryorder="total ascending")  # largest bar at top

    return ts_fig, dist_fig

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
            html.Div("No ward selected", style={"fontWeight": "bold", "marginBottom": "10px"}),
            html.Div("Click an area on the map to view brokerage details.")
        ])

    point = clickData["points"][0]
    ward_code, ward_name, score, risk_level, crimes, predicted_risk, action, units = point["customdata"]

    return html.Div([
        html.Div(lsoa_name, style={"fontSize": "18px", "fontWeight": "bold", "marginBottom": "12px"}),
        html.Div(f"Ward code: {lsoa_code}", style={"marginBottom": "8px"}),
        html.Div(f"Brokerage score: {score}", style={"marginBottom": "8px"}),
        html.Div(f"Risk level: {risk_level}", style={"marginBottom": "8px"}),
        html.Div(f"Identified brokerage crimes: {crimes}", style={"marginBottom": "8px"}),
        html.Div(f"Predicted risk: {predicted_risk}", style={"marginBottom": "8px"}),
        html.Div(f"Suggested action: {action}", style={"marginBottom": "8px"}),
        html.Div(f"Recommended units: {units}", style={"marginBottom": "8px"}),
    ])


@app.callback(
    Output("forecast-map", "figure"),
    Input("police-force-select", "value"),
    prevent_initial_call=True
)
def update_forecast_map(police_force):
    return forecast_per_force_map(police_force)

if __name__ == "__main__":
    app.run(debug=True)