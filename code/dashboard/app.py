import dash
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json ## for the UK Map 
from pathlib import Path

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
BASE_DIR = Path(__file__).resolve().parent
LSOA_PATH = BASE_DIR / "data" / "lsoa_boundaries.geojson"

with open(LSOA_PATH, "r", encoding="utf-8") as f:
    LSOA_GEOJSON = json.load(f)

def extract_lsoa_base(geojson):
    rows = []

    for feature in geojson["features"]:
        props = feature["properties"]

        # adjust these names if your GeoJSON uses different property names
        code = props.get("LSOA21CD") or props.get("LSOA11CD")
        name = props.get("LSOA21NM") or props.get("LSOA11NM") or code

        if code:
            rows.append({
                "lsoa_code": code,
                "lsoa_name": name
            })

    return pd.DataFrame(rows)

def make_fake_lsoa_data(geojson):
    df = extract_lsoa_base(geojson)

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

## Making actual map 
def make_lsoa_map():
    fig = px.choropleth_mapbox(
        LSOA_DF,
        geojson=LSOA_GEOJSON,
        locations="lsoa_code",
        featureidkey="properties.LSOA21CD",   # change if needed
        color="brokerage_score",
        hover_name="lsoa_name",
        hover_data={
            "lsoa_code": True,
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
        mapbox_style="carto-darkmatter",
        zoom=5.1,
        center={"lat": 54.5, "lon": -2.5},
        opacity=0.65
    )

    fig.update_traces(
        marker_line_width=0.3,
        marker_line_color=BORDER,
        customdata=np.stack([
            LSOA_DF["lsoa_code"],
            LSOA_DF["lsoa_name"],
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

FORCES = [
    "London", "Greater Manchester", "West Midlands", "West Yorkshire",
    "Thames Valley", "Hampshire", "Merseyside", "South Yorkshire",
    "Avon and Somerset", "Kent", "Essex", "Hertfordshire"
]

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
    np.random.seed(hash(force) % 999)
    months = pd.date_range("2017-02", "2026-01", freq="MS")
    base = np.random.randint(3000, 12000)
    trend = np.linspace(0, base * 0.3, len(months))
    season = np.sin(np.linspace(0, len(months) * 2 * np.pi / 12, len(months))) * base * 0.12
    noise = np.random.normal(0, base * 0.04, len(months))
    covid = np.where((months >= "2020-03") & (months <= "2021-06"), -base * 0.22, 0)
    values = base + trend + season + noise + covid
    return months, np.clip(values, 0, None).astype(int)

def make_crime_distribution(force):
    np.random.seed((hash(force) + 1) % 999)
    weights = np.random.dirichlet(np.ones(len(CRIME_TYPES)) * 2)
    counts = (weights * 500_000).astype(int)
    return pd.DataFrame({"crime": CRIME_TYPES, "count": counts}).sort_values("count", ascending=True)


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
                        html.Div("UK LSOAs coloured by brokerage-risk score", className="card-subtitle"),
                    ]),
                    html.Div("Sample data", className="card-badge warning"),
                ]),
                dcc.Graph(
                    id="lsoa-map",
                    figure=make_lsoa_map(),
                    config={"displayModeBar": False},
                    style={"height": "520px"}
                ),
            ]),

            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Selected LSOA", className="card-title"),
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
                    children="Click an LSOA to view brokerage details."
                )
            ]),
        ]),

        # Leaderboard row
        html.Div(className="card", style={"marginTop": "24px"}, children=[
            html.Div(className="card-header", children=[
                html.Div([
                    html.Div("Highest Risk LSOAs", className="card-title"),
                    html.Div("Top areas ranked by brokerage-risk score", className="card-subtitle"),
                ]),
                html.Div("Sample data", className="card-badge warning"),
            ]),
            make_lsoa_leaderboard(10)
        ])
    ])

# Force Explorer page
def brokerage_network_page():
    return placeholder_page(
        "Brokerage Network",
        "Network",
        "Content TBD · network analysis output"
    )

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
                value="London",
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
            html.Div(row["lsoa_name"], className="broker-name"),
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
    html.Div(className="main-content", id="page-content"),
])

# Routing
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(path):
    if path == "/brokerage-network":
        return brokerage_network_page()
    elif path == "/forecast":
        return placeholder_page("Crime Forecast", "Forecast",
                                "Coming soon · Prophet models in progress")
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
    months, values = make_time_series(force)

    # Time series
    ts_fig = go.Figure()
    ts_fig.add_vrect(x0="2020-03-01", x1="2021-07-01",
                     fillcolor="rgba(255,184,79,0.06)",
                     layer="below", line_width=0,
                     annotation_text="COVID", annotation_position="top left",
                     annotation_font=dict(size=9, color=WARNING))
    ts_fig.add_trace(go.Scatter(
        x=months, y=values, mode="lines",
        line=dict(color=ACCENT, width=1.5),
        fill="tozeroy", fillcolor="rgba(79,124,255,0.06)",
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} crimes<extra></extra>",
    ))

    # Crime distribution
    df = make_crime_distribution(force)
    dist_fig = go.Figure(go.Bar(
        x=df["count"], y=df["crime"], orientation="h",
        marker=dict(color=ACCENT, opacity=0.7,
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        hovertemplate="<b>%{y}</b><br>%{x:,} crimes<extra></extra>",
    ))
    ts_fig.update_layout(**PLOTLY_LAYOUT, height=280)

    dist_fig.update_layout(**PLOTLY_LAYOUT, height=280)
    dist_fig.update_xaxes(showgrid=True, gridcolor=BORDER)
    dist_fig.update_yaxes(tickfont=dict(size=9), gridcolor="rgba(0,0,0,0)")

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
            html.Div("No LSOA selected", style={"fontWeight": "bold", "marginBottom": "10px"}),
            html.Div("Click an area on the map to view brokerage details.")
        ])

    point = clickData["points"][0]
    lsoa_code, lsoa_name, score, risk_level, crimes, predicted_risk, action, units = point["customdata"]

    return html.Div([
        html.Div(lsoa_name, style={"fontSize": "18px", "fontWeight": "bold", "marginBottom": "12px"}),
        html.Div(f"LSOA code: {lsoa_code}", style={"marginBottom": "8px"}),
        html.Div(f"Brokerage score: {score}", style={"marginBottom": "8px"}),
        html.Div(f"Risk level: {risk_level}", style={"marginBottom": "8px"}),
        html.Div(f"Identified brokerage crimes: {crimes}", style={"marginBottom": "8px"}),
        html.Div(f"Predicted risk: {predicted_risk}", style={"marginBottom": "8px"}),
        html.Div(f"Suggested action: {action}", style={"marginBottom": "8px"}),
        html.Div(f"Recommended units: {units}", style={"marginBottom": "8px"}),
    ])

if __name__ == "__main__":
    app.run(debug=True)