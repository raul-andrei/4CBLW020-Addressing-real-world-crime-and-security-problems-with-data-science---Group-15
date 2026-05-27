import dash
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

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

def make_heatmap():
    np.random.seed(42)
    months_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    crimes_shown = ["Anti-social behaviour","Bicycle theft","Burglary",
                     "Robbery","Shoplifting","Violence and sexual offences"]
    data = np.array([
        [0.70,0.75,0.86,0.96,1.00,0.97,0.99,0.96,0.84,0.85,0.75,0.67],
        [0.58,0.60,0.70,0.75,0.89,0.95,1.00,0.97,0.97,0.97,0.78,0.54],
        [0.97,0.94,0.98,0.90,0.91,0.90,0.91,0.93,0.92,1.00,1.00,0.94],
        [0.90,0.85,0.91,0.86,0.92,0.93,0.96,0.94,0.94,1.00,0.98,0.91],
        [0.92,0.91,0.98,0.93,0.96,0.93,0.96,0.99,0.95,1.00,0.96,0.84],
        [0.88,0.81,0.92,0.87,0.96,0.96,1.00,0.94,0.91,0.94,0.90,0.88],
    ])
    return crimes_shown, months_labels, data

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
def overview_page():
    crimes_shown, months_labels, heatmap_data = make_heatmap()

    heatmap_fig = go.Figure(go.Heatmap(
        z=heatmap_data, x=months_labels, y=crimes_shown,
        colorscale=[[0, "#1e2640"], [0.5, ACCENT], [1, "#ff4f4f"]],
        showscale=True,
        colorbar=dict(thickness=8, len=0.8, tickfont=dict(size=9, color=TEXT_SEC),
                      tickcolor=BORDER, outlinecolor="rgba(0,0,0,0)"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
    ))
    heatmap_fig.update_layout(**PLOTLY_LAYOUT, height=240)
    heatmap_fig.update_yaxes(tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)")
    heatmap_fig.update_xaxes(tickfont=dict(size=10))

    broker_bars = html.Div(className="broker-list", children=[
        html.Div(className="broker-item", children=[
            html.Div(f"#{i+1}", className="broker-rank"),
            html.Div(name, className="broker-name"),
            html.Div(className="broker-bar-wrap", children=[
                html.Div(className="broker-bar",
                         style={"width": f"{score/0.280*100:.0f}%"})
            ]),
            html.Div(f"{score:.3f}", className="broker-score"),
        ]) for i, (name, score) in enumerate(BROKERS)
    ])

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Overview", className="page-tag"),
            html.Div("UK Crime & Policing Dashboard", className="page-title"),
            html.Div("50M+ records · 44 forces · Feb 2017 – Jan 2026 · 14 crime categories",
                     className="page-subtitle"),
        ]),

        # Stat cards
        html.Div(className="stat-grid", children=[
            html.Div(className="stat-card", children=[
                html.Div("Crime Records", className="stat-label"),
                html.Div("50M+", className="stat-value"),
                html.Div("After cleaning & deduplication", className="stat-sub"),
            ]),
            html.Div(className="stat-card", children=[
                html.Div("Police Forces", className="stat-label"),
                html.Div("44", className="stat-value"),
                html.Div("England, Wales & BTP", className="stat-sub"),
            ]),
            html.Div(className="stat-card", children=[
                html.Div("Time Span", className="stat-label"),
                html.Div("9 yrs", className="stat-value"),
                html.Div("Feb 2017 – Jan 2026", className="stat-sub"),
            ]),
            html.Div(className="stat-card", children=[
                html.Div("Crime Categories", className="stat-label"),
                html.Div("14", className="stat-value"),
                html.Div("As defined by data.police.uk", className="stat-sub"),
            ]),
        ]),

        # Charts row
        html.Div(className="chart-grid-3", children=[
            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Seasonality Heatmap", className="card-title"),
                        html.Div("Normalised monthly activity · excl. COVID 2020–21",
                                 className="card-subtitle"),
                    ]),
                    html.Div("Real data", className="card-badge"),
                ]),
                dcc.Graph(figure=heatmap_fig, config={"displayModeBar": False},
                          style={"height": "240px"}),
            ]),

            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Top Brokerage Crimes", className="card-title"),
                        html.Div("Betweenness centrality scores", className="card-subtitle"),
                    ]),
                    html.Div("Network results", className="card-badge"),
                ]),
                broker_bars,
            ]),
        ]),
    ])

# Force Explorer page
def explorer_page():
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Explorer", className="page-tag"),
            html.Div("Force Explorer", className="page-title"),
            html.Div("Drill into crime trends by police force", className="page-subtitle"),
        ]),

        html.Div(className="force-selector", children=[
            html.Label("Select Police Force"),
            dcc.Dropdown(
                id="force-dropdown",
                options=[{"label": f, "value": f} for f in FORCES],
                value="London",
                clearable=False,
                style={"background": BG_CARD, "border": f"1px solid {BORDER}",
                       "color": TEXT_PRI, "borderRadius": "8px"}
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
                dcc.Graph(id="time-series-chart", config={"displayModeBar": False},
                          style={"height": "280px"}),
            ]),

            html.Div(className="card", children=[
                html.Div(className="card-header", children=[
                    html.Div([
                        html.Div("Crime Type Breakdown", className="card-title"),
                        html.Div("Distribution across categories", className="card-subtitle"),
                    ]),
                    html.Div("Sample data", className="card-badge warning"),
                ]),
                dcc.Graph(id="crime-dist-chart", config={"displayModeBar": False},
                          style={"height": "280px"}),
            ]),
        ]),
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
    if path == "/explorer":
        return explorer_page()
    elif path == "/brokers":
        return placeholder_page("Brokerage Analysis", "Network",
                                "Coming soon · awaiting network analysis output")
    elif path == "/forecast":
        return placeholder_page("Crime Forecast", "Forecast",
                                "Coming soon · Prophet models in progress")
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
        ("/explorer", "Force Explorer"),
        ("/brokers", "Brokerage Analysis"),
        ("/forecast", "Forecast"),
        ("/allocation", "Resource Allocation"),
    ]
    return [
        dcc.Link(
            className="nav-item active" if path == href else "nav-item",
            href=href,
            children=[html.Div(className="nav-dot"), label]
        )
        for href, label in pages
    ]

if __name__ == "__main__":
    app.run(debug=True)