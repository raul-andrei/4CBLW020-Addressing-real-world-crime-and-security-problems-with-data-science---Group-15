"""EDA over the full crimes.db dataset (~49M rows, England & Wales, 2017-2026).

Optimised for scale: every chart is driven by a DuckDB aggregation that runs
inside the database (columnar, multi-threaded, out-of-core) and returns only a
tiny result set (a few hundred rows at most) to pandas for plotting. The 49M-row
table is never materialised in Python -- the one larger pull is a SQL-side random
*sample* of points for the hexbin map.

crimes.db schema (cleaned/merged):
    lsoa_code, crime_type, year, month_num, longitude, latitude,
    last_outcome, reported_by
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SRC_ROOT = Path(__file__).resolve().parent.parent      # .../src
PROJECT_ROOT = SRC_ROOT.parent                          # repo root
DB_PATH = PROJECT_ROOT / "data" / "crimes.db"
OUTPUT_DIR = SRC_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def save_fig(name: str, dpi: int = 150) -> None:
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close()


# ── Data access ──────────────────────────────────────────────────────────────
def get_connection() -> duckdb.DuckDBPyConnection:
    """Open crimes.db read-only. DuckDB parallelises across all cores by default,
    so the GROUP BYs below scan the 49M rows once, in parallel, per query."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found")
    return duckdb.connect(str(DB_PATH), read_only=True)


# ── Null analysis ──────────────────────────────────────────────────────────────
def summarize_nulls(con):
    """Per-column null share over the whole table (single aggregate scan).

    crimes.db is the cleaned dataset, so in practice only `last_outcome` is ever
    null -- this is the trimmed-down equivalent of the raw-data null audit (the
    old Crime ID / Context / missingness-correlation charts don't apply, those
    columns aren't in crimes.db)."""
    cols = ["lsoa_code", "crime_type", "year", "month_num",
            "longitude", "latitude", "last_outcome", "reported_by"]
    sel = ", ".join(f"100.0 * (COUNT(*) - COUNT({c})) / COUNT(*) AS {c}" for c in cols)
    pct = con.execute(f"SELECT {sel} FROM crimes").df().iloc[0]
    print(f"Percentage of null values in each column:\n{pct}\n")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = sns.color_palette("viridis", len(pct))
    ax.bar(range(len(pct)), pct.values, color=colors)
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels(pct.index, rotation=45, ha="right")
    ax.set_title("Percentage of Null Values in Each Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("% Null")
    ax.set_ylim(0, max(50, pct.max() * 1.1))
    plt.tight_layout()
    save_fig("null_percentage")


# ── 1. Crime type counts ───────────────────────────────────────────────────────
def plot_crime_type_counts(con):
    counts = con.execute(
        "SELECT crime_type, COUNT(*) AS count "
        "FROM crimes GROUP BY crime_type ORDER BY count DESC"
    ).df()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(counts))
    ax.barh(counts["crime_type"], counts["count"], color=colors)
    ax.invert_yaxis()  # largest at top
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.set_title("Crime Type Counts (all forces, 2017-2026)")
    plt.tight_layout()
    save_fig("crime_type_counts")
    counts.to_csv(OUTPUT_DIR / "crime_type_counts.csv", index=False)


# ── 2. Monthly time series ─────────────────────────────────────────────────────
def plot_monthly_timeseries(con):
    monthly = con.execute(
        "SELECT make_date(year, month_num, 1) AS month_dt, COUNT(*) AS count "
        "FROM crimes GROUP BY 1 ORDER BY 1"
    ).df().set_index("month_dt")["count"]

    fig, ax = plt.subplots(figsize=(14, 5))
    monthly.plot(ax=ax, alpha=0.45, label="Monthly count")
    monthly.rolling(12, center=True).mean().plot(
        ax=ax, linewidth=2.5, color="crimson", label="12-month rolling avg")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.set_title("Monthly Crime Counts")
    ax.set_xlabel("")
    ax.set_ylabel("Crimes")
    ax.legend()
    plt.tight_layout()
    save_fig("monthly_timeseries")
    monthly.to_csv(OUTPUT_DIR / "monthly_timeseries.csv")


# ── 3. Outcome distribution by crime type (resolution rates) ──────────────────
def _bucket_outcomes(outcome: pd.Series) -> np.ndarray:
    """Collapse the 27 raw outcomes into 6 readable buckets (first match wins)."""
    is_null = outcome.isna()
    low = outcome.fillna("").str.lower()
    return np.select(
        [
            is_null,
            low.str.contains("investigation complete; no suspect identified"),
            low.str.contains("unable to prosecute|no further action"),
            low.str.contains(r"charged|imprisoned|crown court|prison|suspended sentence"
                             r"|awaiting court|found guilty|sent to crown"),
            low.str.contains(r"caution|penalty notice|community sentence|discharge"
                             r"|otherwise dealt with|local resolution"),
        ],
        [
            "No outcome recorded",
            "No suspect identified",
            "Unable to prosecute",
            "Charged / Sentenced",
            "Non-custodial outcome",
        ],
        default="Pending / Under investigation",
    )


def plot_outcome_by_crime_type(con):
    # 13 crime types x <=27 outcomes -> <=364 rows; bucket + crosstab in pandas.
    g = con.execute(
        "SELECT crime_type, last_outcome, COUNT(*) AS count "
        "FROM crimes GROUP BY 1, 2"
    ).df()
    g["bucket"] = _bucket_outcomes(g["last_outcome"])

    ct = g.pivot_table(index="crime_type", columns="bucket",
                       values="count", aggfunc="sum", fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    col_order = [
        "Charged / Sentenced", "Non-custodial outcome", "Unable to prosecute",
        "No suspect identified", "Pending / Under investigation", "No outcome recorded",
    ]
    ct_pct = ct_pct.reindex(columns=[c for c in col_order if c in ct_pct.columns])

    fig, ax = plt.subplots(figsize=(14, 7))
    ct_pct.plot(kind="barh", stacked=True, ax=ax, colormap="tab10", width=0.8)
    ax.set_xlabel("% of crimes")
    ax.set_ylabel("")
    ax.set_title("Outcome Distribution by Crime Type")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    save_fig("outcome_by_crime_type")
    ct_pct.to_csv(OUTPUT_DIR / "outcome_by_crime_type.csv")


# ── 4. Crime type × calendar month seasonality heatmap ────────────────────────
def plot_crime_type_month_heatmap(con):
    g = con.execute(
        "SELECT crime_type, month_num, COUNT(*) AS count "
        "FROM crimes GROUP BY 1, 2"
    ).df()
    pivot = (g.pivot_table(index="crime_type", columns="month_num",
                           values="count", fill_value=0)
              .reindex(columns=range(1, 13), fill_value=0))
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Normalize each row so we see within-type seasonal share, not volume
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot_norm, cmap="YlOrRd", ax=ax, fmt=".1f", annot=True,
                cbar_kws={"label": "% share within crime type"}, linewidths=0.5)
    ax.set_title("Crime Type Seasonality (% share by calendar month)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    save_fig("crime_seasonality_heatmap")
    pivot_norm.to_csv(OUTPUT_DIR / "crime_seasonality.csv")


# ── 5. Annual crime trend by type ──────────────────────────────────────────────
def plot_annual_trend_by_type(con):
    g = con.execute(
        "SELECT year, crime_type, COUNT(*) AS count "
        "FROM crimes GROUP BY 1, 2 ORDER BY 1"
    ).df()
    annual = g.pivot_table(index="year", columns="crime_type",
                           values="count", fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("tab20", len(annual.columns))
    annual.plot(ax=ax, marker="o", markersize=3, color=colors)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.set_title("Annual Crime Count by Type (2026 is a partial year)")
    ax.set_xlabel("")
    ax.set_ylabel("Crimes")
    ax.legend(loc="upper left", fontsize=7, ncols=2)
    plt.tight_layout()
    save_fig("annual_trend_by_type")
    annual.to_csv(OUTPUT_DIR / "annual_trend_by_type.csv")


# ── 6a. Geographic: top LSOAs by crime count ──────────────────────────────────
def plot_top_lsoas(con, n: int = 20):
    # crimes.db has no LSOA name column -> label by LSOA code.
    top = con.execute(
        "SELECT lsoa_code, COUNT(*) AS count "
        f"FROM crimes GROUP BY 1 ORDER BY count DESC LIMIT {n}"
    ).df()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(top))
    ax.barh(top["lsoa_code"], top["count"], color=colors)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.set_xlabel("Crime count")
    ax.set_ylabel("")
    ax.set_title(f"Top {n} LSOAs by Crime Count")
    plt.tight_layout()
    save_fig("top_lsoas")
    top.to_csv(OUTPUT_DIR / "top_lsoas.csv", index=False)


# ── 6b. Geographic: hexbin density map ────────────────────────────────────────
def plot_hexbin_density(con, sample_n: int = 500_000):
    # Sample points inside DuckDB (reservoir) instead of pulling 49M coords.
    geo = con.execute(
        "SELECT longitude, latitude FROM crimes "
        "WHERE longitude IS NOT NULL AND latitude IS NOT NULL "
        f"USING SAMPLE {sample_n} ROWS"
    ).df()

    fig, ax = plt.subplots(figsize=(10, 12))
    hb = ax.hexbin(
        geo["longitude"], geo["latitude"],
        gridsize=120, cmap="YlOrRd", mincnt=1, bins="log",
    )
    plt.colorbar(hb, ax=ax, label="log10(crime count)")
    ax.set_title(f"Crime Density Hexbin — England & Wales  (n = {len(geo):,} sampled points)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    save_fig("hexbin_density")


# ── 7. Crime type mix within top LSOAs ────────────────────────────────────────
def plot_crime_mix_top_lsoas(con, n: int = 10):
    g = con.execute(f"""
        WITH top AS (
            SELECT lsoa_code FROM crimes
            GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT {n}
        )
        SELECT lsoa_code, crime_type, COUNT(*) AS count
        FROM crimes
        WHERE lsoa_code IN (SELECT lsoa_code FROM top)
        GROUP BY 1, 2
    """).df()
    mix = g.pivot_table(index="lsoa_code", columns="crime_type",
                        values="count", fill_value=0)
    mix_pct = mix.div(mix.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    mix_pct.plot(kind="barh", stacked=True, ax=ax, colormap="tab20", width=0.8)
    ax.set_xlabel("% of crimes in LSOA")
    ax.set_ylabel("")
    ax.set_title(f"Crime Type Mix in Top {n} LSOAs")
    ax.legend(loc="lower right", fontsize=7, ncols=2)
    plt.tight_layout()
    save_fig("crime_mix_top_lsoas")
    mix_pct.to_csv(OUTPUT_DIR / "crime_mix_top_lsoas.csv")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    con = get_connection()
    n_rows = con.execute("SELECT COUNT(*) FROM crimes").fetchone()[0]
    print(f"crimes.db rows: {n_rows:,}\n")

    print("=== Null analysis ===")
    summarize_nulls(con)

    print("=== 1. Crime type counts ===")
    plot_crime_type_counts(con)

    print("=== 2. Monthly time series ===")
    plot_monthly_timeseries(con)

    print("=== 3. Outcome distribution by crime type ===")
    plot_outcome_by_crime_type(con)

    print("=== 4. Crime type × month seasonality heatmap ===")
    plot_crime_type_month_heatmap(con)

    print("=== 5. Annual trend by crime type ===")
    plot_annual_trend_by_type(con)

    print("=== 6a. Top LSOAs ===")
    plot_top_lsoas(con)

    print("=== 6b. Hexbin density map ===")
    plot_hexbin_density(con)

    print("=== 7. Crime mix in top LSOAs ===")
    plot_crime_mix_top_lsoas(con)

    con.close()
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
