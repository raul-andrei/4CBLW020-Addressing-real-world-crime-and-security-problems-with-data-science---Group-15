import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def save_fig(name: str, dpi: int = 150) -> None:
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close()


# ── Data loading ───────────────────────────────────────────────────────────────
path = ROOT / "data" / "london_crime_full.parquet"
df = pd.read_parquet(path)

cat_cols = ["Crime type", "Reported by", "Falls within", "Last outcome category", "LSOA src", "LSOA name"]
for col in cat_cols:
    df[col] = df[col].astype("category")

# Context is 100% null — drop at load time so all functions see a clean df
df = df.drop(columns=["Context"])

# Derived time columns used throughout
df["month_dt"] = pd.to_datetime(df["Month"], format="%Y-%m")
df["year"] = df["month_dt"].dt.year.astype("int16")
df["month_num"] = df["month_dt"].dt.month.astype("int8")


# ── Null analysis ──────────────────────────────────────────────────────────────
def handle_and_explore_nulls(df):
    missing_df = df.isna()
    nulls = missing_df.sum()
    print(f"Null values in each column:\n{nulls}\n")

    nulls_percentage = (nulls / len(df)) * 100
    print(f"Percentage of null values in each column:\n{nulls_percentage}\n")

    # Per-column null % bar chart
    null_pct_df = nulls_percentage.rename("pct_null").reset_index()
    null_pct_df.columns = ["column", "pct_null"]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = sns.color_palette("viridis", len(null_pct_df))
    ax.bar(range(len(null_pct_df)), null_pct_df["pct_null"], color=colors)
    ax.set_xticks(range(len(null_pct_df)))
    ax.set_xticklabels(null_pct_df["column"], rotation=45, ha="right")
    ax.set_title("Percentage of Null Values in Each Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("% Null")
    ax.set_ylim(0, 50)
    plt.tight_layout()
    save_fig("null_percentage")

    # Missingness correlation (only columns with partial nulls)
    varying = missing_df.columns[(missing_df.sum() > 0) & (missing_df.sum() < len(df))]
    corr = missing_df[varying].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={"label": "Correlation"}, ax=ax)
    ax.set_title("Correlation of Missingness Between Columns")
    plt.tight_layout()
    save_fig("null_correlation_heatmap")

    # Null % over time
    month_period = df["month_dt"].dt.to_period("M")
    nulls_by_month = df.isna().groupby(month_period).sum()
    nulls_by_month = nulls_by_month.loc[:, nulls_by_month.sum() > 0]
    counts_by_month = df.groupby(month_period).size()
    null_pct = nulls_by_month.div(counts_by_month, axis=0) * 100
    null_pct.index = null_pct.index.astype(str)
    fig, ax = plt.subplots(figsize=(14, 5))
    null_pct.plot(ax=ax)
    step = 12
    ax.set_xticks(range(0, len(null_pct), step))
    ax.set_xticklabels(null_pct.index[::step], rotation=45, ha="right")
    ax.set_ylabel("% missing")
    ax.set_title("Missing data % over time")
    plt.tight_layout()
    save_fig("null_over_time")

    # Verify: null Crime IDs should only be anti-social behaviour
    null_crimeID_counts = (
        df[df["Crime ID"].isna()]["Crime type"]
        .value_counts()
        .reset_index()
    )
    null_crimeID_counts.columns = ["crime_type", "count"]
    # Categorical value_counts includes all categories (even with count=0) — drop them
    null_crimeID_counts = null_crimeID_counts[null_crimeID_counts["count"] > 0]
    print(f"Crime types with null Crime ID:\n{null_crimeID_counts}\n")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(null_crimeID_counts))
    ax.barh(null_crimeID_counts["crime_type"], null_crimeID_counts["count"], color=colors)
    ax.set_xlabel("Count of records with null Crime ID")
    ax.set_ylabel("")
    ax.set_title("Crime Types with Null Crime ID")
    plt.tight_layout()
    save_fig("null_crime_id_by_type")

    # Location attribute missingness breakdown
    loc_cols = ["Longitude", "Latitude", "LSOA src", "LSOA name"]
    missing_locations = df[loc_cols].isna()
    print(f"Missing values in location columns:\n{missing_locations.sum()}\n")
    rows_any_missing = df[missing_locations.any(axis=1)]
    print(
        f"Rows with missing location data by crime type:\n"
        f"{rows_any_missing.groupby('Crime type', observed=True).size()}\n"
    )
    print(
        f"Rows with missing location data by outcome:\n"
        f"{rows_any_missing.groupby('Last outcome category', observed=True).size()}\n"
    )


def handle_and_explore_duplicates(df):
    duplicate_crimes = df["Crime ID"].duplicated()
    print(f"Number of duplicate Crime IDs: {duplicate_crimes.sum()}")
    print(f"Duplicate crime IDs: {df[duplicate_crimes]['Crime ID'].unique()}")


# ── 1. Crime type counts ───────────────────────────────────────────────────────
def plot_crime_type_counts(df):
    counts = df["Crime type"].value_counts().reset_index()
    counts.columns = ["crime_type", "count"]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(counts))
    ax.barh(counts["crime_type"], counts["count"], color=colors)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.set_title("Crime Type Counts (all years)")
    plt.tight_layout()
    save_fig("crime_type_counts")
    counts.to_csv(OUTPUT_DIR / "crime_type_counts.csv", index=False)


# ── 2. Monthly time series ─────────────────────────────────────────────────────
def plot_monthly_timeseries(df):
    monthly = df.groupby("month_dt").size().rename("count")

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
def plot_outcome_by_crime_type(df):
    is_null = df["Last outcome category"].isna()
    outcome_str = df["Last outcome category"].astype(str)

    # Vectorized bucketing — order of conditions matters (first match wins)
    outcome_group = np.select(
        [
            is_null,
            outcome_str.str.contains("Investigation complete; no suspect identified"),
            outcome_str.str.contains("Unable to prosecute|no further action", case=False),
            outcome_str.str.contains(
                r"charged|imprisoned|Crown Court|prison|suspended sentence"
                r"|awaiting court|found guilty|sent to Crown", case=False),
            outcome_str.str.contains(
                r"caution|penalty notice|community sentence|discharge"
                r"|otherwise dealt with|local resolution", case=False),
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

    ct = pd.crosstab(df["Crime type"], pd.Series(outcome_group, name="Outcome group"))
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
def plot_crime_type_month_heatmap(df):
    pivot = (
        df.groupby(["Crime type", "month_num"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
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
def plot_annual_trend_by_type(df):
    annual = (
        df.groupby(["year", "Crime type"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("tab20", len(annual.columns))
    annual.plot(ax=ax, marker="o", markersize=3, color=colors)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.set_title("Annual Crime Count by Type")
    ax.set_xlabel("")
    ax.set_ylabel("Crimes")
    ax.legend(loc="upper left", fontsize=7, ncols=2)
    plt.tight_layout()
    save_fig("annual_trend_by_type")
    annual.to_csv(OUTPUT_DIR / "annual_trend_by_type.csv")


# ── 6a. Geographic: top LSOAs by crime count ──────────────────────────────────
def plot_top_lsoas(df, n: int = 20):
    top = df["LSOA name"].value_counts().head(n).reset_index()
    top.columns = ["lsoa_name", "count"]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(top))
    ax.barh(top["lsoa_name"], top["count"], color=colors)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.set_xlabel("Crime count")
    ax.set_ylabel("")
    ax.set_title(f"Top {n} LSOAs by Crime Count")
    plt.tight_layout()
    save_fig("top_lsoas")
    top.to_csv(OUTPUT_DIR / "top_lsoas.csv", index=False)


# ── 6b. Geographic: hexbin density map ────────────────────────────────────────
def plot_hexbin_density(df, sample_n: int = 500_000):
    geo = df[["Longitude", "Latitude"]].dropna()
    if len(geo) > sample_n:
        geo = geo.sample(sample_n, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 12))
    hb = ax.hexbin(
        geo["Longitude"], geo["Latitude"],
        gridsize=120, cmap="YlOrRd", mincnt=1, bins="log",
    )
    plt.colorbar(hb, ax=ax, label="log₁₀(crime count)")
    ax.set_title(f"Crime Density Hexbin  (n = {len(geo):,} sampled points)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    save_fig("hexbin_density")


# ── 7. Crime type mix within top LSOAs ────────────────────────────────────────
def plot_crime_mix_top_lsoas(df, n: int = 10):
    top_lsoas = df["LSOA name"].value_counts().head(n).index
    sub = df[df["LSOA name"].isin(top_lsoas)]
    mix = pd.crosstab(sub["LSOA name"], sub["Crime type"])
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
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB\n")

    print("=== Null analysis ===")
    handle_and_explore_nulls(df)

    print("=== 1. Crime type counts ===")
    plot_crime_type_counts(df)

    print("=== 2. Monthly time series ===")
    plot_monthly_timeseries(df)

    print("=== 3. Outcome distribution by crime type ===")
    plot_outcome_by_crime_type(df)

    print("=== 4. Crime type × month seasonality heatmap ===")
    plot_crime_type_month_heatmap(df)

    print("=== 5. Annual trend by crime type ===")
    plot_annual_trend_by_type(df)

    print("=== 6a. Top LSOAs ===")
    plot_top_lsoas(df)

    print("=== 6b. Hexbin density map ===")
    plot_hexbin_density(df)

    print("=== 7. Crime mix in top LSOAs ===")
    plot_crime_mix_top_lsoas(df)

    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
