import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent
FIGURES_DIR = Path(__file__).parent / "figures"

plt.rcParams.update({'font.size': 11, 'figure.dpi': 100})

BROKERAGE_MEASURES = [
    ('betweenness', False),
    ('current_flow_betweenness', False),
    ('constraint', True),
]

ALL_CENTRALITY_MEASURES = BROKERAGE_MEASURES + [('eigenvector', False)]

MEASURE_LABELS = {
    'betweenness': 'betweenness',
    'current_flow_betweenness': 'current flow\nbetweenness',
    'constraint': 'constraint',
    'eigenvector': 'eigenvector',
}


def add_rank_column(df, score, ascending=False):
    df = df.copy()
    df[f'{score}_rank'] = df.groupby('config_id')[score].rank(ascending=ascending, method='min')
    return df


def mean_rank_across_brokerage_measures(df):
    """Returns a Series indexed by crime_type with mean rank across the 3 brokerage measures."""
    rank_cols = []
    for m, asc in BROKERAGE_MEASURES:
        df = add_rank_column(df, m, ascending=asc)
        rank_cols.append(f'{m}_rank')
    return df.groupby('crime_type')[rank_cols].mean().mean(axis=1)

def mean_centrality_rank(df):
    """Mean rank across all 4 centrality measures (incl. eigenvector)."""
    rank_cols = []
    for m, asc in ALL_CENTRALITY_MEASURES:
        df = add_rank_column(df, m, ascending=asc)
        rank_cols.append(f'{m}_rank')
    return df.groupby('crime_type')[rank_cols].mean().mean(axis=1)


# ── Figure 1 ─────────────────────────────────────────────────────────────────

def make_figure_1(coocc_df, corr_df):
    coocc_df = coocc_df[coocc_df['presence_threshold'] == 3].copy()
    corr_df = corr_df[
        (corr_df['correlation_method'] == 'spearman') &
        (corr_df['normalisation'] == 'proportions')
    ].copy()

    mean_cooc_ranks = mean_rank_across_brokerage_measures(coocc_df)
    mean_corr_ranks = mean_rank_across_brokerage_measures(corr_df)

    n = 13
    cooc_score = (n + 1) - mean_cooc_ranks  # flip so high = central
    corr_score = (n + 1) - mean_corr_ranks

    sorted_crimes = cooc_score.sort_values(ascending=False).index.tolist()
    cooc_vals = cooc_score[sorted_crimes].values
    corr_vals = corr_score.reindex(sorted_crimes).values

    y = np.arange(len(sorted_crimes))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y + height / 2, cooc_vals, height,
            label='Co-occurrence (threshold=3)', color='steelblue')
    ax.barh(y - height / 2, corr_vals, height,
            label='Spearman + proportions', color='darkorange')
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_crimes)
    ax.invert_yaxis()  # most central at top
    ax.set_xlabel('Centrality score (higher = stronger broker)')
    ax.legend(loc='lower right')
    ax.set_title(
        'Top brokers across brokerage measures:\n'
        'lift-based co-occurrence vs Spearman+proportions correlation'
    )
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        path = FIGURES_DIR / f'fig1_top_brokers_comparison.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)



# ── Figure 2 ─────────────────────────────────────────────────────────────────

def make_figure_2(cooc_df, corr_df):
    cooc_t3 = cooc_df[cooc_df['presence_threshold'] == 3].copy()
    corr_sp = corr_df[
        (corr_df['correlation_method'] == 'spearman') &
        (corr_df['normalisation'] == 'proportions')
    ].copy()

    crimes = sorted(cooc_df['crime_type'].unique())
    heatmap_data = {}

    for m, asc in ALL_CENTRALITY_MEASURES:
        for label_prefix, df in [('Co-occurrence', cooc_t3), ('Correlation', corr_sp)]:
            col_label = f'{label_prefix}\n{MEASURE_LABELS[m]}'
            df2 = add_rank_column(df, m, ascending=asc)
            crime_config_counts = df2.groupby('crime_type')['config_id'].nunique()
            top3_counts = (
                df2[df2[f'{m}_rank'] <= 3]
                .groupby('crime_type')['config_id'].nunique()
            )
            pct = (
                (top3_counts / crime_config_counts * 100)
                .reindex(crimes, fill_value=0)
                .fillna(0)
                .round(0)
                .astype(int)
            )
            heatmap_data[col_label] = pct

    matrix = pd.DataFrame(heatmap_data, index=crimes)
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        matrix, ax=ax, cmap='YlOrRd', annot=True, fmt='d',
        cbar_kws={'label': 'Top-3 frequency (%)'}, vmin=0, vmax=100
    )
    ax.set_title('Top-3 broker frequency by centrality measure and method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    for ext in ('png', 'pdf'):
        path = FIGURES_DIR / f'fig2_per_measure_heatmap.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


# ── Figure 3 ─────────────────────────────────────────────────────────────────

def make_figure_3(cooc_df, corr_df):
    headline_crimes = ['Robbery', 'Theft from the person', 'Possession of weapons']
    
    df = cooc_df.copy()
    df['covid_included'] = df['covid_included'].astype(str)
    df['ASB_included'] = df['ASB_included'].astype(str)
    
    for m, asc in BROKERAGE_MEASURES:
        df = add_rank_column(df, m, ascending=asc)
    rank_cols = [f'{m}_rank' for m, _ in BROKERAGE_MEASURES]
    df['mean_rank'] = df[rank_cols].mean(axis=1)
    
    sensitivity_axes = [
        ('presence_threshold', [2, 3, 5], 'Presence threshold'),
        ('time_frame', sorted(cooc_df['time_frame'].unique()), 'Time window'),
        ('covid_included', ['True', 'False'], 'COVID years'),
        ('ASB_included', ['True', 'False'], 'ASB included'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    
    for ax, crime in zip(axes, headline_crimes):
        crime_df = df[df['crime_type'] == crime]
        overall = crime_df['mean_rank'].mean()
        
        rows = []
        for axis_name, levels, axis_label in sensitivity_axes:
            for level in levels:
                level_mean = crime_df[crime_df[axis_name].astype(str) == str(level)]['mean_rank'].mean()
                rows.append((f'{axis_label}={level}', level_mean))
        
        labels, values = zip(*rows)
        y = np.arange(len(labels))
        ax.barh(y, values, color='steelblue', alpha=0.8)
        ax.axvline(overall, color='red', linestyle='--', linewidth=1.5, label=f'Overall mean ({overall:.2f})')
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mean rank (lower = stronger broker)')
        ax.set_title(crime, fontsize=11)
        ax.legend(loc='lower right', fontsize=8)
    
    fig.suptitle('Sensitivity of headline-broker rankings across configuration axes', fontsize=13)
    plt.tight_layout()
    
    for ext in ('png', 'pdf'):
        path = FIGURES_DIR / f'fig3_broker_sensitivity.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


# ── Figure 4 ─────────────────────────────────────────────────────────────────

def make_figure_4(cooc_df, corr_df):
    crimes = sorted(cooc_df['crime_type'].unique())
    n = len(crimes)
    configs = cooc_df['config_id'].unique()
    n_configs = len(configs)

    count_matrix = np.zeros((n, n))

    for config_id in configs:
        config_df = cooc_df[cooc_df['config_id'] == config_id]
        community_map = dict(zip(config_df['crime_type'], config_df['community_id']))
        crimes_in_config = set(community_map.keys())
        for i, c1 in enumerate(crimes):
            for j, c2 in enumerate(crimes):
                ci = community_map.get(c1)
                cj = community_map.get(c2)
                if c1 in crimes_in_config and c2 in crimes_in_config:
                    if ci is not None and cj is not None and not pd.isna(ci) and not pd.isna(cj):
                        if ci == cj:
                            count_matrix[i, j] += 1

    co_membership = count_matrix / n_configs
    np.fill_diagonal(co_membership, 1.0)

    co_df = pd.DataFrame(co_membership, index=crimes, columns=crimes)

    g = sns.clustermap(
        co_df, cmap='Blues', annot=True, fmt='.2f',
        method='average', figsize=(10, 10),
        vmin=0, vmax=1, annot_kws={'size': 8}
    )
    g.fig.suptitle(
        'Crime community co-membership fraction across co-occurrence configurations',
        y=1.02, fontsize=12
    )

    for ext in ('png', 'pdf'):
        path = FIGURES_DIR / f'fig4_community_co_membership.{ext}'
        g.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(g.fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cooc_df = pd.read_csv(DATA_DIR / 'sensitivity_results_per_crime_cooccurrence.csv')
    corr_df = pd.read_csv(DATA_DIR / 'sensitivity_results_per_crime_correlation.csv')
    print(f"Co-occurrence: {cooc_df.shape}, Correlation: {corr_df.shape}")

    print("\nGenerating Figure 1...")
    make_figure_1(cooc_df, corr_df)

    print("\nGenerating Figure 2...")
    make_figure_2(cooc_df, corr_df)

    print("\nGenerating Figure 3...")
    make_figure_3(cooc_df, corr_df)

    print("\nGenerating Figure 4...")
    make_figure_4(cooc_df, corr_df)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()