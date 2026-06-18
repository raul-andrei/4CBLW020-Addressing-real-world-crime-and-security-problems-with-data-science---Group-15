import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from src.network_analysis.preparation import build_cooccurrence_network, connect
from src.network_analysis.analysis import brokerage_analysis
from src.network_analysis.sensitivity_analysis import create_graph, knn_backbone, to_distance_graph
from pathlib import Path
import pandas as pd


MEASURES = [
    ('betweenness', False),
    ('current_flow_betweenness', False),
    ('constraint', True),
]


def __add_rank_column(df, score, ascending=False):
    df = df.copy()
    df[f'{score}_rank'] = df[score].rank(ascending=ascending, method='min')
    return df


def __mean_rank_across_measures(df):
    rank_cols = []
    for m, asc in MEASURES:
        df = __add_rank_column(df, m, ascending=asc)
        rank_cols.append(f'{m}_rank')
    return df.set_index('crime_type')[rank_cols].mean(axis=1)


def build_brokerage_graph(where_sql=None, con=None):
    """Build the crime co-occurrence brokerage graph + per-node metrics + layout.

    Heavy: pulls from the DB and runs the brokerage analysis. This is the single
    source of truth for the graph construction — both the matplotlib presentation
    figure below and the dashboard's plotly artifact (src/dashboard/artifacts.py)
    reuse it, so the network params (kNN k, presence threshold, lift) live in one
    place.

    Parameters
    ----------
    where_sql : str, optional
        SQL filter passed through to build_cooccurrence_network to restrict the
        graph to a slice (e.g. "reported_by = 'Metropolitan Police Service'" for a
        single force). None -> all data (the global network).
    con : duckdb connection, optional
        Reuse an already-open connection so callers building many force graphs
        don't reopen the DB each time. If None, a read-only connection is opened
        and closed here.

    Returns
    -------
    G_sim : nx.Graph          kNN-backbone similarity graph, self-loops removed (edges carry 'weight')
    metrics_pd : pd.DataFrame per-crime brokerage metrics, indexed by crime_type
    mean_ranks : pd.Series    mean broker rank across MEASURES, indexed by crime_type
    pos : dict                kamada-kawai layout positions {crime_type: (x, y)}
    """
    own_conn = con is None
    if own_conn:
        con = connect()

    cooccurrence_matrix = build_cooccurrence_network(con=con, where_sql=where_sql)
    G_sim = knn_backbone(create_graph(cooccurrence_matrix, matrix='cooccurrence'), k=3)
    G_distance = to_distance_graph(G_sim)
    result = brokerage_analysis(G_sim, G_distance)
    if own_conn:
        con.close()

    G_sim.remove_edges_from(nx.selfloop_edges(G_sim))

    # Build metrics DataFrame
    metric_results = [
        {"crime_type": crime_type, **node_metrics}
        for crime_type, node_metrics in result.items()
    ]
    metrics_pd = pd.DataFrame(metric_results)
    mean_ranks = __mean_rank_across_measures(metrics_pd)
    metrics_pd = metrics_pd.set_index('crime_type')

    pos = nx.kamada_kawai_layout(G_sim, weight='weight')
    return G_sim, metrics_pd, mean_ranks, pos


def visualize_graph(graph_id: int = 0):
    G_sim, metrics_pd, mean_ranks, pos = build_brokerage_graph()

    if G_sim.number_of_edges() == 0:
        print("Graph has no edges, skipping visualization.")
        return

    # Node sizes — bigger = stronger broker (low mean rank)
    n_crimes = len(G_sim.nodes())
    node_sizes = [3000 * (n_crimes + 1 - mean_ranks[n]) / n_crimes for n in G_sim.nodes()]

    # Node colours — two communities, blue vs orange
    color_map = {0: '#1f77b4', 1: '#ff7f0e'}
    node_colors = []
    for n in G_sim.nodes():
        cid = metrics_pd.loc[n, 'community_id']
        node_colors.append(color_map.get(int(cid) if not pd.isna(cid) else -1, 'gray'))

    # Edge widths — boosted to emphasise differences (power < 1 amplifies weak edges)
    edge_weights = [G_sim[u][v]['weight'] for u, v in G_sim.edges()]
    max_weight = max(edge_weights)
    edge_widths = [6 * (w / max_weight) ** 0.7 for w in edge_weights]
    edge_alphas = [0.3 + 0.5 * (w / max_weight) for w in edge_weights]

    # Label positions — offset slightly below nodes for readability
    label_pos = {n: (x, y - 0.08) for n, (x, y) in pos.items()}

    # Draw
    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw edges individually so each can have its own alpha
    for (u, v), w, a in zip(G_sim.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G_sim, pos, edgelist=[(u, v)], width=w, alpha=a,
            edge_color='gray', ax=ax
        )

    nx.draw_networkx_nodes(
        G_sim, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='black',
        linewidths=1.5,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G_sim, label_pos,
        font_size=10,
        font_weight='bold',
        ax=ax,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color='#1f77b4', label='Interpersonal and public-disorder crime cluster'),
        mpatches.Patch(color='#ff7f0e', label='Acquisitive and property crime cluster'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=10, framealpha=0.9)

    ax.set_axis_off()
    ax.set_title(
        'Crime co-occurrence network (lift, presence threshold=3, kNN k=3)\n'
        'Node size = mean broker rank across three brokerage measures',
        fontsize=12, pad=12,
    )

    plt.tight_layout()
    #plt.savefig('network.pdf', bbox_inches='tight')
    plt.savefig(f'network{graph_id}.png', bbox_inches='tight', dpi=300)
    print(f"Saved network{graph_id}.png")
    plt.close(fig)


if __name__ == "__main__":
    visualize_graph(graph_id=2)