from code.network_analysis.preparation import connect, get_global_corr_matrix
import networkx as nx
import pandas as pd
from pathlib import Path
import numpy as np
import igraph as ig
import leidenalg

ROOT = Path(__file__).parent.parent.parent


def create_graph(corr_matrix: pd.DataFrame, threshold: float) -> nx.Graph:
    arr = corr_matrix.to_numpy(copy=True)
    np.fill_diagonal(arr, 0)
    W = pd.DataFrame(arr, index=corr_matrix.index, columns=corr_matrix.columns)
    W = W.where(W > threshold, 0)
    return nx.from_pandas_adjacency(W)


def brokerage_analysis(G_similarity: nx.Graph, G_distance: nx.Graph) -> dict:
    """
    Compute per-node brokerage metrics for every node in G.

    Returns dict mapping node_name -> {metric: value}.
    Metrics: betweenness, current_flow_betweenness, constraint,
             eigenvector, community_id, degree.
    """
    nodes = list(G_similarity.nodes())
    result = {n: {} for n in nodes}

    # Betweenness centrality
    try:
        bc = nx.betweenness_centrality(G_distance, weight='weight')
        for n, v in bc.items():
            result[n]['betweenness'] = v
    except Exception as e:
        print(f"[brokerage_analysis] betweenness failed: {e}")
        for n in nodes:
            result[n]['betweenness'] = np.nan

    # Current-flow betweenness — requires connected graph; computed on LCC only
    components = list(nx.connected_components(G_distance))
    lcc_nodes = max(components, key=len) if components else set()
    lcc = G_distance.subgraph(lcc_nodes).copy()
    try:
        cfbc = nx.current_flow_betweenness_centrality(lcc, weight='weight')
        for n in nodes:
            result[n]['current_flow_betweenness'] = cfbc.get(n, np.nan)
    except Exception as e:
        print(f"[brokerage_analysis] current_flow_betweenness failed: {e}")
        for n in nodes:
            result[n]['current_flow_betweenness'] = np.nan

    # Burt's constraint
    try:
        constraint = nx.constraint(G_similarity, weight='weight')
        for n, v in constraint.items():
            result[n]['constraint'] = v if v is not None else np.nan
    except Exception as e:
        print(f"[brokerage_analysis] constraint failed: {e}")
        for n in nodes:
            result[n]['constraint'] = np.nan

    # Eigenvector centrality
    try:
        ec = nx.eigenvector_centrality(G_similarity, weight='weight', max_iter=1000)
        for n, v in ec.items():
            result[n]['eigenvector'] = v
    except nx.PowerIterationFailedConvergence:
        for n in nodes:
            result[n]['eigenvector'] = np.nan
    except Exception as e:
        print(f"[brokerage_analysis] eigenvector failed: {e}")
        for n in nodes:
            result[n]['eigenvector'] = np.nan

    # Degree (raw, not normalised)
    for n in nodes:
        result[n]['degree'] = G_similarity.degree(n)

    # Leiden community assignment
    if G_similarity.number_of_edges() > 0:
        try:
            edges_sorted = sorted(G_similarity.edges(data='weight'), key=lambda e: (e[0], e[1]))
            G_ig = ig.Graph.TupleList(edges_sorted, weights=True, directed=False)
            partition = leidenalg.find_partition(
                G_ig, leidenalg.ModularityVertexPartition,
                weights='weight', seed=42,
            )
            community_map = {
                G_ig.vs[vi]['name']: comm_id
                for comm_id, community in enumerate(partition)
                for vi in community
            }
            for n in nodes:
                result[n]['community_id'] = community_map.get(n, np.nan)
        except Exception as e:
            print(f"[brokerage_analysis] Leiden community detection failed: {e}")
            for n in nodes:
                result[n]['community_id'] = np.nan
    else:
        for n in nodes:
            result[n]['community_id'] = np.nan

    return result


