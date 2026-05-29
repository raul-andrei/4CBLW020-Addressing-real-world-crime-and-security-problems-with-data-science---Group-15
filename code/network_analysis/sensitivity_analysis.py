import itertools
import duckdb
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
from pathlib import Path

from sklearn import metrics
from code.network_analysis.preparation import (
    connect, get_global_corr_matrix, build_cooccurrence_network
)
from code.network_analysis.analysis import brokerage_analysis

ROOT                   = Path(__file__).parent.parent.parent
OUTPUT                 = Path(__file__).parent / "sensitivity_results.csv"
OUTPUT_PER_CRIME_CORR  = Path(__file__).parent / "sensitivity_results_per_crime_correlation.csv"
OUTPUT_PER_CRIME_COOC  = Path(__file__).parent / "sensitivity_results_per_crime_cooccurrence.csv"

COVID_YEARS = {2020, 2021}

PARAM_GRID = {
    "time_frame":         [(2017, 2020), (2021, 2026), (2017, 2026)],
    "covid_included":     [True, False],
    "ASB_included":       [True, False],
    "correlation_method": ["cosine", "spearman", "pearson"],
    # Each entry is (method_name, parameter)
    "edge_selection": [
        ("threshold", 0.5),
        ("threshold", 0.7),
        ("knn", 3),
        ("knn", 4),
    ],
}


# ---------- edge selection ----------

def threshold_filter(G: nx.Graph, threshold: float) -> nx.Graph:
    """Keep edges whose absolute weight exceeds threshold.
    Absolute value is used so this also works for Pearson/Spearman, which
    can produce negative correlations."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if abs(data.get("weight", 0.0)) > threshold:
            H.add_edge(u, v, **data)
    return H


def knn_backbone(G: nx.Graph, k: int) -> nx.Graph:
    """
    Keep each node's k strongest similarity edges.
    Edge survives if selected by either endpoint.
    """

    keep = set()

    for node in G.nodes():
        edges = sorted(
            G.edges(node, data=True),
            key=lambda e: e[2].get("weight", 0.0),
            reverse=True,
        )

        for u, v, _ in edges[:k]:
            keep.add(tuple(sorted((u, v))))

    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    for u, v in keep:
        H.add_edge(u, v, **G[u][v])

    return H


# ---------- graph construction ----------

def create_graph(sim_matrix: pd.DataFrame, drop_negatives: bool = True, matrix='correlation') -> nx.Graph:

    """
    Creates a graph from a Pandas DF Similarity matrix.

    :param sim_matrix: square DataFrame with index and columns as node labels, values as similarities
    :param drop_negatives: if True, set negative similarities to zero (for correlation matrices)
    :param matrix: type of matrix ('correlation' or 'cooccurrence') to determine how to handle values
    :return: NetworkX Graph with edges weighted by similarity
    """

    if matrix=='correlation':
        arr = sim_matrix.to_numpy(copy=True)
        np.fill_diagonal(arr, 0)
        if drop_negatives:
            arr = np.where(arr < 0, 0, arr)
        W = pd.DataFrame(arr, index=sim_matrix.index, columns=sim_matrix.columns)
        return nx.from_pandas_adjacency(W)
    elif matrix=='cooccurrence':
        return nx.from_pandas_adjacency(sim_matrix)

def to_distance_graph(G_sim):
    """Convert a similarity graph to a distance graph for shortest-path algorithms."""
    G_dist = G_sim.copy()
    for u, v, d in G_dist.edges(data=True):
        w = d['weight']
        d['similarity'] = w  # keep original
        d['weight'] = 1.0 / w if w > 0 else float('inf')
    inf_edges = [(u, v) for u, v, d in G_dist.edges(data=True) if np.isinf(d['weight'])]
    G_dist.remove_edges_from(inf_edges)
    return G_dist

def compute_metrics(G_similarity: nx.Graph, G_distance: nx.Graph) -> dict:
    if G_similarity.number_of_nodes() == 0:
        return {}

    components = list(nx.connected_components(G_similarity))
    lcc = G_similarity.subgraph(max(components, key=len)).copy()

    metrics = {
        "n_nodes":      G_similarity.number_of_nodes(),
        "n_edges":      G_similarity.number_of_edges(),
        "density":      nx.density(G_similarity),
        "n_components": len(components),
        "lcc_size":     lcc.number_of_nodes(),
    }

    # Leiden on similarity
    if G_similarity.number_of_edges() > 0:
        try:
            edges_sorted = sorted(G_similarity.edges(data="weight"), key=lambda e: (e[0], e[1]))
            G_ig = ig.Graph.TupleList(edges_sorted, weights=True, directed=False)
            partition = leidenalg.find_partition(
                G_ig, leidenalg.ModularityVertexPartition,
                weights="weight", seed=42,
            )
            metrics["n_communities"] = len(partition)
            metrics["modularity"]    = partition.modularity
        except Exception:
            metrics["n_communities"] = np.nan
            metrics["modularity"]    = np.nan
    else:
        metrics["n_communities"] = G_similarity.number_of_nodes()
        metrics["modularity"]    = 0.0
    
    # Betweenness on distance
    bc = nx.betweenness_centrality(G_distance, weight="weight")
    metrics["mean_betweenness"] = float(np.mean(list(bc.values())))
    
    # Degree on either (same nodes, same edges, just different weights — degree is structural)
    dc = nx.degree_centrality(G_similarity)
    metrics["mean_degree"] = float(np.mean(list(dc.values())))
    
    # Constraint on similarity
    constraint_vals = [
        v for v in nx.constraint(G_similarity, weight="weight").values()
        if v is not None and not np.isnan(v)
    ]

    metrics["mean_constraint"] = (
        float(np.mean(constraint_vals)) if constraint_vals else np.nan
    )
    
    # Eigenvector on similarity (using LCC of similarity graph)
    
    try:
        ec = nx.eigenvector_centrality(lcc, weight="weight", max_iter=1000)
        metrics["mean_eigenvector"] = float(np.mean(list(ec.values())))
    except nx.PowerIterationFailedConvergence:
        metrics["mean_eigenvector"] = np.nan

    return metrics

# ---------- shared SQL filtering ----------

def _build_where_sql(params: dict) -> str:
    year_start, year_end = params["time_frame"]
    conditions = [f"year BETWEEN {year_start} AND {year_end}"]
    if not params.get("covid_included", True):
        conditions.append("year NOT IN (2020, 2021)")
    if not params.get("ASB_included", True):
        conditions.append("crime_type != 'Anti-social behaviour'")
    return " AND ".join(conditions)


def _query_stats(con, where_sql: str) -> tuple:
    """Return (n_rows, n_distinct_crime_types) for the filtered slice."""
    row = con.execute(f"""
        SELECT COUNT(*) AS n_rows, COUNT(DISTINCT crime_type) AS n_types
        FROM crimes WHERE {where_sql}
    """).fetchone()
    return row[0], row[1]


def network_sensibility_test():
    con = connect()
    combos = list(itertools.product(*PARAM_GRID.values()))
    print(f"Running {len(combos)} parameter combinations...\n")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(PARAM_GRID.keys(), combo))
        year_start, year_end    = params["time_frame"]
        edge_method, edge_param = params["edge_selection"]

        where_sql          = _build_where_sql(params)
        n_rows, n_types    = _query_stats(con, where_sql)

        print(f"[{i}/{len(combos)}] tf={year_start}-{year_end} "
              f"covid={params['covid_included']} asb={params['ASB_included']} "
              f"method={params['correlation_method']} "
              f"edge={edge_method}({edge_param}) "
              f"-> rows={n_rows:,}, types={n_types}")

        try:
            sim       = get_global_corr_matrix(con, method=params["correlation_method"],
                                               where_sql=where_sql)
            G_pre     = create_graph(sim)
            edges_pre = G_pre.number_of_edges()

            if edge_method == "threshold":
                G_similarity = threshold_filter(G_pre, edge_param)
            elif edge_method == "knn":
                G_similarity = knn_backbone(G_pre, edge_param)
            else:
                raise ValueError(f"unknown edge_method={edge_method}")

            G_distance = to_distance_graph(G_similarity)
            metrics = compute_metrics(G_similarity, G_distance)
            metrics["edges_pre_filter"]  = edges_pre
            metrics["edges_post_filter"] = G_similarity.number_of_edges()
            metrics["n_rows_used"]       = n_rows
        except Exception as e:
            print(f"  -> Error: {e}")
            metrics = {}

        results.append({
            "time_frame":         f"{year_start}-{year_end}",
            "covid_included":     params["covid_included"],
            "ASB_included":       params["ASB_included"],
            "correlation_method": params["correlation_method"],
            "edge_method":        edge_method,
            "edge_param":         edge_param,
            **metrics,
        })

    con.close()
    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False)
    print(f"\nDone. {len(out)} rows saved to {OUTPUT}")

# ---------- per-crime sensitivity ----------

PARAM_GRID_CORR = {
    "time_frame":         [(2017, 2020), (2021, 2026), (2017, 2026)],
    "covid_included":     [True, False],
    "ASB_included":       [True, False],
    "correlation_method": ["cosine", "spearman", "pearson"],
    "normalisation": ["counts", "proportions"],
}

PARAM_GRID_COOC = {
    "time_frame":     [(2017, 2020), (2021, 2026), (2017, 2026)],
    "covid_included": [True, False],
    "ASB_included":   [True, False],
    "presence_threshold": [2, 3, 5],
}


def run_sensitivity_per_crime():
    con = connect()

    # --- correlation graphs ---
    combos = list(itertools.product(*PARAM_GRID_CORR.values()))
    print(f"=== Correlation graphs: {len(combos)} configurations ===\n")
    corr_results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(PARAM_GRID_CORR.keys(), combo))
        year_start, year_end = params["time_frame"]
        where_sql         = _build_where_sql(params)
        n_rows, n_types   = _query_stats(con, where_sql)
        print(f"[{i}/{len(combos)}] tf={year_start}-{year_end} "
              f"covid={params['covid_included']} asb={params['ASB_included']} "
              f"method={params['correlation_method']} normalisation={params['normalisation']} "
              f"edge=knn(3) -> rows={n_rows:,}, types={n_types}")
        try:
            sim      = get_global_corr_matrix(con, method=params["correlation_method"],
                                              normalisation=params["normalisation"],
                                              where_sql=where_sql)
            G_similarity = knn_backbone(create_graph(sim), k=3)
            G_distance   = to_distance_graph(G_similarity)
            per_node = brokerage_analysis(G_similarity, G_distance)
            config = {
                "config_id":          i,
                "time_frame":         f"{year_start}-{year_end}",
                "covid_included":     params["covid_included"],
                "ASB_included":       params["ASB_included"],
                "correlation_method": params["correlation_method"],
                "normalisation":      params["normalisation"],
                "edge_method":        "knn",
                "edge_param":         3,
                "n_rows_used":        n_rows,
            }
            for crime_type, node_metrics in per_node.items():
                corr_results.append({**config, "crime_type": crime_type, **node_metrics})
        except Exception as e:
            print(f"  -> Error: {e}")

    pd.DataFrame(corr_results).to_csv(OUTPUT_PER_CRIME_CORR, index=False)
    print(f"\nCorrelation done. {len(corr_results)} rows saved to {OUTPUT_PER_CRIME_CORR}\n")

    # --- co-occurrence graphs ---
    combos = list(itertools.product(*PARAM_GRID_COOC.values()))
    print(f"=== Co-occurrence graphs: {len(combos)} configurations ===\n")
    cooc_results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(PARAM_GRID_COOC.keys(), combo))
        year_start, year_end = params["time_frame"]
        where_sql         = _build_where_sql(params)
        n_rows, n_types   = _query_stats(con, where_sql)
        print(f"[{i}/{len(combos)}] tf={year_start}-{year_end} "
              f"covid={params['covid_included']} asb={params['ASB_included']} "
              f"presence_threshold={params['presence_threshold']} "
              f"edge=knn(3) -> rows={n_rows:,}, types={n_types}")
        try:
            cooc     = build_cooccurrence_network(con,
                                                  presence_threshold=params["presence_threshold"],
                                                  where_sql=where_sql)
            G_similarity = knn_backbone(create_graph(cooc, matrix="cooccurrence"), k=3)
            G_distance   = to_distance_graph(G_similarity)
            per_node = brokerage_analysis(G_similarity, G_distance)
            config = {
                "config_id":          i,
                "time_frame":         f"{year_start}-{year_end}",
                "covid_included":     params["covid_included"],
                "ASB_included":       params["ASB_included"],
                "presence_threshold": params["presence_threshold"],
                "edge_method":        "knn",
                "edge_param":         3,
                "n_rows_used":        n_rows,
            }
            for crime_type, node_metrics in per_node.items():
                cooc_results.append({**config, "crime_type": crime_type, **node_metrics})
        except Exception as e:
            print(f"  -> Error: {e}")

    con.close()
    pd.DataFrame(cooc_results).to_csv(OUTPUT_PER_CRIME_COOC, index=False)
    print(f"\nCo-occurrence done. {len(cooc_results)} rows saved to {OUTPUT_PER_CRIME_COOC}")

def run_primary_brokerage_analysis(where_sql=None, con: duckdb.DuckDBPyConnection = None) -> pd.DataFrame | None:
    """
    Run brokerage analysis under the primary specification:
      - Co-occurrence network (lift)
      - Presence threshold = 3
      - kNN backbone k = 3
    
    Parameters
    ----------
    where_sql : str, optional
        SQL filter for restricting the data (e.g. by time, force, region).
        Passed through to build_cooccurrence_network. If None, uses all data.
    
    Returns
    -------
    pd.DataFrame
        One row per crime type with columns: crime_type, betweenness,
        current_flow_betweenness, constraint, eigenvector, community_id, degree.
    """
    try:
        cooc = build_cooccurrence_network(
            con,
            presence_threshold=3,
            where_sql=where_sql,
        )
        G_sim = knn_backbone(create_graph(cooc, matrix='cooccurrence'), k=3)
        G_dist = to_distance_graph(G_sim)
        per_node = brokerage_analysis(G_sim, G_dist)
    except Exception as e:
        print(f"Error in run_primary_brokerage_analysis: {e}")
        per_node = {}
    rows = [
        {'crime_type': crime, **metrics}
        for crime, metrics in per_node.items()
    ]
    return pd.DataFrame(rows) if rows else None

def run_brokerage_per_force():
    """For each police force, run the per-crime co-occurrence analysis using
    the primary specification (lift, threshold=3, kNN k=3, ASB included,
    full time period). Outputs one row per (force, crime_type)."""
    con = connect()
    
    forces = con.execute("""
        SELECT DISTINCT reported_by 
        FROM crimes 
        WHERE reported_by IS NOT NULL
        ORDER BY reported_by
    """).df()['reported_by'].tolist()
    
    print(f"Running per-force analysis for {len(forces)} forces\n")
    
    results = pd.DataFrame()
    for i, force in enumerate(forces, 1):
        # Escape single quotes for SQL safety
        force_sql = force.replace("'", "''")
        where_sql = f"year BETWEEN 2017 AND 2026 AND reported_by = '{force_sql}'"
        
        n_rows = con.execute(
            f"SELECT COUNT(*) FROM crimes WHERE {where_sql}"
        ).fetchone()[0]
        
        print(f"[{i}/{len(forces)}] {force}: {n_rows:,} rows")
        
        if n_rows < 10000:
            print(f"  -> skipping, too few rows")
            continue
        
        res = run_primary_brokerage_analysis(where_sql=where_sql, con=con)
        if res is None or res.empty:
            print(f"  -> no valid graph could be built, skipping")
            continue
        
        res['force'] = force
        res['n_rows_used'] = n_rows

        results = pd.concat([results, res], ignore_index=True)

    con.close()
    out = results
    out_path = Path(__file__).parent / 'per_force_brokerage.csv'
    out.to_csv(out_path, index=False)
    print(f"\nDone. {len(out)} rows saved to {out_path}")
    return out


if __name__ == "__main__":
    # network_sensibility_test()
    
    out_path = Path(__file__).parent / 'global_brokerage.csv'
    con = connect()
    where_sql = "year BETWEEN 2017 AND 2026"
    df = run_primary_brokerage_analysis(where_sql=where_sql, con=con)
    con.close()
    if df is not None:
        df.to_csv(out_path, index=False)
        print(f"Global brokerage analysis done. {len(df)} rows saved to {out_path}")