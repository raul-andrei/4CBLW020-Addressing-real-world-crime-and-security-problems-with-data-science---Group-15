import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "crimes.db"


def connect():
    """Open the prebuilt DB read-only. Multiple analysis scripts can share it."""
    return duckdb.connect(str(DB_PATH), read_only=True)


def build_lsoa_crime_matrix(con, min_total=100, normalisation='counts', where_sql=None):
    extra = f"AND {where_sql}" if where_sql else ""
    long = con.execute(f"""
        SELECT lsoa_code, crime_type, COUNT(*) AS n
        FROM crimes
        WHERE TRUE {extra}
        GROUP BY lsoa_code, crime_type
    """).df()

    counts = (long.pivot(index='lsoa_code', columns='crime_type', values='n')
                  .fillna(0))
    counts = counts.loc[counts.sum(axis=1) >= min_total]

    if normalisation == 'proportions':
        counts = counts.div(counts.sum(axis=1), axis=0)
    return counts


def get_global_corr_matrix(con, method='cosine', normalisation='counts', where_sql=None):
    matrix = build_lsoa_crime_matrix(con, normalisation=normalisation, where_sql=where_sql)
    if method == 'cosine':
        return pd.DataFrame(
            cosine_similarity(matrix.T.values),
            index=matrix.columns, columns=matrix.columns,
        )
    elif method in ('spearman', 'pearson'):
        return matrix.corr(method=method)
    raise ValueError(f"Invalid method: {method}")


def build_cooccurrence_network(
    con,
    presence_threshold=3,
    where_sql=None,
    min_lift=1.0,
    log_transform=False,
):
    extra = f"AND {where_sql}" if where_sql else ""

    long = con.execute(f"""
        SELECT lsoa_code, month_num, crime_type, COUNT(*) AS n
        FROM crimes
        WHERE TRUE {extra}
        GROUP BY lsoa_code, month_num, crime_type
    """).df()

    # Binary presence matrix
    present = (
        long.pivot_table(
            index=['lsoa_code', 'month_num'],
            columns='crime_type',
            values='n',
            fill_value=0,
        ) >= presence_threshold
    )

    n_cells = len(present)

    # Marginal probabilities
    p_marginal = present.mean()

    # Joint probabilities
    p_joint = (
        present.T.astype(int) @ present.astype(int)
    ) / n_cells

    # Expected under independence
    expected = np.outer(p_marginal, p_marginal)

    # Lift
    lift = np.where(expected > 0, p_joint / expected, 0)

    # Remove diagonal
    np.fill_diagonal(lift, 0)

    # Convert to DataFrame
    lift_df = pd.DataFrame(
        lift,
        index=present.columns,
        columns=present.columns,
    )

    # Keep only positive associations
    lift_df = lift_df.where(lift_df > min_lift, 0)

    # Optional stabilization
    if log_transform:
        lift_df = np.log1p(lift_df)

    return lift_df
