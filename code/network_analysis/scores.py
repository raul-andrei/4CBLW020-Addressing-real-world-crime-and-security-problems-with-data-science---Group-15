import numpy as np
import pandas as pd
from pathlib import Path
from code.network_analysis.preparation import connect

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / 'data'

def log_normalize(bc_log: dict) -> dict:
    """Log-normalize the betweenness centrality scores.

    Args:
        bc_log (dict): A dictionary containing the log-transformed betweenness centrality scores.
    Returns:
        dict: A dictionary containing the log-normalized betweenness centrality scores.
    """

    bc_log = {crime: np.log1p(score) for crime, score in bc_log.items()}
    max_log = max(bc_log.values())

    bc_log_normalized = {crime: score / max_log for crime, score in bc_log.items()}

    return bc_log_normalized

def load_scores(file_path: Path) -> dict:
    """Load betweenness centrality scores from a CSV file.

    Args:
        file_path (Path): The path to the CSV file containing the betweenness centrality scores.
    Returns:
        dict: A dictionary containing the betweenness centrality scores.
    """
    df = pd.read_csv(file_path)
    bc_scores = dict(zip(df['crime_type'], df['current_flow_betweenness']))
    return bc_scores

def get_crime_data() -> pd.DataFrame:
    con = connect()
    sql = "SELECT lsoa_code, crime_type, month_num FROM crimes WHERE year BETWEEN 2017 AND 2026"
    df = con.execute(sql).df()
    con.close()
    return df

def aggregate_lsoas_to_wards(df_lsoas: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LSOA-level data to ward-level data.

    Args:
        df_lsoas (pd.DataFrame): A DataFrame containing LSOA-level data with columns 
    Returns:
        pd.DataFrame: A DataFrame containing ward-level aggregated scores.
    """

    

    # First Load the csv files with mappings between LSOAS and wards:
    df_ward = pd.read_csv(DATA / 'lsoa_ward_mapping.csv', usecols=['LSOA11CD', 'WD21CD'])

    # Now merge the LSOA-level data with the ward mapping:
    df_lsoas = df_lsoas.merge(df_ward, left_on='lsoa_code', right_on='LSOA11CD', how='left')

    #Now we first groupby ward, month, crime types to get counts per ward, month, crime type:
    df_ward_agg = df_lsoas.groupby(['WD21CD', 'month_num', 'crime_type']).size().reset_index(name='count')

    return df_ward_agg

def calculate_brokerage_scores(crimes_df: pd.DataFrame, bc_normalised: dict) -> pd.DataFrame:
    """Calculate brokerage scores for each ward based on the aggregated data.

    Args:
        df_ward_agg (pd.DataFrame): A DataFrame containing ward-level aggregated data with columns ['WD21CD', 'month_num', 'crime_type', 'count'].
    Returns:
        pd.DataFrame: A DataFrame containing brokerage scores for each ward.
    """
    # Placeholder for actual brokerage score calculation logic
    # This would involve creating a network based on the crime types and calculating centrality measures
    # For demonstration, we will just return a DataFrame with dummy brokerage scores

    totals = totals = crimes_df.groupby(['WD21CD', 'month_num'])['count'].sum().reset_index(name='total')
    crimes_df = crimes_df.merge(totals, on=['WD21CD', 'month_num'])
    crimes_df['share'] = crimes_df['count'] / crimes_df['total']


    crimes_df['brokerage_score'] = crimes_df['crime_type'].map(bc_normalised)
    crimes_df['weighted'] = crimes_df['brokerage_score'] * crimes_df['share']

    avg_betweenness = crimes_df.groupby(['WD21CD', 'month_num'])['weighted'].sum().reset_index(name='avg_betweenness')
    return avg_betweenness

if __name__ == "__main__":
    df = get_crime_data()
    df = aggregate_lsoas_to_wards(df)

    # Diagnostics
    print(f"Ward-month-crime rows: {len(df)}")

    bc_scores = load_scores(DATA / 'global_brokerage.csv')
    bc_normalised = log_normalize(bc_scores)

    # Warn on unmapped crime types
    crimes_in_data = set(df['crime_type'].unique())
    crimes_in_scores = set(bc_normalised.keys())
    missing = crimes_in_data - crimes_in_scores
    if missing:
        print(f"Crime types with no brokerage score (will be excluded): {missing}")
        df = df[df['crime_type'].isin(crimes_in_scores)]

    df = calculate_brokerage_scores(df, bc_normalised)
    print(df.head(15))
    print(f"\nAvg betweenness range: [{df['avg_betweenness'].min():.3f}, {df['avg_betweenness'].max():.3f}]")