import duckdb
import pandas as pd
import os
import re

# Paths
BASE        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL_PATH  = os.path.join(BASE, "data", "cleaned_data", "panels", "london_panel.parquet")
OUTPUT_PATH = os.path.join(BASE, "data", "cleaned_data", "panels", "london_panel_borough.parquet")

LONDON_BOROUGHS = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
    "Camden", "City of London", "Croydon", "Ealing", "Enfield",
    "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey",
    "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington",
    "Kensington and Chelsea", "Kingston upon Thames", "Lambeth",
    "Lewisham", "Merton", "Newham", "Redbridge", "Richmond upon Thames",
    "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest", "Wandsworth",
    "Westminster"
]

print("Building borough panel...")

con = duckdb.connect()
df = con.execute(f"""
    SELECT lsoa_name, month, crime_type, SUM(crime_count) as crime_count
    FROM read_parquet('{PANEL_PATH}')
    GROUP BY lsoa_name, month, crime_type
    ORDER BY lsoa_name, month, crime_type
""").df()
con.close()

# Extract borough name from lsoa_name
df["borough"] = df["lsoa_name"].apply(lambda x: re.sub(r'\s+\d+\w*$', '', x).strip())
df = df[df["borough"].isin(LONDON_BOROUGHS)]



# Aggregate to borough level
borough_df = (
    df.groupby(["borough", "month", "crime_type"])["crime_count"]
    .sum()
    .reset_index()
)

borough_df.to_parquet(OUTPUT_PATH, index=False)

print(f"Done! Saved to {OUTPUT_PATH}")
print(f"Total rows: {len(borough_df):,}")
print(f"Boroughs: {borough_df['borough'].nunique()}")
print(f"Crime types: {borough_df['crime_type'].nunique()}")