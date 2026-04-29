import pandas as pd
import os

parquet_path = os.path.join(os.path.dirname(__file__), "..", "data", "london_crime_full.parquet")

print("Loading data...")
df = pd.read_parquet(parquet_path)

print("\n--- BASIC INFO ---")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")

print("\n--- DATE RANGE ---")
print(f"From: {df['Month'].min()}")
print(f"To:   {df['Month'].max()}")

print("\n--- POLICE FORCES ---")
print(df['Reported by'].value_counts())

print("\n--- CRIME TYPES ---")
print(df['Crime type'].value_counts())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- CRIMES PER YEAR ---")
df['Year'] = df['Month'].str[:4]
print(df['Year'].value_counts().sort_index())