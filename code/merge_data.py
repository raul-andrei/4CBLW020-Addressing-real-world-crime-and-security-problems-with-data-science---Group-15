import pandas as pd
import glob
import os

# Path to raw data
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "london_raw_data")

# Find all CSV files
all_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)

print(f"Found {len(all_files)} CSV files.")

chunks = []
for i, f in enumerate(all_files):
    print(f"Reading {i+1}/{len(all_files)}: {os.path.basename(f)}")
    df = pd.read_csv(f)
    chunks.append(df)

# Merge everything
print("Merging all files...")
final = pd.concat(chunks, ignore_index=True)

# Remove duplicates
print("Removing duplicates...")
final = final.drop_duplicates()

print(f"Total rows: {len(final)}")
print(f"Date range: {final['Month'].min()} to {final['Month'].max()}")
print(f"Columns: {list(final.columns)}")

# Save as Parquet next to the data folder
output_path = os.path.join(os.path.dirname(__file__), "..", "data", "london_crime_full.parquet")
final.to_parquet(output_path, index=False)
print(f"Done! Saved to {output_path}")