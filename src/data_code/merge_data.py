import pandas as pd
import glob
import os

base_path = "/Volumes/Extreme SSD/Proiect CBL Raul/Data"
zip_names = ["2017-01", "2020-01", "2023-01", "2026-01"]

# Step 1 - Create one parquet per zip
for zip_name in zip_names:
    data_path = os.path.join(base_path, "uk_raw_data", zip_name)
    output_path = os.path.join(base_path, f"{zip_name}.parquet")

    if os.path.exists(output_path):
        print(f"Skipping {zip_name} - parquet already exists")
        continue

    all_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)
    print(f"\nProcessing {zip_name} - Found {len(all_files)} CSV files...")

    chunks = []
    for i, f in enumerate(all_files):
        print(f"Reading {i+1}/{len(all_files)}: {os.path.basename(f)}")
        try:
            df = pd.read_csv(f)
            if "Crime type" in df.columns:
                chunks.append(df)
            else:
                print(f"Skipping {os.path.basename(f)} - not a crime file")
        except pd.errors.EmptyDataError:
            print(f"Skipping {os.path.basename(f)} - empty file")

    print(f"Merging {zip_name}...")
    final = pd.concat(chunks, ignore_index=True)
    final.to_parquet(output_path, index=False)
    print(f"Done! Saved {len(final)} rows to {output_path}")

# Step 2 - Merge all 4 parquets into one
print("\nMerging all parquets into one...")
dfs = []
for zip_name in zip_names:
    path = os.path.join(base_path, f"{zip_name}.parquet")
    df = pd.read_parquet(path)
    print(f"{zip_name}: {len(df)} rows")
    dfs.append(df)

final = pd.concat(dfs, ignore_index=True)

print("Removing duplicates...")
final = final.drop_duplicates()

print("Removing Context column...")
final = final.drop(columns=["Context"], errors="ignore")

print(f"Total rows: {len(final)}")
print(f"Date range: {final['Month'].min()} to {final['Month'].max()}")

final.to_parquet(os.path.join(base_path, "uk_crime_full.parquet"), index=False)
print("Done! Saved uk_crime_full.parquet")