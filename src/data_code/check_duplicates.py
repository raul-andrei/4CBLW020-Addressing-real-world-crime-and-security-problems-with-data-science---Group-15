import pandas as pd
import os

output_dir = "/Volumes/Extreme SSD/Proiect CBL Raul/Data/by_force_cleaned"
files = [f for f in os.listdir(output_dir) if f.endswith(".parquet") and not f.startswith(".")]

for f in sorted(files):
    filepath = os.path.join(output_dir, f)
    df = pd.read_parquet(filepath)
    
    total = len(df)
    unique = len(df.drop_duplicates())
    duplicates = total - unique
    
    if duplicates > 0:
        print(f"⚠️ {f}: {total:,} total vs {unique:,} unique — {duplicates:,} duplicates!")
    else:
        print(f"✅ {f}: no duplicates")