import pandas as pd
import os

input_dir = "/Volumes/Extreme SSD/Proiect CBL Raul/Data/by_force_cleaned"
output_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_preview.csv")

files = [f for f in os.listdir(input_dir) if f.endswith(".parquet") and not f.startswith(".")]

chunks = []
for f in sorted(files):
    filepath = os.path.join(input_dir, f)
    df = pd.read_parquet(filepath)
    sample = df.sample(min(200, len(df)), random_state=42)
    chunks.append(sample)
    print(f"Sampled {f}")

final = pd.concat(chunks, ignore_index=True)
final = final.sort_values("Month")

final.to_csv(output_path, index=False)
print(f"\nDone! {len(final)} rows saved to sample_preview.csv")