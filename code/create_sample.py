import pandas as pd
import os

parquet_path = os.path.join(os.path.dirname(__file__), "..", "data", "london_crime_full.parquet")
output_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_preview.csv")

df = pd.read_parquet(parquet_path)

# Export just a random sample of 1000 rows as CSV so anyone can open it
df.sample(1000, random_state=42).sort_values("Month").to_csv(output_path, index=False)
print("Sample saved!")