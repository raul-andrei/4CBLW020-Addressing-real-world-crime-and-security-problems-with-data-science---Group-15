import duckdb
import os

input_dir = "/Volumes/Extreme SSD/Proiect CBL Raul/Data/by_force"
output_dir = "/Volumes/Extreme SSD/Proiect CBL Raul/Data/by_force_cleaned"
os.makedirs(output_dir, exist_ok=True)

files = [f for f in os.listdir(input_dir) if f.endswith(".parquet") and not f.startswith(".")]

con = duckdb.connect()

for f in sorted(files):
    input_path = os.path.join(input_dir, f)
    output_path = os.path.join(output_dir, f)

    if os.path.exists(output_path):
        print(f"Skipping {f} - already exists")
        continue

    print(f"Cleaning {f}...")
    con.execute(f"""
        COPY (
            SELECT DISTINCT *
            FROM read_parquet('{input_path}')
        ) TO '{output_path}' (FORMAT PARQUET)
    """)
    print(f"Done {f}!")

con.close()
print("\nAll done!")