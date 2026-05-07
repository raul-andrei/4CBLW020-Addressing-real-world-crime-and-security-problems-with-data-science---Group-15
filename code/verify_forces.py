import duckdb
import os

output_dir = "/Volumes/Extreme SSD/Proiect CBL Raul/Data/by_force"

# Get all parquet files
files = [f for f in os.listdir(output_dir) if f.endswith(".parquet") and not f.startswith(".")]
print(f"Total force files: {len(files)}\n")

con = duckdb.connect()

total_rows = 0
for f in sorted(files):
    filepath = os.path.join(output_dir, f)
    result = con.execute(f"""
        SELECT 
            COUNT(*) as rows,
            MIN("Month") as earliest,
            MAX("Month") as latest,
            COUNT(DISTINCT "Crime type") as crime_types,
            SUM(CASE WHEN "COVID period" = true THEN 1 ELSE 0 END) as covid_rows
        FROM read_parquet('{filepath}')
    """).fetchone()
    
    print(f"{f}")
    print(f"  Rows: {result[0]:,}")
    print(f"  Date range: {result[1]} to {result[2]}")
    print(f"  Crime types: {result[3]}")
    print(f"  COVID rows: {result[4]:,}")
    print()
    
    total_rows += result[0]

con.close()
print(f"Total rows across all forces: {total_rows:,}")