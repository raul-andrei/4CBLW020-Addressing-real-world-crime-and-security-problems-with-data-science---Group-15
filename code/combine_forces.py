import duckdb
import os

input_dir = "/Volumes/Extreme SSD/Proiect CBL Raul/Data/by_force_cleaned"
output_path = "/Volumes/Extreme SSD/Proiect CBL Raul/Data/uk_crime_full_cleaned.parquet"

# Get all valid parquet files 
files = [
    os.path.join(input_dir, f) 
    for f in os.listdir(input_dir) 
    if f.endswith(".parquet") and not f.startswith(".")
]

con = duckdb.connect()

file_list = ", ".join([f"'{f}'" for f in files])

con.execute(f"""
    COPY (
        SELECT * FROM read_parquet([{file_list}])
    ) TO '{output_path}' (FORMAT PARQUET)
""")

print("Done!")
con.close()