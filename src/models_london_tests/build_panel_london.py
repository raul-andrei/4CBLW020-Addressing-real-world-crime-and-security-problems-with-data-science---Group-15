import duckdb
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH  = os.path.join(BASE, "data", "cleaned_data", "by_force_cleaned", "london.parquet")
OUTPUT_PATH = os.path.join(BASE, "data", "cleaned_data", "panels", "london_panel.parquet")


os.makedirs("output", exist_ok=True)

# Build panel
print("Building London panel...")

con = duckdb.connect()

con.execute(f"""
    COPY (
        SELECT
            "LSOA code"        AS lsoa_code,
            "LSOA name"        AS lsoa_name,
            strptime(Month, '%Y-%m') AS month,
            "Crime type"       AS crime_type,
            COUNT(*)           AS crime_count
        FROM read_parquet('{INPUT_PATH}')
        WHERE "COVID period" = false
        GROUP BY
            "LSOA code",
            "LSOA name",
            Month,
            "Crime type"
        ORDER BY
            lsoa_code, month, crime_type
    ) TO '{OUTPUT_PATH}' (FORMAT PARQUET)
""")

con.close()

print(f"Done! Panel saved to {OUTPUT_PATH}")

# Quick check
con = duckdb.connect()
result = con.execute(f"""
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT lsoa_code) as num_lsoas,
        COUNT(DISTINCT crime_type) as num_crime_types,
        MIN(month) as start_date,
        MAX(month) as end_date
    FROM read_parquet('{OUTPUT_PATH}')
""").fetchone()
con.close()

print(f"Total rows: {result[0]:,}")
print(f"LSOAs: {result[1]:,}")
print(f"Crime types: {result[2]}")
print(f"Date range: {result[3]} → {result[4]}")