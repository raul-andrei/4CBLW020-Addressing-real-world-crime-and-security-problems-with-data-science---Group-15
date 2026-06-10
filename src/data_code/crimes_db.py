import duckdb
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "crimes.db"
PARQUET_GLOB = ROOT / "data" / "*.parquet"

DB_PATH.unlink(missing_ok=True)
conn = duckdb.connect(database=DB_PATH, read_only=False)

conn.execute(f"""
    CREATE TABLE crimes AS
    SELECT
        "LSOA code"                              AS lsoa_code,
        "Crime type"                             AS crime_type,
        CAST(SUBSTR("Month", 1, 4) AS SMALLINT)  AS year,
        CAST(SUBSTR("Month", 6, 2) AS TINYINT)   AS month_num,
        "Longitude"                              AS longitude,
        "Latitude"                               AS latitude,
        "Last outcome category"                  AS last_outcome,
        "Reported by"                            AS reported_by
    FROM read_parquet('{PARQUET_GLOB}', union_by_name=true)
    WHERE "LSOA code" IS NOT NULL
      AND CAST(SUBSTR("Month", 1, 4) AS INTEGER) BETWEEN 2017 AND 2026
      AND "Crime type" != 'Other crime'
    ORDER BY lsoa_code, year, month_num
""")

# Sanity report
n_rows, n_lsoa, n_types = conn.execute("""
    SELECT COUNT(*), COUNT(DISTINCT lsoa_code), COUNT(DISTINCT crime_type)
    FROM crimes
""").fetchone()
print(f"Built {DB_PATH.name}: {n_rows:,} rows, {n_lsoa:,} LSOAs, {n_types} crime types")

conn.close()