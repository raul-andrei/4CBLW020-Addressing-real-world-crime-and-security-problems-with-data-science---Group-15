import duckdb

parquet_path = "/Volumes/Extreme SSD/Proiect CBL Raul/Data/uk_crime_full_cleaned.parquet"

con = duckdb.connect()

print("\n--- BASIC INFO ---")
result = con.execute(f"""
    SELECT COUNT(*) as rows, COUNT(DISTINCT "Reported by") as forces
    FROM read_parquet('{parquet_path}')
""").fetchone()
print(f"Total rows: {result[0]:,}")
print(f"Total forces: {result[1]}")

print("\n--- DATE RANGE ---")
result = con.execute(f"""
    SELECT MIN("Month"), MAX("Month")
    FROM read_parquet('{parquet_path}')
""").fetchone()
print(f"From: {result[0]}")
print(f"To: {result[1]}")

print("\n--- CRIME TYPES ---")
results = con.execute(f"""
    SELECT "Crime type", COUNT(*) as count
    FROM read_parquet('{parquet_path}')
    GROUP BY "Crime type"
    ORDER BY count DESC
""").fetchall()
for r in results:
    print(f"  {r[0]}: {r[1]:,}")

print("\n--- CRIMES PER YEAR ---")
results = con.execute(f"""
    SELECT LEFT("Month", 4) as year, COUNT(*) as count
    FROM read_parquet('{parquet_path}')
    GROUP BY year
    ORDER BY year
""").fetchall()
for r in results:
    print(f"  {r[0]}: {r[1]:,}")

print("\n--- TOP 10 FORCES BY CRIME COUNT ---")
results = con.execute(f"""
    SELECT "Reported by", COUNT(*) as count
    FROM read_parquet('{parquet_path}')
    GROUP BY "Reported by"
    ORDER BY count DESC
    LIMIT 10
""").fetchall()
for r in results:
    print(f"  {r[0]}: {r[1]:,}")

con.close()