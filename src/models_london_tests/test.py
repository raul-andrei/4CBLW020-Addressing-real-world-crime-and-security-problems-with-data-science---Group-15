import duckdb

path = "/Users/raulmociornita/Library/Mobile Documents/com~apple~CloudDocs/University/Year 2/Quarter 4/CBL/Coding Project/4CBLW020-Addressing-real-world-crime-and-security-problems-with-data-science---Group-15/data/cleaned_data/by_force_cleaned/london.parquet"

con = duckdb.connect()
df = con.execute(f"""
    SELECT DISTINCT regexp_extract("LSOA name", '^[A-Za-z ]+', 0) as area
    FROM read_parquet('{path}')
    ORDER BY area
""").df()
print(df)
con.close()