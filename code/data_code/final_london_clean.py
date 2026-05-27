import duckdb

con = duckdb.connect()

filepath = "/Volumes/Extreme SSD/Proiect CBL Raul/Data/by_force_cleaned/london.parquet"

con.execute(f"""
    COPY (
        SELECT * FROM read_parquet('{filepath}')
        WHERE "Crime type" NOT IN ('Hillingdon 011C', 'Hillingdon 009E')
    ) TO '{filepath}.tmp' (FORMAT PARQUET)
""")

import os
os.replace(f"{filepath}.tmp", filepath)
print("Done! London cleaned.")

con.close()