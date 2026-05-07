import duckdb
import os

base_path = "/Volumes/Extreme SSD/Proiect CBL Raul/Data"
raw_data_path = os.path.join(base_path, "uk_raw_data")
output_dir = os.path.join(base_path, "by_force")
os.makedirs(output_dir, exist_ok=True)

london_forces = ("Metropolitan Police Service", "City of London Police")

con = duckdb.connect()

# Get all unique forces from raw CSVs
print("Getting all forces...")
forces = con.execute(f"""
    SELECT DISTINCT "Reported by" 
    FROM read_csv_auto('{raw_data_path}/**/*street*.csv',
        filename=true,
        union_by_name=true,
        ignore_errors=true)
    WHERE "Month" >= '2017-01'
    AND "LSOA code" IS NOT NULL
""").fetchall()

print(f"Found {len(forces)} forces")

# Save London separately
print("Saving london.parquet...")
con.execute(f"""
    COPY (
        SELECT DISTINCT
            "Crime ID",
            "Month",
            "Reported by",
            "Falls within",
            "Longitude",
            "Latitude",
            "Location",
            "LSOA code",
            "LSOA name",
            "Crime type",
            "Last outcome category",
            CASE 
                WHEN "Month" >= '2020-01' AND "Month" <= '2021-12' 
                THEN true 
                ELSE false 
            END AS "COVID period"
        FROM read_csv_auto('{raw_data_path}/**/*street*.csv',
            filename=true,
            union_by_name=true,
            ignore_errors=true)
        WHERE "Month" >= '2017-01'
        AND "Reported by" IN ('Metropolitan Police Service', 'City of London Police')
        AND "LSOA code" IS NOT NULL
        AND "Longitude" IS NOT NULL
        AND "Latitude" IS NOT NULL
    ) TO '{output_dir}/london.parquet' (FORMAT PARQUET)
""")
print("London saved!")

# Save each other force
for (force,) in forces:
    if force in london_forces:
        continue

    filename = force.lower().replace(" ", "_").replace("/", "_").replace("&", "and") + ".parquet"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"Skipping {force} - already exists")
        continue

    print(f"Saving {force}...")
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                "Crime ID",
                "Month",
                "Reported by",
                "Falls within",
                "Longitude",
                "Latitude",
                "Location",
                "LSOA code",
                "LSOA name",
                "Crime type",
                "Last outcome category",
                CASE 
                    WHEN "Month" >= '2020-01' AND "Month" <= '2021-12' 
                    THEN true 
                    ELSE false 
                END AS "COVID period"
            FROM read_csv_auto('{raw_data_path}/**/*street*.csv',
                filename=true,
                union_by_name=true,
                ignore_errors=true)
            WHERE "Month" >= '2017-01'
            AND "Reported by" = '{force}'
            AND "LSOA code" IS NOT NULL
            AND "Longitude" IS NOT NULL
            AND "Latitude" IS NOT NULL
        ) TO '{filepath}' (FORMAT PARQUET)
    """)
    print(f"Saved {force} → {filename}")

con.close()
print("\nAll done!")