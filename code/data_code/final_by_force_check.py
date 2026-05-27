import duckdb
import os

output_dir = "/Volumes/Extreme SSD/Proiect CBL Raul/Data/by_force_cleaned"
files = [f for f in os.listdir(output_dir) if f.endswith(".parquet") and not f.startswith(".")]

con = duckdb.connect()

total_rows_all = 0
issues = []

expected_crime_types = {
    'Anti-social behaviour', 'Bicycle theft', 'Burglary',
    'Criminal damage and arson', 'Drugs', 'Other crime',
    'Other theft', 'Possession of weapons', 'Public order',
    'Robbery', 'Shoplifting', 'Theft from the person',
    'Vehicle crime', 'Violence and sexual offences'
}

print("=" * 60)
print("FINAL DATA CHECK")
print("=" * 60)

for f in sorted(files):
    filepath = os.path.join(output_dir, f)
    force_name = f.replace(".parquet", "")
    print(f"\n--- {force_name} ---")

    # Basic info
    basic = con.execute(f"""
        SELECT 
            COUNT(*) as rows,
            MIN("Month") as earliest,
            MAX("Month") as latest
        FROM read_parquet('{filepath}')
    """).fetchone()

    total_rows_all += basic[0]
    print(f"Rows: {basic[0]:,}")
    print(f"Date range: {basic[1]} to {basic[2]}")

    # Missing values
    nulls = con.execute(f"""
        SELECT
            SUM(CASE WHEN "LSOA code" IS NULL THEN 1 ELSE 0 END) as lsoa_nulls,
            SUM(CASE WHEN "Longitude" IS NULL THEN 1 ELSE 0 END) as lon_nulls,
            SUM(CASE WHEN "Latitude" IS NULL THEN 1 ELSE 0 END) as lat_nulls,
            SUM(CASE WHEN "Crime type" IS NULL THEN 1 ELSE 0 END) as crime_nulls,
            SUM(CASE WHEN "Month" IS NULL THEN 1 ELSE 0 END) as month_nulls
        FROM read_parquet('{filepath}')
    """).fetchone()

    if any(n > 0 for n in nulls):
        print(f"⚠️ Missing values: LSOA={nulls[0]}, Lon={nulls[1]}, Lat={nulls[2]}, Crime type={nulls[3]}, Month={nulls[4]}")
        issues.append(f"{force_name}: has unexpected null values")
    else:
        print("Missing values: none ✅")

    # Crime types
    crime_types = con.execute(f"""
        SELECT DISTINCT "Crime type"
        FROM read_parquet('{filepath}')
        ORDER BY "Crime type"
    """).fetchall()
    crime_type_set = {ct[0] for ct in crime_types}

    extra = crime_type_set - expected_crime_types
    missing = expected_crime_types - crime_type_set

    if extra:
        print(f"⚠️ Extra crime types: {extra}")
        issues.append(f"{force_name}: has extra crime types {extra}")
    if missing:
        print(f"⚠️ Missing crime types: {missing}")
        issues.append(f"{force_name}: missing crime types {missing}")
    if not extra and not missing:
        print("Crime types: all correct ✅")

    # COVID flag check
    covid_check = con.execute(f"""
        SELECT
            SUM(CASE WHEN "COVID period" = true AND "Month" < '2020-01' THEN 1 ELSE 0 END) as wrong_before,
            SUM(CASE WHEN "COVID period" = true AND "Month" > '2021-12' THEN 1 ELSE 0 END) as wrong_after,
            SUM(CASE WHEN "COVID period" = false AND "Month" >= '2020-01' AND "Month" <= '2021-12' THEN 1 ELSE 0 END) as wrong_during
        FROM read_parquet('{filepath}')
    """).fetchone()

    if any(n > 0 for n in covid_check):
        print(f"⚠️ COVID flag errors: {covid_check[0]} wrong before, {covid_check[1]} wrong after, {covid_check[2]} wrong during")
        issues.append(f"{force_name}: COVID flag incorrectly assigned")
    else:
        print("COVID flag: correct ✅")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total force files: {len(files)}")
print(f"Total rows across all forces: {total_rows_all:,}")

if issues:
    print(f"\n⚠️ Issues found ({len(issues)}):")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n✅ No issues found!")

con.close()