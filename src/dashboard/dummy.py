import pandas as pd
from pathlib import Path
import json
from duckdb import connect

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DASHBOARD_ASSETS = BASE_DIR / "src" / "dashboard" / "assets"
WARD_PATH = BASE_DIR / "data" / "wards_dec2021_uk_bgc_4326.geojson"

forecasts = pd.read_parquet(DASHBOARD_ASSETS / "forecast_snapshot.parquet")
mapping_df = pd.read_parquet(DASHBOARD_ASSETS / "ward_force_mapping.parquet")

with open(WARD_PATH, "r", encoding="utf-8") as f:
    WARD_GEOJSON = json.load(f)

wards = pd.read_parquet(DASHBOARD_ASSETS / "ward_snapshot.parquet")

print(wards.info())