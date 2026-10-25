#!/usr/bin/env python3
"""One-off export of MekongPulse static datasets for external analysis.

Reads data/commune_gazetteer_2025.json, data/stations.json and
data/raw/mef_commune_boundaries.geojson, and writes CSV + GeoJSON outputs
next to this script. Run from projects/: python3 dataset_export/export.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
DATA = PROJ / "data"

gaz = json.loads((DATA / "commune_gazetteer_2025.json").read_text(encoding="utf-8"))
stations = json.loads((DATA / "stations.json").read_text(encoding="utf-8"))
bounds = json.loads((DATA / "raw" / "mef_commune_boundaries.geojson").read_text(encoding="utf-8"))

# --- communes.csv: flat gazetteer, one row per commune ----------------------
commune_fields = [
    "code", "name_en", "name_kh",
    "district_code", "district_en", "district_kh",
    "province_code", "province_en", "province_kh",
    "lat", "lng", "has_boundary", "ground_elevation_m",
    "nearest_station_id", "distance_to_station_km",
]
with (HERE / "communes.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=commune_fields)
    w.writeheader()
    w.writerows(gaz["communes"])

# --- stations.csv -----------------------------------------------------------
station_fields = [
    "id", "name_en", "name_kh", "river", "lat", "lng",
    "alarm_stage_m", "flood_stage_m", "zero_gauge_elevation_m", "fallback_id",
]
with (HERE / "stations.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=station_fields)
    w.writeheader()
    w.writerows(stations["stations"])

# --- communes.geojson: boundaries joined with gazetteer attributes ---------
by_pcode = {f["properties"]["ADM3_PCODE"]: f for f in bounds["features"]}
joined, missing = [], []
for c in gaz["communes"]:
    feat = by_pcode.get("KH" + c["code"])
    if feat is None:
        missing.append(c["code"])
        continue
    joined.append({
        "type": "Feature",
        "properties": {k: c[k] for k in commune_fields if k not in ("lat", "lng")},
        "geometry": feat["geometry"],
    })

geojson = {
    "name": "cambodia_communes_mekongpulse",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "gazetteer_version": gaz["version"],
    "features": joined,
}
(HERE / "communes.geojson").write_text(
    json.dumps(geojson, ensure_ascii=False), encoding="utf-8"
)

print(f"communes.csv: {len(gaz['communes'])} rows")
print(f"stations.csv: {len(stations['stations'])} rows")
print(f"communes.geojson: {len(joined)} features ({len(missing)} communes lack boundaries)")
if missing:
    print("without boundary:", ",".join(missing[:20]), "..." if len(missing) > 20 else "")
