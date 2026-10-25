#!/usr/bin/env python3
"""Scrape the MRC RFMMC hydrograph backend for multi-year daily water levels.

ffw.mrcmekong.org renders amCharts hydrographs from fetchwet_new.php (wet
season Jun-Oct, year-by-year columns) and fetchdry_new.php (dry season
Nov-May, Min/AVG/Max envelope + per-year columns). The payloads are JS object
literals, not JSON, so this parser handles unquoted keys and numeric-only keys
("1992") directly.

Usage (from projects/): python3 dataset_export/scrape_mrc_history.py
Outputs CSVs under dataset_export/historical/.
"""
import csv
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "historical"
BASE = "https://ffw.mrcmekong.org"

# FFWC short codes for all 22 RFMMC stations, with our telemetry IDs where they
# overlap the MekongPulse watch set.
STATIONS = {
    "CSA": ("Chiang Saen", None), "CKH": ("Chiang Khan", None),
    "LUA": ("Luang Prabang", None), "NON": ("Nongkhai", None),
    "PAK": ("Paksane", None), "NAK": ("Nakhon Phanom", None),
    "THA": ("Thakhek", None), "SAV": ("Savannakhet", None),
    "MUK": ("Mukdahan", None), "PKS": ("Pakse", None),
    "STR": ("Stung Treng", None), "KRA": ("Kratie", None),
    "KOM": ("Kompong Cham", "019802"),
    "NEA": ("Neak Luong", None), "PPP": ("Phnom Penh Port", None),
    "PPB": ("Phnom Penh (Bassac)", "033401"),
    "KOH": ("Koh Khel", "033402"),
    "PRE": ("Prek Kdam", "020102"),
    "TCH": ("Tan Chau", None), "CDO": ("Chau Doc (Bassac)", None),
    "VIE": ("Vientiane", None), "KHO": ("Khong Chiam", None),
}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "mekongpulse-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_js_objects(text: str) -> list[dict]:
    """Parse the amCharts JS array-of-object-literals payload."""
    body = text.strip().lstrip("[").rstrip(",]").strip()
    rows: list[dict] = []
    # each object starts with {date_gmt:"YYYY-MM-DD"
    for chunk in re.finditer(r"\{date_gmt:\"([0-9-]+)\"([^}]*)\}", body):
        date, rest = chunk.group(1), chunk.group(2)
        row: dict = {"date": date}
        for m in re.finditer(r'([A-Za-z0-9_.]+):(-?[0-9.]+)', rest):
            key, val = m.group(1), float(m.group(2))
            row[key] = None if val == 0.0 and key not in ("floodStage", "alarmStage") else val
        rows.append(row)
    return rows


def collapse_wet(rows: list[dict]) -> list[dict]:
    """One row per (year, date): wet payload repeats the same MM-DD per year col."""
    out = []
    for row in rows:
        date = datetime.strptime(row["date"], "%Y-%m-%d")
        meta = {k: v for k, v in row.items() if not _is_year(k) and k not in ("date",)}
        for k, v in row.items():
            if _is_year(k) and v is not None:
                out.append({
                    "date": f"{k}-{date:%m-%d}",
                    "year": int(k),
                    "water_level_m": v,
                    **meta,
                })
    out.sort(key=lambda r: r["date"])
    return out


def _is_year(k: str) -> bool:
    return k.isdigit() and 1980 <= int(k) <= 2100


def collapse_dry(rows: list[dict]) -> list[dict]:
    """Dry payload: rows dated of the current season, per-year columns like 1992.93."""
    out = []
    for row in rows:
        try:
            date = datetime.strptime(row["date"], "%Y-%m-%d")
        except ValueError:
            # site emits impossible calendar dates (e.g. Feb 29 in non-leap
            # years) because rows are indexed by day-of-season; drop them
            continue
        meta = {k: v for k, v in row.items() if not _is_dry_year(k) and k not in ("date",)}
        season_keys = [k for k in row if _is_dry_year(k) and row[k] is not None]
        if not season_keys:
            continue
        for k in season_keys:
            start_year = int(k.split(".")[0])
            # dry season spans Nov(YYYY)..May(YYYY+1): normalize to calendar year
            cal_year = start_year if date.month >= 11 else start_year + 1
            out.append({
                "date": f"{cal_year}-{date:%m-%d}",
                "year": cal_year,
                "water_level_m": row[k],
                **meta,
            })
    out.sort(key=lambda r: r["date"])
    return out


def _is_dry_year(k: str) -> bool:
    return re.fullmatch(r"(19|20)\d{2}\.\d{2}", k) is not None


def main() -> None:
    OUT.mkdir(exist_ok=True)
    wet_rows: list[dict] = []
    dry_rows: list[dict] = []
    for code, (name, tele_id) in STATIONS.items():
        for kind, url in (
            ("wet", f"{BASE}/fetchwet_new.php?StCode={code}"),
            ("dry", f"{BASE}/fetchdry_new.php?StCode={code}"),
        ):
            try:
                payload = fetch(url)
                rows = parse_js_objects(payload)
            except Exception as exc:  # keep going; a missing station is not fatal
                print(f"{code} {kind}: FAILED {exc}")
                continue
            collapsed = collapse_wet(rows) if kind == "wet" else collapse_dry(rows)
            for r in collapsed:
                r["station"] = code
                r["station_name"] = name
                r["telemetry_id"] = tele_id or ""
            (wet_rows if kind == "wet" else dry_rows).extend(collapsed)
            print(f"{code} {kind}: {len(collapsed)} rows "
                  f"({collapsed[0]['date']} .. {collapsed[-1]['date']})" if collapsed else f"{code} {kind}: 0 rows")
            time.sleep(1.0)

    for fname, rows in (("mrc_daily_wet.csv", wet_rows), ("mrc_daily_dry.csv", dry_rows)):
        path = OUT / fname
        fields = ["date", "station", "station_name", "telemetry_id", "year",
                  "water_level_m", "floodStage", "alarmStage"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"{path.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
