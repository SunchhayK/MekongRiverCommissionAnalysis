"""MekongDS: data loaders for the MekongPulse dataset export.

All data comes from `dataset_export/` (no network access anywhere in this package).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Live-network station id -> RFMMC short code (from stations_historical telemetry ids)
LIVE_TO_CODE = {"019802": "KOM", "020102": "PRE", "033401": "CHT", "033402": "KOH"}


def find_root() -> Path:
    """Locate the project root (the directory containing dataset_export/)."""
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for c in candidates:
        if (c / "dataset_export").is_dir():
            return c
    raise FileNotFoundError("Could not locate dataset_export/ from cwd or package location")


def data_dir() -> Path:
    return find_root() / "dataset_export"


def processed_dir() -> Path:
    d = find_root() / "data" / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir() -> Path:
    d = find_root() / "reports" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_stage() -> pd.DataFrame:
    """Daily stage (m above local gauge zero), tidy: date, station, stage, thresholds."""
    df = pd.read_csv(data_dir() / "historical" / "mrc_daily_all.csv", parse_dates=["date"])
    df = df.rename(columns={"water_level_m": "stage"})[
        ["date", "station", "stage", "floodStage", "alarmStage"]
    ]
    return df


def load_station_meta() -> pd.DataFrame:
    """Station registry in upstream->downstream order, with flood/alarm thresholds merged."""
    meta = pd.read_csv(data_dir() / "historical" / "stations_historical.csv")
    th = (
        load_stage().groupby("station")[["floodStage", "alarmStage"]].first().reset_index()
    )
    meta = meta.merge(th, on="station", how="left")
    # row order of stations_historical.csv is upstream -> downstream
    meta["order"] = range(len(meta))
    return meta


def hydro_order() -> list[str]:
    return load_station_meta()["station"].tolist()


def upstream_of(station: str, n: int | None = None) -> list[str]:
    """Stations upstream of `station` in the file's hydrologic order."""
    order = hydro_order()
    idx = order.index(station)
    up = order[:idx]
    return up if n is None else up[-n:]


def stage_wide(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Wide stage matrix (date x station) on a complete daily calendar index."""
    df = load_stage()
    if start is not None:
        df = df[df.date >= start]
    if end is not None:
        df = df[df.date <= end]
    w = df.pivot_table(index="date", columns="station", values="stage", aggfunc="first")
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="D"))
    w.index.name = "date"
    return w


def load_communes() -> pd.DataFrame:
    cm = pd.read_csv(
        data_dir() / "communes.csv",
        dtype={
            "code": str, "district_code": str, "province_code": str,
            "nearest_station_id": str, "district_en": str, "province_en": str,
        },
    )
    return cm


def load_live_hydro() -> pd.DataFrame:
    """15-min live telemetry, 2026-10-17..2026-10-25, station_id mapped to RFMMC code."""
    hy = pd.read_csv(
        data_dir() / "timeseries" / "hydro_observations.csv", parse_dates=["recorded_at"]
    )
    hy["station"] = hy.station_id.astype(int).astype(str).str.zfill(6).map(LIVE_TO_CODE)
    return hy


def load_alerts() -> pd.DataFrame:
    al = pd.read_csv(
        data_dir() / "timeseries" / "sentinel_alerts.csv",
        dtype={"commune_id": str, "station_id": str},
        parse_dates=["dispatched_at"],
    )
    al["station_id"] = al.station_id.str.zfill(6)
    return al


def load_weather_forecasts() -> pd.DataFrame:
    wf = pd.read_csv(
        data_dir() / "timeseries" / "weather_forecasts.csv",
        dtype={"province_code": str},
        parse_dates=["forecast_time", "fetched_at"],
    )
    return wf


def load_patrol_runs() -> pd.DataFrame:
    pr = pd.read_csv(data_dir() / "timeseries" / "sentinel_patrol_runs.csv", parse_dates=["run_at"])
    return pr


def load_live_station_thresholds() -> pd.DataFrame:
    """Thresholds + zero-gauge datums for the 4 live stations (from stations.csv)."""
    st = pd.read_csv(data_dir() / "stations.csv", dtype={"id": str, "fallback_id": str})
    st["id"] = st.id.str.zfill(6)
    st["station"] = st.id.map(LIVE_TO_CODE)
    return st
