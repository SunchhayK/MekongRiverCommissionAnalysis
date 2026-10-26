# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %% [markdown]
# # 01 — Preprocessing and data quality
#
# **MekongDS: from river forecast to farm action.**
#
# This notebook turns the raw `dataset_export/` files into clean, analysis-ready tables:
#
# - daily stage per station on a complete calendar, with a documented reliability mask
# - commune exposure attributes (elevation, station assignment, gauge datum)
# - live 15-minute telemetry for the 4 live stations
# - the 23 production DANGER alerts joined to commune metadata
#
# Every cleaning decision is stated explicitly and justified. Outputs go to
# `data/processed/` as parquet for the downstream notebooks.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from mekongds import data
from mekongds.plotting import set_style

set_style()
pd.set_option("display.width", 160)

RAW = data.data_dir()
OUT = data.processed_dir()

# %% [markdown]
# ## 1. Daily stage record (MRC RFMMC, 22 stations, 1992–2026)
#
# Raw file: `historical/mrc_daily_all.csv`, 172,829 rows, one per station-day,
# stage in meters above local gauge zero.

# %%
stage = data.load_stage()
print(f"rows={len(stage):,}  stations={stage.station.nunique()}  "
      f"range {stage.date.min().date()} → {stage.date.max().date()}")

# %% [markdown]
# ### 1.1 Integrity checks
#
# Checks: duplicate keys, nulls, impossible values, threshold consistency,
# calendar coverage. Results feed the cleaning policy below.

# %%
qa = {}
qa["duplicate (station, date) keys"] = stage.duplicated(["station", "date"]).sum()
qa["null stage values"] = stage.stage.isna().sum()
qa["negative stage values"] = (stage.stage < 0).sum()
th = stage.groupby("station")[["floodStage", "alarmStage"]].nunique()
qa["stations with inconsistent thresholds"] = int((th > 1).any(axis=1).sum())
qa = {k: int(v) for k, v in qa.items()}
qa

# %%
neg = stage[stage.stage < 0].groupby("station").size().sort_values(ascending=False)
neg

# %% [markdown]
# **Findings.**
#
# - No duplicates, no nulls, thresholds consistent for all 22 stations. The export is
#   already well-formed at the row level.
# - 334 negative stages, all at the Vietnamese delta stations (TCH, CDO) and one
#   Lao station (SAV). These are deltas relative to a local gauge zero; tides push
#   the delta stations below gauge zero in the dry season. **Decision: keep them.**
#   They are real, and none of our target stations are affected.

# %% [markdown]
# ### 1.2 Calendar coverage and the reliability mask
#
# The record is not uniformly complete: 1992–2000 contain only reference wet years
# (~214 days), 2001–2006 are partial, the 2014 wet season is missing at source
# (the RFMMC 2014 wet column was zeroed), and the last year is in progress.
# Annual-maxima and trend analyses must know this.

# %%
w = data.stage_wide()
cov = w.notna().groupby(w.index.year).mean()
cov.iloc[:, :11].round(2).style.background_gradient(cmap="Blues", vmin=0, vmax=1)

# %%
months_2014 = sorted(stage[stage.date.dt.year == 2014].date.dt.month.unique())
f"months present in 2014: {months_2014}  (Jun–Oct wet season missing at source)"

# %% [markdown]
# **Coverage policy.**
#
# - **Reliable era = 2007-01-01 → 2025-12-31**, minus the 2014 wet season
#   (2014-06-01 → 2014-10-31). In this window every station has near-complete
#   daily coverage (see matrix above; only TCH/CDO have minor gaps).
# - 1992–2006 rows are kept in the tidy table but flagged `reliable=False`:
#   annual statistics there would be biased toward the wet season because the
#   dry-season rows are missing in the reference years.
# - 2026 is in progress and is treated as a test/case-study year only.

# %%
stage_tidy = stage.copy()
stage_tidy["reliable"] = (
    (stage_tidy.date >= "2007-01-01")
    & (stage_tidy.date <= "2025-12-31")
    & ~((stage_tidy.date >= "2014-06-01") & (stage_tidy.date <= "2014-10-31"))
)
stage_tidy["reliable"].value_counts()

# %% [markdown]
# ### 1.3 Gap structure within the reliable era
#
# How many missing days remain inside the reliable era, and are they isolated
# (interpolation is defensible) or block gaps (interpolation would fabricate a flood wave)?

# %%
rel = w.loc["2007":"2025"]
gaps = rel.isna().sum().sort_values(ascending=False)
gaps

# %%
# longest run of consecutive missing days per station inside the reliable era
def longest_gap(s: pd.Series) -> int:
    isnan = s.isna().to_numpy()
    best = cur = 0
    for v in isnan:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

longest = {c: longest_gap(rel[c]) for c in rel.columns}
pd.Series(longest).sort_values(ascending=False).head(8)

# %% [markdown]
# Missing days inside the reliable era are a few hundred per station at most and
# runs are short (a handful of days), except TCH/CDO where they are larger but
# those are delta reference stations, not forecast targets.
#
# **Interpolation policy.** We do **not** fill gaps in the stage matrix. Model
# training uses feature rows only where the needed lags exist (LightGBM also
# tolerates missing values natively), and annual-maxima statistics use a
# coverage filter instead (notebook 03). Interpolating stage across even 3–4
# days can erase the peak of a flood wave, which is exactly the signal we care
# about, so we prefer fewer, truthful rows to a smooth, partly invented series.

# %%
w.to_parquet(OUT / "stage_wide.parquet")
stage_tidy.to_parquet(OUT / "stage_tidy.parquet", index=False)

# %% [markdown]
# ## 2. Station metadata and hydrologic order
#
# `stations_historical.csv` lists stations upstream → downstream. We attach the
# live-network telemetry ids and the flood/alarm thresholds, and define the
# station set used by the forecast model (Cambodian stations, the ones whose
# communes are in the gazetteer).

# %%
meta = data.load_station_meta()
meta

# %%
# The live network maps to these historical codes
LIVE = {"KOM": "019802", "PRE": "020102", "CHT*": "033401", "KOH": "033402"}
# *Chaktomuk (live 033401) has no historical twin; Phnom Penh Port (PPP) is the
# nearest historical gauge at the same confluence and stands in for it.
live_st = data.load_live_station_thresholds()
live_st[["station", "name_en", "river", "alarm_stage_m", "flood_stage_m", "zero_gauge_elevation_m"]]

# %% [markdown]
# **Gauge datums.** `zero_gauge_elevation_m` is published for PRE (+0.08 m),
# Chaktomuk (−1.02 m) and Koh Khel (−1.00 m), but missing for Kompong Cham.
# This matters for the decision layer: converting a gauge stage to an absolute
# water elevation (and hence to "does this commune flood") is only possible
# where the datum is known. The commune mapping in notebook 05 therefore covers
# PRE, CHT and KOH (1,116 communes) and excludes KOM.

# %%
meta.to_parquet(OUT / "station_meta.parquet", index=False)

# %% [markdown]
# ## 3. Commune exposure attributes
#
# 1,661 communes with coordinates, DEM elevation (Copernicus GLO-90 at the
# representative point), nearest MRC station and distance. Cleaning: enforce
# string PCODEs, flag the 117 communes without boundaries/station assignment,
# and validate the join to station metadata.

# %%
cm = data.load_communes()
print(f"rows={len(cm)}  provinces={cm.province_code.nunique()}")
cm.head(3)

# %%
qa_cm = {
    "null nearest_station_id": int(cm.nearest_station_id.isna().sum()),
    "null ground_elevation_m": int(cm.ground_elevation_m.isna().sum()),
    "zero ground_elevation_m": int((cm.ground_elevation_m == 0).sum()),
    "duplicate codes": int(cm.code.duplicated().sum()),
}
qa_cm

# %%
# Zero-elevation communes: legitimate (Tonle Sap floodplain islands/marshes) or bad DEM samples?
zero = cm[cm.ground_elevation_m == 0]
print(zero[["name_en", "district_en", "province_en", "nearest_station_id"]]
      .groupby("province_en").size().sort_values(ascending=False).to_string())

# %% [markdown]
# Zero elevations cluster around the Tonle Sap lake and the Bassac floodplain
# (Kampong Thom, Pursat, Battambang, Kampong Chhnang). These are real wetland
# communes sitting at or near sea level, not DEM artifacts. **Decision: keep.**
# They are, in fact, the most flood-exposed communes in the country.

# %%
cm["has_station"] = cm.nearest_station_id.notna()
cm_by_station = cm.groupby("nearest_station_id").size().sort_values(ascending=False)
cm_by_station

# %%
# join station names + datums for the exposure table
st = live_st[["id", "station", "alarm_stage_m", "flood_stage_m", "zero_gauge_elevation_m"]]
cm_exp = cm.merge(st, left_on="nearest_station_id", right_on="id", how="left")
cm_exp.to_parquet(OUT / "communes_clean.parquet", index=False)
cm_exp[cm_exp.station.notna()].groupby("station").agg(
    communes=("code", "count"),
    elev_p10=("ground_elevation_m", lambda s: s.quantile(0.10)),
    elev_median=("ground_elevation_m", "median"),
)

# %% [markdown]
# ## 4. Live telemetry (15-minute, 4 stations)
#
# `hydro_observations.csv`: 2,628 rows, 2026-10-17 → 2026-10-25. Columns
# `rainfall_mm` and `temperature_c` are entirely null in this export; `source`
# is all `live` (no stale fallbacks). Rate of rise is derivable from stage, so
# we recompute it as a consistency check.

# %%
hy = data.load_live_hydro()
print(f"rows={len(hy)}  stations={sorted(hy.station.unique())}  "
      f"{hy.recorded_at.min()} → {hy.recorded_at.max()}")
hy[["water_level_m", "rate_of_rise_m_per_hr"]].describe().round(3)

# %%
# consistency: reported rate_of_rise vs recomputed from consecutive observations
hy2 = hy.sort_values(["station", "recorded_at"]).copy()
hy2["dt_hr"] = hy2.groupby("station").recorded_at.diff().dt.total_seconds() / 3600
hy2["stage_diff"] = hy2.groupby("station").water_level_m.diff()
hy2["ror_recomputed"] = hy2.stage_diff / hy2.dt_hr
chk = hy2.dropna(subset=["ror_recomputed", "rate_of_rise_m_per_hr"])
err = (chk.ror_recomputed - chk.rate_of_rise_m_per_hr).abs()
f"median abs diff reported vs recomputed rate-of-rise: {err.median():.5f} m/h over {len(chk)} rows"

# %%
# 15-min wide stage matrix
live_wide = hy.pivot_table(index="recorded_at", columns="station", values="water_level_m")
live_wide = live_wide.reindex(
    pd.date_range(live_wide.index.min(), live_wide.index.max(), freq="15min")
)
live_wide.index.name = "recorded_at"
live_wide.isna().sum()

# %%
# map live codes to their historical counterparts for model compatibility
CODEMAP = {"KOM": "KOM", "PRE": "PRE", "KOH": "KOH", "CHT": "PPP"}
live_hist = live_wide.rename(columns=CODEMAP)
live_hist = live_hist.resample("D").mean()
live_hist.round(3).tail(6)

# %%
live_wide.to_parquet(OUT / "live_hydro_15min.parquet")
live_hist.to_parquet(OUT / "live_hydro_daily.parquet")

# %% [markdown]
# **Note for downstream.** The live window (Oct 17–25, 2026) is a slow recession:
# all four stations are falling gently. Production dispatched 23 DANGER alerts
# on Oct 21–22 during this recession, which notebook 05 examines critically.

# %% [markdown]
# ## 5. Production alerts (the system we improve on)
#
# 23 DANGER-tier "Emergency harvest acceleration" bulletins dispatched
# Oct 21–22, 2026, each tied to a commune and a station. Join to commune
# metadata for the comparison in notebook 05.

# %%
al = data.load_alerts()
al["station"] = al.station_id.map({"033402": "KOH", "020102": "PRE", "019802": "KOM", "033401": "CHT"})
al_enriched = al.merge(
    cm[["code", "name_en", "district_en", "province_en", "ground_elevation_m",
        "nearest_station_id", "distance_to_station_km"]],
    left_on="commune_id", right_on="code", how="left",
)
al_enriched[["dispatched_at", "station", "name_en", "province_en",
             "ground_elevation_m", "distance_to_station_km"]].head(8)

# %%
print(al_enriched.groupby(["station", "province_en"]).size().to_string())

# %%
al_enriched.to_parquet(OUT / "alerts_enriched.parquet", index=False)

# %% [markdown]
# ## 6. Weather forecast snapshots — scope decision
#
# `weather_forecasts.csv` holds province-point precipitation/soil-moisture
# forecast snapshots, but only for 2026-10-17 → 2026-10-25 (the deployment
# window). There is no historical weather covariate anywhere in the export.

# %%
wf = data.load_weather_forecasts()
print(f"rows={len(wf)}  provinces={wf.province_code.nunique()}  "
      f"range {wf.forecast_time.min():%Y-%m-%d} → {wf.forecast_time.max():%Y-%m-%d}")

# %% [markdown]
# **Scope decision.** The forecast model (notebook 04) is therefore built on
# river-stage dynamics alone: upstream lags, seasonality and memory. This is the
# classic river-routing setting and is honest about what the data supports.
# Rainfall-driven runoff is the main unmodeled driver, and the quantile
# forecasts absorb some of that uncertainty. We state this as a limitation
# rather than pretending rainfall features exist.

# %%
wf.to_parquet(OUT / "weather_forecasts.parquet", index=False)

# %% [markdown]
# ## 7. Summary of outputs
#
# | File | Content |
# | --- | --- |
# | `stage_wide.parquet` | daily stage, date × 22 stations, complete calendar, NaN gaps preserved |
# | `stage_tidy.parquet` | tidy stage with `reliable` flag (2007–2025 minus 2014 wet) |
# | `station_meta.parquet` | 22 stations, order, thresholds, telemetry ids |
# | `communes_clean.parquet` | 1,661 communes + station join, datum, thresholds |
# | `live_hydro_15min.parquet` | 15-min stage, 4 live stations, 2026-10-17→25 |
# | `live_hydro_daily.parquet` | daily means, live codes mapped to historical codes |
# | `alerts_enriched.parquet` | 23 DANGER alerts + commune metadata |
# | `weather_forecasts.parquet` | province forecast snapshots (deployment window only) |
#
# **Cleaning decisions recap.**
#
# 1. Negative delta-station stages kept (real tidal signal below gauge zero).
# 2. Reliable era = 2007–2025 minus the 2014 wet season; earlier years flagged, not deleted.
# 3. No gap interpolation anywhere; analyses filter by coverage instead.
# 4. Zero-elevation communes kept (real Tonle Sap floodplain communes).
# 5. Chaktomuk (live) ↔ Phnom Penh Port (historical) stand-in documented.
# 6. No rainfall features anywhere: none exist in the export; models are stage-dynamics only.
