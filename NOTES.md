# NOTES — Mekong flood ADS final project (working notes, delete before submission)

## Objective
Course project for Advanced Data Science (CADT MSc, Term 4). Problem: flood forecasts stop at the river, farmers need the next action. Build EDA + models from `dataset_export/` ONLY (no new data fetching).

## Data facts (verified)
- `historical/mrc_daily_all.csv`: 172,829 rows, 22 MRC stations, 1992-06-01 → 2026-10-25, daily stage m above local gauge zero. No nulls, no dups. Negative stages at TCH(166)/CDO(154)/SAV(14) — delta tide, plausible, keep. 2014 wet season missing at source (Jun–Oct absent, only dry months present). 1992–2000 partial (reference years only, ~214 days). 2007+ near-complete.
- Thresholds per station in file (floodStage/alarmStage) consistent. KOM flood 16.2/alarm 15.2. PRE 10.0/9.5. KOH 8.4/7.9. NEA 8.0/7.5.
- KOM alarm exceedance years: 2000(28d), 2011(16), 2013(10), 2018(22), 2019(8). KOM never crossed flood stage 16.2 in record (max 16.02 in 2011).
- PRE alarm years: 2000(53d), 2011(41). KOH alarm: none in record.
- PRE lags KOM by ~15 days, r=0.957 (Tonle Sap filling).
- 2022 at Cambodian stations: KOM max 13.81 (below alarm!) — 2022 damage was tributary/pluvial, mainstream moderate. Honest limitation to state.
- `stations.csv` zero-gauge elevations: KOM missing, PRE +0.08, CHT -1.02, KOH -1.0. → absolute-elevation commune mapping works for PRE/KOH/CHT (1,116 communes), not KOM.
- `communes.csv`: 1,661 rows; 117 have no nearest_station (no boundary). nearest_station: PRE 541, KOM 428, KOH 399, CHT 176.
- `timeseries/`: hydro_observations 2,628 rows, 4 stations, 15-min, 2026-10-17→10-25; rainfall/temp ALL NULL. weather_forecasts: province snapshots Oct 2026 only. sentinel_alerts: 23 DANGER alerts Oct 21–22 2026, "Emergency harvest acceleration", stations KOH(17)/PRE(6), commune_ids join to communes.csv. patrol_runs 675.
- Live telemetry: stations falling steadily Oct 17–25 (KOM 6.9→6.0, KOH 5.45→4.90). Alerts fired Oct 21–22 as levels fell — production rule presumably used rate_of_rise + forecast; check alert communes' elevation profile.

## Decisions
- Forecast features: stage lags (own + upstream stations by hydrologic order), day-of-year Fourier, rolling means, rate of change. NO rainfall (null/unavailable historically). SARIMA dropped in favor of persistence/climatology/ridge baselines + LightGBM point & quantile + exceedance classifier.
- Targets for models: KOM, PRE, KOH (the stations behind the Oct 2026 alerts + richest records).
- CV: expanding window, 5 folds, wet-season focus, no shuffling. Metrics: MAE, RMSE, NSE, KGE, persistence skill.
- EVA: GEV (MLE + L-moments) on annual maxima, POT-GPD declustered, bootstrap CIs, Cambodian stations.
- Decision layer: critical stage per commune = ground_elevation - zero_gauge (+ margin param). Cost-loss trigger: act if P(exceed) > C/L, C=$100/ha emergency harvest, L=$452/ha (midpoint 405–500). Sensitivity 0.1–0.5.
- Case study: forecast from 2026-10-20 features, compare commune trigger set vs the 23 production alerts.

## Structure
- `src/mekongds/` shared lib (data load, metrics NSE/KGE, Mann-Kendall, L-moments GEV, plotting style)
- `notebooks/01..05` (ipynb executed + py percent source in notebooks/src/)
- `reports/figures/`, `REPORT.md`, README, CHANGELOG

## Status
- [x] Env scaffolded, deps resolved
- [ ] nb01 preprocessing
- [ ] nb02 EDA
- [ ] nb03 EVA
- [ ] nb04 model
- [ ] nb05 decision layer
- [ ] report + README

## Next step
Write src/mekongds lib, then notebook 01.
