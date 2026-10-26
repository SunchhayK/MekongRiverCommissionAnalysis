# From river forecast to farm action — a flood decision system for Cambodia's Lower Mekong

*Advanced Data Science final project, CADT MSc Term 4, October 2026.*

## 1. The problem

Flood forecasts stop at the river. A gauge bulletin says "Kompong Cham is at
14.9 metres and rising" — and a farmer 30 km from the gauge still has to guess
whether to pump, drain, harvest early, or wait. The stakes, from the problem
statement:

- 152,386 ha of farmland flooded in Cambodia's Lower Mekong Basin in 2022;
  85,482 households affected; 15 deaths.
- US$405–500/ha of wet-season rice production cost is at risk in a lost season.
- About 28% of indebted rural households have pledged land titles, so a crop
  loss can become land loss.

This project builds the missing layer between the river forecast and the farm
decision, using only the MekongPulse dataset export: the MRC daily stage record
for 22 stations (1992–2026), 1,661 communes with DEM elevations and nearest-gauge
assignments, station thresholds and gauge datums, plus 9 days of 15-minute live
telemetry and the 23 DANGER alerts a production deployment dispatched in
October 2026. No external data is fetched anywhere.

## 2. What was built

Five executed notebooks (jupytext-paired `.py` + `.ipynb`), a shared library
(`src/mekongds/`), 17 figures, and 18 trained model bundles:

| Stage | Notebook | Headline output |
| --- | --- | --- |
| Preprocessing | `01_preprocessing` | 8 clean tables in `data/processed/`, documented QA policy |
| EDA | `02_eda` | flood pulse, flood years, wave lags, trends, regimes |
| Extremes | `03_extreme_value` | GEV + POT return levels with bootstrap CIs |
| Forecasting | `04_forecast_model` | 1–10 day stage forecasts, quantile intervals, exceedance probabilities |
| Decision layer | `05_decision_layer` | per-commune flood probability → cost-loss action tiers |

## 3. Data and preprocessing decisions

The source record is strong but imperfect, and every imperfection got an
explicit policy rather than silent handling:

- **Reliable era = 2007–2025, minus the 2014 wet season** (zeroed at source).
  1992–2006 contain only reference wet years; they are kept but flagged
  `reliable=False` because annual statistics there would be wet-season-biased.
- **No gap interpolation.** Filling even a 3–4 day stage gap can erase a flood
  peak — the exact signal the project exists to detect. Models train only on
  rows whose lags exist; extreme-value statistics use a wet-season coverage
  filter (≥90%).
- **Negative stages at the delta stations are real** (tides below gauge zero)
  and stay. None of the target stations are affected.
- **Zero-elevation communes are real** Tonle Sap floodplain communes, not DEM
  artifacts — and they are the most flood-exposed ones.
- **No rainfall features exist anywhere in the export** (the live telemetry's
  rainfall/temperature columns are entirely null; weather-forecast snapshots
  cover only the 9-day deployment window). The models are therefore built on
  river-stage dynamics alone, and the report treats rainfall-driven runoff as
  an unmodeled driver absorbed (partly) by the quantile forecasts, not as a
  solved problem.

## 4. EDA findings that drive the design

1. **The flood wave is fast and observable.** Kratie's stage anomaly explains
   92% of Kompong Cham's next-day anomaly variance (R²=0.92); Stung Treng at a
   3-day lag explains 86%. Celerity through Cambodia is roughly 60–80 km/day.
   This is the exploitable lead: upstream gauges are the forecast.
2. **The Tonle Sap has two weeks of memory.** Prek Kdam lags Kompong Cham by
   ~15 days at r=0.957, making it the most forecastable station in the system.
3. **Flood years are discrete and recent.** Kompong Cham crossed its alarm
   stage (15.2 m) in 2000, 2011, 2013, 2018 and 2019 — and never reached its
   16.2 m flood stage in the record (max 16.02 m, 2011).
4. **No trend in annual maxima** over the reliable era (Mann-Kendall, all
   Cambodian stations p>0.45). With 18–19 years the tests have limited power,
   but a stationary model is not obviously wrong.
5. **2022 was not a mainstream flood.** Kompong Cham peaked at 13.81 m — 1.4 m
   *below* alarm. The 2022 damage in the problem statement came from tributary
   flash floods and local rainfall (typhoon Noru, September storms) that no
   mainstream gauge can see. This bounds what any model built on this data can
   claim, and the decision layer says so in its output.

## 5. Extreme-value analysis

GEV fitted by MLE and L-moments, plus POT/GPD with 7-day declustering at the
98th-percentile threshold, all with wet-season-coverage filtering:

| Station | RL10 (m) | RL100 (m) | Alarm ≈ return period | Observed max |
| --- | --- | --- | --- | --- |
| Kratie | 22.7 | 22.9 | — | 22.88 (2011) |
| Kompong Cham | 15.8 | 16.0 | alarm ≈ 5 yr, flood ≈ 100 yr | 16.02 (2011) |
| Neak Luong | 7.5 | 8.1 | alarm ≈ 10 yr | 8.06 (2011) |
| Prek Kdam | 9.3 | 10.2 | alarm ≈ 10 yr, flood ≈ 50 yr | 10.19 (2011) |
| Koh Khel | 7.8 | 7.9 | alarm ≈ 100 yr | 7.89 (2011) |

The three estimators agree within centimetres at T≤20. Bootstrap 90%
intervals at T=100 are wide (e.g. Prek Kdam 9.1–11.2 m): return levels are
design numbers, not forecasts, which is exactly why the decision layer
consumes forecast probabilities instead.

## 6. Forecast model

**Setup.** LightGBM (point + quantile) and ridge, with own-station lags,
rollings and rises; upstream stations at EDA-derived lags; day-of-year Fourier
terms. Targets: Kompong Cham, Prek Kdam, Koh Khel. Validation: expanding-window
CV, wet seasons 2018–2025 one fold per year, no shuffling, all statistics
(climatology, imputation, residual distributions) refit inside each fold.

**Results (mean MAE, metres, across 8 folds):**

| Lead | KOM lgbm | KOM ridge | KOM persistence | PRE lgbm | PRE ridge | KOH lgbm |
| --- | --- | --- | --- | --- | --- | --- |
| 1 d | 0.148 | 0.081 | 0.192 | 0.067 | 0.049 | 0.064 |
| 3 d | 0.476 | 0.362 | 0.519 | 0.191 | 0.158 | 0.186 |
| 5 d | 0.747 | 0.614 | 0.783 | 0.314 | 0.273 | 0.297 |
| 7 d | 0.919 | 0.806 | 1.012 | 0.405 | 0.368 | 0.383 |
| 10 d | 1.134 | 1.028 | 1.322 | 0.519 | 0.482 | 0.468 |

Stated honestly: **ridge beats LightGBM on point MAE everywhere** — with
well-constructed lag features the routing problem is close to linear, and the
linear model's variance advantage wins at ~2,300 training rows per fold.
LightGBM is retained because it produces the machinery the decision layer
actually needs: a full residual distribution for exceedance probabilities at
*any* threshold, not just one point track. It beats persistence at every
station and lead (skill scores 0.14–0.48 vs persistence; both beat
climatology by wide margins).

**Uncertainty.** Direct quantile-GBM intervals under-cover badly (0.52–0.75
against 0.90 nominal) — the pinball loss lets both quantile heads drift toward
the median in small samples. A conformal-style recalibration (point forecast
plus inner-validation residual quantiles) restores coverage to ≈0.88 across
stations and leads. Every probability the decision layer uses comes from this
recalibrated residual CDF.

**Exceedance forecasts.** Residual-CDF probabilities are strongly reliable at
short leads (Brier skill score vs climatology: 0.97 at KOM lead-1, 0.83–0.93
at lead 2–3 across stations), degrading with lead as they should. At KOM the
alarm-stage events are rare in the test years (3.75 events/fold), so lead-5+
BSS turns negative — the model is honest enough to say "I don't know" beyond
about 5 days at mainstream alarm level, while Prek Kdam's long lake memory
keeps BSS at 0.65 even at 10 days.

**Ablation.** Removing the upstream network costs 10–20 cm MAE at KOM by
lead 5–10 (`fig04_ablation`): the upstream gauges carry most of the
long-lead signal that own-station persistence cannot.

## 7. The decision layer

**Exposure.** Each commune's critical stage is
`DEM elevation − gauge datum + 0.5 m margin`, valid for the 1,116 communes
assigned to Prek Kdam, Chaktomuk or Koh Khel (Kompong Cham's datum is
unpublished — 428 communes get forecasts but not absolute triggers). The
station alarm stages sit inside the bulk of their communes' critical-stage
distributions: a single gauge alert is structurally too coarse.

**Economics.** Act when P(critical stage exceeded within lead) > C/L, with
C = US$100/ha (emergency harvest, pumping, moving stores) and L = US$452/ha
(midpoint of the 405–500 production cost at risk), giving a trigger of 0.22.
A C/L sweep from 0.10 to 0.50 shows the trigger is stable on a recession day.

**Tiers.** ACT (harvest/pump) at P≥0.22 with ≤3 d lead; prepare at 4–7 d;
monitor beyond — plus two states the production system lacked: ALREADY WET
(commune underwater at issue time; hold, don't alert) and RECEDE (drain/pump
as water drops).

## 8. Validation against the production system

The October 2026 deployment dispatched 23 DANGER "Emergency harvest
acceleration" alerts on Oct 21–22. The data shows what those alerts were:

- All 17 Koh Khel alert communes sit at exactly 4.0 m elevation; the river
  stood at 4.06 m absolute on Oct 21. **They were already underwater when
  alerted.** All 23 alerts went out with `lead_time_hours = 0`, during a
  steady recession, at zero lead time.
- The decision layer, issued from the same network state (Oct 20), produces:
  zero ACT triggers for dry communes at any C/L, 70 communes correctly marked
  ALREADY WET/RECEDE instead of alarmed, and a confident recession forecast
  (the models' 90% intervals contain the observed October fall).

**Retrospective replay, 2019 wet season** (demonstration of the logic on a
real rising limb, using final models): commune-level triggers fire days before
the gauge crosses its alarm stage, and they fire for a subset of communes —
the ones whose fields actually sit below the coming water — rather than as a
blanket station alert. First ACT trigger: Koh Khel communes from July 1,
Prek Kdam from July 28.

| | Production (Oct 2026) | This decision layer |
| --- | --- | --- |
| Trigger | level vs elevation at sync time | forecast P(exceed) vs cost-loss ratio |
| Lead time | 0 h, alerts during recession | 1–10 days with actionable tiers |
| Spatial | blanket DANGER, incl. already-wet communes | per-commune critical stage |
| Already-flooded communes | alerted again | hold/drain directive |
| Economics | none | US$/ha at risk; land-loss context |

## 9. Limitations

1. **Mainstream-only coverage.** The 2022 disaster was tributary flash flood
   and rainfall driven; this system addresses mainstream floodplain flooding.
   Extending to tributaries needs rainfall and tributary gauges.
2. **No rainfall features.** On a rain-driven rising limb the forecasts lag
   reality; the residual CDF widens honestly but the point track can miss.
3. **One DEM point per commune.** The 0.5 m margin partially absorbs within-
   commune variation; a field-scale DEM would sharpen triggers.
4. **Kompong Cham's missing datum** excludes 428 communes from absolute
   triggers until the gauge datum is surveyed.
5. **Rare-event validation.** Only 4 alarm years in 8 test folds at KOM;
   BSS at long leads is unreliable with so few events. More seasons (or
   stochastic resampling of the flood wave) would tighten this.
6. **Residual bank approximation.** Live inference uses in-sample residuals
   widened by the CV width ratio; per-fold residual CDFs are the honest
   version and the coverage table reflects them.

## 10. Reproducing

```bash
uv sync
uv run jupyter execute --inplace notebooks/01_preprocessing.ipynb
uv run jupyter execute --inplace notebooks/02_eda.ipynb
uv run jupyter execute --inplace notebooks/03_extreme_value.ipynb
uv run jupyter execute --inplace notebooks/04_forecast_model.ipynb   # ~10 min, trains 18 models
uv run jupyter execute --inplace notebooks/05_decision_layer.ipynb   # loads bundles, no training
```

All data comes from `dataset_export/` (provenance in `dataset_export/README.md`).
Figures land in `reports/figures/`, model bundles in `models/`, intermediate
tables in `data/processed/`.
