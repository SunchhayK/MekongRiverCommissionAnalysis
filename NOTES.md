# NOTES — Mekong flood ADS final project (working notes)

Status: COMPLETE (2026-10-26). Delete this file before submission if unwanted.

## Delivered
- 5 executed notebooks (01 preprocessing, 02 EDA, 03 EVA, 04 forecast, 05 decision layer), paired .py/.ipynb via jupytext.
- src/mekongds: data.py, metrics.py, stats_ext.py (GEV L-moments: scipy c == Hosking k, same sign), forecast.py (shared build_features), plotting.py.
- REPORT.md, README.md, CHANGELOG.md (0.1.0), 17 figures, 18 model bundles in models/, 8 clean tables in data/processed/.

## Key results
- CV MAE lgbm KOM: 0.148 (1d) → 1.134 (10d); ridge beats lgbm on point MAE everywhere (stated honestly in report).
- Persistence skill 0.14–0.48; conformal coverage ≈0.88 vs 0.60 for raw quantile-GBM.
- Decision layer vs production Oct 2026: production alerted 17 already-submerged communes at 0 h lead during recession; decision layer issues 0 ACT for dry communes, marks ALREADY WET/RECEDE instead.
- 2019 replay: commune triggers fire before gauge alarm (KOH from Jul 1, PRE from Jul 28).

## Gotchas (if re-running)
- jupyter execute needs --inplace to save outputs.
- nb04 takes ~10 min (cache-fold design; don't restructure without keeping lru_cache).
- communes_clean.parquet merge must not include station name_en (collides with commune name_en → _x/_y).
- action_tier needs currently_flooded flag; retro and sweep exclude already-wet communes.
- LightGBM verbose=-1 set in make_lgbm and the classifier.

## Remaining polish (optional)
- notebook 05 §8: bulletin example uses 2019-09-15 issue (synthetic rising) — p_exceed=100% for top communes shows "prepare" not ACT because lead 5 > 3; could add a lead-3 variant for the ACT branch.
- geopandas/communes.geojson unused; could add a map figure if desired.
