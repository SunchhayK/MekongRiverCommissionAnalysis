# Changelog

## 0.1.0 — 2026-10-26

Added:
- `src/mekongds/` shared library: data loaders, forecast metrics (NSE, KGE,
  Brier, pinball, skill), Mann-Kendall + L-moments GEV/GPD extreme-value
  statistics, shared feature builder, matplotlib style.
- `notebooks/01_preprocessing` — QA and cleaning policy for all 8 source
  files; reliable-era mask (2007–2025 minus 2014 wet season); no-gap-fill
  policy documented.
- `notebooks/02_eda` — flood pulse climatology, flood-year hydrographs,
  upstream lag structure (Kratie→Kompong Cham R²=0.92), Tonle Sap memory
  (~15 d), Mann-Kendall trend tests, wet-season regime clustering.
- `notebooks/03_extreme_value` — GEV (MLE + L-moments) and POT/GPD return
  levels for 8 Cambodian stations with bootstrap CIs.
- `notebooks/04_forecast_model` — 1–10 day stage forecasts at KOM/PRE/KOH;
  expanding-window CV 2018–2025; persistence/climatology/ridge baselines;
  LightGBM point + quantile; conformal-recalibrated exceedance probabilities;
  ablation of the upstream network; 18 saved model bundles.
- `notebooks/05_decision_layer` — per-commune critical stages from DEM +
  gauge datum; residual-CDF commune exceedance probabilities; cost-loss
  action tiers (C/L = 0.22); ALREADY WET / RECEDE states; validation against
  the 23 production alerts of Oct 2026; 2019 retrospective replay.
- `REPORT.md` full write-up with results tables.
