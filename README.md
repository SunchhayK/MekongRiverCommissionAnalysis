# MekongDS — from river forecast to farm action

Advanced Data Science final project, CADT MSc Term 4 (2026).

**Problem.** Flood forecasts stop at the river. Farmers need the next action. In 2022,
152,386 ha of farmland in Cambodia's Lower Mekong Basin flooded, 85,482 households were
affected and 15 people died. A lost wet-season rice crop costs US$405–500/ha, and with
~28% of indebted rural households having pledged land titles, crop loss becomes land loss.
Gauge forecasts alone don't tell a farmer whether to pump, drain, harvest early, or replant.

**What this project does.** Using the MekongPulse dataset export (MRC daily stage record
1992–2026 for 22 stations, live 15-minute telemetry, commune gazetteer with DEM elevations
and station assignments), it builds:

1. **EDA** of the Mekong stage regime — hydroclimatology, flood years, threshold exceedance,
   upstream→downstream wave propagation, Tonle Sap filling, wet-season clustering, trend tests.
2. **Extreme-value analysis** — GEV and peaks-over-threshold return levels per station,
   with bootstrap uncertainty.
3. **A multi-lead forecast model** (1–10 days) for stage at Kampong Cham, Prek Kdam and
   Koh Khel: LightGBM with upstream lag features, quantile forecasts, alarm-exceedance
   classifier, evaluated by expanding-window time-series CV against persistence,
   climatology and ridge baselines.
4. **A farm-action decision layer** — per-commune critical stage from gauge datum and DEM
   elevation, cost-loss action triggers (act when P(exceed) > C/L), and a retrospective
   case study on the October 2026 event that produced 23 DANGER alerts in production.

## Layout

```
dataset_export/          source data (read-only, not modified)
data/processed/          cleaned tables written by notebook 01
notebooks/
  01_preprocessing.ipynb
  02_eda.ipynb
  03_extreme_value.ipynb
  04_forecast_model.ipynb
  05_decision_layer.ipynb
src/mekongds/            shared library (loaders, metrics, statistics, plotting)
reports/figures/         all figures, PNG 150 dpi
REPORT.md                full write-up
```

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). Everything runs offline from `dataset_export/`:

```bash
uv sync
uv run jupyter execute notebooks/01_preprocessing.ipynb   # or open in Jupyter
uv run jupyter lab
```

## Data sources

All from the MekongPulse dataset export (see `dataset_export/README.md` for provenance):
MRC RFMMC daily water levels (22 stations, 1992–2026), MRC telemetry API snapshots,
Cambodia 2025 commune gazetteer (MEF DataEF), Copernicus DEM GLO-90 elevations.
No external data is fetched by this project.
