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
# # 03 — Extreme-value analysis
#
# **MekongDS: from river forecast to farm action.**
#
# Return levels answer the design question behind the decision layer: how rare
# is a given stage, and what should a commune's drainage or levee be built for?
#
# Methods:
#
# - **GEV on annual maxima** (block maxima), fitted two ways: MLE and
#   L-moments. L-moments are more robust in small samples; MLE gives better
#   tail estimates when the fit is good. We report both plus bootstrap CIs.
# - **Peaks-over-threshold (POT/GPD)** with runs declustering, which uses the
#   data more efficiently and separates "one long flood" from "several peaks".
# - Both are computed on the **reliable era** (2007–2025 minus the 2014 wet
#   season) with a wet-season coverage filter, per notebook 01.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from mekongds import data
from mekongds.stats_ext import (
    fit_gev_mle, fit_gev_lmom, gev_return_level, pot_gpd, gpd_return_level, l_moments,
)
from mekongds.plotting import set_style, savefig, STAGE_COLORS, PALETTE

set_style()
rng = np.random.default_rng(42)

OUT = data.processed_dir()
w = pd.read_parquet(OUT / "stage_wide.parquet")
meta = data.load_station_meta().set_index("station")
TH = meta[["floodStage", "alarmStage"]].dropna()

REL = w.loc["2007-01-01":"2025-12-31"]
REL = REL.loc[~((REL.index >= "2014-06-01") & (REL.index <= "2014-10-31"))]

CAM = ["STR", "KRA", "KOM", "NEA", "PPP", "PPB", "KOH", "PRE"]

# %% [markdown]
# ## 1. Annual maxima sample
#
# A year enters the sample only if its wet season (Jun–Nov) is at least 90%
# observed, so a missing peak can never masquerade as a calm year.

# %%
def annual_maxima(stn: str, min_cov: float = 0.9) -> pd.Series:
    grp = REL[stn].groupby(REL.index.year)
    wet_cov = grp.apply(lambda g: g.loc[g.index.month.isin([6, 7, 8, 9, 10, 11])].notna().mean())
    amax = grp.max()
    return amax[wet_cov >= min_cov].dropna()

AM = {stn: annual_maxima(stn) for stn in CAM}
pd.DataFrame({stn: {"n_years": len(s), "min": round(s.min(), 2), "median": round(s.median(), 2),
                    "max": round(s.max(), 2)} for stn, s in AM.items()}).T

# %% [markdown]
# ## 2. GEV fits and return levels

# %%
RETURNS = [2, 5, 10, 20, 50, 100]

rows = []
for stn in CAM:
    am = AM[stn].values
    if len(am) < 10:
        continue
    pm = fit_gev_mle(am)
    pl = fit_gev_lmom(am)
    rows.append({
        "station": stn,
        "n": len(am),
        "MLE c": round(pm["c"], 3), "Lmom c": round(pl["c"], 3),
        **{f"RL{T}_MLE": round(gev_return_level(pm, T), 2) for T in RETURNS},
        **{f"RL{T}_Lmom": round(gev_return_level(pl, T), 2) for T in RETURNS},
    })
gev_tab = pd.DataFrame(rows).set_index("station")
gev_tab[[c for c in gev_tab.columns if "MLE" in c or c in ("n", "MLE c")]]

# %%
gev_tab[[c for c in gev_tab.columns if "Lmom" in c or c in ("n", "Lmom c")]]

# %% [markdown]
# The two estimators agree closely on return levels for the Cambodian stations,
# usually within 10–20 cm up to T=50. Where they disagree (heavy tail vs
# bounded), the bootstrap intervals in §4 will show the uncertainty honestly.

# %%
# sanity: return levels vs thresholds
for stn in ["KOM", "PRE", "KOH", "NEA"]:
    am = AM[stn].values
    pm = fit_gev_mle(am)
    print(f"{stn}: alarm {TH.loc[stn,'alarmStage']:.1f} ~ RL{RETURNS[np.argmin([abs(gev_return_level(pm,T)-TH.loc[stn,'alarmStage']) for T in RETURNS])]}, "
          f"flood {TH.loc[stn,'floodStage']:.1f} ~ RL{RETURNS[np.argmin([abs(gev_return_level(pm,T)-TH.loc[stn,'floodStage']) for T in RETURNS])]}, "
          f"obs max {am.max():.2f} (RL100 {gev_return_level(pm,100):.2f})")

# %% [markdown]
# ## 3. Peaks over threshold

# %%
pot_rows = []
POT_FITS = {}
for stn in CAM:
    seg = REL[stn].dropna()
    u = float(np.quantile(seg, 0.98))  # ~7 days/year exceedance before declustering
    pot = pot_gpd(seg.values, u=float(u))
    pot["n_years"] = len(np.unique(seg.index.year))
    POT_FITS[stn] = pot
    row = {
        "station": stn, "u": round(u, 2), "peaks": pot["npeaks"],
        "peaks/yr": round(pot["npeaks"] / pot["n_years"], 2),
        "GPD c": round(pot["c"], 3), "GPD scale": round(pot["scale"], 3),
    }
    # POT can only resolve T with lambda*T >= 1 (T-year level at or above u)
    lam = pot["npeaks"] / pot["n_years"]
    for T in RETURNS:
        row[f"RL{T}"] = round(gpd_return_level(pot, T), 2) if lam * T >= 1 else np.nan
    pot_rows.append(row)
pot_tab = pd.DataFrame(pot_rows).set_index("station")
pot_tab

# %% [markdown]
# ## 4. Bootstrap uncertainty
#
# Nonparametric bootstrap of the annual-maxima sample (resample years with
# replacement, 1000 draws), refit GEV-MLE each draw, take return-level
# quantiles. This captures parameter uncertainty; it does not capture
# nonstationarity (§2 of the EDA found no trend, so this is defensible for
# the reliable era).

# %%
def bootstrap_rl(stn: str, T: int, n_boot: int = 1000) -> np.ndarray:
    am = AM[stn].values
    out = []
    for _ in range(n_boot):
        sample = rng.choice(am, size=len(am), replace=True)
        try:
            p = fit_gev_mle(sample)
            out.append(gev_return_level(p, T))
        except Exception:
            continue
    return np.array(out)

boot = {}
for stn in ["KOM", "KRA", "PRE", "KOH"]:
    boot[stn] = {T: bootstrap_rl(stn, T, 500) for T in [10, 100]}

rows = []
for stn, d in boot.items():
    for T, vals in d.items():
        rows.append({
            "station": stn, "T": T,
            "RL": round(np.median(vals), 2),
            "CI5": round(np.quantile(vals, 0.05), 2),
            "CI95": round(np.quantile(vals, 0.95), 2),
        })
boot_tab = pd.DataFrame(rows)
boot_tab

# %%
fig, ax = plt.subplots(figsize=(7.5, 4))
for i, stn in enumerate(boot):
    for T, vals in boot[stn].items():
        ax.errorbar(T + (i - 1.5) * 1.5, np.median(vals),
                    yerr=[[np.median(vals) - np.quantile(vals, 0.05)],
                          [np.quantile(vals, 0.95) - np.median(vals)]],
                    fmt="o", color=PALETTE[i], capsize=3, label=stn if T == 10 else None)
ax.set_xscale("log")
ax.set_xticks([10, 100], ["T=10", "T=100"])
ax.set_xlabel("return period")
ax.set_ylabel("stage (m above gauge zero)")
ax.set_title("GEV return levels with bootstrap 90% intervals")
ax.legend()
savefig(fig, "fig03_bootstrap_rl")
plt.show()

# %% [markdown]
# ## 5. Return-level plots per station

# %%
emp_T = 1 / (1 - np.arange(1, len(AM["KOM"]) + 1) / (len(AM["KOM"]) + 1))

fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=False)
for ax, stn in zip(axes.ravel(), CAM):
    am = np.sort(AM[stn].values)
    T_emp = (len(am) + 1) / np.arange(1, len(am) + 1)  # Weibull plotting position
    pm = fit_gev_mle(am)
    Ts = np.logspace(0.7, 2.1, 60)
    ax.scatter(T_emp, am, s=22, color="0.3", zorder=3, label="observed AM")
    ax.plot(Ts, [gev_return_level(pm, T) for T in Ts], color=PALETTE[0], label="GEV MLE")
    if stn in TH.index:
        ax.axhline(TH.loc[stn, "alarmStage"], color=STAGE_COLORS["alarm"], ls="--", lw=1)
        ax.axhline(TH.loc[stn, "floodStage"], color=STAGE_COLORS["flood"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_title(stn)
    ax.set_xlabel("T (yr)")
fig.suptitle("Return-level plots, GEV-MLE on annual maxima 2007–2025 (dashed: alarm/flood stages)", y=1.02)
fig.tight_layout()
savefig(fig, "fig03_return_level_curves")
plt.show()

# %% [markdown]
# ## 6. What this means for the farm-action layer
#
# - At **Kompong Cham**, the MRC alarm stage (15.2 m) corresponds to roughly a
#   2-year event and the flood stage (16.2 m) to something beyond the fitted
#   RL100: the record's 16.02 m (2011) is close to the estimated upper bound.
#   The floodplain near the gauge floods often; the official flood stage is a
#   severe design event.
# - At **Prek Kdam** and **Koh Khel** the thresholds sit similarly high.
#   Communes protected to "alarm" are safe against roughly the 20-year event.
# - The **bootstrap intervals are wide** (±0.5 m or more at T=100): return
#   levels are design numbers, not forecasts, and the decision layer should
#   consume forecast probabilities (notebook 04), not return levels.

# %%
gev_tab.to_parquet(OUT / "gev_return_levels.parquet")
pot_tab.to_parquet(OUT / "pot_return_levels.parquet")
boot_tab.to_parquet(OUT / "bootstrap_rl.parquet")
print("saved: gev_return_levels, pot_return_levels, bootstrap_rl")
