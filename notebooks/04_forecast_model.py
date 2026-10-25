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
# # 04 — River-stage forecast model
#
# **MekongDS: from river forecast to farm action.**
#
# Goal: multi-lead (1–10 day) stage forecasts at the three stations that matter
# for the decision layer — Kompong Cham (mainstream), Prek Kdam (Tonle Sap
# outlet) and Koh Khel (Bassac) — built on stage dynamics only (no rainfall
# covariate exists in the export, per notebook 01).
#
# Design:
#
# - **Features:** own lags/rollings/rises, upstream stations at EDA-informed
#   lags (Kratie leads Kompong Cham by 1–2 days, Stung Treng by ~3, the Lao
#   stations by 5+; Prek Kdam integrates Kompong Cham over ~2 weeks),
#   day-of-year Fourier terms. Everything observable at issue time t.
# - **Models:** persistence, day-of-year climatology, ridge regression, and
#   LightGBM (point). Plus LightGBM quantile (0.05/0.95) for 90% intervals.
# - **Probabilistic exceedance:** point forecast + inner-validation residual
#   distribution → full predictive CDF → P(stage > threshold) at any threshold;
#   compared against a direct gradient-boosting classifier at Kampong Cham.
# - **Validation:** expanding-window CV, wet seasons 2018–2025 as test folds,
#   one fold per year, trained on wet seasons 2007+ (2014 excluded at source).
#   No shuffling, no leakage: climatology, imputers and residual distributions
#   are refit inside each training fold only.

# %%
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier

from mekongds import data
from mekongds.metrics import mae, rmse, nse, kge, skill, brier, brier_skill, interval_coverage, pinball
from mekongds.plotting import set_style, savefig, STAGE_COLORS, PALETTE

warnings.filterwarnings("ignore")
set_style()

OUT = data.processed_dir()
MODELS_DIR = data.find_root() / "models"
MODELS_DIR.mkdir(exist_ok=True)

w = pd.read_parquet(OUT / "stage_wide.parquet")
meta = data.load_station_meta().set_index("station")
TH = meta[["floodStage", "alarmStage"]].dropna()

TARGETS = ["KOM", "PRE", "KOH"]
LEADS = [1, 2, 3, 5, 7, 10]
TEST_YEARS = list(range(2018, 2026))
WET_MONTHS = [6, 7, 8, 9, 10, 11]

# reliable-era mask (notebook 01 policy)
rel_mask = (w.index >= "2007-01-01") & (w.index <= "2025-12-31")
rel_mask &= ~((w.index >= "2014-06-01") & (w.index <= "2014-10-31"))

# %% [markdown]
# ## 1. Feature engineering

# %%
OWN_LAGS = {"KOM": [0, 1, 2, 3, 4, 5, 6], "PRE": [0, 1, 2, 3, 5, 7, 14], "KOH": [0, 1, 2, 3, 5, 7, 14]}
ROLLS = {"KOM": [3, 7, 15], "PRE": [3, 7, 15, 30], "KOH": [3, 7, 15, 30]}
RISES = [1, 3]
# upstream stations and the lags (days back from issue time) that EDA says carry signal
UP = {
    "KOM": {"KRA": [0, 1, 2], "STR": [1, 2, 3], "PKS": [3, 4, 5], "KHO": [3, 4, 5],
            "MUK": [4, 5, 6], "SAV": [4, 5, 6], "NAK": [5, 6, 7], "PAK": [6, 7, 8]},
    "PRE": {"KOM": [0, 5, 10, 15], "KOH": [0, 1, 2], "PPP": [0, 1, 2], "KRA": [0, 1, 2], "NEA": [0, 1, 2]},
    "KOH": {"PPP": [0, 1, 2], "PPB": [0, 1, 2], "KOM": [0, 1, 2], "NEA": [0, 1, 2], "PRE": [0, 1, 2]},
}

def build_features(station: str, include_upstream: bool = True) -> pd.DataFrame:
    s = w[station]
    cols = {}
    for lag in OWN_LAGS[station]:
        cols[f"{station}_lag{lag}"] = s.shift(lag)
    for win in ROLLS[station]:
        cols[f"{station}_mean{win}"] = s.rolling(win).mean()
    for r in RISES:
        cols[f"{station}_rise{r}"] = s - s.shift(r)
    if include_upstream:
        for up, lags in UP[station].items():
            for lag in lags:
                cols[f"{up}_lag{lag}"] = w[up].shift(lag)
    doy = w.index.dayofyear.values
    for k in (1, 2, 3):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return pd.DataFrame(cols, index=w.index)

FEATS = {stn: build_features(stn) for stn in TARGETS}

# evaluation rows: wet season, reliable era, target and today's stage observed
def eval_rows(station: str, lead: int) -> pd.Series:
    y = w[station].shift(-lead)
    ok = (rel_mask
          & w.index.month.isin(WET_MONTHS)
          & w[station].notna()      # persistence needs stage(t); keeps rows comparable
          & y.notna())
    return ok

f"{len(FEATS['KOM'].columns)} features for KOM; wet rows 2007–2025: {int(eval_rows('KOM', 1).sum())}"

# %% [markdown]
# ## 2. Models

# %%
def make_lgbm(**kw):
    return lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=25,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=7, **kw,
    )

def fit_predict_fold(station: str, lead: int, test_year: int) -> dict:
    """One CV fold: train on wet seasons < test_year, predict the test_year wet season."""
    X, s = FEATS[station], w[station]
    y = s.shift(-lead)
    ok = eval_rows(station, lead)
    tr = ok & (w.index.year < test_year)
    te = ok & (w.index.year == test_year)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    out = {"date": Xte.index, "y": yte.values, "persistence": s[te].values}

    # DOY climatology from training years only
    doy_train = X[tr].index.dayofyear
    clim_map = pd.Series(ytr.values, index=doy_train).groupby(level=0).mean()
    out["climatology"] = clim_map.reindex(Xte.index.dayofyear).values

    # ridge on the same features (median imputation + scaling, fit on train only)
    ridge = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ]).fit(Xtr, ytr)
    out["ridge"] = ridge.predict(Xte)

    # LightGBM point
    m = make_lgbm().fit(Xtr, ytr)
    out["lgbm"] = m.predict(Xte)

    # LightGBM quantiles for the 90% interval
    for q, name in [(0.05, "q05"), (0.95, "q95")]:
        mq = make_lgbm(objective="quantile", alpha=q).fit(Xtr, ytr)
        out[name] = mq.predict(Xte)
    # enforce monotone quantiles per row
    lo, hi = np.minimum(out["q05"], out["q95"]), np.maximum(out["q05"], out["q95"])
    out["q05"], out["q95"] = lo, hi

    # residual-CDF exceedance: point model refit without the last 2 training years;
    # residuals of that model on those 2 years give the predictive spread.
    years_tr = X[tr].index.year
    cut = years_tr.max() - 1
    inner_tr, inner_va = tr & (w.index.year <= cut - 1), tr & (w.index.year >= cut)
    if inner_va.sum() > 60 and inner_tr.sum() > 200:
        mi = make_lgbm().fit(X[inner_tr], y[inner_tr])
        res = (y[inner_va] - mi.predict(X[inner_va])).values
        probs_grid = np.linspace(0.001, 0.999, 199)
        rq = np.quantile(res, probs_grid)
        out["_resid_q"] = rq
        out["_resid_probs"] = probs_grid
    out["_train_mask"] = tr
    out["_test_mask"] = te
    out["_Xtr_index"] = Xtr.index
    out["_ytr"] = ytr
    return out, Xtr, ytr, Xte, yte

# %% [markdown]
# ## 3. Point-forecast CV

# %%
rows = []
for stn in TARGETS:
    for lead in LEADS:
        for ty in TEST_YEARS:
            fold, *_ = fit_predict_fold(stn, lead, ty)
            y = fold["y"]
            for model in ["persistence", "climatology", "ridge", "lgbm"]:
                yhat = fold[model]
                rows.append({
                    "station": stn, "lead": lead, "test_year": ty, "model": model,
                    "n": len(y),
                    "MAE": mae(y, yhat), "RMSE": rmse(y, yhat),
                    "NSE": nse(y, yhat), "KGE": kge(y, yhat),
                    "skill_vs_pers": skill(y, yhat, fold["persistence"]),
                })
cv_point = pd.DataFrame(rows)
cv_point.to_parquet(OUT / "cv_point.parquet")

agg = (cv_point.groupby(["station", "lead", "model"])[["MAE", "RMSE", "NSE", "KGE", "skill_vs_pers"]]
        .mean().round(3).reset_index())
agg.head(12)

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=False)
for ax, stn in zip(axes, TARGETS):
    sub = agg[agg.station == stn]
    for model, ls in [("persistence", ":"), ("climatology", "--"), ("ridge", "-."), ("lgbm", "-")]:
        ss = sub[sub.model == model]
        ax.plot(ss.lead, ss.MAE, ls, marker="o", ms=3.5, label=model)
    ax.set_title(stn)
    ax.set_xlabel("lead (days)")
    ax.set_xticks(LEADS)
axes[0].set_ylabel("MAE (m), mean of 8 folds")
axes[0].legend()
fig.suptitle("Point forecast error by lead, wet seasons 2018–2025", y=1.03)
fig.tight_layout()
savefig(fig, "fig04_skill_vs_lead")
plt.show()

# %%
# skill vs persistence, pooled folds
piv = (cv_point[cv_point.model == "lgbm"]
       .pivot_table(index="lead", columns="station", values="skill_vs_pers", aggfunc="mean")
       .round(3))
piv.style.background_gradient(cmap="RdYlGn", vmin=-0.2, vmax=0.6).set_caption(
    "LightGBM skill score vs persistence (1 = perfect, 0 = tie)")

# %% [markdown]
# LightGBM beats persistence at every lead and station once the lead reaches
# about 2 days; at 1 day persistence is competitive (as it should be — the
# river has inertia). The upstream features buy the most at Kratie-fed leads:
# by day 5–10, persistence is beaten by wide margins at Kompong Cham and
# Prek Kdam.

# %%
# headline numbers for the report
head = (agg[agg.model == "lgbm"]
        .pivot_table(index="station", columns="lead", values="MAE").round(3))
head

# %% [markdown]
# ## 4. Quantile forecasts: interval coverage and pinball

# %%
rows = []
for stn in TARGETS:
    for lead in LEADS:
        for ty in TEST_YEARS:
            fold, *_ = fit_predict_fold(stn, lead, ty)
            y, lo, hi = fold["y"], fold["q05"], fold["q95"]
            rows.append({
                "station": stn, "lead": lead, "test_year": ty,
                "coverage90": interval_coverage(y, lo, hi),
                "width": float(np.mean(hi - lo)),
                "pinball05": pinball(y, lo, 0.05),
                "pinball95": pinball(y, hi, 0.95),
            })
cv_q = pd.DataFrame(rows)
cv_q.to_parquet(OUT / "cv_quantile.parquet")
(cv_q.groupby(["station", "lead"])[["coverage90", "width"]].mean().round(3)
   .style.background_gradient(subset="coverage90", cmap="RdYlGn", vmin=0.7, vmax=1.0)
   .set_caption("90% interval coverage (nominal 0.90) and mean width (m)"))

# %% [markdown]
# ## 5. Probabilistic exceedance
#
# For each fold, the predictive CDF comes from the point forecast plus the
# residual distribution of an inner-validation refit (last 2 training years
# held out). P(stage(t+h) > threshold) = 1 − F(threshold − ŷ).
#
# Thresholds: the MRC alarm stage where test folds contain events (Kompong
# Cham); the training-period 95th-percentile wet-day stage for Prek Kdam and
# Koh Khel, whose alarm stages are never crossed in the record.

# %%
def exceed_thresholds(station: str, train_mask: pd.Series) -> dict[str, float]:
    if station == "KOM":
        return {"alarm": float(TH.loc["KOM", "alarmStage"])}
    wet_vals = w.loc[train_mask, station].dropna()
    wet_vals = wet_vals[wet_vals.index.month.isin(WET_MONTHS)]
    return {"p95": float(wet_vals.quantile(0.95))}

rows = []
reliability_rows = []
for stn in TARGETS:
    for lead in LEADS:
        for ty in TEST_YEARS:
            fold, *_ = fit_predict_fold(stn, lead, ty)
            if "_resid_q" not in fold:
                continue
            y = fold["y"]
            yhat = fold["lgbm"]
            thr_map = exceed_thresholds(stn, fold["_train_mask"])
            # climatology base rate for Brier skill, from training years
            tr_y = fold["_ytr"].values
            for tname, thr in thr_map.items():
                p = 1 - np.interp(thr - yhat, fold["_resid_q"], fold["_resid_probs"])
                e = (y > thr).astype(float)
                base = float((tr_y > thr).mean())
                rows.append({
                    "station": stn, "lead": lead, "test_year": ty, "threshold": tname,
                    "thr": round(thr, 2), "n": len(y), "n_events": int(e.sum()),
                    "brier": brier(e, p), "brier_clim": brier(e, np.full_like(p, base)),
                    "bss": brier_skill(e, p, np.full_like(p, base)),
                })
                if stn == "KOM" and lead in (1, 3, 5, 7, 10):
                    reliability_rows.append(pd.DataFrame({
                        "p": p, "e": e, "lead": lead, "year": ty,
                    }))
cv_exc = pd.DataFrame(rows)
cv_exc.to_parquet(OUT / "cv_exceed.parquet")
(cv_exc.groupby(["station", "threshold", "lead"])[["n_events", "brier", "bss"]]
   .mean().round(4).head(20))

# %%
rel_df = pd.concat(reliability_rows, ignore_index=True)
bins = [0, 0.05, 0.15, 0.3, 0.6, 1.0]
rel_df["bin"] = pd.cut(rel_df.p, bins, include_lowest=True)
rel_tab = rel_df.groupby("bin", observed=True).agg(
    n=("e", "size"), mean_p=("p", "mean"), freq=("e", "mean")).round(3)
rel_tab

# %%
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="perfect reliability")
ax.plot(rel_tab.mean_p, rel_tab.freq, "o-", color=PALETTE[0], label="residual-CDF, KOM pooled leads")
for _, r in rel_tab.iterrows():
    ax.annotate(f"n={int(r.n)}", (r.mean_p, r.freq), fontsize=7, xytext=(5, -10),
                textcoords="offset points")
ax.set_xlabel("forecast probability")
ax.set_ylabel("observed frequency")
ax.set_title("Reliability of P(stage > alarm), Kompong Cham 2018–2025")
ax.legend(loc="upper left")
savefig(fig, "fig04_reliability_kom")
plt.show()

# %% [markdown]
# ## 6. Direct classifier comparison (Kompong Cham)
#
# A gradient-boosted binary classifier on the same features, versus the
# residual-CDF probabilities.

# %%
rows = []
for lead in [1, 3, 5, 7, 10]:
    for ty in TEST_YEARS:
        fold, Xtr, ytr, Xte, yte = fit_predict_fold("KOM", lead, ty)
        thr = float(TH.loc["KOM", "alarmStage"])
        etr, ete = (ytr > thr).astype(int), (yte > thr).astype(int)
        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=25,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
            scale_pos_weight=max(1.0, (1 - etr.mean()) / max(etr.mean(), 1e-6)),
            random_state=7,
        ).fit(Xtr, etr)
        pclf = clf.predict_proba(Xte)[:, 1]
        if "_resid_q" in fold:
            pres = 1 - np.interp(thr - fold["lgbm"], fold["_resid_q"], fold["_resid_probs"])
            rows.append({
                "lead": lead, "test_year": ty, "n_events": int(ete.sum()),
                "classifier_PR_AUC": average_precision_score(ete, pclf) if ete.any() else np.nan,
                "residCDF_PR_AUC": average_precision_score(ete, pres) if ete.any() else np.nan,
                "classifier_Brier": brier(ete, pclf),
                "residCDF_Brier": brier(ete, pres),
            })
clf_cmp = pd.DataFrame(rows)
clf_cmp.groupby("lead")[["classifier_PR_AUC", "residCDF_PR_AUC",
                          "classifier_Brier", "residCDF_Brier"]].mean().round(4)

# %% [markdown]
# The two approaches are close; the residual-CDF has the advantage of a full
# predictive distribution at any threshold (the decision layer needs
# commune-specific thresholds, not one fixed class), so it is what notebook 05
# consumes.

# %% [markdown]
# ## 7. Ablation: how much do upstream stations buy?

# %%
FEATS_NOUP = {stn: build_features(stn, include_upstream=False) for stn in ["KOM"]}

rows = []
for lead in LEADS:
    for ty in TEST_YEARS:
        X, s = FEATS["KOM"], w["KOM"]
        y = s.shift(-lead)
        ok = eval_rows("KOM", lead)
        tr, te = ok & (w.index.year < ty), ok & (w.index.year == ty)
        for tag, XX in [("with_upstream", X), ("own_only", FEATS_NOUP["KOM"])]:
            m = make_lgbm().fit(XX[tr], y[tr])
            rows.append({"lead": lead, "test_year": ty, "features": tag,
                         "MAE": mae(y[te], m.predict(XX[te]))})
abl = pd.DataFrame(rows)
abl_tab = abl.pivot_table(index="lead", columns="features", values="MAE", aggfunc="mean").round(3)
abl_tab["gain_m"] = (abl_tab["own_only"] - abl_tab["with_upstream"]).round(3)
abl_tab

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(abl_tab.index, abl_tab.own_only, "o--", color=PALETTE[1], label="own station + seasonality only")
ax.plot(abl_tab.index, abl_tab.with_upstream, "o-", color=PALETTE[0], label="+ upstream network")
ax.set_xlabel("lead (days)")
ax.set_ylabel("MAE (m)")
ax.set_title("Value of the upstream network, Kompong Cham")
ax.legend()
savefig(fig, "fig04_ablation")
plt.show()

# %% [markdown]
# ## 8. Example: the 2019 flood at Kompong Cham, lead 3

# %%
fold, *_ = fit_predict_fold("KOM", 3, 2019)
seg = pd.DataFrame({"y": fold["y"], "fc": fold["lgbm"], "lo": fold["q05"], "hi": fold["q95"]},
                   index=fold["date"]).loc["2019-08-15":"2019-11-15"]
fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(seg.index, seg.lo, seg.hi, color=STAGE_COLORS["band"], alpha=0.5, label="90% interval")
ax.plot(seg.index, seg.y, color="0.2", lw=1.6, label="observed")
ax.plot(seg.index, seg.fc, color=STAGE_COLORS["fc"], lw=1.4, label="LightGBM lead-3")
ax.axhline(15.2, color=STAGE_COLORS["alarm"], ls="--", lw=1, label="alarm 15.2 m")
ax.set_ylabel("stage (m)")
ax.set_title("Kompong Cham 2019 wet season: rolling lead-3 forecasts with 90% intervals")
ax.legend(ncol=4, fontsize=8)
savefig(fig, "fig04_hydrograph_2019")
plt.show()

# %% [markdown]
# ## 9. Feature importance (KOM, lead 3, pooled final fit)

# %%
X, y = FEATS["KOM"], w["KOM"].shift(-3)
ok = eval_rows("KOM", 3)
m = make_lgbm().fit(X[ok], y[ok])
imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
imp.iloc[::-1].plot.barh(ax=ax, color=PALETTE[0])
ax.set_title("LightGBM gain importance, KOM lead-3 (top 15)")
ax.set_xlabel("split gain")
savefig(fig, "fig04_featimp_kom")
plt.show()

# %% [markdown]
# ## 10. Final models for the decision layer
#
# Refit point + quantile models on all reliable-era wet seasons (2007–2025)
# and save with the feature spec. Notebook 05 issues forecasts from the
# 2026-10-20 state of the network.

# %%
for stn in TARGETS:
    X = FEATS[stn]
    for lead in LEADS:
        y = w[stn].shift(-lead)
        ok = eval_rows(stn, lead)
        m = make_lgbm().fit(X[ok], y[ok])
        mlo = make_lgbm(objective="quantile", alpha=0.05).fit(X[ok], y[ok])
        mhi = make_lgbm(objective="quantile", alpha=0.95).fit(X[ok], y[ok])
        joblib.dump({"point": m, "q05": mlo, "q95": mhi, "lead": lead,
                     "features": list(X.columns), "station": stn},
                    MODELS_DIR / f"{stn}_lead{lead}.joblib")
print(f"saved {len(TARGETS) * len(LEADS)} model bundles to {MODELS_DIR}")

# %% [markdown]
# ## 11. Summary
#
# 1. LightGBM with upstream-lag features beats persistence, climatology and
#    ridge at all leads ≥2 days and all three stations (see `cv_point.parquet`).
# 2. 90% quantile intervals are close to nominal coverage (see table in §4).
# 3. Residual-CDF exceedance probabilities are reliable at Kompong Cham and
#    competitive with a direct classifier, while also giving a full
#    distribution for arbitrary thresholds.
# 4. The upstream network is worth roughly the MAE gains in the ablation
#    table, growing with lead: at 7–10 days, upstream stations carry most of
#    the signal that persistence cannot.
# 5. Models, feature lists and leads are persisted for the farm-action layer.
