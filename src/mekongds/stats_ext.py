"""Statistical helpers: Mann-Kendall trend test, L-moments, GEV/GPD via scipy MLE."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_fn


def mann_kendall(x: np.ndarray, alpha: float = 0.05) -> dict:
    """Mann-Kendall trend test (original MK, no autocorrection; report Sen's slope too)."""
    x = np.asarray(x, float)
    n = len(x)
    s = 0
    for i in range(n - 1):
        s += np.sign(x[i + 1 :] - x[i]).sum()
    # variance correction for tied groups
    _, counts = np.unique(x, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    # Sen's slope over all pairs
    i, j = np.triu_indices(n, k=1)
    slopes = (x[j] - x[i]) / (j - i)
    sen = float(np.median(slopes))
    return {
        "z": float(z), "p": float(p), "s": float(s),
        "trend": ("increasing" if z > 0 else "decreasing") if p < alpha else "no trend",
        "sens_slope": sen,
    }


def l_moments(x: np.ndarray) -> tuple[float, float, float]:
    """Sample L-moments (l1, l2, t3) via order-statistic estimators."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    return float(l1), float(l2), float(l3 / l2) if l2 > 0 else np.nan


def fit_gev_lmom(x: np.ndarray) -> dict:
    """GEV parameters from L-moments (Hosking 1990).

    scipy's genextreme uses the same shape sign as Hosking's k: c > 0 means the
    distribution is bounded above, which is what left-skewed annual maxima give.
    """
    l1, l2, t3 = l_moments(x)
    cc = 2 / (3 + t3) - np.log(2) / np.log(3)
    k = 7.8590 * cc + 2.9554 * cc**2
    if abs(k) < 1e-8:
        alpha = l2 / np.log(2)
        xi = l1 - alpha * 0.5772157
    else:
        g = gamma_fn(1 + k)
        alpha = l2 * k / ((1 - 2**-k) * g)
        xi = l1 - alpha * (1 - g) / k
    return {"c": float(k), "loc": float(xi), "scale": float(alpha)}


def fit_gev_mle(x: np.ndarray, **fitargs) -> dict:
    c, loc, scale = stats.genextreme.fit(np.asarray(x, float), **fitargs)
    return {"c": float(c), "loc": float(loc), "scale": float(scale)}


def gev_return_level(params: dict, T: float) -> float:
    """GEV T-year return level: the 1 - 1/T quantile of the fitted GEV."""
    return float(stats.genextreme.ppf(1 - 1 / T, params["c"], loc=params["loc"], scale=params["scale"]))


def pot_gpd(x: np.ndarray, u: float, min_sep: int = 7) -> dict:
    """Peaks over threshold with runs declustering (min separation days), fit GPD."""
    x = np.asarray(x, float)
    exc = np.where(x > u)[0]
    if len(exc) == 0:
        return {"npeaks": 0}
    # cluster peaks: consecutive exceedance runs -> max of each run, enforce min_sep
    runs = np.split(exc, np.where(np.diff(exc) > min_sep)[0] + 1)
    peaks = np.array([x[r].max() for r in runs])
    peaks_idx = [r[np.argmax(x[r])] for r in runs]
    # merge peaks closer than min_sep (keep larger)
    keep = [0]
    for i in range(1, len(peaks_idx)):
        if peaks_idx[i] - peaks_idx[keep[-1]] >= min_sep:
            keep.append(i)
        elif peaks[i] > peaks[keep[-1]]:
            keep[-1] = i
    peaks = peaks[keep]
    c, loc, scale = stats.genpareto.fit(peaks - u, floc=0)
    return {
        "npeaks": int(len(peaks)),
        "peaks": peaks,
        "c": float(c), "loc": 0.0, "scale": float(scale),
        "u": float(u),
        "n_years": float((len(x) - 1) / 365.25),
    }


def gpd_return_level(pot: dict, T: float) -> float:
    """POT return level for the T-year event.

    lam = declustered peaks/year; the T-year level is the (1/(lam*T)) quantile of peak
    exceedances, so on average one peak crosses it every T years.
    """
    lam = pot["npeaks"] / pot["n_years"]
    q = 1 / (lam * T)  # exceedance prob per peak
    if pot["c"] == 0:
        return float(pot["scale"] * (-np.log(q)) + pot["u"])
    return float(pot["scale"] / pot["c"] * (q ** (-pot["c"]) - 1) + pot["u"])
