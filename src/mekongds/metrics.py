"""Forecast evaluation metrics: NSE, KGE, skill scores, Brier, pinball loss, coverage."""

from __future__ import annotations

import numpy as np


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    d = np.asarray(y) - np.asarray(yhat)
    return float(np.sqrt(np.mean(d**2)))


def nse(y, yhat) -> float:
    """Nash-Sutcliffe efficiency (1 = perfect, 0 = climatology mean)."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def kge(y, yhat) -> float:
    """Kling-Gupta efficiency (1 = perfect)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    r = np.corrcoef(y, yhat)[0, 1]
    alpha = yhat.std() / y.std()
    beta = yhat.mean() / y.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def skill(y, yhat, yref) -> float:
    """Skill score vs a reference forecast: 1 - MSE/MSE_ref (1 = perfect, 0 = tie)."""
    y, yhat, yref = np.asarray(y), np.asarray(yhat), np.asarray(yref)
    mse = np.mean((y - yhat) ** 2)
    mse_ref = np.mean((y - yref) ** 2)
    return float(1 - mse / mse_ref)


def brier(y_event, p_hat) -> float:
    e, p = np.asarray(y_event), np.asarray(p_hat)
    return float(np.mean((p - e) ** 2))


def brier_skill(y_event, p_hat, p_clim) -> float:
    """Brier skill score vs climatological base rate."""
    return float(1 - brier(y_event, p_hat) / brier(y_event, p_clim))


def pinball(y, q, alpha: float) -> float:
    """Quantile (pinball) loss for the alpha-quantile forecast q."""
    y, q = np.asarray(y, float), np.asarray(q, float)
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return float(np.mean((y >= lo) & (y <= hi)))
