"""Heston calibration to the cleaned OTM surface.

Objective: vega-weighted price errors, (model - mid) / vega ~ implied-vol error,
so calibration matches IVs without inverting Black at every iteration.
"""
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .heston import heston_price

NAMES = ["v0", "kappa", "theta", "sigma", "rho"]
LO = np.array([1e-4, 0.05, 1e-3, 0.05, -0.99])
HI = np.array([0.5, 15.0, 0.5, 3.0, 0.5])


def select_quotes(iv, per_expiry=40, k_sd=(-3.5, 2.0)):
    """Thin each expiry to ~per_expiry strikes spread across the smile."""
    out = []
    for _, g in iv.groupby("expiry"):
        sd = g.iv.median() * np.sqrt(g["T"].iloc[0])
        g = g[(g.k > k_sd[0] * sd) & (g.k < k_sd[1] * sd)].sort_values("k")
        idx = np.unique(np.linspace(0, len(g) - 1, min(per_expiry, len(g))).round().astype(int))
        out.append(g.iloc[idx])
    return pd.concat(out).reset_index(drop=True)


def model_prices(q, params):
    out = np.empty(len(q))
    for _, g in q.groupby("expiry"):
        r = g.iloc[0]
        out[g.index] = heston_price(r.F, g.strike.values, r["T"], r.DF, g.cp.values, params)
    return out


def calibrate(q, x0=(0.04, 1.5, 0.05, 0.6, -0.5)):
    vega = q.vega.values
    mid = q.mid.values

    def resid(p):
        return (model_prices(q, p) - mid) / vega

    t0 = time.time()
    r = least_squares(resid, np.array(x0), bounds=(LO, HI), method="trf", x_scale="jac",
                      xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
    return dict(params=r.x, rmse_iv=float(np.sqrt(np.mean(r.fun ** 2))), nfev=r.nfev,
                seconds=time.time() - t0, result=r)
