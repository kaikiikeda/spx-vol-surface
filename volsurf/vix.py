"""VIX replication following the Cboe VIX white paper.

sigma^2 = (2/T) * sum_i dK_i / K_i^2 * e^{RT} Q(K_i)  -  (1/T) * (F/K0 - 1)^2

Near/next terms are the SPX expirations with 23 < days < 37; their variances are
interpolated in minutes to a constant 30 days and annualised.
"""
import numpy as np
import pandas as pd

from . import config as cfg
from .bs import black_price
from .svi import svi_w

N30, N365 = 30 * 1440.0, 365 * 1440.0


def term_variance(g, T, R):
    """g: raw quotes for one expiry (both cp), crossed quotes already removed."""
    g = g.assign(mid=0.5 * (g.bid + g.ask))
    calls = g[g.cp == 1].set_index("strike").sort_index()
    puts = g[g.cp == -1].set_index("strike").sort_index()
    both = calls.join(puts, lsuffix="_c", rsuffix="_p", how="inner")
    both = both[(both.bid_c > 0) & (both.bid_p > 0)]
    kstar = (both.mid_c - both.mid_p).abs().idxmin()
    F = kstar + np.exp(R * T) * (both.loc[kstar, "mid_c"] - both.loc[kstar, "mid_p"])
    K0 = max(k for k in both.index if k <= F)

    def walk(side, strikes):
        picked, zeros = [], 0
        for k in strikes:
            if side.loc[k, "bid"] <= 0:
                zeros += 1
                if zeros == 2:
                    break
                continue
            zeros = 0
            picked.append((k, side.loc[k, "mid"]))
        return picked

    put_pts = walk(puts, [k for k in puts.index[::-1] if k < K0])
    call_pts = walk(calls, [k for k in calls.index if k > K0])
    q0 = 0.5 * (calls.loc[K0, "mid"] + puts.loc[K0, "mid"])
    pts = sorted(put_pts + [(K0, q0)] + call_pts)
    K = np.array([p[0] for p in pts])
    Qk = np.array([p[1] for p in pts])
    dK = np.empty_like(K)
    dK[1:-1] = 0.5 * (K[2:] - K[:-2])
    dK[0], dK[-1] = K[1] - K[0], K[-1] - K[-2]
    contrib = 2.0 / T * dK / K ** 2 * np.exp(R * T) * Qk
    var = contrib.sum() - (F / K0 - 1) ** 2 / T
    return dict(F=F, K0=K0, sigma2=var, n_strikes=len(K), k_lo=K.min(), k_hi=K.max(),
                strikes=K, dK=dK, contrib=contrib)


def svi_term_variance(params, F, T, R, k_range=(-3.0, 1.5), n=20001):
    """Same replication but integrating the fitted SVI smile over a continuum of strikes
    (no discretisation, no wing truncation) -- isolates those two error sources."""
    k = np.linspace(*k_range, n)
    K = F * np.exp(k)
    vol = np.sqrt(np.maximum(svi_w(k, *params), 1e-12) / T)
    df = np.exp(-R * T)
    otm = np.where(K < F, black_price(F, K, T, vol, df, -1), black_price(F, K, T, vol, df, 1))
    integrand = otm / K ** 2 * K           # dK = K dk
    return 2.0 / T * np.exp(R * T) * np.trapezoid(integrand, k)


def pick_terms(expiries):
    days = {e: (e - cfg.SNAPSHOT).total_seconds() / 86400 for e, *_ in expiries}
    window = sorted(e for e, d in days.items() if 23 < d < 37)
    return window[0], window[-1]


def vix_from_terms(s1, T1, s2, T2):
    N1, N2 = T1 * cfg.MIN_PER_YEAR, T2 * cfg.MIN_PER_YEAR
    w1 = (N2 - N30) / (N2 - N1)
    w2 = (N30 - N1) / (N2 - N1)
    return 100.0 * np.sqrt((T1 * s1 * w1 + T2 * s2 * w2) * N365 / N30)
