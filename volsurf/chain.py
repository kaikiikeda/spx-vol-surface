"""Synthetic SPX chain snapshot, cleaning, parity forwards, implied vols."""
import numpy as np
import pandas as pd

from . import config as cfg
from .bs import black_vega, implied_vol
from .heston import heston_price


def _tick(p):
    return np.where(p < 3.0, 0.05, 0.10)


def _strikes(F, T):
    sd = 0.20 * np.sqrt(T)                       # rough ATM scale for the strike range
    lo, hi = F * np.exp(-6.0 * sd - 0.04), F * np.exp(3.5 * sd + 0.02)
    near_lo, near_hi = F * np.exp(-1.5 * sd), F * np.exp(1.0 * sd)
    fine = 5 if T < 0.15 else 25
    wide = 25 if T < 0.15 else 50
    ks = set(np.arange(np.floor(near_lo / fine) * fine, near_hi + fine, fine))
    ks |= set(np.arange(np.floor(lo / wide) * wide, hi + wide, wide))
    return np.array(sorted(k for k in ks if k > 0), float)


def dress_quotes(true, rng, noise=0.12):
    """Microstructure model: spread grows with price (floor 0.10), mid jittered by
    noise x spread, bid/ask rounded outward to the SPX tick, bid set to 0 when the
    option is worth less than a tick. noise=0 gives unbiased mids (rounding only)."""
    spread = np.clip(0.05 + 0.015 * true, 0.10, 0.08 * true + 0.10)
    mid = true + rng.normal(0.0, noise, true.size) * spread if noise > 0 else true.copy()
    bid = np.floor((mid - spread / 2) / _tick(mid)) * _tick(mid)
    ask = np.ceil((mid + spread / 2) / _tick(mid)) * _tick(mid)
    cheap = mid - spread / 2 < 0.05
    bid = np.where(cheap, 0.0, bid)
    ask = np.where(cheap, np.maximum(ask, 0.05), ask)
    return bid, ask


def generate_chain(seed=20260213):
    """Quote every strike/expiry from the hidden Heston model, then add realistic
    bid/ask structure, tick rounding, zero bids in the far wings and ~1% junk quotes."""
    rng = np.random.default_rng(seed)
    p = cfg.TRUE_HESTON
    params = (p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
    rows = []
    for expiry, style, label in cfg.EXPIRIES:
        T = cfg.year_frac(expiry)
        df = np.exp(-cfg.R * T)
        F = cfg.SPOT * np.exp((cfg.R - cfg.Q) * T)
        K = _strikes(F, T)
        for cp in (1, -1):
            true = np.maximum(heston_price(F, K, T, df, cp, params), 0.0)
            bid, ask = dress_quotes(true, rng)
            for i in range(K.size):
                rows.append(dict(expiry=expiry, label=label, settle=style, T=T, strike=K[i],
                                 cp=cp, bid=round(bid[i], 2), ask=round(ask[i], 2),
                                 true_price=true[i]))
    df_ = pd.DataFrame(rows)
    # junk: crossed markets and stale wide quotes, as you'd see in a raw vendor snapshot
    junk = rng.choice(df_.index, size=int(0.01 * len(df_)), replace=False)
    half = len(junk) // 2
    df_.loc[junk[:half], ["bid", "ask"]] = df_.loc[junk[:half], ["ask", "bid"]].values + [0.5, 0]
    df_.loc[junk[half:], "ask"] = df_.loc[junk[half:], "ask"] * 3 + 5
    return df_


def clean(chain):
    """Return (clean, report). Drops crossed, zero-bid, absurdly wide and below-intrinsic quotes."""
    c = chain.copy()
    c["mid"] = 0.5 * (c.bid + c.ask)
    rep = {"raw": len(c)}
    m_crossed = c.ask < c.bid
    m_zero = (c.bid <= 0) & ~m_crossed
    rel = (c.ask - c.bid) / c.mid.where(c.mid > 0)
    m_wide = (rel > 0.5) & (c.ask - c.bid > 1.0) & ~m_crossed & ~m_zero
    rep.update(crossed=int(m_crossed.sum()), zero_bid=int(m_zero.sum()), too_wide=int(m_wide.sum()))
    c = c[~(m_crossed | m_zero | m_wide)].copy()
    rep["after_basic"] = len(c)
    return c, rep


def parity_forwards(c):
    """Per expiry: regress C - P = DF*F - DF*K on near-ATM strikes with both sides quoted."""
    out = []
    for (exp, T), g in c.groupby(["expiry", "T"]):
        piv = g.pivot_table(index="strike", columns="cp", values="mid").dropna()
        piv = piv[np.abs(piv.index / cfg.SPOT - 1) < 0.04]
        y = piv[1] - piv[-1]
        A = np.column_stack([np.ones(len(y)), -piv.index.values])
        (a, dfac), *_ = np.linalg.lstsq(A, y.values, rcond=None)
        F = a / dfac
        out.append(dict(expiry=exp, T=T, F=F, DF=dfac, r_implied=-np.log(dfac) / T,
                        F_true=cfg.SPOT * np.exp((cfg.R - cfg.Q) * T), n_pairs=len(y)))
    return pd.DataFrame(out)


def implied_vols(c, fwd):
    """OTM quotes only (puts below F, calls at/above F); Black IVs with parity forwards."""
    c = c.merge(fwd[["expiry", "F", "DF"]], on="expiry")
    otm = ((c.cp == -1) & (c.strike < c.F)) | ((c.cp == 1) & (c.strike >= c.F))
    c = c[otm].copy()
    intrinsic = c.DF * np.maximum(c.cp * (c.F - c.strike), 0)
    c = c[c.mid > intrinsic + 0.01]
    c["iv"] = [implied_vol(r.mid, r.F, r.strike, r.T, r.DF, r.cp) for r in c.itertuples()]
    c["iv_true"] = [implied_vol(r.true_price, r.F, r.strike, r.T, r.DF, r.cp) for r in c.itertuples()]
    c = c.dropna(subset=["iv"])
    c["vega"] = black_vega(c.F, c.strike, c["T"], c.iv, c.DF)
    c["iv_halfspread"] = 0.5 * (c.ask - c.bid) / np.maximum(c.vega, 1e-8)
    c["k"] = np.log(c.strike / c.F)
    c["w"] = c.iv ** 2 * c["T"]
    return c.reset_index(drop=True)
