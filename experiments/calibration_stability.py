"""Experiment B: how well do the quotes identify Heston's parameters?

For N independent quote-noise seeds, regenerate the chain, clean it, rebuild parity
forwards and IVs, and recalibrate Heston on
  (full)  all nine expiries, and
  (short) only expiries <= 35 days, and
  (full_true_fwd) all expiries with the true forward/discount factor (control for
          forward-estimation error).
Reports dispersion of the estimates, their correlation, and compares the seed-to-seed
dispersion with the Gauss-Newton standard errors from a single calibration.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from volsurf import config as cfg
from volsurf.bs import black_price, black_vega
from volsurf.calibrate import NAMES, calibrate, select_quotes
from volsurf.chain import clean, generate_chain, parity_forwards

OUT = Path(__file__).resolve().parents[1] / "output" / "experiments"
N_SEEDS = 40


def iv_vec(price, F, K, T, df, cp, lo=1e-4, hi=5.0, iters=80):
    """Vectorised bisection Black implied vol (prices are monotone in sigma)."""
    lo, hi = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = black_price(F, K, T, mid, df, cp) > price
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    return 0.5 * (lo + hi)


def build_iv(seed, true_forward=False):
    c, _ = clean(generate_chain(seed))
    fwd = parity_forwards(c)
    if true_forward:   # control: remove forward-estimation error
        fwd = fwd.assign(F=fwd.F_true, DF=np.exp(-cfg.R * fwd["T"]))
    c = c.merge(fwd[["expiry", "F", "DF"]], on="expiry")
    otm = ((c.cp == -1) & (c.strike < c.F)) | ((c.cp == 1) & (c.strike >= c.F))
    c = c[otm & (c.mid > c.DF * np.maximum(c.cp * (c.F - c.strike), 0) + 0.01)].copy()
    c["iv"] = iv_vec(c.mid.values, c.F.values, c.strike.values, c["T"].values, c.DF.values, c.cp.values)
    c = c[(c.iv > 1e-3) & (c.iv < 4.9)]
    c["vega"] = black_vega(c.F, c.strike, c["T"], c.iv, c.DF)
    c["k"] = np.log(c.strike / c.F)
    return c.reset_index(drop=True)


def gauss_newton_se(cal):
    r = cal["result"]
    J, n, p = r.jac, r.fun.size, r.x.size
    s2 = 2 * r.cost / (n - p)
    cov = s2 * np.linalg.inv(J.T @ J)
    return np.sqrt(np.diag(cov)), cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, gn = [], {}
    t0 = time.time()
    for seed in range(N_SEEDS):
        iv = build_iv(10_000 + seed)
        iv_tf = build_iv(10_000 + seed, true_forward=True)
        for scope in ("full", "short", "full_true_fwd"):
            q_all = {"full": iv, "short": iv[iv["T"] <= 36 / 365], "full_true_fwd": iv_tf}[scope]
            q = select_quotes(q_all)
            cal = calibrate(q)
            rows.append(dict(seed=seed, scope=scope, n=len(q), rmse_bp=1e4 * cal["rmse_iv"],
                             **dict(zip(NAMES, map(float, cal["params"])))))
            if seed == 0:
                se, corr = gauss_newton_se(cal)
                gn[scope] = dict(se=dict(zip(NAMES, map(float, se))), corr=corr.tolist())
        if seed % 10 == 9:
            print(f"  {seed+1}/{N_SEEDS} seeds, {time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration_seeds.csv", index=False)
    summary = {}
    for scope, g in df.groupby("scope"):
        s = {}
        for nm in NAMES:
            x, tru = g[nm].values, cfg.TRUE_HESTON[nm]
            s[nm] = dict(true=tru, median=float(np.median(x)), sd=float(x.std(ddof=1)),
                         q05=float(np.quantile(x, .05)), q95=float(np.quantile(x, .95)),
                         bias_pct=float(100 * (x.mean() - tru) / abs(tru)),
                         rel_sd_pct=float(100 * x.std(ddof=1) / abs(tru)),
                         gn_se=gn[scope]["se"][nm])
        summary[scope] = dict(params=s, rmse_bp_median=float(g.rmse_bp.median()),
                              empirical_corr=g[NAMES].corr().values.tolist(), gn_corr=gn[scope]["corr"])
        print(scope, {nm: f"{s[nm]['median']:.4f} (sd {s[nm]['rel_sd_pct']:.1f}%)" for nm in NAMES})
    (OUT / "calibration_stability.json").write_text(json.dumps(dict(n_seeds=N_SEEDS, names=NAMES,
                                                                    summary=summary), indent=2))


if __name__ == "__main__":
    main()
