"""Robustness checks quoted in the paper (run after run_all.py):

1. SVI: an independent quasi-explicit fit (De Marco & Martini 2009) on every slice.
2. Heston: calibration from three different starting points.
3. Monte Carlo: per-expiry mean z-score across seeds 1-4, and with coarse time steps.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from volsurf.calibrate import NAMES, calibrate, model_prices, select_quotes  # noqa: E402
from volsurf.svi import fit_slice, svi_w  # noqa: E402

OUT = ROOT / "output" / "experiments"


def qe_svi_fit(k, iv, T, wts):
    """Quasi-explicit SVI: for fixed (m, s) the slice is linear in (a, d, c); search (m, s)."""
    wm = iv ** 2 * T
    ww = wts / (2 * iv * T)

    def inner(ms):
        m, s = ms[0], abs(ms[1]) + 1e-5
        y = (k - m) / s
        A = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1)])
        (a, d, c), *_ = np.linalg.lstsq(A * ww[:, None], wm * ww, rcond=None)
        rho = np.clip(d / c, -0.999, 0.999) if c > 0 else 0.0
        return np.array([a, max(c / s, 1e-6), rho, m, s])

    def obj(ms):
        p = inner(ms)
        return np.sum((wts * (np.sqrt(np.maximum(svi_w(k, *p), 1e-12) / T) - iv)) ** 2)

    best = min((minimize(obj, [m0, s0], method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-12, maxiter=4000))
                for m0 in (-0.1, 0.0, 0.05, 0.1) for s0 in (0.01, 0.05, 0.2)), key=lambda r: r.fun)
    lo, hi = [-1, 1e-6, -0.999, -1, 1e-4], [1, 5, 0.999, 1, 2]
    r = least_squares(lambda p: wts * (np.sqrt(np.maximum(svi_w(k, *p), 1e-12) / T) - iv),
                      np.clip(inner(best.x), lo, hi), bounds=(lo, hi), x_scale="jac")
    return r.x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    iv = pd.read_csv(ROOT / "data" / "spx_iv_2026-02-13.csv", parse_dates=["expiry"])
    res = {}

    # 1. SVI cross-check
    svi = []
    for (lab, T), g in iv.groupby(["label", "T"], sort=False):
        wts = 1 / np.maximum(g.iv_halfspread.values, 0.002)
        p1, _ = fit_slice(g.k.values, g.iv.values, T, wts)
        p2 = qe_svi_fit(g.k.values, g.iv.values, T, wts)
        r1 = 1e4 * np.sqrt(np.mean((np.sqrt(svi_w(g.k.values, *p1) / T) - g.iv) ** 2))
        r2 = 1e4 * np.sqrt(np.mean((np.sqrt(svi_w(g.k.values, *p2) / T) - g.iv) ** 2))
        kk = np.linspace(g.k.min(), g.k.max(), 400)
        dvol = 1e4 * np.max(np.abs(np.sqrt(svi_w(kk, *p1) / T) - np.sqrt(svi_w(kk, *p2) / T)))
        svi.append(dict(label=lab, rmse_multistart=r1, rmse_quasi_explicit=r2, max_curve_gap_bp=dvol))
    res["svi_cross_check"] = svi
    print("SVI max curve gap (bp):", round(max(s["max_curve_gap_bp"] for s in svi), 3),
          " max RMSE diff:", round(max(abs(s["rmse_multistart"] - s["rmse_quasi_explicit"]) for s in svi), 3))

    # 2. Heston multi-start
    q = select_quotes(iv)
    starts = [(0.04, 1.5, 0.05, 0.6, -0.5), (0.01, 5.0, 0.02, 0.3, -0.2), (0.10, 0.5, 0.10, 1.5, -0.9)]
    cal = []
    for x0 in starts:
        c = calibrate(q, x0)
        cal.append(dict(start=list(x0), params=dict(zip(NAMES, map(float, c["params"]))),
                        rmse_bp=1e4 * c["rmse_iv"], seconds=c["seconds"]))
    P = np.array([[c["params"][n] for n in NAMES] for c in cal])
    res["heston_multistart"] = dict(runs=cal, max_rel_spread=float(np.max(np.ptp(P, axis=0) / np.abs(P.mean(0)))))
    print("Heston multistart max relative spread:", res["heston_multistart"]["max_rel_spread"])

    # 3. Monte Carlo seeds and step sizes
    p = cal[0]["params"]
    pv = [p[n] for n in NAMES]
    q["id"] = np.arange(len(q))
    mc_in = OUT / "mc_check_in.csv"
    q[["id", "T", "F", "DF", "strike", "cp"]].rename(columns={"strike": "K"}).to_csv(mc_in, index=False)
    an = model_prices(q, pv)
    exe = ROOT / "cpp" / "heston_mc"
    if not exe.exists():
        subprocess.run(["make", "-C", str(ROOT / "cpp")], check=True)
    mc = []
    for seed, spy in [(1, 1460), (2, 1460), (3, 1460), (4, 1460), (1, 365), (1, 52)]:
        out = OUT / "mc_check_out.csv"
        subprocess.run([str(exe), str(mc_in), str(out), *map(repr, pv), "200000", str(spy), str(seed)],
                       check=True, capture_output=True)
        m = pd.read_csv(out).sort_values("id")
        z = (m.price.values - an) / m.stderr.values
        per_exp = pd.Series(z).groupby(q["T"].values).mean()
        mc.append(dict(seed=seed, steps_per_year=spy, max_abs_mean_z=float(per_exp.abs().max()),
                       share_within_2se=float(np.mean(np.abs(z) < 2))))
        print(f"  MC seed {seed} steps/yr {spy}: max |mean z| per expiry {per_exp.abs().max():.2f}")
    res["mc_checks"] = mc
    for f in (mc_in, OUT / "mc_check_out.csv"):
        f.unlink(missing_ok=True)
    (OUT / "robustness.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
