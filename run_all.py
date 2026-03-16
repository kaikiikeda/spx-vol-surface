"""End-to-end pipeline: chain -> IVs -> SVI + arbitrage -> VIX -> Heston + C++ MC.

    python run_all.py            # everything (builds the C++ pricer if needed)
    python run_all.py --no-mc    # skip the Monte Carlo stage
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volsurf import config as cfg
from volsurf.calibrate import NAMES, calibrate, model_prices, select_quotes
from volsurf.chain import clean, generate_chain, implied_vols, parity_forwards
from volsurf.heston import expected_avg_variance
from volsurf.svi import butterfly_report, calendar_report, fit_slice, gatheral_g, svi_w
from volsurf.vix import pick_terms, svi_term_variance, term_variance, vix_from_terms

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "output"

# reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titlecolor": INK,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def stage_chain(res):
    chain = generate_chain()
    c, rep = clean(chain)
    fwd = parity_forwards(c)
    iv = implied_vols(c, fwd)
    res["chain"] = dict(snapshot=str(cfg.SNAPSHOT), spot=cfg.SPOT, cleaning=rep, n_iv=len(iv),
                        iv_err_vs_truth_bp=dict(median=float(1e4 * (iv.iv - iv.iv_true).abs().median()),
                                                p95=float(1e4 * (iv.iv - iv.iv_true).abs().quantile(.95))),
                        forwards=[dict(label=lab, T=float(r.T), F=float(r.F), F_true=float(r.F_true),
                                       r_implied=float(r.r_implied))
                                  for lab, r in zip([e[2] for e in cfg.EXPIRIES], fwd.itertuples())])
    return chain, iv


def stage_svi(iv, res):
    slices, rows = [], []
    for (lab, T), g in iv.groupby(["label", "T"], sort=False):
        wts = 1 / np.maximum(g.iv_halfspread.values, 0.002)
        p, _ = fit_slice(g.k.values, g.iv.values, T, wts)
        model = np.sqrt(svi_w(g.k.values, *p) / T)
        br = butterfly_report(p, T, g.k.min(), g.k.max())
        rows.append(dict(label=lab, T=float(T), params=dict(zip("a b rho m s".split(), map(float, p))),
                         n=len(g), rmse_bp=float(1e4 * np.sqrt(np.mean((model - g.iv) ** 2))),
                         rmse_vs_true_bp=float(1e4 * np.sqrt(np.mean((model - g.iv_true) ** 2))),
                         in_spread=float(np.mean(np.abs(model - g.iv) <= g.iv_halfspread)), **br))
        slices.append((T, p, lab))
    cal = calendar_report(slices, np.linspace(-1.2, 0.5, 1701))
    res["svi"] = dict(slices=rows, calendar=cal,
                      butterfly_free=all(not r["neg_region_extended"] for r in rows),
                      lee_ok=all(max(r["lee_left"], r["lee_right"]) <= 2 for r in rows),
                      calendar_free=all(c["min_dw"] >= 0 for c in cal))
    return slices


def stage_vix(chain, iv, slices, res):
    mid = (chain.bid + chain.ask) / 2
    rel = (chain.ask - chain.bid) / mid.where(mid > 0)
    bad = (chain.ask < chain.bid) | ((rel > 0.5) & (chain.ask - chain.bid > 1) & (chain.bid > 0))
    ch = chain[~bad]
    e1, e2 = pick_terms(cfg.EXPIRIES)
    T1, T2 = cfg.year_frac(e1), cfg.year_frac(e2)
    r1 = term_variance(ch[ch.expiry == e1], T1, cfg.R)
    r2 = term_variance(ch[ch.expiry == e2], T2, cfg.R)
    tv1, tv2 = (expected_avg_variance(T, **cfg.TRUE_HESTON) for T in (T1, T2))
    p_by_T = {round(T, 10): p for T, p, _ in slices}
    s1 = svi_term_variance(p_by_T[round(T1, 10)], r1["F"], T1, cfg.R)
    s2 = svi_term_variance(p_by_T[round(T2, 10)], r2["F"], T2, cfg.R)
    vix = vix_from_terms(r1["sigma2"], T1, r2["sigma2"], T2)
    truth = 100 * np.sqrt(expected_avg_variance(30 / 365, **cfg.TRUE_HESTON))
    res["vix"] = dict(
        near=dict(expiry=str(e1), days=T1 * 365, F=r1["F"], K0=r1["K0"], n_strikes=r1["n_strikes"],
                  strike_range=[r1["k_lo"], r1["k_hi"]], vol=100 * np.sqrt(r1["sigma2"]),
                  vol_svi_continuum=100 * np.sqrt(s1), vol_true=100 * np.sqrt(tv1)),
        next=dict(expiry=str(e2), days=T2 * 365, F=r2["F"], K0=r2["K0"], n_strikes=r2["n_strikes"],
                  strike_range=[r2["k_lo"], r2["k_hi"]], vol=100 * np.sqrt(r2["sigma2"]),
                  vol_svi_continuum=100 * np.sqrt(s2), vol_true=100 * np.sqrt(tv2)),
        vix_replicated=vix, vix_true_30d=truth, error_vol_pts=vix - truth,
        vix_from_true_term_variances=vix_from_terms(tv1, T1, tv2, T2),
        vix_from_svi_continuum=vix_from_terms(s1, T1, s2, T2),
    )
    return r1, r2


def stage_heston(iv, res, run_mc=True):
    q = select_quotes(iv)
    cal = calibrate(q)
    p = cal["params"]
    res["heston"] = dict(n_quotes=len(q), params=dict(zip(NAMES, map(float, p))), true=cfg.TRUE_HESTON,
                         rmse_iv_bp=1e4 * cal["rmse_iv"], nfev=cal["nfev"], seconds=cal["seconds"],
                         feller=float(2 * p[1] * p[2] - p[3] ** 2))
    q["model"] = model_prices(q, p)
    if run_mc:
        exe = ROOT / "cpp" / "heston_mc"
        if not exe.exists():
            subprocess.run(["make", "-C", str(ROOT / "cpp")], check=True)
        q["id"] = np.arange(len(q))
        q[["id", "T", "F", "DF", "strike", "cp"]].rename(columns={"strike": "K"}).to_csv(DATA / "mc_in.csv", index=False)
        t0 = time.time()
        r = subprocess.run([str(exe), str(DATA / "mc_in.csv"), str(DATA / "mc_out.csv"),
                            *map(repr, map(float, p)), "200000", "1460", "7"],
                           capture_output=True, text=True, check=True)
        mc = pd.read_csv(DATA / "mc_out.csv").sort_values("id")
        q["mc"], q["mc_se"] = mc.price.values, mc.stderr.values
        z = (q.mc - q.model) / q.mc_se
        res["mc"] = dict(paths=200000, steps_per_year=1460, seconds=time.time() - t0, log=r.stderr.strip(),
                         share_within_2se=float((z.abs() < 2).mean()), max_abs_z=float(z.abs().max()),
                         mean_abs_price_err=float((q.mc - q.model).abs().mean()))
    return q


# ---------------------------------------------------------------- figures
def fig_smiles(iv, slices):
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.6), sharey=False)
    for ax, (T, p, lab) in zip(axes.flat, slices):
        g = iv[iv["T"] == T]
        ax.errorbar(g.k, 100 * g.iv, yerr=100 * g.iv_halfspread, fmt="none", ecolor="#c3c2b7", lw=0.8, zorder=1)
        ax.scatter(g.k, 100 * g.iv, s=9, color=BLUE, zorder=2, label="Market mid IV")
        kk = np.linspace(g.k.min(), g.k.max(), 400)
        ax.plot(kk, 100 * np.sqrt(svi_w(kk, *p) / T), color=ORANGE, zorder=3, label="SVI fit")
        ax.set_title(f"{lab}  ·  {T*365:.0f}d", loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("log-moneyness  k = ln(K/F)")
    for ax in axes[:, 0]:
        ax.set_ylabel("implied vol (%)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("SPX implied-volatility smiles, 13 Feb 2026 close — raw SVI per expiry", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_smiles.png", dpi=150)
    plt.close(fig)


def fig_svi_error(iv, slices):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for col, lab in zip([BLUE, ORANGE, AQUA], ["Feb 20 (monthly)", "Apr 17 (monthly)", "Dec 18 (monthly)"]):
        T, p, _ = next(s for s in slices if s[2] == lab)
        g = iv[iv["T"] == T].sort_values("k")
        sd = g.iv.median() * np.sqrt(T)
        ax.plot(g.k / sd, 1e4 * (np.sqrt(svi_w(g.k, *p) / T) - g.iv_true), color=col, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("standardised moneyness  k / (σ√T)")
    ax.set_ylabel("SVI − true IV (bp)")
    ax.set_title("SVI shape error against the noiseless Heston smile", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_svi_error.png", dpi=150)
    plt.close(fig)


def fig_arbitrage(iv, slices):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    k = np.linspace(-1.2, 0.5, 1000)
    for col, (T, p, lab) in zip(SEQ, slices):
        a1.plot(k, svi_w(k, *p), color=col, lw=1.6, label=f"{T*365:.0f}d")
        a2.plot(k, gatheral_g(k, p), color=col, lw=1.6)
    a1.set_title("Total variance w(k) — curves never cross (no calendar arb)", loc="left")
    a1.set_xlabel("k = ln(K/F)"); a1.set_ylabel("w = σ²T")
    a1.legend(ncol=3, fontsize=8, title="days to expiry", title_fontsize=8)
    a2.axhline(0, color="#e34948", lw=1.0)
    a2.set_title("Gatheral g(k) — stays above 0 (no butterfly arb)", loc="left")
    a2.set_xlabel("k = ln(K/F)"); a2.set_ylabel("g(k)")
    a2.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_arbitrage.png", dpi=150)
    plt.close(fig)


def fig_vix(r1, r2):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for col, r, lab in [(BLUE, r1, "Near term (Mar 13, 28d)"), (ORANGE, r2, "Next term (Mar 20, 34.7d)")]:
        ax.plot(r["strikes"], 1e6 * r["contrib"] / r["dK"], color=col, label=lab, lw=1.6)
    ax.set_xlabel("strike")
    ax.set_ylabel("σ² per index point  (×10⁻⁶)")
    ax.set_title("Where the VIX comes from: 2·e^(RT)·Q(K) / (T·K²) by strike", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_vix.png", dpi=150)
    plt.close(fig)


def fig_heston(q, has_mc):
    fig, axes = plt.subplots(1, 2 if has_mc else 1, figsize=(11 if has_mc else 6, 3.8), squeeze=False)
    a1 = axes[0, 0]
    err = 1e4 * (q.model - q.mid) / q.vega
    sd = q.iv * np.sqrt(q["T"])
    a1.scatter(q.k / sd, err, s=9, color=BLUE)
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_title("Calibrated Heston − market (IV bp, all 9 expiries)", loc="left")
    a1.set_xlabel("standardised moneyness  k / (σ√T)"); a1.set_ylabel("IV error (bp)")
    if has_mc:
        a2 = axes[0, 1]
        z = (q.mc - q.model) / q.mc_se
        a2.scatter(np.arange(len(q)), z, s=9, color=BLUE)
        for y in (-2, 2):
            a2.axhline(y, color="#c3c2b7", lw=1.0, ls="--")
        a2.set_ylim(-4, 4)
        a2.set_title("C++ Monte Carlo vs semi-analytic: z = Δprice / std-error", loc="left")
        a2.set_xlabel("option (sorted by expiry, then strike)"); a2.set_ylabel("z-score")
    fig.tight_layout()
    fig.savefig(OUT / "fig_heston.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mc", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    style()
    res = {}
    t = time.time()
    chain, iv = stage_chain(res); print(f"[1] chain + IVs      {time.time()-t:5.1f}s"); t = time.time()
    chain.to_csv(DATA / "spx_chain_2026-02-13.csv", index=False)
    iv.to_csv(DATA / "spx_iv_2026-02-13.csv", index=False)
    slices = stage_svi(iv, res); print(f"[2] SVI + arbitrage  {time.time()-t:5.1f}s"); t = time.time()
    r1, r2 = stage_vix(chain, iv, slices, res); print(f"[3] VIX              {time.time()-t:5.1f}s"); t = time.time()
    q = stage_heston(iv, res, run_mc=not args.no_mc); print(f"[4] Heston + MC      {time.time()-t:5.1f}s")
    fig_smiles(iv, slices); fig_svi_error(iv, slices); fig_arbitrage(iv, slices); fig_vix(r1, r2)
    fig_heston(q, not args.no_mc)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    v = res["vix"]
    print(f"\nVIX replicated {v['vix_replicated']:.3f}  vs true 30d {v['vix_true_30d']:.3f}  "
          f"(error {v['error_vol_pts']:+.3f} vol pts)")
    print("Heston calibrated:", {k: round(x, 4) for k, x in res["heston"]["params"].items()})
    if "mc" in res:
        print(f"MC within 2 s.e.: {100*res['mc']['share_within_2se']:.1f}%")


if __name__ == "__main__":
    main()
