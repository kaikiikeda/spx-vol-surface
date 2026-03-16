"""Build the paper's experiment figures and generated/numbers.tex from pipeline outputs.

Run after run_all.py, experiments/vix_error.py and experiments/calibration_stability.py.
Every number quoted in paper.tex comes from a macro defined here.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_all import BLUE, INK2, ORANGE, AQUA, style  # noqa: E402
from volsurf import config as cfg  # noqa: E402

OUT = ROOT / "output"
EXP = OUT / "experiments"
GEN = Path(__file__).parent / "generated"
FIG = Path(__file__).parent / "figures"


def theory_disc_bias(step, A):
    """Leading-order VIX bias (vol pts) from dK^2/(6 T F^2) per term, added on top of V1."""
    N1, N2, N30 = A["T_near"] * cfg.MIN_PER_YEAR, A["T_next"] * cfg.MIN_PER_YEAR, 30 * 1440.0
    w1, w2 = (N2 - N30) / (N2 - N1), (N30 - N1) / (N2 - N1)
    d_vix2 = 1e4 * (365 * 1440.0 / N30) * step ** 2 / 6 * (w1 / A["F_near"] ** 2 + w2 / A["F_next"] ** 2)
    return np.sqrt(A["V1"] ** 2 + d_vix2) - A["V1"]


def fig_vix_experiment(A):
    rows = pd.DataFrame(A["rows"])
    steps = rows.step.values.astype(float)
    grid = np.geomspace(4, 120, 200)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.loglog(grid, theory_disc_bias(grid, A), color=INK2, lw=1.2, ls="--", label="theory  ΔK²/(6TF²)")
    a1.loglog(steps, rows.disc, "o", color=BLUE, ms=7, label="measured (true mids)")
    a1.set_xlabel("strike spacing ΔK (index points)"); a1.set_ylabel("discretisation bias (vol pts)")
    a1.set_title("(a) Discretisation bias is second order in ΔK", loc="left")
    a1.legend()
    a2.fill_between(steps, rows.total_q05, rows.total_q95, color=BLUE, alpha=0.18, lw=0,
                    label="5–95% over 200 quote seeds")
    a2.plot(steps, rows.total_mean, "o-", color=BLUE, ms=6, label="total error (mean)")
    a2.plot(grid, A["V1"] - A["V0"] + rows.trunc.mean() + theory_disc_bias(grid, A), color=ORANGE, lw=1.6,
            ls="--", label="interp. + trunc. + theory")
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xscale("log"); a2.set_xticks(steps); a2.set_xticklabels([str(int(s)) for s in steps])
    a2.set_xlabel("strike spacing ΔK (index points)"); a2.set_ylabel("replicated − true VIX (vol pts)")
    a2.set_title("(b) Total error and its predicted value", loc="left")
    a2.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig_exp_vix.png", dpi=200); plt.close(fig)


def fig_calib_experiment(B, seeds):
    names = B["names"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for col, scope, lab in [(BLUE, "full", "all 9 expiries"), (ORANGE, "short", "expiries ≤ 35 days")]:
        g = seeds[seeds.scope == scope]
        a1.scatter(g.kappa, g.theta, s=22, color=col, label=lab, edgecolor="white", linewidth=0.6)
    a1.axvline(cfg.TRUE_HESTON["kappa"], color=INK2, lw=0.8, ls=":")
    a1.axhline(cfg.TRUE_HESTON["theta"], color=INK2, lw=0.8, ls=":")
    a1.set_xlabel("κ (mean-reversion speed)"); a1.set_ylabel("θ (long-run variance)")
    a1.set_title(f"(a) κ–θ estimates over {B['n_seeds']} quote-noise seeds", loc="left")
    a1.legend()
    x = np.arange(len(names)); wdt = 0.26
    for i, (col, scope, lab) in enumerate([(BLUE, "full", "all expiries"),
                                           (AQUA, "full_true_fwd", "all, true forward"),
                                           (ORANGE, "short", "≤ 35 days")]):
        p = B["summary"][scope]["params"]
        a2.bar(x + (i - 1) * wdt, [p[n]["sd"] / p[n]["gn_se"] for n in names], wdt * 0.92, color=col, label=lab)
    a2.axhline(1, color=INK2, lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels(["v₀", "κ", "θ", "σ", "ρ"])
    a2.set_ylabel("empirical s.d. / Gauss–Newton s.e.")
    a2.set_title("(b) Naive standard errors understate dispersion", loc="left")
    a2.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig_exp_calib.png", dpi=200); plt.close(fig)


def macros(R, A, B):
    m = {}
    def put(name, val, fmt):
        txt = format(val, fmt)
        m[name] = r"\ensuremath{-}" + txt[1:] if txt.startswith("-") else txt
    c, s, v, h, mc = R["chain"], R["svi"], R["vix"], R["heston"], R["mc"]
    put("NRaw", c["cleaning"]["raw"], ",d"); put("NCrossed", c["cleaning"]["crossed"], "d")
    put("NZeroBid", c["cleaning"]["zero_bid"], ",d"); put("NWide", c["cleaning"]["too_wide"], "d")
    put("NIV", c["n_iv"], ",d")
    put("IVErrMed", c["iv_err_vs_truth_bp"]["median"], ".1f"); put("IVErrPnf", c["iv_err_vs_truth_bp"]["p95"], ".1f")
    fe = [abs(f["F"] - f["F_true"]) for f in c["forwards"]]; ri = [100 * f["r_implied"] for f in c["forwards"]]
    put("FwdErrMax", max(fe), ".2f"); put("RImpMin", min(ri), ".2f"); put("RImpMax", max(ri), ".2f")
    sl = s["slices"]
    put("SVIRmseMax", max(x["rmse_bp"] for x in sl), ".1f"); put("SVIgMin", min(x["g_min_extended"] for x in sl), ".2f")
    put("SVILeeMax", max(max(x["lee_left"], x["lee_right"]) for x in sl), ".3f")
    put("CalMinGap", min(x["min_dw"] for x in s["calendar"]) * 1e4, ".1f")
    put("VIXRep", v["vix_replicated"], ".3f"); put("VIXTrue", v["vix_true_30d"], ".3f")
    put("VIXErr", v["error_vol_pts"], "+.4f"); put("VIXErrAbs", abs(v["error_vol_pts"]), ".3f"); put("VIXInterp", v["vix_from_true_term_variances"], ".3f")
    put("VIXSVI", v["vix_from_svi_continuum"], ".3f")
    put("NearN", v["near"]["n_strikes"], "d"); put("NextN", v["next"]["n_strikes"], "d")
    for nm in ["v0", "kappa", "theta", "sigma", "rho"]:
        key = {"v0": "Vzero", "kappa": "Kappa", "theta": "Theta", "sigma": "Sigma", "rho": "Rho"}[nm]
        put("Cal" + key, h["params"][nm], ".4f")
    put("CalRmse", h["rmse_iv_bp"], ".1f"); put("CalSec", h["seconds"], ".1f"); put("CalNfev", h["nfev"], "d")
    put("Feller", h["feller"], ".2f")
    put("MCWithin", 100 * mc["share_within_2se"], ".0f"); put("MCMaxZ", mc["max_abs_z"], ".2f")
    put("MCSec", mc["seconds"], ".1f"); put("MCAbsErr", mc["mean_abs_price_err"], ".2f")
    rows = {r["step"]: r for r in A["rows"]}
    put("ASeeds", A["n_seeds"], "d")
    put("AInterp", A["V1"] - A["V0"], "+.4f"); put("ATruncFive", rows[5]["trunc"], "+.4f")
    put("ATruncHundred", rows[100]["trunc"], "+.4f")
    put("ADiscFive", rows[5]["disc"], "+.4f"); put("ADiscTwentyFive", rows[25]["disc"], "+.4f")
    put("ADiscFifty", rows[50]["disc"], "+.4f"); put("ADiscHundred", rows[100]["disc"], "+.3f")
    put("ADiscHundredAbs", rows[100]["disc"], ".2f")
    put("ANoiseSdFive", rows[5]["quote_sd"], ".4f"); put("ANoiseSdHundred", rows[100]["quote_sd"], ".4f")
    put("ATotalFive", rows[5]["total_mean"], "+.4f"); put("ATotalHundred", rows[100]["total_mean"], "+.3f")
    put("AThetaCoef", theory_disc_bias(100.0, A) / 1e4 * 1e5, ".2f")                 # x 1e-5 per pt^2
    put("AEmpCoef", rows[100]["disc"] / 1e4 * 1e5, ".2f")
    ratios = [x["ratio"] for x in A["forward_position_check"]]
    put("FPCMin", min(ratios), ".3f"); put("FPCMax", max(ratios), ".3f")
    idrel = max(abs(x["full_continuum"] - x["exact"]) / x["exact"] for x in A["identity"])
    m["IdentityRel"] = f"{idrel:.0e}".replace("e-0", r"\times 10^{-") + "}"
    sf, ss, st = (B["summary"][k]["params"] for k in ("full", "short", "full_true_fwd"))
    put("BSeeds", B["n_seeds"], "d")
    put("BKappaSdFull", sf["kappa"]["rel_sd_pct"], ".1f"); put("BKappaSdShort", ss["kappa"]["rel_sd_pct"], ".1f")
    put("BThetaSdFull", sf["theta"]["rel_sd_pct"], ".1f"); put("BThetaSdShort", ss["theta"]["rel_sd_pct"], ".1f")
    put("BRhoSdShort", ss["rho"]["rel_sd_pct"], ".1f"); put("BSigmaSdShort", ss["sigma"]["rel_sd_pct"], ".1f")
    names = B["names"]
    corr_s = np.array(B["summary"]["short"]["empirical_corr"]); corr_f = np.array(B["summary"]["full"]["empirical_corr"])
    put("BCorrKTShort", corr_s[1, 2], "+.2f"); put("BCorrKTFull", corr_f[1, 2], "+.2f")
    rat = lambda p, n: p[n]["sd"] / p[n]["gn_se"]
    put("BRatioMinFull", min(rat(sf, n) for n in names), ".1f"); put("BRatioMaxFull", max(rat(sf, n) for n in names), ".1f")
    put("BRatioMaxShort", max(rat(ss, n) for n in names), ".1f")
    put("BRatioThetaFull", rat(sf, "theta"), ".2f"); put("BRatioThetaTF", rat(st, "theta"), ".2f")
    put("BRatioVzeroFull", rat(sf, "v0"), ".2f"); put("BRatioVzeroTF", rat(st, "v0"), ".2f")
    put("BRatioSigmaTF", rat(st, "sigma"), ".2f"); put("BRatioRhoTF", rat(st, "rho"), ".2f")
    put("BThetaSdTF", st["theta"]["rel_sd_pct"], ".2f"); put("BThetaSdFullP", sf["theta"]["rel_sd_pct"], ".2f")
    # outlier-robust spread (1.4826 x MAD) from the seed file, and the outlier diagnosis
    seeds = pd.read_csv(EXP / "calibration_seeds.csv")
    for scope, tag in [("short", "Short"), ("full", "Full")]:
        g = seeds[seeds.scope == scope]
        for nm, key in [("kappa", "Kappa"), ("theta", "Theta")]:
            mad = 1.4826 * np.median(np.abs(g[nm] - g[nm].median()))
            put(f"B{key}MAD{tag}", 100 * mad / cfg.TRUE_HESTON[nm], ".1f")
    O = json.loads((EXP / "outlier.json").read_text())
    put("OSeed", O["seed"], "d"); put("ORmse", O["rmse_bp"], ".1f"); put("ORmseMed", O["median_rmse_bp"], ".1f")
    put("OKappa", O["params"]["kappa"], ".2f"); put("OTheta", O["params"]["theta"], ".4f")
    put("OStrike", O["quote"]["strike"], ".0f"); put("OBid", O["quote"]["bid"], ".2f"); put("OAsk", O["quote"]["ask"], ".2f")
    put("OTrue", O["quote"]["true_price"], ".2f"); put("OMidErr", O["quote"]["mid_err_bp"], ".0f")
    put("OResid", O["quote"]["residual_bp"], ".0f")
    put("ODropKappa", O["drop_params"]["kappa"], ".2f"); put("ORobKappa", O["robust_params"]["kappa"], ".2f")
    # martingale check: largest |E[F_T/F_0]-1| and its size in MC standard errors,
    # with sd(F_T/F_0) = sqrt(exp(E int v dt) - 1) under the hidden model
    import re as _re
    from volsurf.heston import expected_avg_variance
    devs = [(float(t), float(d)) for t, d in _re.findall(r"T=([\d.]+)\s+E\[F_T/F_0\]-1 = ([+-][\d.e+-]+)", mc["log"])]
    t_w, d_w = max(devs, key=lambda x: abs(x[1]))
    se = np.sqrt(np.exp(t_w * expected_avg_variance(t_w, **cfg.TRUE_HESTON)) - 1) / np.sqrt(mc["paths"])
    m["MCMartMax"] = f"{abs(d_w)/1e-4:.1f}" + r"\times 10^{-4}"
    put("MCMartZ", abs(d_w) / se, ".1f"); put("MCMartDays", t_w * 365, ".0f")
    Rb = json.loads((EXP / "robustness.json").read_text())
    put("RSVIGap", max(x["max_curve_gap_bp"] for x in Rb["svi_cross_check"]), ".3f")
    sp = Rb["heston_multistart"]["max_rel_spread"]
    m["RHestonSpread"] = f"{sp:.0e}".replace("e-0", r"\times 10^{-") + "}"
    mcs = Rb["mc_checks"]
    put("RMCSeedMax", max(x["max_abs_mean_z"] for x in mcs if x["steps_per_year"] == 1460), ".2f")
    put("RMCDaily", next(x["max_abs_mean_z"] for x in mcs if x["steps_per_year"] == 365), ".2f")
    put("RMCWeekly", next(x["max_abs_mean_z"] for x in mcs if x["steps_per_year"] == 52), ".2f")
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (GEN / "numbers.tex").write_text("% generated by build_assets.py -- do not edit\n" + "\n".join(lines) + "\n")
    return m


import re


def _minus(path):
    txt = path.read_text()
    path.write_text(re.sub(r"(?<![\w.$])-(?=\d)", r"\\ensuremath{-}", txt))


def tables(R, A, B):
    # SVI slice table
    t = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Expiry & $n$ & RMSE & RMSE & in & $\min g$ & Lee \\",
         r" & & quotes (bp) & truth (bp) & spread & & slope \\", r"\midrule"]
    for x in R["svi"]["slices"]:
        lab = x["label"].replace(" (monthly)", "").replace(" (weekly)", "w").replace(" (monthly, Thu)", "")
        t.append(f"{lab} ({x['T']*365:.0f}d) & {x['n']} & {x['rmse_bp']:.1f} & {x['rmse_vs_true_bp']:.1f} & "
                 f"{100*x['in_spread']:.0f}\\% & {x['g_min_extended']:.2f} & {max(x['lee_left'], x['lee_right']):.3f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_svi.tex").write_text("\n".join(t) + "\n")
    # VIX decomposition table
    t = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
         r"$\Delta K$ & strikes & interp. & trunc. & discret. & quote noise & total \\",
         r" & near/next & & & & mean $\pm$ s.d. & mean \\", r"\midrule"]
    for r in A["rows"]:
        t.append(f"{r['step']} & {r['n_strikes_near']}/{r['n_strikes_next']} & {r['interp']:+.4f} & {r['trunc']:+.4f} & "
                 f"{r['disc']:+.4f} & ${r['quote_mean']:+.4f} \\pm {r['quote_sd']:.4f}$ & {r['total_mean']:+.4f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_vix.tex").write_text("\n".join(t) + "\n")
    # calibration table
    sym = {"v0": r"$v_0$", "kappa": r"$\kappa$", "theta": r"$\theta$", "sigma": r"$\sigma$", "rho": r"$\rho$"}
    t = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{All 9 expiries} & \multicolumn{3}{c}{Expiries $\le$ 35 days} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Param. & True & Median & s.d. (\%) & s.d./GN & Median & s.d. (\%) & s.d./GN \\", r"\midrule"]
    for n in B["names"]:
        f, s_ = B["summary"]["full"]["params"][n], B["summary"]["short"]["params"][n]
        t.append(f"{sym[n]} & {f['true']:.4g} & {f['median']:.4f} & {f['rel_sd_pct']:.2f} & {f['sd']/f['gn_se']:.2f} & "
                 f"{s_['median']:.4f} & {s_['rel_sd_pct']:.2f} & {s_['sd']/s_['gn_se']:.2f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_calib.tex").write_text("\n".join(t) + "\n")
    for f in ("tab_svi.tex", "tab_vix.tex", "tab_calib.tex"):
        _minus(GEN / f)


def main():
    GEN.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    style()
    R = json.loads((OUT / "results.json").read_text())
    A = json.loads((EXP / "vix_error.json").read_text())
    B = json.loads((EXP / "calibration_stability.json").read_text())
    seeds = pd.read_csv(EXP / "calibration_seeds.csv")
    fig_vix_experiment(A); fig_calib_experiment(B, seeds)
    for f in ["fig_smiles", "fig_svi_error", "fig_arbitrage", "fig_vix", "fig_heston"]:
        (FIG / f"{f}.png").write_bytes((OUT / f"{f}.png").read_bytes())
    m = macros(R, A, B); tables(R, A, B)
    print(f"wrote {len(m)} macros, 3 tables, 7 figures")
    print({k: m[k] for k in ["AThetaCoef", "AEmpCoef", "FPCMin", "FPCMax", "IdentityRel", "BRatioMaxShort"]})


if __name__ == "__main__":
    main()
