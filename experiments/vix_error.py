"""Experiment A: where does VIX replication error come from?

Sequential decomposition, in VIX (vol) points:
  V0  exact 30-day fair variance  theta + (v0-theta)(1-e^{-k tau})/(k tau)
  V1  Cboe time interpolation of the two *exact* term variances  -> interpolation error
  V2  continuum strike integral of true prices over the strike range the Cboe
      two-zero-bid rule keeps                                     -> truncation error
  V3  Cboe discrete sum on that strike grid, true mid prices      -> discretisation error
  V4  Cboe sum on dressed quotes (noise, tick rounding, zero bids)-> quote error
Swept over uniform strike spacings, 200 noise seeds each.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from volsurf import config as cfg
from volsurf.chain import dress_quotes
from volsurf.heston import expected_avg_variance, heston_price
from volsurf.vix import pick_terms, term_variance, vix_from_terms

OUT = Path(__file__).resolve().parents[1] / "output" / "experiments"
SPACINGS = [5, 10, 25, 50, 100]
N_SEEDS = 200
P = cfg.TRUE_HESTON
PARAMS = (P["v0"], P["kappa"], P["theta"], P["sigma"], P["rho"])


def continuum_var(F, T, k_lo, k_hi, n=20001):
    """(2/T) e^{RT} int Q(K)/K^2 dK with Q = OTM price about F (so K0 = F, no correction)."""
    k = np.linspace(k_lo, k_hi, n)
    K = F * np.exp(k)
    df = np.exp(-cfg.R * T)
    Q = np.where(K < F, heston_price(F, K, T, df, -1, PARAMS), heston_price(F, K, T, df, 1, PARAMS))
    return 2.0 / T * np.exp(cfg.R * T) * np.trapezoid(Q / K, k)   # dK/K^2 = dk/K


def chain_for(F, T, step, true_c, true_p, K, bid_ask=None):
    if bid_ask is None:            # true mids: bid = ask = true price (bid>0 where worth >= 1 tick)
        bc = np.where(true_c >= 0.05, true_c, 0.0); bp = np.where(true_p >= 0.05, true_p, 0.0)
        ac, ap = true_c, true_p
    else:
        (bc, ac), (bp, ap) = bid_ask
    return pd.DataFrame(dict(strike=np.r_[K, K], cp=np.r_[np.ones(K.size), -np.ones(K.size)].astype(int),
                             bid=np.r_[bc, bp], ask=np.r_[ac, ap]))


def forward_position_check(steps=(25, 50, 100), offsets=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Proposition check: discrete-minus-continuum bias / (dK^2 / (6 T F^2)) should be ~1
    wherever the forward sits between strikes. Uses true mid prices, near term only."""
    e1, _ = pick_terms(cfg.EXPIRIES)
    T = cfg.year_frac(e1); F = cfg.SPOT * np.exp((cfg.R - cfg.Q) * T); df = np.exp(-cfg.R * T)
    out = []
    for step in steps:
        for u in offsets:
            anchor = F - u * step                       # a strike sits u*step below F
            K = anchor + step * np.arange(np.ceil((0.6 * F - anchor) / step), np.floor((1.25 * F - anchor) / step) + 1)
            c = np.maximum(heston_price(F, K, T, df, 1, PARAMS), 0)
            p = np.maximum(heston_price(F, K, T, df, -1, PARAMS), 0)
            r = term_variance(chain_for(F, T, step, c, p, K), T, cfg.R)
            cont = continuum_var(F, T, np.log(r["k_lo"] / F), np.log(r["k_hi"] / F))
            out.append(dict(step=step, u=float((r["F"] - r["K0"]) / step),
                            ratio=float((r["sigma2"] - cont) / (step ** 2 / (6 * T * F ** 2)))))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e1, e2 = pick_terms(cfg.EXPIRIES)
    terms = []
    for e in (e1, e2):
        T = cfg.year_frac(e)
        F = cfg.SPOT * np.exp((cfg.R - cfg.Q) * T)
        terms.append(dict(T=T, F=F, df=np.exp(-cfg.R * T), tv=expected_avg_variance(T, **P)))
    (T1, T2), (tv1, tv2) = [t["T"] for t in terms], [t["tv"] for t in terms]
    V0 = 100 * np.sqrt(expected_avg_variance(30 / 365, **P))
    V1 = vix_from_terms(tv1, T1, tv2, T2)

    # identity check: full continuum replication reproduces the exact term variance
    identity = [dict(T=t["T"], full_continuum=continuum_var(t["F"], t["T"], -4.0, 2.0), exact=t["tv"])
                for t in terms]

    rows, seeds_out = [], []
    for step in SPACINGS:
        per_term = []
        for t in terms:
            K = np.arange(np.ceil(0.3 * t["F"] / step) * step, 2.0 * t["F"], step, dtype=float)
            c = np.maximum(heston_price(t["F"], K, t["T"], t["df"], 1, PARAMS), 0)
            p = np.maximum(heston_price(t["F"], K, t["T"], t["df"], -1, PARAMS), 0)
            # strike range kept by the Cboe rule on noise-free (rounded) quotes
            base = term_variance(chain_for(t["F"], t["T"], step, c, p, K,
                                           (dress_quotes(c, None, 0), dress_quotes(p, None, 0))), t["T"], cfg.R)
            kr = (np.log(base["k_lo"] / t["F"]), np.log(base["k_hi"] / t["F"]))
            s_cont = continuum_var(t["F"], t["T"], *kr)
            g = chain_for(t["F"], t["T"], step, c, p, K)
            g = g[(g.strike >= base["k_lo"]) & (g.strike <= base["k_hi"])]
            s_disc = term_variance(g, t["T"], cfg.R)["sigma2"]
            per_term.append(dict(K=K, c=c, p=p, s_cont=s_cont, s_disc=s_disc, n=base["n_strikes"],
                                 k_lo=base["k_lo"], k_hi=base["k_hi"], s_round=base["sigma2"]))
        V2 = vix_from_terms(per_term[0]["s_cont"], T1, per_term[1]["s_cont"], T2)
        V3 = vix_from_terms(per_term[0]["s_disc"], T1, per_term[1]["s_disc"], T2)
        V3r = vix_from_terms(per_term[0]["s_round"], T1, per_term[1]["s_round"], T2)
        v4 = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 * step + seed)
            s = []
            for t, pt in zip(terms, per_term):
                ba = (dress_quotes(pt["c"], rng), dress_quotes(pt["p"], rng))
                s.append(term_variance(chain_for(t["F"], t["T"], step, pt["c"], pt["p"], pt["K"], ba),
                                       t["T"], cfg.R)["sigma2"])
            v4.append(vix_from_terms(s[0], T1, s[1], T2))
        v4 = np.array(v4)
        seeds_out += [dict(step=step, seed=i, vix=float(v)) for i, v in enumerate(v4)]
        rows.append(dict(step=step, n_strikes_near=per_term[0]["n"], n_strikes_next=per_term[1]["n"],
                         range_near=[per_term[0]["k_lo"], per_term[0]["k_hi"]],
                         interp=V1 - V0, trunc=V2 - V1, disc=V3 - V2, rounding=V3r - V3,
                         quote_mean=v4.mean() - V3, quote_sd=v4.std(ddof=1),
                         total_mean=v4.mean() - V0, total_sd=v4.std(ddof=1),
                         total_q05=np.quantile(v4 - V0, 0.05), total_q95=np.quantile(v4 - V0, 0.95)))
        r = rows[-1]
        print(f"step {step:3d}: interp {r['interp']:+.4f} trunc {r['trunc']:+.4f} disc {r['disc']:+.4f} "
              f"quote {r['quote_mean']:+.4f}±{r['quote_sd']:.4f}  total {r['total_mean']:+.4f}  "
              f"n={r['n_strikes_near']}/{r['n_strikes_next']}")
    pd.DataFrame(seeds_out).to_csv(OUT / "vix_error_seeds.csv", index=False)
    fpc = forward_position_check()
    print("forward-position check ratios:", [round(x["ratio"], 3) for x in fpc])
    (OUT / "vix_error.json").write_text(json.dumps(dict(V0=V0, V1=V1, identity=identity, rows=rows,
                                                        forward_position_check=fpc,
                                                        n_seeds=N_SEEDS, T_near=T1, T_next=T2,
                                                        F_near=terms[0]["F"], F_next=terms[1]["F"]),
                                                   indent=2, default=float))


if __name__ == "__main__":
    main()
