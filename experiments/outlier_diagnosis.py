"""Diagnose the worst short-expiry calibration from Experiment B.

Finds the seed with the largest calibration RMSE in the 'short' scope, identifies the
quote with the largest vega-scaled residual, then refits (i) without that quote and
(ii) with a soft-L1 robust loss. Run after calibration_stability.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_stability import build_iv  # noqa: E402
from volsurf.calibrate import HI, LO, NAMES, calibrate, model_prices, select_quotes  # noqa: E402

EXP = Path(__file__).resolve().parents[1] / "output" / "experiments"


def main():
    seeds = pd.read_csv(EXP / "calibration_seeds.csv")
    s = seeds[seeds.scope == "short"]
    worst = s.loc[s.rmse_bp.idxmax()]
    seed = int(worst.seed)
    q = select_quotes(build_iv(10_000 + seed)[lambda d: d["T"] <= 36 / 365])
    base = calibrate(q)
    res = (model_prices(q, base["params"]) - q.mid.values) / q.vega.values
    i = int(np.abs(res).argmax())
    r = q.iloc[i]
    drop = calibrate(q.drop(q.index[i]).reset_index(drop=True))
    rob = least_squares(lambda p: (model_prices(q, p) - q.mid.values) / q.vega.values,
                        [0.04, 1.5, 0.05, 0.6, -0.5], bounds=(LO, HI), loss="soft_l1",
                        f_scale=10e-4, x_scale="jac")
    out = dict(seed=seed, rmse_bp=1e4 * base["rmse_iv"], median_rmse_bp=float(s.rmse_bp.median()),
               params=dict(zip(NAMES, map(float, base["params"]))),
               quote=dict(label=r.label, strike=float(r.strike), cp=int(r.cp), bid=float(r.bid),
                          ask=float(r.ask), true_price=float(r.true_price), vega=float(r.vega),
                          residual_bp=float(1e4 * res[i]),
                          mid_err_bp=float(1e4 * (r.mid - r.true_price) / r.vega)),
               drop_params=dict(zip(NAMES, map(float, drop["params"]))), drop_rmse_bp=1e4 * drop["rmse_iv"],
               robust_params=dict(zip(NAMES, map(float, rob.x))))
    (EXP / "outlier.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
