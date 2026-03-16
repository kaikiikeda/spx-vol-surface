"""Raw SVI slices and static-arbitrage diagnostics.

w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + s^2)),   k = ln(K/F),  w = sigma_imp^2 * T
"""
import numpy as np
from scipy.optimize import least_squares


def svi_w(k, a, b, rho, m, s):
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + s * s))


def svi_derivs(k, a, b, rho, m, s):
    x = k - m
    r = np.sqrt(x * x + s * s)
    w = a + b * (rho * x + r)
    w1 = b * (rho + x / r)
    w2 = b * s * s / r ** 3
    return w, w1, w2


def gatheral_g(k, params):
    """Butterfly condition: g(k) >= 0  <=>  risk-neutral density >= 0 (Gatheral & Jacquier 2014)."""
    w, w1, w2 = svi_derivs(k, *params)
    return (1 - k * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1 / w + 0.25) + w2 / 2


def fit_slice(k, iv, T, weights, n_starts=12, seed=0):
    """Weighted least squares on implied vol, with the no-negative-variance constraint
    a + b*s*sqrt(1-rho^2) >= 0 imposed through a penalty. Multi-start for robustness."""
    rng = np.random.default_rng(seed)
    w_mkt = iv ** 2 * T
    lo = np.array([-0.5 * w_mkt.max(), 1e-5, -0.999, -1.0, 1e-4])
    hi = np.array([w_mkt.max(), 5.0, 0.999, 1.0, 2.0])

    def resid(p):
        a, b, rho, m, s = p
        w = svi_w(k, *p)
        model_iv = np.sqrt(np.maximum(w, 1e-12) / T)
        pen = max(0.0, -(a + b * s * np.sqrt(1 - rho ** 2))) * 1e3
        return np.append(weights * (model_iv - iv), pen)

    best = None
    for i in range(n_starts):
        x0 = np.array([0.5 * w_mkt.min(), 0.1, -0.6, 0.0, 0.1]) if i == 0 else \
            lo + rng.uniform(0.05, 0.95, 5) * (hi - lo)
        x0[1:] = np.clip(x0[1:], lo[1:] + 1e-6, hi[1:] - 1e-6)
        r = least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=4000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x, best


def butterfly_report(params, T, k_lo, k_hi):
    """Min of g on the quoted range and on an extended range; Lee wing-slope check."""
    a, b, rho, m, s = params
    kq = np.linspace(k_lo, k_hi, 2001)
    kx = np.linspace(k_lo - 0.5, k_hi + 0.5, 4001)
    gq, gx = gatheral_g(kq, params), gatheral_g(kx, params)
    return dict(
        g_min_quoted=float(gq.min()),
        g_min_extended=float(gx.min()),
        k_at_g_min=float(kx[gx.argmin()]),
        neg_region_extended=bool((gx < 0).any()),
        lee_right=float(b * (1 + rho)),   # must be <= 2
        lee_left=float(b * (1 - rho)),    # must be <= 2
        min_w=float(a + b * s * np.sqrt(1 - rho ** 2)),
    )


def calendar_report(slices, k_grid):
    """Total variance must be non-decreasing in T at fixed forward log-moneyness."""
    out = []
    for (T1, p1, lab1), (T2, p2, lab2) in zip(slices[:-1], slices[1:]):
        diff = svi_w(k_grid, *p2) - svi_w(k_grid, *p1)
        out.append(dict(near=lab1, far=lab2, min_dw=float(diff.min()),
                        k_at_min=float(k_grid[diff.argmin()]),
                        violated_k_share=float((diff < 0).mean())))
    return out
