"""Black-76 on the forward: prices, vega, implied vol."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black_price(F, K, T, sigma, df, cp):
    """cp = +1 call, -1 put."""
    F, K, sigma = np.asarray(F, float), np.asarray(K, float), np.asarray(sigma, float)
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    return df * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))


def black_vega(F, K, T, sigma, df):
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    return df * F * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, F, K, T, df, cp):
    intrinsic = df * max(cp * (F - K), 0.0)
    upper = df * (F if cp == 1 else K)
    if not (intrinsic + 1e-12 < price < upper):
        return np.nan
    f = lambda s: black_price(F, K, T, s, df, cp) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-12, maxiter=200)
    except ValueError:
        return np.nan
