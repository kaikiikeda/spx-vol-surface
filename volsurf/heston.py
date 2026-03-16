"""Semi-analytic Heston pricing (Lewis single-integral form).

Uses the "little Heston trap" characteristic function (Albrecher et al. 2007),
which avoids the branch-cut discontinuity of the original formulation.
"""
import numpy as np


def cf_logforward(u, T, v0, kappa, theta, sigma, rho):
    """Characteristic function of X_T = ln(F_T / F_0) (a martingale exponent)."""
    iu = 1j * u
    beta = kappa - rho * sigma * iu
    d = np.sqrt(beta ** 2 + sigma ** 2 * (iu + u ** 2))
    g = (beta - d) / (beta + d)
    edT = np.exp(-d * T)
    C = kappa * theta / sigma ** 2 * ((beta - d) * T - 2.0 * np.log((1 - g * edT) / (1 - g)))
    D = (beta - d) / sigma ** 2 * (1 - edT) / (1 - g * edT)
    return np.exp(C + D * v0)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(600)


def heston_call(F, K, T, df, params):
    """Vector of call prices for one expiry. params = (v0, kappa, theta, sigma, rho).

    Lewis (2001):  C = df * [F - sqrt(FK)/pi * int_0^inf Re(e^{iuk} phi(u - i/2)) / (u^2 + 1/4) du],
    k = ln(F/K).
    """
    v0, kappa, theta, sigma, rho = params
    K = np.atleast_1d(np.asarray(K, float))
    vbar = max(min(v0, theta), 1e-3)
    # Heston's cf decays only exponentially in u, so integrate well past the Gaussian scale
    umax = min(5000.0, max(250.0, np.sqrt(250.0 / (vbar * T))))
    u = 0.5 * umax * (_GL_X + 1.0)
    w = 0.5 * umax * _GL_W
    phi = cf_logforward(u - 0.5j, T, v0, kappa, theta, sigma, rho)
    k = np.log(F / K)[:, None]
    integrand = np.real(np.exp(1j * u[None, :] * k) * phi[None, :]) / (u ** 2 + 0.25)[None, :]
    integral = integrand @ w
    return df * (F - np.sqrt(F * K) / np.pi * integral)


def heston_price(F, K, T, df, cp, params):
    """Calls and puts (put via parity) for one expiry; cp array of +1/-1."""
    K = np.atleast_1d(np.asarray(K, float))
    cp = np.broadcast_to(np.asarray(cp), K.shape)
    c = heston_call(F, K, T, df, params)
    return np.where(cp == 1, c, c - df * (F - K))


def expected_avg_variance(tau, v0, kappa, theta, **_):
    """E[(1/tau) int_0^tau v_t dt] under Q: the fair variance-swap strike."""
    return theta + (v0 - theta) * (1 - np.exp(-kappa * tau)) / (kappa * tau)
