"""
True conditional expectile targets for the simulation study.

Motivation
----------
The estimand of tau-expectile regression is the conditional tau-expectile
function q_tau(x), NOT individual noisy responses y. Scoring predictions
against the noisy test responses y_test is problematic for two reasons:

  (1) It is bounded below by the irreducible noise level, which dominates and
      masks the (small) differences in estimation quality between methods.
  (2) For tau != 0.5 it targets the WRONG functional. The minimizer of
      E|y - c| over c is the conditional median and of E(y - c)^2 is the
      conditional mean; neither equals the conditional tau-expectile. Hence
      MAE/RMSE against y reward estimators that are biased toward the centre
      of the conditional distribution, which is exactly what happens at the
      tails (tau = 0.1, 0.9): a method that under-shoots the expectile and sits
      closer to the bulk of the y's scores a *smaller* error while being a
      *worse* expectile estimator.

Under the multiplicative-noise data-generating mechanism used here,
    Y = m(x) + s(x) * eps,   s(x) > 0,
the conditional tau-expectile is
    q_tau(x) = m(x) + s(x) * e_tau,
where e_tau is the tau-expectile of the *base* error distribution eps. This
factorization holds because the tau-expectile is translation-equivariant and
positively-scale-equivariant, and s(x) > 0 throughout (s(x) in [0.04, 0.36] for
Example 1; s(x) = 1 for Examples 2-3).

The error distributions are used exactly as drawn in data_generation.py (NO
centering): N(0, 1), standard t with 3 df, and uncentered chi^2 with 2 df. In
particular, for tau = 0.5 the expectile equals the mean, so e_0.5 = 0 for the
normal and t3 errors and e_0.5 = 2 for the (uncentered) chi^2(2) errors.

Computing e_tau
---------------
e_tau is the unique root of the expectile first-order condition
    tau * E[(eps - q)_+] = (1 - tau) * E[(q - eps)_+].
We solve it by high-accuracy numerical integration of the error density
(scipy.integrate.quad) combined with Brent root-finding (scipy.optimize.brentq),
and cross-check against a large Monte-Carlo estimate. Results are cached.
"""
import warnings
from functools import lru_cache

import numpy as np
from scipy import stats
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq

from data_generation import conditional_moments


def _base_distribution(error_type: str):
    """Return the scipy frozen distribution for the base error eps."""
    if error_type == "normal":
        return stats.norm(loc=0.0, scale=1.0)
    elif error_type == "t3":
        return stats.t(df=3)
    elif error_type == "chi2":
        # Uncentered chi^2(2), matching data_generation.generate_errors.
        return stats.chi2(df=2)
    else:
        raise ValueError(f"Unknown error type: {error_type}")


def _expectile_foc(q: float, dist, tau: float, a: float, b: float) -> float:
    """
    Expectile first-order condition residual:
        g(q) = tau * E[(eps - q)_+] - (1 - tau) * E[(q - eps)_+].
    The tau-expectile is the unique root g(q) = 0. g is strictly decreasing in q,
    positive for small q and negative for large q.

    Integration uses finite bounds [a, b] taken at extreme quantiles of `dist`.
    The truncated tail mass is < 1e-10 and the omitted contribution is far below
    the root-finding tolerance (verified by the Monte-Carlo cross-check in
    error_expectile); finite bounds keep the integrands smooth and avoid scipy's
    infinite-interval quadrature warnings.
    """
    def upper_integrand(x):
        return (x - q) * dist.pdf(x)

    def lower_integrand(x):
        return (q - x) * dist.pdf(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        e_plus, _ = quad(upper_integrand, q, b, limit=200)
        e_minus, _ = quad(lower_integrand, a, q, limit=200)
    return tau * e_plus - (1.0 - tau) * e_minus


@lru_cache(maxsize=None)
def error_expectile(error_type: str, tau: float, mc_check: bool = True) -> float:
    """
    tau-expectile of the base error distribution eps (cached).

    Args:
        error_type: "normal", "t3", or "chi2"
        tau: expectile level in (0, 1)
        mc_check: if True, verify the quadrature-based root against a large
            Monte-Carlo estimate and raise if they disagree materially.

    Returns:
        e_tau (float)
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")

    dist = _base_distribution(error_type)

    # Finite integration / bracketing bounds at extreme quantiles.
    a = float(dist.ppf(1e-11))
    b = float(dist.ppf(1.0 - 1e-11))
    lo, hi = a, b  # the tau-expectile lies strictly inside this range

    e_tau = brentq(_expectile_foc, lo, hi, args=(dist, tau, a, b),
                   xtol=1e-12, rtol=1e-12, maxiter=200)

    if mc_check:
        rng = np.random.default_rng(20240617)
        z = dist.rvs(size=8_000_000, random_state=rng)
        # Empirical FOC residual at e_tau should be ~0.
        plus = np.mean(np.clip(z - e_tau, 0.0, None))
        minus = np.mean(np.clip(e_tau - z, 0.0, None))
        resid = tau * plus - (1.0 - tau) * minus
        scale = max(1.0, abs(e_tau))
        if abs(resid) > 5e-3 * scale:
            raise RuntimeError(
                f"error_expectile MC cross-check failed for {error_type}, "
                f"tau={tau}: residual={resid:.3e}")

    return float(e_tau)


def true_conditional_expectile(example: int, error_type: str,
                               X: np.ndarray, tau: float) -> np.ndarray:
    """
    Exact conditional tau-expectile q_tau(x) = m(x) + s(x) * e_tau on the rows
    of X, for the given example and base error distribution.

    Args:
        example: 1, 2, or 3
        error_type: "normal", "t3", or "chi2"
        X: (N, p) covariates
        tau: expectile level

    Returns:
        (N,) float64 array of true conditional tau-expectiles
    """
    loc, scale = conditional_moments(example, X)
    e_tau = error_expectile(error_type, tau)
    return loc + scale * e_tau
