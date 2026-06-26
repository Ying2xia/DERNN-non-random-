"""
Data generation for simulation studies (Section 4.1).

Three examples with increasing structural complexity.
Three error distributions: N(0,1), t(3), chi^2(2).

IMPORTANT: chi^2(2) is NOT centered. The latest manuscript removed the
centering term. Do NOT subtract 2 from chi-square errors anywhere.
"""
import torch
import numpy as np


def generate_errors(N: int, error_type: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate error terms.

    Args:
        N: number of observations
        error_type: one of "normal", "t3", "chi2"
        rng: numpy random generator

    Returns:
        (N,) array of errors
    """
    if error_type == "normal":
        return rng.standard_normal(N)
    elif error_type == "t3":
        return rng.standard_t(df=3, size=N)
    elif error_type == "chi2":
        # IMPORTANT: No centering! The manuscript uses chi^2(2) directly.
        # Do NOT subtract 2.
        return rng.chisquare(df=2, size=N)
    else:
        raise ValueError(f"Unknown error type: {error_type}")


def example1_mean(X: np.ndarray) -> np.ndarray:
    """
    Example 1 regression function (univariate):
        m(x) = (1 - x + 2x^2) * exp(-x^2)
    """
    return (1.0 - X + 2.0 * X**2) * np.exp(-X**2)


def example1_sigma(X: np.ndarray) -> np.ndarray:
    """
    Example 1 heteroscedastic scale:
        sigma(x) = (1 + 0.2x) / 5
    """
    return (1.0 + 0.2 * X) / 5.0


def generate_example1(N: int, error_type: str, rng: np.random.Generator):
    """
    Example 1: Univariate nonlinear heteroscedastic model.
        Y = (1 - X + 2X^2) exp(-X^2) + sigma(X) * eps
        X ~ U(-4, 4)
        sigma(x) = (1 + 0.2x) / 5

    Returns:
        X: (N, 1) covariates
        y: (N,) responses
    """
    X = rng.uniform(-4, 4, size=N)
    eps = generate_errors(N, error_type, rng)
    y = example1_mean(X) + example1_sigma(X) * eps
    return X.reshape(-1, 1), y


def generate_example2(N: int, error_type: str, rng: np.random.Generator):
    """
    Example 2: Additive bivariate nonlinear model.
        Y = sin(pi * X1) + sin(pi * X2) + eps
        X1, X2 ~ U(0, 1)

    Returns:
        X: (N, 2) covariates
        y: (N,) responses
    """
    X1 = rng.uniform(0, 1, size=N)
    X2 = rng.uniform(0, 1, size=N)
    eps = generate_errors(N, error_type, rng)
    y = np.sin(np.pi * X1) + np.sin(np.pi * X2) + eps
    return np.column_stack([X1, X2]), y


def generate_example3(N: int, error_type: str, rng: np.random.Generator):
    """
    Example 3: Complex bivariate interaction model.
        Y = 40*exp{-8[(X1-0.5)^2 + (X2-0.7)^2]}
              * exp{-8[(X1-0.2)^2 + (X2-0.5)^2]}
          +    exp{-8[(X1-0.7)^2 + (X2-0.2)^2]}
          + eps
        X1, X2 ~ U(0, 1)

    NOTE: The first two exponential terms are MULTIPLIED (not added).

    Returns:
        X: (N, 2) covariates
        y: (N,) responses
    """
    X1 = rng.uniform(0, 1, size=N)
    X2 = rng.uniform(0, 1, size=N)
    eps = generate_errors(N, error_type, rng)
    y = (40.0 * np.exp(-8.0 * ((X1 - 0.5)**2 + (X2 - 0.7)**2))
              * np.exp(-8.0 * ((X1 - 0.2)**2 + (X2 - 0.5)**2))
         + np.exp(-8.0 * ((X1 - 0.7)**2 + (X2 - 0.2)**2))
         + eps)
    return np.column_stack([X1, X2]), y


GENERATORS = {
    1: generate_example1,
    2: generate_example2,
    3: generate_example3,
}


def example2_location(X: np.ndarray) -> np.ndarray:
    """Example 2 conditional location: sin(pi X1) + sin(pi X2)."""
    x1, x2 = X[:, 0], X[:, 1]
    return np.sin(np.pi * x1) + np.sin(np.pi * x2)


def example3_location(X: np.ndarray) -> np.ndarray:
    """Example 3 conditional location (multiplicative first two terms)."""
    x1, x2 = X[:, 0], X[:, 1]
    return (40.0 * np.exp(-8.0 * ((x1 - 0.5) ** 2 + (x2 - 0.7) ** 2))
                 * np.exp(-8.0 * ((x1 - 0.2) ** 2 + (x2 - 0.5) ** 2))
            + np.exp(-8.0 * ((x1 - 0.7) ** 2 + (x2 - 0.2) ** 2)))


def conditional_moments(example: int, X: np.ndarray):
    """
    Return the conditional location m(x) and scale s(x) of the data-generating
    mechanism, i.e. the (location, scale) such that Y = m(x) + s(x) * eps.

    These are the exact DGP functions and are used to construct the TRUE
    conditional tau-expectile target q_tau(x) = m(x) + s(x) * e_tau for
    simulation evaluation (see targets.py). Examples 2 and 3 are homoscedastic
    (s(x) = 1); Example 1 is heteroscedastic with s(x) = sigma(x) > 0.

    Args:
        example: 1, 2, or 3
        X: (N, p) covariate array

    Returns:
        (location, scale): two (N,) float arrays
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if example == 1:
        return example1_mean(X[:, 0]), example1_sigma(X[:, 0])
    elif example == 2:
        return example2_location(X), np.ones(N)
    elif example == 3:
        return example3_location(X), np.ones(N)
    else:
        raise ValueError(f"Unknown example: {example}")


def generate_data(example: int, N: int, error_type: str,
                  seed: int = 42, return_truth: bool = False):
    """
    Generate data for a given example, sample size, and error type.

    Args:
        example: 1, 2, or 3
        N: number of observations
        error_type: "normal", "t3", or "chi2"
        seed: RNG seed
        return_truth: if True, also return the exact conditional location and
            scale of the data-generating mechanism (for building the true
            conditional expectile target used in simulation evaluation).

    Returns:
        If return_truth is False (default, backward compatible):
            X: (N, p) float32 covariates
            y: (N,) float32 responses
        If return_truth is True:
            X, y, truth where truth = {"location": (N,) float32,
                                       "scale": (N,) float32}
    """
    rng = np.random.default_rng(seed)
    gen_fn = GENERATORS[example]
    X, y = gen_fn(N, error_type, rng)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    if return_truth:
        loc, scale = conditional_moments(example, X)
        truth = {"location": loc.astype(np.float32),
                 "scale": scale.astype(np.float32)}
        return X, y, truth
    return X, y
