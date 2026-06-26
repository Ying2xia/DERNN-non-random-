"""
Data partitioning (storage) strategies (Section 4.2).

Strategy 1: Random storage
Strategy 2: Non-random by covariate summary S_i = sum(X_il)
Strategy 3: Non-random by first covariate X_i1

NOTE: For Example 1 (univariate), Strategies 2 and 3 are identical
because S_i = X_i1 when p = 1. This is expected behavior per the paper.
"""
import numpy as np
from typing import List, Tuple


def partition_data(X: np.ndarray, M: int, strategy: int,
                   rng: np.random.Generator = None
                   ) -> List[np.ndarray]:
    """
    Partition N observations into M workers.

    Args:
        X: (N, p) covariate matrix
        M: number of workers
        strategy: 1, 2, or 3
        rng: random generator (needed for strategy 1)

    Returns:
        List of M index arrays, one per worker
    """
    N = X.shape[0]

    if strategy == 1:
        # Strategy 1: Random storage
        if rng is None:
            rng = np.random.default_rng(0)
        perm = rng.permutation(N)
        return _split_indices(perm, M)

    elif strategy == 2:
        # Strategy 2: Sort by covariate summary S_i = sum_l X_il
        S = X.sum(axis=1)
        order = np.argsort(S)
        return _split_indices(order, M)

    elif strategy == 3:
        # Strategy 3: Sort by first covariate X_i1
        order = np.argsort(X[:, 0])
        return _split_indices(order, M)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _split_indices(indices: np.ndarray, M: int) -> List[np.ndarray]:
    """Split an array of indices into M roughly equal parts."""
    return [chunk.copy() for chunk in np.array_split(indices, M)]


def poisson_pilot_sample(N: int, M: int, worker_indices: List[np.ndarray],
                         pilot_ratio: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Poisson pilot sampling (Section 3.1).

    Each worker independently includes each of its observations with
    probability n/N = pilot_ratio. This is equivalent to global Poisson
    sampling and represents the global empirical distribution.

    Args:
        N: total sample size
        M: number of workers
        worker_indices: list of index arrays per worker
        pilot_ratio: r = n/N
        rng: random generator

    Returns:
        pilot_indices: 1D array of selected global indices
    """
    pilot_list = []
    for m in range(M):
        local_idx = worker_indices[m]
        # Bernoulli(pilot_ratio) independently for each observation
        include = rng.random(len(local_idx)) < pilot_ratio
        pilot_list.append(local_idx[include])

    return np.concatenate(pilot_list) if pilot_list else np.array([], dtype=int)
