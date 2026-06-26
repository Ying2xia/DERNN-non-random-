"""
Wrappers for the five methods compared in the paper:

1. Centralized ERNN - full-data penalized ERNN (benchmark)
2. Pilot ERNN       - penalized ERNN on pilot sample only
3. OS-ERNN          - one-shot PARAMETER averaging (NOT prediction averaging)
4. CSL-ERNN         - surrogate with worker 1 as reference
5. DPS-ERNN         - surrogate with Poisson pilot as reference (proposed)

CRITICAL:
    - OS-ERNN averages parameter vectors, then predicts with averaged network.
      It does NOT average predictions from local models.
    - DPS-ERNN uses exact full correction: no eta, no alpha, no trust-region,
      no backtracking.
"""
import torch
import numpy as np
import time
from typing import Dict, Any, List

from models import ERNN, init_model, copy_model, get_flat_params, set_flat_params
from losses import penalized_expectile_objective
from train import train_penalized_ernn, train_model
from surrogate import run_dps_ernn, run_csl_ernn
from partition import poisson_pilot_sample


def run_centralized(X_train: torch.Tensor, y_train: torch.Tensor,
                    tau: float, lambda_: float,
                    p: int, J: int, seed: int,
                    train_kwargs: dict,
                    n_inits: int = 1) -> Dict[str, Any]:
    """Method 1: Centralized ERNN - benchmark trained on full data."""
    t0 = time.time()
    if n_inits > 1:
        from train import train_penalized_ernn_multi_init
        model, info = train_penalized_ernn_multi_init(
            p, J, X_train, y_train, tau, lambda_, n_inits, seed, **train_kwargs)
    else:
        model = init_model(p, J, seed)
        info = train_penalized_ernn(model, X_train, y_train, tau, lambda_,
                                    **train_kwargs)
    total_time = time.time() - t0
    info["time_seconds"] = total_time
    return {"model": model, "info": info, "method": "Centralized"}


def run_pilot(pilot_X: torch.Tensor, pilot_y: torch.Tensor,
              tau: float, lambda_: float,
              p: int, J: int, seed: int,
              train_kwargs: dict,
              n_inits: int = 1) -> Dict[str, Any]:
    """Method 2: Pilot ERNN - trained only on pilot sample."""
    t0 = time.time()
    if n_inits > 1:
        from train import train_penalized_ernn_multi_init
        model, info = train_penalized_ernn_multi_init(
            p, J, pilot_X, pilot_y, tau, lambda_, n_inits, seed, **train_kwargs)
    else:
        model = init_model(p, J, seed)
        info = train_penalized_ernn(model, pilot_X, pilot_y, tau, lambda_,
                                    **train_kwargs)
    total_time = time.time() - t0
    info["time_seconds"] = total_time
    return {"model": model, "info": info, "method": "Pilot"}


def run_os_ernn(worker_data: List[dict],
                tau: float, lambda_: float,
                p: int, J: int, seed: int, N: int,
                train_kwargs: dict) -> Dict[str, Any]:
    """
    Method 3: OS-ERNN / One-Shot Parameter Averaging.

    CRITICAL: This is PARAMETER averaging, NOT prediction averaging.

    1. All workers start from the SAME initial parameters (required for
       meaningful parameter averaging).
    2. Each worker trains a local penalized ERNN on its own data.
    3. Master computes sample-size-weighted average of parameter vectors:
        theta_OS = sum_m (N_m / N) * theta_m
    4. Prediction uses the averaged parameter network:
        y_hat = f(x; theta_OS)

    This is NOT: (1/M) * sum_m f(x; theta_m)
    """
    M = len(worker_data)

    # Same initialization for all workers (essential for parameter averaging)
    template_model = init_model(p, J, seed)
    init_params = get_flat_params(template_model)

    t0 = time.time()
    local_params_list = []
    local_weights = []

    for m in range(M):
        local_model = copy_model(template_model)  # Same init for all
        X_m = worker_data[m]["X"]
        y_m = worker_data[m]["y"]
        N_m = worker_data[m]["N_m"]

        info_m = train_penalized_ernn(local_model, X_m, y_m, tau, lambda_,
                                      **train_kwargs)
        local_params_list.append(get_flat_params(local_model))
        local_weights.append(N_m / N)

    # Sample-size-weighted parameter averaging
    avg_params = torch.zeros_like(init_params)
    for m in range(M):
        avg_params += local_weights[m] * local_params_list[m]

    # Create model with averaged parameters
    os_model = copy_model(template_model)
    set_flat_params(os_model, avg_params)

    total_time = time.time() - t0
    return {
        "model": os_model,
        "info": {"time_seconds": total_time, "failed": False, "failure_reason": ""},
        "method": "OS-ERNN",
    }


def run_csl(worker_data: List[dict],
            tau: float, lambda_: float,
            p: int, J: int, seed: int, N: int,
            train_kwargs: dict,
            n_inits: int = 1,
            prox_rho: float = 0.0) -> Dict[str, Any]:
    """
    Method 4: CSL-ERNN.

    Uses worker 1's local loss as reference/curvature loss.
    Expansion point: penalized estimator trained on worker 1's data.
    """
    t0 = time.time()
    if n_inits > 1:
        from train import train_penalized_ernn_multi_init
        w1_model, _ = train_penalized_ernn_multi_init(
            p, J, worker_data[0]["X"], worker_data[0]["y"],
            tau, lambda_, n_inits, seed, **train_kwargs)
    else:
        w1_model = init_model(p, J, seed)
        train_penalized_ernn(w1_model, worker_data[0]["X"], worker_data[0]["y"],
                             tau, lambda_, **train_kwargs)

    result = run_csl_ernn(
        worker1_model=w1_model,
        worker1_X=worker_data[0]["X"],
        worker1_y=worker_data[0]["y"],
        worker_data=worker_data,
        tau=tau,
        lambda_=lambda_,
        N=N,
        train_kwargs=train_kwargs,
        prox_rho=prox_rho,
    )

    total_time = time.time() - t0
    result["info"]["time_seconds"] = total_time
    result["method"] = "CSL-ERNN"
    return result


def run_dps(worker_data: List[dict],
            pilot_X: torch.Tensor, pilot_y: torch.Tensor,
            tau: float, lambda_: float,
            p: int, J: int, seed: int, N: int,
            train_kwargs: dict,
            n_inits: int = 1,
            T: int = 1,
            prox_rho: float = 0.0) -> Dict[str, Any]:
    """
    Method 5: DPS-ERNN (proposed).

    Uses Poisson pilot loss as reference/curvature loss.
    Expansion point: penalized pilot estimator.
    T: number of communication rounds (T=1 is standard one-step).
    prox_rho: proximal/trust-region coefficient for the surrogate minimization.
    """
    t0 = time.time()
    if n_inits > 1:
        from train import train_penalized_ernn_multi_init
        pilot_model, _ = train_penalized_ernn_multi_init(
            p, J, pilot_X, pilot_y, tau, lambda_, n_inits, seed, **train_kwargs)
    else:
        pilot_model = init_model(p, J, seed)
        train_penalized_ernn(pilot_model, pilot_X, pilot_y, tau, lambda_,
                             **train_kwargs)

    result = run_dps_ernn(
        pilot_model=pilot_model,
        pilot_X=pilot_X,
        pilot_y=pilot_y,
        worker_data=worker_data,
        tau=tau,
        lambda_=lambda_,
        N=N,
        train_kwargs=train_kwargs,
        T=T,
        prox_rho=prox_rho,
    )

    total_time = time.time() - t0
    result["info"]["time_seconds"] = total_time
    result["method"] = f"DPS-ERNN(T={T})" if T > 1 else "DPS-ERNN"
    return result
