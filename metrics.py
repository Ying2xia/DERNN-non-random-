"""
Evaluation metrics: MAE, RMSE, RACT, and diagnostic metrics.
"""
import torch
import numpy as np
from models import ERNN, get_flat_params


def compute_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return (y_true - y_pred).abs().mean().item()


def compute_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return ((y_true - y_pred) ** 2).mean().sqrt().item()


def compute_ract(time_centralized: float, time_dps: float) -> float:
    """
    Relative Average Computing Time:
        RACT = Time_Centralized / Time_DPS
    """
    if time_dps <= 0:
        return float("nan")
    return time_centralized / time_dps


def compute_residual_proportions(model: ERNN, X: torch.Tensor,
                                 y: torch.Tensor) -> dict:
    """
    Compute proportions of near-zero residuals.
    Helps diagnose residual-zero issues relevant to expectile loss smoothness.
    """
    with torch.no_grad():
        y_pred = model(X)
        residuals = (y - y_pred).abs()

    return {
        "residual_prop_1e_minus_6": (residuals < 1e-6).float().mean().item(),
        "residual_prop_1e_minus_4": (residuals < 1e-4).float().mean().item(),
        "residual_prop_1e_minus_3": (residuals < 1e-3).float().mean().item(),
    }


def compute_param_distance(model_a: ERNN, model_b: ERNN) -> float:
    """L2 distance between parameter vectors of two models."""
    params_a = get_flat_params(model_a)
    params_b = get_flat_params(model_b)
    return (params_a - params_b).norm().item()


def compute_prediction_gap(model: ERNN, ref_model: ERNN,
                           X: torch.Tensor, y: torch.Tensor) -> dict:
    """
    Compute prediction gap relative to a reference model (typically centralized).
    """
    with torch.no_grad():
        pred = model(X)
        ref_pred = ref_model(X)

    mae = compute_mae(y, pred)
    ref_mae = compute_mae(y, ref_pred)
    rmse = compute_rmse(y, pred)
    ref_rmse = compute_rmse(y, ref_pred)

    return {
        "prediction_gap_MAE": mae - ref_mae,
        "prediction_gap_RMSE": rmse - ref_rmse,
    }


def evaluate_model(model: ERNN, X_test: torch.Tensor,
                   y_test: torch.Tensor) -> dict:
    """
    Evaluate a model against an observed response vector.

    NOTE: In the SIMULATION study, predictions should be scored against the
    true conditional tau-expectile (see evaluate_against_target), not against
    the noisy y_test, because the latter targets the wrong functional for
    tau != 0.5 and is dominated by irreducible noise. This y-based version is
    retained for the REAL-DATA experiments, where no ground-truth conditional
    expectile exists and held-out predictive error against observed y is the
    appropriate (and only available) metric.
    """
    with torch.no_grad():
        y_pred = model(X_test)
    return {
        "MAE": compute_mae(y_test, y_pred),
        "RMSE": compute_rmse(y_test, y_pred),
    }


def evaluate_against_target(model: ERNN, X_test: torch.Tensor,
                            target: torch.Tensor) -> dict:
    """
    Evaluate a model against a provided target vector.

    In the simulation study `target` is the TRUE conditional tau-expectile
    q_tau(x) on the test covariates (known by construction; see
    targets.true_conditional_expectile). MAE/RMSE against this target measure
    the estimation error of the conditional expectile function directly, with
    the irreducible noise removed and the correct functional as the reference.
    """
    with torch.no_grad():
        y_pred = model(X_test)
    return {
        "MAE": compute_mae(target, y_pred),
        "RMSE": compute_rmse(target, y_pred),
    }


def compute_bic(model: ERNN, X: torch.Tensor, y: torch.Tensor,
                tau: float) -> float:
    """
    BIC-type criterion for ERNN model selection (paper Section 4):

        BIC(J, lambda) = log[ (1/N) sum rho_tau(Y_i - f(X_i; theta_hat)) ]
                         + log(N) / (2N) * df

    where df = pJ + J + J + 1.

    Args:
        model: trained ERNN
        X: (N, p) training covariates
        y: (N,) training responses
        tau: expectile level

    Returns:
        BIC value (lower is better)
    """
    from losses import expectile_loss
    import math

    N = X.shape[0]
    with torch.no_grad():
        y_pred = model(X)

    avg_loss = expectile_loss(y, y_pred, tau, reduction="mean").item()
    # A non-finite average loss means the fit diverged (parameters blew up so the
    # predictions overflowed) or produced NaNs; a non-positive average loss is
    # likewise degenerate. In all these cases the candidate is unusable, so its
    # BIC is +inf (it will never be selected, and the selector counts it as a
    # failure rather than silently treating it as a valid score).
    if not math.isfinite(avg_loss) or avg_loss <= 0:
        return float("inf")

    df = model.d  # pJ + J + J + 1
    bic = math.log(avg_loss) + math.log(N) / (2.0 * N) * df
    return bic


def select_hyperparams_bic(X: torch.Tensor, y: torch.Tensor, tau: float,
                           J_grid: list, lambda_grid: list,
                           p: int, seed: int = 42,
                           train_kwargs: dict = None,
                           n_inits: int = 1,
                           verbose: bool = False) -> dict:
    """
    Select (J, lambda) by minimizing BIC over a grid.

    For each (J, lambda) pair, trains a centralized penalized ERNN on the
    full training data and evaluates BIC. Returns the best combination.

    Args:
        X: (N, p) training covariates
        y: (N,) responses
        tau: expectile level
        J_grid: list of hidden-node counts to try
        lambda_grid: list of regularization strengths to try
        p: input dimension
        seed: base random seed
        train_kwargs: training keyword arguments
        n_inits: number of random initializations per (J, lambda) pair
        verbose: print progress

    Returns:
        dict with "best_J", "best_lambda", "best_bic", "all_results"
    """
    from models import init_model
    from train import train_penalized_ernn, train_penalized_ernn_multi_init

    if train_kwargs is None:
        train_kwargs = dict(adam_epochs=200, adam_lr=1e-3,
                            lbfgs_max_iter=50, grad_clip=10.0,
                            early_stop_patience=30)

    best_bic = float("inf")
    best_J = J_grid[0]
    best_lambda = lambda_grid[0]
    all_results = []
    n_finite = 0          # candidates with a usable (finite) BIC
    fail_reasons = {}     # reason -> count, for diagnostics

    total = len(J_grid) * len(lambda_grid)
    count = 0

    for J in J_grid:
        for lam in lambda_grid:
            count += 1
            if n_inits > 1:
                model, info = train_penalized_ernn_multi_init(
                    p, J, X, y, tau, lam, n_inits, seed, **train_kwargs)
            else:
                model = init_model(p, J, seed)
                info = train_penalized_ernn(model, X, y, tau, lam, **train_kwargs)

            if info.get("failed", False):
                bic_val = float("inf")
                reason = info.get("failure_reason", "training failed") or "training failed"
            else:
                bic_val = compute_bic(model, X, y, tau)
                reason = "non-finite/degenerate loss (diverged)" if bic_val == float("inf") else None

            import math as _math
            if _math.isfinite(bic_val):
                n_finite += 1
            elif reason is not None:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

            all_results.append({
                "J": J, "lambda": lam, "BIC": bic_val,
                "train_loss": info.get("final_loss", float("nan")),
            })

            if bic_val < best_bic:
                best_bic = bic_val
                best_J = J
                best_lambda = lam

            if verbose:
                print(f"  [{count}/{total}] J={J}, lam={lam:.2f}  "
                      f"BIC={bic_val:.4f}  {'<-- best' if bic_val == best_bic else ''}")

    # Guard against the silent-fallback failure mode: if NO candidate yielded a
    # finite BIC, every fit diverged or failed, so best_(J,lambda) would just be
    # the first grid entry with best_bic = inf. Returning that quietly would run
    # the entire study at J_grid[0] with no signal that selection failed. Raise a
    # clear error instead so the cause (usually a too-large adam_lr causing the
    # fits to blow up) gets fixed before the main experiments run.
    if n_finite == 0:
        raise RuntimeError(
            f"BIC selection failed: all {total} (J, lambda) candidates produced a "
            f"non-finite BIC (every fit diverged or failed). Failure reasons: "
            f"{fail_reasons if fail_reasons else 'unknown'}. This usually means "
            f"the optimizer diverged -- lower adam_lr (e.g. 0.01) and/or check "
            f"grad_clip. NOT falling back to J={J_grid[0]}, lambda={lambda_grid[0]} "
            f"silently."
        )

    return {
        "best_J": best_J,
        "best_lambda": best_lambda,
        "best_bic": best_bic,
        "n_finite": n_finite,
        "n_failed": total - n_finite,
        "fail_reasons": fail_reasons,
        "all_results": all_results,
    }
