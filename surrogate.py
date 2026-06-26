"""
Surrogate loss construction and optimization for DPS-ERNN and CSL-ERNN.

DPS-ERNN surrogate (eq. 14 of paper):
    L_tilde_{N,tau,lambda}(theta; theta_bar)
      = L_{P,tau}(theta)
        - <grad L_{P,tau}(theta_bar) - grad L_{N,tau}(theta_bar), theta>
        + lambda * R(theta)

CSL-ERNN surrogate (analogous, using worker 1's loss as reference):
    L_tilde_{CSL,tau,lambda}(theta; theta_bar)
      = L_{1,tau}(theta)
        - <grad L_{1,tau}(theta_bar) - grad L_{N,tau}(theta_bar), theta>
        + lambda * R(theta)

CRITICAL DESIGN CHOICES:
    1. Correction uses ONLY data-loss gradients (no penalty)
    2. No eta step-size, no alpha damping, no trust-region, no backtracking
    3. Final estimator = direct minimizer of the surrogate objective
    4. Penalty added separately on master side
"""
import torch
from typing import List, Dict, Any

from models import (ERNN, copy_model, get_flat_params, set_flat_params,
                    compute_flat_grad, flatten_params_with_grad)
from losses import data_expectile_objective, hidden_weight_penalty
from train import train_model


def compute_data_gradient(model: ERNN, X: torch.Tensor, y: torch.Tensor,
                          tau: float) -> torch.Tensor:
    """
    Compute the flat gradient of the DATA-ONLY expectile loss at current params.
    Does NOT include the penalty gradient.

    Returns:
        (d,) tensor of gradients in paper ordering
    """
    model.zero_grad()
    loss = data_expectile_objective(model, X, y, tau)
    return compute_flat_grad(loss, model)


def aggregate_full_data_gradient(model: ERNN,
                                 worker_data: List[dict],
                                 tau: float,
                                 N: int) -> torch.Tensor:
    """
    Compute the sample-size-weighted full-data gradient:
        grad L_{N,tau}(theta) = sum_m (N_m / N) * grad L_{m,tau}(theta)

    Args:
        model: ERNN model at the expansion point
        worker_data: list of dicts with keys "X", "y", "N_m"
        tau: expectile level
        N: total sample size

    Returns:
        (d,) tensor - full-data gradient (data loss only, no penalty)
    """
    full_grad = torch.zeros(model.d, device=next(model.parameters()).device)
    for wd in worker_data:
        local_grad = compute_data_gradient(model, wd["X"], wd["y"], tau)
        full_grad += (wd["N_m"] / N) * local_grad
    return full_grad


def build_surrogate_objective(model_template: ERNN,
                              reference_X: torch.Tensor,
                              reference_y: torch.Tensor,
                              correction_vector: torch.Tensor,
                              tau: float,
                              lambda_: float,
                              theta_bar_vec: torch.Tensor = None,
                              prox_rho: float = 0.0):
    """
    Build the surrogate objective function for DPS-ERNN or CSL-ERNN.

    The surrogate objective is:
        L_tilde(theta) = L_ref(theta) - <correction, theta> + lambda * R(theta)
                         + (prox_rho / 2) * ||theta - theta_bar||^2

    where:
        correction = grad L_ref(theta_bar) - grad L_{N,tau}(theta_bar)

    The optional proximal term (prox_rho > 0, anchored at theta_bar_vec) keeps the
    minimizer in a neighborhood of the expansion point. The linear correction
    term is unbounded below, and the reference loss of a (nonconvex) ERNN has
    many near-flat directions, so without the proximal term the minimizer can
    drift arbitrarily far along low-curvature directions, yielding a lower
    surrogate value but worse generalization. With the proximal term the one-step
    update solves (H + prox_rho * I) Delta theta = correction, a well-conditioned
    Tikhonov-regularized Newton step (the DANE/proximal-surrogate form). prox_rho
    = 0 recovers the unconstrained objective.

    Args:
        model_template: not used directly, just for reference
        reference_X: pilot data or worker-1 data (for the curvature term)
        reference_y: corresponding responses
        correction_vector: grad L_ref(theta_bar) - grad L_{N,tau}(theta_bar)
        tau: expectile level
        lambda_: regularization parameter
        theta_bar_vec: flat parameter vector of the expansion point (required if
            prox_rho > 0); the proximal term is anchored here.
        prox_rho: proximal coefficient (>= 0).

    Returns:
        Callable objective function: model -> scalar loss
    """
    # Detach the correction vector (it's a constant w.r.t. optimization)
    corr = correction_vector.detach().clone()

    use_prox = prox_rho and prox_rho > 0.0
    if use_prox:
        if theta_bar_vec is None:
            raise ValueError("prox_rho > 0 requires theta_bar_vec (the "
                             "expansion point) to anchor the proximal term.")
        tbar = theta_bar_vec.detach().clone()

    def surrogate_obj(model: ERNN) -> torch.Tensor:
        # 1. Reference data loss (provides curvature)
        ref_loss = data_expectile_objective(model, reference_X, reference_y, tau)

        # 2. Linear correction: - <correction, theta>
        #    NOTE: theta MUST be obtained via flatten_params_with_grad (graph-
        #    connected). Using the detached get_flat_params here would give the
        #    linear term a zero gradient, silently removing the gradient
        #    correction from the optimized surrogate (it would collapse to the
        #    penalized reference-loss objective).
        theta = flatten_params_with_grad(model)
        linear_term = torch.dot(corr, theta)

        # 3. Penalty (added on master side)
        penalty = hidden_weight_penalty(model)

        obj = ref_loss - linear_term + lambda_ * penalty

        # 4. Optional proximal (trust-region) term anchored at theta_bar
        if use_prox:
            obj = obj + 0.5 * prox_rho * torch.sum((theta - tbar) ** 2)

        return obj

    return surrogate_obj


def run_dps_ernn(pilot_model: ERNN,
                 pilot_X: torch.Tensor,
                 pilot_y: torch.Tensor,
                 worker_data: List[dict],
                 tau: float,
                 lambda_: float,
                 N: int,
                 train_kwargs: dict,
                 T: int = 1,
                 prox_rho: float = 0.0) -> Dict[str, Any]:
    """
    Run the DPS-ERNN procedure with T communication rounds.

    Following Wang et al. (2022) Algorithm 1:
      for t = 0, ..., T-1:
        1. At current expansion point theta^(t), compute pilot gradient
        2. Aggregate full-data gradient from workers at theta^(t)
        3. Construct correction and surrogate
        4. Minimize surrogate -> theta^(t+1)

    T=1 is the standard one-step surrogate estimator.
    T>1 iteratively rebuilds the surrogate at each new solution,
    shrinking the approximation error by O(n^{-1/2}) per round.

    prox_rho: coefficient of the proximal term (rho/2)||theta - theta^(t)||^2
        anchored at the current expansion point. rho > 0 keeps the surrogate
        minimizer in the neighborhood where the first-order expansion is valid
        (a Tikhonov-regularized one-step Newton update). rho = 0 reproduces the
        unconstrained minimization.

    Returns:
        dict with "model", "info", "diagnostics"
    """
    diagnostics = {}
    current_model = copy_model(pilot_model)
    train_info = {}

    for t in range(T):
        # Current expansion point for this round
        ref_X = pilot_X
        ref_y = pilot_y

        # Compute pilot data-loss gradient at current expansion point
        grad_pilot = compute_data_gradient(current_model, ref_X, ref_y, tau)

        # Full-data gradient via worker aggregation at current expansion point
        grad_full = aggregate_full_data_gradient(
            current_model, worker_data, tau, N)

        # Correction vector (data-loss gradients ONLY)
        correction = grad_pilot - grad_full

        # Expansion point parameters theta^(t): anchor for the proximal term.
        theta_bar_vec = get_flat_params(current_model).clone()

        # Build surrogate at current expansion point
        surrogate_obj = build_surrogate_objective(
            model_template=current_model,
            reference_X=ref_X,
            reference_y=ref_y,
            correction_vector=correction,
            tau=tau,
            lambda_=lambda_,
            theta_bar_vec=theta_bar_vec,
            prox_rho=prox_rho,
        )

        # Snapshot the expansion point theta_bar of THIS round (current_model is
        # about to be overwritten by the optimized iterate). After the loop this
        # holds the final round's expansion point, at which the surrogate-
        # gradient-matching identity is supposed to hold exactly.
        theta_bar = copy_model(current_model)

        # Minimize surrogate -> next iterate
        next_model = copy_model(current_model)
        train_info = train_model(next_model, surrogate_obj, **train_kwargs)

        # Record diagnostics for this round
        diagnostics[f"round_{t}_correction_norm"] = correction.norm().item()
        diagnostics[f"round_{t}_grad_pilot_norm"] = grad_pilot.norm().item()
        diagnostics[f"round_{t}_grad_full_norm"] = grad_full.norm().item()

        # Update expansion point for next round
        current_model = next_model

    # Store final-round diagnostics under standard keys for compatibility
    diagnostics["correction_norm"] = diagnostics[f"round_{T-1}_correction_norm"]
    diagnostics["grad_pilot_norm"] = diagnostics[f"round_{T-1}_grad_pilot_norm"]
    diagnostics["grad_full_norm"] = diagnostics[f"round_{T-1}_grad_full_norm"]
    diagnostics["T"] = T

    # Surrogate gradient matching check (at the final expansion point theta_bar).
    # This GENUINELY tests the implementation: it differentiates the actual
    # surrogate objective callable via autograd at theta_bar and compares it with
    # the full-data penalized gradient at theta_bar. The identity
    #     grad L_tilde(theta_bar) = grad L_{N,tau}(theta_bar) + lambda * grad R(theta_bar)
    # holds only if (i) the linear correction term is graph-connected (so it
    # contributes -correction to the gradient) and (ii) the flat-parameter
    # ordering used to form <correction, theta> matches compute_flat_grad. A
    # nonzero value here flags a broken surrogate (e.g. a detached correction).
    from losses import hidden_weight_penalty
    try:
        surrogate_grad = compute_flat_grad(surrogate_obj(theta_bar), theta_bar)
        penalty_grad = compute_flat_grad(
            lambda_ * hidden_weight_penalty(theta_bar), theta_bar)
        grad_full_bar = aggregate_full_data_gradient(
            theta_bar, worker_data, tau, N)
        expected_grad = grad_full_bar + penalty_grad
        diagnostics["surrogate_gradient_matching_error"] = \
            (surrogate_grad - expected_grad).norm().item()
    except Exception:
        diagnostics["surrogate_gradient_matching_error"] = float("nan")

    # Final gradient norm at solution
    try:
        final_loss = surrogate_obj(current_model)
        final_grad = compute_flat_grad(final_loss, current_model)
        diagnostics["grad_norm_at_solution"] = final_grad.norm().item()
    except Exception:
        diagnostics["grad_norm_at_solution"] = float("nan")

    return {
        "model": current_model,
        "info": train_info,
        "diagnostics": diagnostics,
    }


def run_csl_ernn(worker1_model: ERNN,
                 worker1_X: torch.Tensor,
                 worker1_y: torch.Tensor,
                 worker_data: List[dict],
                 tau: float,
                 lambda_: float,
                 N: int,
                 train_kwargs: dict,
                 prox_rho: float = 0.0) -> Dict[str, Any]:
    """
    Run the CSL-ERNN procedure.

    Uses the first worker's local loss as the curvature/reference loss.
    The correction is: grad L_{1,tau}(theta_bar) - grad L_{N,tau}(theta_bar).

    Structurally parallel to DPS-ERNN; only the reference loss differs (worker-1
    local loss instead of the global pilot loss). The same proximal coefficient
    prox_rho applies, so the two surrogate methods are optimized identically and
    differ only in their reference/curvature loss.

    Returns:
        dict with "model", "info", "diagnostics"
    """
    diagnostics = {}

    # Worker 1's data-loss gradient at expansion point
    grad_w1 = compute_data_gradient(worker1_model, worker1_X, worker1_y, tau)
    diagnostics["grad_ref_norm"] = grad_w1.norm().item()

    # Full-data gradient via worker aggregation
    grad_full = aggregate_full_data_gradient(worker1_model, worker_data, tau, N)
    diagnostics["grad_full_norm"] = grad_full.norm().item()

    # Correction vector
    correction = grad_w1 - grad_full
    diagnostics["correction_norm"] = correction.norm().item()

    # Expansion point parameters: anchor for the proximal term.
    theta_bar_vec = get_flat_params(worker1_model).clone()

    # Build and minimize surrogate
    surrogate_model = copy_model(worker1_model)
    surrogate_obj = build_surrogate_objective(
        model_template=worker1_model,
        reference_X=worker1_X,
        reference_y=worker1_y,
        correction_vector=correction,
        tau=tau,
        lambda_=lambda_,
        theta_bar_vec=theta_bar_vec,
        prox_rho=prox_rho,
    )

    train_info = train_model(surrogate_model, surrogate_obj, **train_kwargs)

    # Genuine surrogate-gradient-matching check at the expansion point theta_bar
    # (= worker1_model, which is left unmodified above). Mirrors the DPS check.
    from losses import hidden_weight_penalty
    try:
        surrogate_grad = compute_flat_grad(
            surrogate_obj(worker1_model), worker1_model)
        penalty_grad = compute_flat_grad(
            lambda_ * hidden_weight_penalty(worker1_model), worker1_model)
        expected_grad = grad_full + penalty_grad
        diagnostics["surrogate_gradient_matching_error"] = \
            (surrogate_grad - expected_grad).norm().item()
    except Exception:
        diagnostics["surrogate_gradient_matching_error"] = float("nan")

    # Final surrogate gradient norm AT THE SOLUTION. Previously this was never
    # computed for CSL-ERNN, and the results writer defaulted the missing value
    # to 0.0 -- making an unconverged CSL surrogate indistinguishable from a
    # perfectly converged one. Compute it explicitly here, consistently with
    # run_dps_ernn, so a reported grad_norm reflects actual stationarity.
    try:
        final_loss = surrogate_obj(surrogate_model)
        final_grad = compute_flat_grad(final_loss, surrogate_model)
        diagnostics["grad_norm_at_solution"] = final_grad.norm().item()
    except Exception:
        diagnostics["grad_norm_at_solution"] = float("nan")

    return {
        "model": surrogate_model,
        "info": train_info,
        "diagnostics": diagnostics,
    }
