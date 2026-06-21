"""
Loss functions for the DPS-ERNN framework.

Expectile loss (eq. 1):
    rho_tau(u) = |tau - I(u < 0)| * u^2

Hidden-weight penalty (eq. 5):
    R(theta) = (1 / (pJ)) * sum_{l,j} (w_{lj}^(h))^2

CRITICAL DESIGN:
    - data_expectile_objective: does NOT include penalty
    - penalized_expectile_objective: includes penalty
    - Surrogate gradient correction uses ONLY data_expectile_objective
    - Penalty is added SEPARATELY on the master side
"""
import torch
from models import ERNN


def expectile_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                   tau: float, reduction: str = "mean") -> torch.Tensor:
    """
    Asymmetric squared loss (expectile loss).

    rho_tau(u) = |tau - I(u < 0)| * u^2

    Args:
        y_true: (N,) true responses
        y_pred: (N,) predicted values
        tau: expectile level in (0, 1)
        reduction: "mean", "sum", or "none"

    Returns:
        Scalar loss (if reduction != "none") or (N,) vector
    """
    u = y_true - y_pred                          # residual
    weight = torch.where(u < 0, 1.0 - tau, tau)  # |tau - I(u < 0)|
    loss = weight * u ** 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def hidden_weight_penalty(model: ERNN) -> torch.Tensor:
    """
    L2 penalty on hidden-layer weights only (eq. 5):
        R(theta) = (1 / (pJ)) * sum (w_lj^(h))^2

    The normalization (pJ)^{-1} ensures comparability across architectures.
    """
    W_h = model.hidden.weight  # (J, p) in nn.Linear storage
    p, J = model.p, model.J
    return (W_h ** 2).sum() / (p * J)


def data_expectile_objective(model: ERNN, X: torch.Tensor,
                             y: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Data-only expectile loss (NO penalty):
        L_{N,tau}(theta) = (1/N) sum rho_tau(y_i - f(x_i; theta))

    Used for:
        - Computing worker gradients in surrogate construction
        - The gradient correction term
        - MUST NOT include the penalty term
    """
    y_pred = model(X)
    return expectile_loss(y, y_pred, tau, reduction="mean")


def penalized_expectile_objective(model: ERNN, X: torch.Tensor,
                                  y: torch.Tensor, tau: float,
                                  lambda_: float) -> torch.Tensor:
    """
    Penalized expectile objective (eq. 7):
        L_{N,tau,lambda}(theta) = L_{N,tau}(theta) + lambda * R(theta)

    Used for:
        - Centralized ERNN training
        - Pilot ERNN training
        - Master-side surrogate optimization (penalty added here)
    """
    data_loss = data_expectile_objective(model, X, y, tau)
    penalty = hidden_weight_penalty(model)
    return data_loss + lambda_ * penalty
