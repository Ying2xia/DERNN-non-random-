"""
Training routines for ERNN models.

Supports:
- Adam warm-up followed by L-BFGS refinement (paper default)
- Gradient clipping, NaN detection, early stopping
- Generic objective function interface for centralized, pilot, and surrogate training
"""
import torch
import torch.optim as optim
import time
import math
from typing import Callable, Optional, Dict, Any

from models import ERNN, copy_model, get_flat_params, set_flat_params


def train_model(model: ERNN,
                objective_fn: Callable[[ERNN], torch.Tensor],
                adam_epochs: int = 300,
                adam_lr: float = 1e-3,
                lbfgs_max_iter: int = 50,
                grad_clip: float = 10.0,
                early_stop_patience: int = 30,
                device: str = "cpu",
                verbose: bool = False) -> Dict[str, Any]:
    """
    Train an ERNN model by minimizing objective_fn(model).

    Uses Adam warm-up followed by L-BFGS refinement.

    Args:
        model: ERNN model to train (modified in-place)
        objective_fn: callable that takes model and returns scalar loss
        adam_epochs: number of Adam epochs
        adam_lr: Adam learning rate
        lbfgs_max_iter: L-BFGS maximum iterations
        grad_clip: maximum gradient norm for clipping
        early_stop_patience: epochs without improvement before stopping Adam
        device: "cpu" or "cuda"
        verbose: print progress

    Returns:
        dict with training info: final_loss, time, failed, reason, etc.
    """
    model = model.to(device)
    info = {"failed": False, "failure_reason": "", "time_seconds": 0.0}
    t0 = time.time()

    # ---- Phase 1: Adam warm-up ----
    best_loss = float("inf")
    best_params = get_flat_params(model).clone()
    patience_counter = 0

    optimizer_adam = optim.Adam(model.parameters(), lr=adam_lr)

    for epoch in range(adam_epochs):
        optimizer_adam.zero_grad()
        try:
            loss = objective_fn(model)
        except Exception as e:
            info["failed"] = True
            info["failure_reason"] = f"Adam forward: {e}"
            break

        if not torch.isfinite(loss):
            info["failed"] = True
            info["failure_reason"] = f"Adam epoch {epoch}: non-finite loss"
            break

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Check for NaN gradients
        has_nan = any(
            torch.isnan(p.grad).any() for p in model.parameters()
            if p.grad is not None
        )
        if has_nan:
            info["failed"] = True
            info["failure_reason"] = f"Adam epoch {epoch}: NaN gradient"
            break

        optimizer_adam.step()

        loss_val = loss.item()
        if loss_val < best_loss - 1e-8:
            best_loss = loss_val
            best_params = get_flat_params(model).clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"  Adam early stop at epoch {epoch}, loss={best_loss:.6f}")
            break

    # Restore best Adam parameters
    if not info["failed"]:
        set_flat_params(model, best_params)

    # ---- Phase 2: L-BFGS refinement ----
    if not info["failed"] and lbfgs_max_iter > 0:
        lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )
        lbfgs_steps = [0]

        def closure():
            lbfgs.zero_grad()
            loss = objective_fn(model)
            if not torch.isfinite(loss):
                return loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            lbfgs_steps[0] += 1
            return loss

        try:
            lbfgs.step(closure)
        except Exception as e:
            if verbose:
                print(f"  L-BFGS exception (non-fatal): {e}")
            # Restore best Adam params on L-BFGS failure
            set_flat_params(model, best_params)

        # Check if L-BFGS improved
        with torch.no_grad():
            try:
                final_loss = objective_fn(model).item()
            except:
                final_loss = float("inf")

        if not math.isfinite(final_loss) or final_loss > best_loss + 1e-6:
            # L-BFGS made things worse -> revert
            set_flat_params(model, best_params)
            final_loss = best_loss
        else:
            best_loss = final_loss

    info["final_loss"] = best_loss
    info["time_seconds"] = time.time() - t0

    if verbose:
        print(f"  Training done: loss={best_loss:.6f}, "
              f"time={info['time_seconds']:.1f}s, failed={info['failed']}")

    return info


def train_penalized_ernn(model: ERNN, X: torch.Tensor, y: torch.Tensor,
                         tau: float, lambda_: float, **kwargs) -> Dict[str, Any]:
    """Train penalized ERNN: minimize L_{N,tau}(theta) + lambda * R(theta)."""
    from losses import penalized_expectile_objective

    def obj(m):
        return penalized_expectile_objective(m, X, y, tau, lambda_)

    return train_model(model, obj, **kwargs)


def train_penalized_ernn_multi_init(p: int, J: int, X: torch.Tensor,
                                    y: torch.Tensor, tau: float,
                                    lambda_: float, n_inits: int,
                                    base_seed: int,
                                    activation: str = "tanh",
                                    **kwargs) -> tuple:
    """
    Train penalized ERNN with multiple random initializations.
    Returns the model with the lowest final training loss.

    Returns:
        (best_model, best_info)
    """
    from models import init_model, copy_model
    from losses import penalized_expectile_objective

    best_model = None
    best_info = None
    best_loss = float("inf")

    for k in range(n_inits):
        model_k = init_model(p, J, seed=base_seed + k * 7919, activation=activation)

        def obj(m):
            return penalized_expectile_objective(m, X, y, tau, lambda_)

        info_k = train_model(model_k, obj, **kwargs)
        loss_k = info_k.get("final_loss", float("inf"))

        if not info_k["failed"] and loss_k < best_loss:
            best_loss = loss_k
            best_model = model_k
            best_info = info_k

    # Fallback: if all failed, return the last one
    if best_model is None:
        best_model = model_k
        best_info = info_k

    return best_model, best_info
