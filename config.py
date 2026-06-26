"""
Configuration system for DPS-ERNN experiments.
Supports quick mode (fast validation) and full mode (paper experiments).
"""
import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # --- Sample sizes ---
    N_train: int = 200_000
    N_test: int = 2_000
    n_replications: int = 100

    # --- Distributed ---
    M_list: List[int] = field(default_factory=lambda: [10])
    pilot_ratios: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    default_pilot_ratio: float = 0.05

    # --- Expectile levels (paper: five levels) ---
    tau_list: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])

    # --- Model ---
    J: int = 5                    # hidden nodes
    lambda_: float = 0.01         # regularization strength
    activation: str = "tanh"

    # --- BIC grid ---
    J_grid: List[int] = field(default_factory=lambda: list(range(1, 11)))
    lambda_grid: List[float] = field(
        default_factory=lambda: [round(0.01 * k, 2) for k in range(1, 11)]
    )
    use_bic: bool = False         # if True, select J,lambda via BIC
    bic_subsample_N: int = 20000  # subsample size for BIC (saves time)
    bic_n_inits: int = 10         # more inits for BIC to avoid local-min traps

    # --- Optimizer ---
    adam_epochs: int = 300
    adam_lr: float = 1e-3
    lbfgs_max_iter: int = 50
    grad_clip: float = 10.0
    early_stop_patience: int = 30
    n_inits: int = 1              # number of random initializations

    # --- Surrogate proximal (trust-region) coefficient ---
    # Coefficient rho of the proximal term (rho/2)||theta - theta_bar||^2 added
    # to the surrogate objective of the surrogate-based methods (DPS-ERNN and
    # CSL-ERNN). The ERNN objective is nonconvex with many near-flat directions;
    # without this term the unbounded linear correction -<corr, theta> drives the
    # surrogate minimizer far from the expansion point along low-curvature
    # directions, which degrades generalization (the one-step solution leaves the
    # neighborhood in which the first-order expansion is valid). rho > 0 keeps the
    # step well-conditioned (Delta theta = (H + rho*I)^{-1} corr). Set rho = 0 to
    # recover the unconstrained behavior.
    prox_rho: float = 1.0

    # --- Examples and errors ---
    examples: List[int] = field(default_factory=lambda: [1, 2, 3])
    errors: List[str] = field(default_factory=lambda: ["normal", "t3", "chi2"])
    strategies: List[int] = field(default_factory=lambda: [1, 2, 3])

    # --- Methods ---
    methods: List[str] = field(
        default_factory=lambda: [
            "Centralized", "Pilot", "OS-ERNN", "CSL-ERNN", "DPS-ERNN"
        ]
    )

    # --- Reproducibility ---
    base_seed: int = 42
    device: str = "cpu"

    # --- Output ---
    results_dir: str = "results"
    logs_dir: str = "logs"

    # --- Real data ---
    household_data_path: str = "data/household_power_consumption.txt"
    airquality_data_path: str = "data/PRSA_Data/"


def get_quick_config() -> Config:
    """Quick mode for fast validation."""
    return Config(
        N_train=20_000,
        N_test=1_000,
        n_replications=3,
        M_list=[10],
        pilot_ratios=[0.05],
        default_pilot_ratio=0.05,
        tau_list=[0.1, 0.5, 0.9],
        J=5,
        lambda_=0.01,
        adam_epochs=150,
        lbfgs_max_iter=30,
        n_inits=1,
        examples=[1, 2],
        errors=["normal"],
        strategies=[1, 2],
        use_bic=False,
    )


def get_full_config() -> Config:
    """Full mode for paper experiments."""
    return Config()


def load_config(path: str) -> Config:
    """Load config from YAML file."""
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return Config(**d)


def save_config(cfg: Config, path: str):
    """Save config to YAML file."""
    from dataclasses import asdict
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)
