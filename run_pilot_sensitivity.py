"""
Pilot-ratio sensitivity study for DPS-ERNN.

Reads BIC-selected hyperparameters from a previous run, then sweeps pilot
ratios r in {0.5%, 1%, 2%, 5%, 10%, 20%} to show that DPS-ERNN remains stable
as the pilot ratio shrinks (it uses the full-data gradient correction, not just
the pilot sample). Only DPS-ERNN is evaluated.

All fixes from the main pipeline are applied here:
  - the surrogate proximal term (prox_rho) is passed to run_dps, without which
    the one-step surrogate minimizer drifts along flat directions;
  - the converged optimizer settings (adam_lr=0.01, adam_epochs=2000) are used,
    without which the bivariate examples under-fit to a near-constant;
  - all tensors are placed on the configured device (cpu/cuda);
  - MAE/RMSE are computed against the true conditional tau-expectile q_tau(x),
    not the noisy y.

Usage:
    python run_pilot_sensitivity.py --bic results/bic_selected.csv --quick
    python run_pilot_sensitivity.py --bic results/bic_selected.csv
    python run_pilot_sensitivity.py --bic results/bic_selected.csv --device cuda
"""
import argparse
import os
import csv
import time
import torch
import numpy as np
from collections import defaultdict

from data_generation import generate_data
from targets import true_conditional_expectile
from partition import partition_data, poisson_pilot_sample
from methods import run_dps
from metrics import evaluate_against_target


def load_bic(path: str) -> dict:
    """Load BIC-selected (J, lambda) per (example, error, tau)."""
    bic = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["example"]), row["error"], float(row["tau"]))
            bic[key] = (int(row["best_J"]), float(row["best_lambda"]))
    return bic


def main():
    parser = argparse.ArgumentParser(
        description="Pilot-ratio sensitivity for DPS-ERNN")
    parser.add_argument("--bic", type=str, required=True,
                        help="Path to bic_selected.csv from a previous run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer reps, ratios, taus")
    parser.add_argument("--n-train", type=int, default=200000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-reps", type=int, default=None,
                        help="Override number of replications")
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--T", type=int, nargs="+", default=[1],
                        help="Communication rounds for DPS-ERNN (e.g., --T 1 2 3)")
    parser.add_argument("--prox-rho", type=float, default=1.0,
                        help="Surrogate proximal coefficient (must be > 0)")
    parser.add_argument("--adam-lr", type=float, default=0.01)
    parser.add_argument("--adam-epochs", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda")
    parser.add_argument("--output", type=str, default="results/pilot_sensitivity.csv")
    args = parser.parse_args()

    # ---- Resolve device ----
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: device='{device}' requested but CUDA unavailable; using CPU.")
        device = "cpu"
    print(f"Using device: {device}")

    # ---- Configuration ----
    if args.quick:
        pilot_ratios = [0.01, 0.02, 0.05, 0.10]
        tau_list = [0.1, 0.5, 0.9]
        n_reps = args.n_reps or 3
        examples = [1, 2, 3]
        errors = ["normal"]
        strategies = [2]  # non-random only (where the story matters)
    else:
        pilot_ratios = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
        tau_list = [0.5]
        n_reps = args.n_reps or 10
        examples = [1, 2, 3]
        errors = ["normal"]
        strategies = [2]
    N_train = args.n_train
    N_test = args.n_test

    train_kwargs = dict(
        adam_epochs=args.adam_epochs, adam_lr=args.adam_lr, lbfgs_max_iter=50,
        grad_clip=10.0, early_stop_patience=200, device=device,
    )
    n_inits = 3
    base_seed = 42
    M = args.M
    T_list = args.T
    prox_rho = args.prox_rho
    p_map = {1: 1, 2: 2, 3: 2}

    # ---- Load BIC ----
    bic = load_bic(args.bic)
    print(f"Loaded BIC selections: {len(bic)} settings")
    print()

    # ---- Output CSV ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fields = ["example", "error", "storage_strategy", "tau", "M",
              "pilot_ratio", "method", "replication", "J", "lambda",
              "actual_pilot_size", "MAE", "RMSE", "time_seconds"]
    out_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=fields)
    writer.writeheader()

    t_start = time.time()

    for example in examples:
        p = p_map[example]
        for error_type in errors:
            for tau in tau_list:
                bic_key = (example, error_type, tau)
                if bic_key not in bic:
                    print(f"  [SKIP] No BIC for Ex{example} {error_type} tau={tau}")
                    continue
                J, lam = bic[bic_key]

                for strategy in strategies:
                    print(f"\n{'='*60}")
                    print(f"Ex{example} {error_type} S{strategy} tau={tau} J={J} lam={lam}")
                    print(f"{'='*60}")

                    for rep in range(n_reps):
                        seed = base_seed + rep * 1000 + example * 100

                        X_tr_np, y_tr_np = generate_data(example, N_train, error_type, seed)
                        X_te_np, y_te_np = generate_data(example, N_test, error_type, seed + 100000)
                        X_tr = torch.tensor(X_tr_np, dtype=torch.float32).to(device)
                        y_tr = torch.tensor(y_tr_np, dtype=torch.float32).to(device)
                        X_te = torch.tensor(X_te_np, dtype=torch.float32).to(device)
                        # True conditional tau-expectile target (the estimand).
                        target_te = torch.tensor(
                            true_conditional_expectile(example, error_type, X_te_np, tau),
                            dtype=torch.float32).to(device)

                        rng_part = np.random.default_rng(seed + 200000)
                        worker_indices = partition_data(X_tr_np, M, strategy, rng_part)
                        worker_data = [
                            {"X": X_tr[idx], "y": y_tr[idx], "N_m": len(idx)}
                            for idx in worker_indices
                        ]

                        for r in pilot_ratios:
                            rng_pilot = np.random.default_rng(seed + 300000 + int(r * 10000))
                            pilot_idx = poisson_pilot_sample(N_train, M, worker_indices, r, rng_pilot)
                            actual_n = len(pilot_idx)
                            pilot_X = X_tr[pilot_idx]
                            pilot_y = y_tr[pilot_idx]

                            for T in T_list:
                                method_name = f"DPS-ERNN(T={T})" if T > 1 else "DPS-ERNN"
                                try:
                                    res_d = run_dps(
                                        worker_data, pilot_X, pilot_y,
                                        tau, lam, p, J, seed, N_train,
                                        train_kwargs, n_inits=n_inits, T=T,
                                        prox_rho=prox_rho)
                                    ev_d = evaluate_against_target(res_d["model"], X_te, target_te)
                                    writer.writerow({
                                        "example": example, "error": error_type,
                                        "storage_strategy": strategy, "tau": tau,
                                        "M": M, "pilot_ratio": r,
                                        "method": method_name, "replication": rep,
                                        "J": J, "lambda": lam,
                                        "actual_pilot_size": actual_n,
                                        "MAE": f"{ev_d['MAE']:.6f}",
                                        "RMSE": f"{ev_d['RMSE']:.6f}",
                                        "time_seconds": f"{res_d['info']['time_seconds']:.2f}",
                                    })
                                except Exception as e:
                                    print(f"    DPS T={T} r={r} FAILED: {e}")

                        out_file.flush()
                        print(f"  rep {rep+1}/{n_reps} done ({time.time()-t_start:.0f}s)")

    out_file.close()
    print(f"\nDone! {time.time()-t_start:.0f}s total. Results: {args.output}")
    summarize(args.output)


def summarize(path: str):
    """Print mean MAE by (example, error, strategy, tau, r)."""
    import statistics
    groups = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["example"], row["error"], row["storage_strategy"],
                   row["tau"], row["pilot_ratio"], row["method"])
            groups[key].append(float(row["MAE"]))

    print(f"\n{'='*60}")
    print("Summary: mean DPS-ERNN MAE by (example, error, strategy, tau, r)")
    print(f"{'='*60}")
    examples = sorted(set(k[0] for k in groups))
    for ex in examples:
        print(f"\n--- Example {ex} ---")
        settings = sorted(set((k[1], k[2], k[3]) for k in groups if k[0] == ex))
        for err, strat, tau in settings:
            print(f"  {err} S{strat} tau={tau}:")
            ratios = sorted(set(k[4] for k in groups
                                if k[0] == ex and k[1] == err and k[2] == strat and k[3] == tau),
                            key=float)
            for r in ratios:
                line = f"    r={float(r):5.1%}: "
                for method in sorted(set(k[5] for k in groups
                                         if k[0] == ex and k[1] == err and k[2] == strat
                                         and k[3] == tau and k[4] == r)):
                    mk = (ex, err, strat, tau, r, method)
                    if mk in groups:
                        line += f" {method}={statistics.mean(groups[mk]):.4f}"
                print(line)


if __name__ == "__main__":
    main()
