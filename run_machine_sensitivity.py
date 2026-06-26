"""
Number-of-machines (M) sensitivity study for DPS-ERNN.

Fixes the pilot ratio and sweeps the number of worker machines
M in {5, 10, 20, 50, 100} to show that DPS-ERNN's accuracy is essentially
invariant to M (the Poisson pilot sample is globally representative regardless
of how the data is split), while the relative acceleration RACT improves as M
grows. Only DPS-ERNN is evaluated; the centralized model is run once per
replication purely as the timing/accuracy reference (it does not depend on M).

    RACT = time(Centralized) / time(DPS-ERNN)

This is the companion to run_pilot_sensitivity.py. All pipeline fixes apply:
the proximal term (prox_rho) is passed to run_dps, the converged optimizer
settings are used (adam_lr=0.01, adam_epochs=2000), tensors are placed on the
configured device, and MAE/RMSE are computed against the true conditional
tau-expectile q_tau(x).

Usage:
    python run_machine_sensitivity.py --bic results/bic_selected.csv --quick
    python run_machine_sensitivity.py --bic results/bic_selected.csv
    python run_machine_sensitivity.py --bic results/bic_selected.csv --M-list 5 10 30
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
from methods import run_centralized, run_dps
from metrics import evaluate_against_target


def load_bic(path: str) -> dict:
    bic = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            bic[(int(row["example"]), row["error"], float(row["tau"]))] = (
                int(row["best_J"]), float(row["best_lambda"]))
    return bic


def main():
    parser = argparse.ArgumentParser(description="Machine-number sensitivity for DPS-ERNN")
    parser.add_argument("--bic", type=str, required=True,
                        help="Path to bic_selected.csv from a previous run")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--M-list", type=int, nargs="+", default=[5, 10, 20, 50, 100],
                        help="Worker counts to sweep")
    parser.add_argument("--n-train", type=int, default=200000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-reps", type=int, default=None)
    parser.add_argument("--pilot-ratio", type=float, default=0.05)
    parser.add_argument("--prox-rho", type=float, default=1.0)
    parser.add_argument("--adam-lr", type=float, default=0.01)
    parser.add_argument("--adam-epochs", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="results/machine_sensitivity.csv")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: device='{device}' requested but CUDA unavailable; using CPU.")
        device = "cpu"
    print(f"Using device: {device}")

    if args.quick:
        M_list = args.M_list
        tau_list = [0.1, 0.5, 0.9]
        n_reps = args.n_reps or 3
        examples = [1, 2, 3]
        errors = ["normal"]
        strategies = [2]  # non-random only
    else:
        M_list = args.M_list
        tau_list = [0.5]
        n_reps = args.n_reps or 10
        examples = [2]
        errors = ["normal"]
        strategies = [1, 2, 3]
    N_train, N_test = args.n_train, args.n_test
    r = args.pilot_ratio

    train_kwargs = dict(adam_epochs=args.adam_epochs, adam_lr=args.adam_lr,
                        lbfgs_max_iter=50, grad_clip=10.0,
                        early_stop_patience=200, device=device)
    n_inits = 3
    base_seed = 42
    p_map = {1: 1, 2: 2, 3: 2}

    bic = load_bic(args.bic)
    print(f"Loaded BIC selections: {len(bic)} settings | M sweep: {M_list} | r={r}\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fields = ["example", "error", "storage_strategy", "tau", "M", "pilot_ratio",
              "method", "replication", "J", "lambda", "MAE", "RMSE",
              "time_seconds", "RACT"]
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
                    print(f"\n{'='*60}\nEx{example} {error_type} S{strategy} "
                          f"tau={tau} J={J} lam={lam}\n{'='*60}")
                    for rep in range(n_reps):
                        seed = base_seed + rep * 1000 + example * 100
                        X_tr_np, y_tr_np = generate_data(example, N_train, error_type, seed)
                        X_te_np, _ = generate_data(example, N_test, error_type, seed + 100000)
                        X_tr = torch.tensor(X_tr_np, dtype=torch.float32).to(device)
                        y_tr = torch.tensor(y_tr_np, dtype=torch.float32).to(device)
                        X_te = torch.tensor(X_te_np, dtype=torch.float32).to(device)
                        target_te = torch.tensor(
                            true_conditional_expectile(example, error_type, X_te_np, tau),
                            dtype=torch.float32).to(device)

                        # Centralized reference (M-independent): for RACT + accuracy anchor
                        try:
                            res_c = run_centralized(X_tr, y_tr, tau, lam, p, J, seed,
                                                    train_kwargs, n_inits=n_inits)
                            t_cen = res_c["info"]["time_seconds"]
                            ev_c = evaluate_against_target(res_c["model"], X_te, target_te)
                            writer.writerow({
                                "example": example, "error": error_type,
                                "storage_strategy": strategy, "tau": tau, "M": 0,
                                "pilot_ratio": r, "method": "Centralized",
                                "replication": rep, "J": J, "lambda": lam,
                                "MAE": f"{ev_c['MAE']:.6f}", "RMSE": f"{ev_c['RMSE']:.6f}",
                                "time_seconds": f"{t_cen:.2f}", "RACT": "1.00"})
                        except Exception as e:
                            print(f"    Centralized FAILED: {e}"); t_cen = float("nan")

                        for M in M_list:
                            rng_part = np.random.default_rng(seed + 200000)
                            worker_indices = partition_data(X_tr_np, M, strategy, rng_part)
                            worker_data = [{"X": X_tr[idx], "y": y_tr[idx], "N_m": len(idx)}
                                           for idx in worker_indices]
                            rng_pilot = np.random.default_rng(seed + 300000)
                            pilot_idx = poisson_pilot_sample(N_train, M, worker_indices, r, rng_pilot)
                            pilot_X, pilot_y = X_tr[pilot_idx], y_tr[pilot_idx]
                            try:
                                res_d = run_dps(worker_data, pilot_X, pilot_y, tau, lam,
                                                p, J, seed, N_train, train_kwargs,
                                                n_inits=n_inits, prox_rho=args.prox_rho)
                                t_d = res_d["info"]["time_seconds"]
                                ev_d = evaluate_against_target(res_d["model"], X_te, target_te)
                                ract = (t_cen / t_d) if (t_d and t_d > 0 and t_cen == t_cen) else float("nan")
                                writer.writerow({
                                    "example": example, "error": error_type,
                                    "storage_strategy": strategy, "tau": tau, "M": M,
                                    "pilot_ratio": r, "method": "DPS-ERNN",
                                    "replication": rep, "J": J, "lambda": lam,
                                    "MAE": f"{ev_d['MAE']:.6f}", "RMSE": f"{ev_d['RMSE']:.6f}",
                                    "time_seconds": f"{t_d:.2f}", "RACT": f"{ract:.2f}"})
                            except Exception as e:
                                print(f"    DPS M={M} FAILED: {e}")
                        out_file.flush()
                        print(f"  rep {rep+1}/{n_reps} done ({time.time()-t_start:.0f}s)")

    out_file.close()
    print(f"\nDone! {time.time()-t_start:.0f}s total. Results: {args.output}")
    summarize(args.output)


def summarize(path: str):
    import statistics
    mae = defaultdict(list); ract = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["method"] != "DPS-ERNN":
                continue
            key = (row["example"], row["error"], row["storage_strategy"], row["tau"], int(row["M"]))
            mae[key].append(float(row["MAE"]))
            try: ract[key].append(float(row["RACT"]))
            except ValueError: pass
    print(f"\n{'='*60}\nSummary: DPS-ERNN mean MAE (and RACT) by M\n{'='*60}")
    examples = sorted(set(k[0] for k in mae))
    for ex in examples:
        print(f"\n--- Example {ex} ---")
        settings = sorted(set((k[1], k[2], k[3]) for k in mae if k[0] == ex))
        for err, strat, tau in settings:
            print(f"  {err} S{strat} tau={tau}:")
            Ms = sorted(set(k[4] for k in mae
                            if k[0]==ex and k[1]==err and k[2]==strat and k[3]==tau))
            for M in Ms:
                k = (ex, err, strat, tau, M)
                rstr = f", RACT={statistics.mean(ract[k]):.2f}" if ract[k] else ""
                print(f"    M={M:4d}: MAE={statistics.mean(mae[k]):.4f}{rstr}")


if __name__ == "__main__":
    main()
