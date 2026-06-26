"""
Main experiment entry point for DPS-ERNN simulation studies.

Usage:
    python run_simulations.py --quick          # fast validation
    python run_simulations.py                  # full experiments
    python run_simulations.py --config configs/full.yaml
"""
import argparse
import os
import sys
import csv
import time
import json
import traceback
import torch
import numpy as np
from datetime import datetime

from config import Config, get_quick_config, get_full_config, load_config
from models import init_model, copy_model, get_flat_params
from data_generation import generate_data
from targets import true_conditional_expectile
from partition import partition_data, poisson_pilot_sample
from methods import run_centralized, run_pilot, run_os_ernn, run_csl, run_dps
from metrics import (evaluate_model, evaluate_against_target, compute_ract,
                     compute_param_distance, compute_prediction_gap,
                     compute_residual_proportions)
from sanity_checks import run_all_checks


RAW_FIELDS = [
    "example", "error", "storage_strategy", "tau", "M", "pilot_ratio",
    "method", "replication", "seed", "J", "lambda",
    "actual_pilot_size", "MAE", "RMSE", "time_seconds", "RACT",
    "param_distance_to_centralized", "prediction_gap_MAE",
    "prediction_gap_RMSE", "correction_norm",
    "surrogate_gradient_matching_error", "full_gradient_aggregation_error",
    "grad_norm", "residual_prop_1e_minus_6", "residual_prop_1e_minus_4",
    "residual_prop_1e_minus_3", "failed", "failure_reason",
]


def run_single_replication(example: int, error_type: str, strategy: int,
                           tau: float, M: int, pilot_ratio: float,
                           cfg: Config, rep: int, seed: int,
                           writer, log_file,
                           J_override: int = None,
                           lambda_override: float = None) -> dict:
    """Run a single replication for all five methods."""

    p_map = {1: 1, 2: 2, 3: 2}
    p = p_map[example]
    J = J_override if J_override is not None else cfg.J
    lam = lambda_override if lambda_override is not None else cfg.lambda_
    N_train = cfg.N_train
    N_test = cfg.N_test

    train_kwargs = dict(
        adam_epochs=cfg.adam_epochs,
        adam_lr=cfg.adam_lr,
        lbfgs_max_iter=cfg.lbfgs_max_iter,
        grad_clip=cfg.grad_clip,
        early_stop_patience=cfg.early_stop_patience,
        device=cfg.device,
    )

    # Generate data (test set also exposes the exact conditional location/scale
    # so we can build the TRUE conditional tau-expectile target for evaluation).
    X_train_np, y_train_np = generate_data(example, N_train, error_type, seed)
    X_test_np, y_test_np = generate_data(example, N_test, error_type, seed + 100000)

    # Place all data tensors on the configured device. The training routine
    # moves models to cfg.device; if the data stayed on CPU while the model was
    # on cuda, every forward pass would raise a device-mismatch error. Worker
    # and pilot tensors below are sliced from X_train/y_train and therefore
    # inherit the device automatically.
    device = getattr(cfg, "device", "cpu")
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

    # True conditional tau-expectile on the test covariates (the estimand).
    # Simulation MAE/RMSE are computed against this, NOT against the noisy
    # y_test, which would target the wrong functional at tau != 0.5.
    target_test_np = true_conditional_expectile(example, error_type, X_test_np, tau)
    target_test = torch.tensor(target_test_np, dtype=torch.float32).to(device)

    # Partition data
    rng_part = np.random.default_rng(seed + 200000)
    worker_indices = partition_data(X_train_np, M, strategy, rng_part)

    worker_data = []
    for idx in worker_indices:
        worker_data.append({
            "X": X_train[idx],
            "y": y_train[idx],
            "N_m": len(idx),
        })

    # Pilot sample
    rng_pilot = np.random.default_rng(seed + 300000)
    pilot_idx = poisson_pilot_sample(N_train, M, worker_indices,
                                     pilot_ratio, rng_pilot)
    actual_pilot_size = len(pilot_idx)
    pilot_X = X_train[pilot_idx]
    pilot_y = y_train[pilot_idx]

    # Compute full-data aggregation error (diagnostic)
    from surrogate import compute_data_gradient, aggregate_full_data_gradient
    diag_model = init_model(p, J, seed).to(device)
    grad_full_direct = compute_data_gradient(diag_model, X_train, y_train, tau)
    grad_full_agg = aggregate_full_data_gradient(diag_model, worker_data, tau, N_train)
    full_grad_agg_error = (grad_full_direct - grad_full_agg).norm().item()

    results_list = []
    centralized_model = None
    centralized_time = 0.0

    # ---- Method 1: Centralized ERNN ----
    if "Centralized" in cfg.methods:
        try:
            res = run_centralized(X_train, y_train, tau, lam, p, J, seed,
                                  train_kwargs, n_inits=cfg.n_inits)
            centralized_model = res["model"]
            centralized_time = res["info"]["time_seconds"]
            ev = evaluate_against_target(res["model"], X_test, target_test)
            resid = compute_residual_proportions(res["model"], X_test, y_test)
            row = _make_row(example, error_type, strategy, tau, M, pilot_ratio,
                            "Centralized", rep, seed, J, lam, actual_pilot_size,
                            ev, res["info"], centralized_time, 1.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, full_grad_agg_error, float("nan"), resid)
            results_list.append(row)
        except Exception as e:
            _write_failed_row(writer, example, error_type, strategy, tau, M,
                              pilot_ratio, "Centralized", rep, seed, J, lam,
                              actual_pilot_size, str(e))

    # ---- Method 2: Pilot ERNN ----
    if "Pilot" in cfg.methods:
        try:
            res = run_pilot(pilot_X, pilot_y, tau, lam, p, J, seed, train_kwargs,
                            n_inits=cfg.n_inits)
            ev = evaluate_against_target(res["model"], X_test, target_test)
            resid = compute_residual_proportions(res["model"], X_test, y_test)
            pd = compute_param_distance(res["model"], centralized_model) if centralized_model else 0.0
            pg = compute_prediction_gap(res["model"], centralized_model, X_test, target_test) if centralized_model else {"prediction_gap_MAE": 0, "prediction_gap_RMSE": 0}
            ract = compute_ract(centralized_time, res["info"]["time_seconds"])
            row = _make_row(example, error_type, strategy, tau, M, pilot_ratio,
                            "Pilot", rep, seed, J, lam, actual_pilot_size,
                            ev, res["info"], res["info"]["time_seconds"], ract,
                            pd, pg["prediction_gap_MAE"], pg["prediction_gap_RMSE"],
                            0.0, 0.0, full_grad_agg_error, float("nan"), resid)
            results_list.append(row)
        except Exception as e:
            _write_failed_row(writer, example, error_type, strategy, tau, M,
                              pilot_ratio, "Pilot", rep, seed, J, lam,
                              actual_pilot_size, str(e))

    # ---- Method 3: OS-ERNN ----
    if "OS-ERNN" in cfg.methods:
        try:
            res = run_os_ernn(worker_data, tau, lam, p, J, seed, N_train,
                              train_kwargs)
            ev = evaluate_against_target(res["model"], X_test, target_test)
            resid = compute_residual_proportions(res["model"], X_test, y_test)
            pd = compute_param_distance(res["model"], centralized_model) if centralized_model else 0.0
            pg = compute_prediction_gap(res["model"], centralized_model, X_test, target_test) if centralized_model else {"prediction_gap_MAE": 0, "prediction_gap_RMSE": 0}
            ract = compute_ract(centralized_time, res["info"]["time_seconds"])
            row = _make_row(example, error_type, strategy, tau, M, pilot_ratio,
                            "OS-ERNN", rep, seed, J, lam, actual_pilot_size,
                            ev, res["info"], res["info"]["time_seconds"], ract,
                            pd, pg["prediction_gap_MAE"], pg["prediction_gap_RMSE"],
                            0.0, 0.0, full_grad_agg_error, float("nan"), resid)
            results_list.append(row)
        except Exception as e:
            _write_failed_row(writer, example, error_type, strategy, tau, M,
                              pilot_ratio, "OS-ERNN", rep, seed, J, lam,
                              actual_pilot_size, str(e))

    # ---- Method 4: CSL-ERNN ----
    if "CSL-ERNN" in cfg.methods:
        try:
            res = run_csl(worker_data, tau, lam, p, J, seed, N_train,
                          train_kwargs, n_inits=cfg.n_inits,
                          prox_rho=getattr(cfg, "prox_rho", 0.0))
            ev = evaluate_against_target(res["model"], X_test, target_test)
            resid = compute_residual_proportions(res["model"], X_test, y_test)
            pd = compute_param_distance(res["model"], centralized_model) if centralized_model else 0.0
            pg = compute_prediction_gap(res["model"], centralized_model, X_test, target_test) if centralized_model else {"prediction_gap_MAE": 0, "prediction_gap_RMSE": 0}
            ract = compute_ract(centralized_time, res["info"]["time_seconds"])
            diag = res.get("diagnostics", {})
            row = _make_row(example, error_type, strategy, tau, M, pilot_ratio,
                            "CSL-ERNN", rep, seed, J, lam, actual_pilot_size,
                            ev, res["info"], res["info"]["time_seconds"], ract,
                            pd, pg["prediction_gap_MAE"], pg["prediction_gap_RMSE"],
                            diag.get("correction_norm", 0),
                            diag.get("surrogate_gradient_matching_error", 0),
                            full_grad_agg_error,
                            diag.get("grad_norm_at_solution", float("nan")), resid)
            results_list.append(row)
        except Exception as e:
            _write_failed_row(writer, example, error_type, strategy, tau, M,
                              pilot_ratio, "CSL-ERNN", rep, seed, J, lam,
                              actual_pilot_size, str(e))

    # ---- Method 5: DPS-ERNN ----
    if "DPS-ERNN" in cfg.methods:
        try:
            res = run_dps(worker_data, pilot_X, pilot_y, tau, lam, p, J, seed,
                          N_train, train_kwargs, n_inits=cfg.n_inits,
                          prox_rho=getattr(cfg, "prox_rho", 0.0))
            ev = evaluate_against_target(res["model"], X_test, target_test)
            resid = compute_residual_proportions(res["model"], X_test, y_test)
            pd = compute_param_distance(res["model"], centralized_model) if centralized_model else 0.0
            pg = compute_prediction_gap(res["model"], centralized_model, X_test, target_test) if centralized_model else {"prediction_gap_MAE": 0, "prediction_gap_RMSE": 0}
            ract = compute_ract(centralized_time, res["info"]["time_seconds"])
            diag = res.get("diagnostics", {})
            row = _make_row(example, error_type, strategy, tau, M, pilot_ratio,
                            "DPS-ERNN", rep, seed, J, lam, actual_pilot_size,
                            ev, res["info"], res["info"]["time_seconds"], ract,
                            pd, pg["prediction_gap_MAE"], pg["prediction_gap_RMSE"],
                            diag.get("correction_norm", 0),
                            diag.get("surrogate_gradient_matching_error", 0),
                            full_grad_agg_error,
                            diag.get("grad_norm_at_solution", float("nan")), resid)
            results_list.append(row)
        except Exception as e:
            _write_failed_row(writer, example, error_type, strategy, tau, M,
                              pilot_ratio, "DPS-ERNN", rep, seed, J, lam,
                              actual_pilot_size, str(e))

    # Write results
    for row in results_list:
        writer.writerow(row)

    return {"n_results": len(results_list)}


def _make_row(example, error_type, strategy, tau, M, pilot_ratio,
              method, rep, seed, J, lam, actual_pilot_size,
              ev, info, time_s, ract, pd, pg_mae, pg_rmse,
              corr_norm, surr_grad_err, full_grad_err, grad_norm, resid):
    return {
        "example": example, "error": error_type, "storage_strategy": strategy,
        "tau": tau, "M": M, "pilot_ratio": pilot_ratio,
        "method": method, "replication": rep, "seed": seed,
        "J": J, "lambda": lam,
        "actual_pilot_size": actual_pilot_size,
        "MAE": f"{ev['MAE']:.6f}", "RMSE": f"{ev['RMSE']:.6f}",
        "time_seconds": f"{time_s:.2f}",
        "RACT": f"{ract:.2f}" if ract == ract else "NA",
        "param_distance_to_centralized": f"{pd:.6f}",
        "prediction_gap_MAE": f"{pg_mae:.6f}",
        "prediction_gap_RMSE": f"{pg_rmse:.6f}",
        "correction_norm": f"{corr_norm:.6f}",
        "surrogate_gradient_matching_error": f"{surr_grad_err:.2e}",
        "full_gradient_aggregation_error": f"{full_grad_err:.2e}",
        "grad_norm": f"{grad_norm:.6f}",
        "residual_prop_1e_minus_6": f"{resid.get('residual_prop_1e_minus_6', 0):.6f}",
        "residual_prop_1e_minus_4": f"{resid.get('residual_prop_1e_minus_4', 0):.6f}",
        "residual_prop_1e_minus_3": f"{resid.get('residual_prop_1e_minus_3', 0):.6f}",
        "failed": info.get("failed", False),
        "failure_reason": info.get("failure_reason", ""),
    }


def _write_failed_row(writer, example, error_type, strategy, tau, M,
                      pilot_ratio, method, rep, seed, J, lam,
                      actual_pilot_size, reason):
    writer.writerow({
        "example": example, "error": error_type, "storage_strategy": strategy,
        "tau": tau, "M": M, "pilot_ratio": pilot_ratio,
        "method": method, "replication": rep, "seed": seed,
        "J": J, "lambda": lam, "actual_pilot_size": actual_pilot_size,
        "MAE": "NA", "RMSE": "NA", "time_seconds": "NA", "RACT": "NA",
        "param_distance_to_centralized": "NA", "prediction_gap_MAE": "NA",
        "prediction_gap_RMSE": "NA", "correction_norm": "NA",
        "surrogate_gradient_matching_error": "NA",
        "full_gradient_aggregation_error": "NA", "grad_norm": "NA",
        "residual_prop_1e_minus_6": "NA", "residual_prop_1e_minus_4": "NA",
        "residual_prop_1e_minus_3": "NA",
        "failed": True, "failure_reason": reason,
    })


def load_bic_selected(path: str, cfg: "Config") -> dict:
    """
    Load previously-computed BIC selections from a CSV written by the BIC phase
    (columns: example, error, tau, best_J, best_lambda, best_BIC) into the
    {(example, error_type, tau): (J, lambda)} dictionary consumed by the main
    loop. This lets a run reuse an existing results/bic_selected.csv instead of
    recomputing the (J, lambda) grid search.

    Types are cast to match the lookup keys exactly (example -> int,
    tau -> float, best_J -> int, best_lambda -> float); error_type stays str.
    Coverage against the current cfg grid is validated and any missing
    (example, error, tau) combination is reported loudly, because the main loop
    silently falls back to the fixed cfg.J / cfg.lambda_ for missing keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"--skip-bic was given but no BIC cache was found at '{path}'. "
            f"Run once without --skip-bic to generate it, or pass the correct "
            f"path with --bic-csv."
        )

    selected = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"example", "error", "tau", "best_J", "best_lambda"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(
                f"BIC cache '{path}' is missing required columns: "
                f"{sorted(missing_cols)}. Found columns: {reader.fieldnames}."
            )
        for row in reader:
            key = (int(row["example"]), str(row["error"]), float(row["tau"]))
            selected[key] = (int(row["best_J"]), float(row["best_lambda"]))

    # Validate coverage against the grid this run will actually iterate over.
    needed = [(ex, err, tau)
              for ex in cfg.examples
              for err in cfg.errors
              for tau in cfg.tau_list]
    missing = [k for k in needed if k not in selected]
    print(f"Loaded {len(selected)} BIC selection(s) from: {path}")
    if missing:
        print("  WARNING: the BIC cache does not cover the following "
              f"(example, error, tau) combinations required by this config:")
        for k in missing:
            print(f"    - example={k[0]}, error={k[1]}, tau={k[2]}")
        print(f"  These will FALL BACK to the fixed hyperparameters "
              f"J={cfg.J}, lambda={cfg.lambda_}. Regenerate the cache (run "
              f"without --skip-bic) if that is not intended.")
    else:
        print(f"  Coverage complete for all {len(needed)} "
              f"(example, error, tau) combinations in this config.")
    return selected


def main():
    parser = argparse.ArgumentParser(description="DPS-ERNN Simulation Studies")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for fast validation")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--sanity-only", action="store_true",
                        help="Run only sanity checks")
    parser.add_argument("--skip-bic", action="store_true",
                        help="Skip the BIC selection phase and load existing "
                             "selections from a CSV (see --bic-csv). Only "
                             "relevant when the config has use_bic: true.")
    parser.add_argument("--bic-csv", type=str, default=None,
                        help="Path to a previously-written bic_selected.csv to "
                             "reuse with --skip-bic. Defaults to "
                             "<results_dir>/bic_selected.csv.")
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = load_config(args.config)
    elif args.quick:
        cfg = get_quick_config()
    else:
        cfg = get_full_config()

    # Resolve compute device. If cuda is requested but unavailable, fall back to
    # cpu with a clear message rather than crashing deep inside training.
    requested_device = getattr(cfg, "device", "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: device='{requested_device}' requested but CUDA is not "
              f"available; falling back to CPU.")
        cfg.device = "cpu"
    print(f"Using device: {cfg.device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if cfg.device.startswith("cuda") else ""))

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.results_dir, "tables"), exist_ok=True)

    # Log file
    log_path = os.path.join(cfg.logs_dir, "run_log.txt")
    log_file = open(log_path, "w")
    log_file.write(f"DPS-ERNN Simulation Run\n")
    log_file.write(f"Start: {datetime.now().isoformat()}\n")
    log_file.write(f"Mode: {'quick' if args.quick else 'full'}\n\n")

    # Sanity checks
    print("=" * 60)
    print("Running sanity checks...")
    print("=" * 60)
    check_results = run_all_checks(verbose=True)
    log_file.write("Sanity Checks:\n")
    for cr in check_results:
        status = "PASS" if cr["passed"] else "FAIL"
        log_file.write(f"  [{status}] {cr['name']}: {cr['message']}\n")
    log_file.write("\n")

    if args.sanity_only:
        log_file.close()
        print("\nSanity checks complete. Exiting.")
        return

    # Open CSV output
    raw_path = os.path.join(cfg.results_dir, "raw_results.csv")
    csv_file = open(raw_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=RAW_FIELDS)
    writer.writeheader()

    # ---- BIC hyperparameter selection phase ----
    bic_selected = {}  # (example, error, tau) -> (J, lambda)
    if cfg.use_bic and not args.skip_bic:
        from metrics import select_hyperparams_bic
        p_map = {1: 1, 2: 2, 3: 2}

        bic_train_kwargs = dict(
            adam_epochs=cfg.adam_epochs,
            adam_lr=cfg.adam_lr,
            lbfgs_max_iter=cfg.lbfgs_max_iter,
            grad_clip=cfg.grad_clip,
            early_stop_patience=cfg.early_stop_patience,
        )

        # Use a subsample for BIC to avoid training on full N for every
        # grid point. BIC model selection does not need the full dataset;
        # a representative subsample gives the same ranking.
        bic_N = min(cfg.N_train, getattr(cfg, "bic_subsample_N", cfg.N_train))

        bic_path = os.path.join(cfg.results_dir, "bic_selected.csv")
        bic_file = open(bic_path, "w", newline="")
        bic_writer = csv.DictWriter(bic_file, fieldnames=[
            "example", "error", "tau", "best_J", "best_lambda", "best_BIC"])
        bic_writer.writeheader()

        n_bic = len(cfg.examples) * len(cfg.errors) * len(cfg.tau_list)
        bic_count = 0

        print(f"\n{'=' * 60}")
        print(f"BIC hyperparameter selection: {n_bic} settings")
        print(f"BIC subsample size: {bic_N} "
              f"{'(full data)' if bic_N >= cfg.N_train else '(subsampled)'}")
        print(f"J grid: {cfg.J_grid}")
        print(f"lambda grid: {cfg.lambda_grid}")
        print(f"{'=' * 60}\n")

        for example in cfg.examples:
            p = p_map[example]

            for error_type in cfg.errors:
                X_bic_np, y_bic_np = generate_data(example, bic_N,
                                                    error_type,
                                                    seed=cfg.base_seed)
                X_bic = torch.tensor(X_bic_np, dtype=torch.float32).to(
                    getattr(cfg, "device", "cpu"))
                y_bic = torch.tensor(y_bic_np, dtype=torch.float32).to(
                    getattr(cfg, "device", "cpu"))

                for tau in cfg.tau_list:
                    bic_count += 1
                    print(f"[BIC {bic_count}/{n_bic}] "
                          f"Ex{example} err={error_type} tau={tau}")

                    result = select_hyperparams_bic(
                        X_bic, y_bic, tau,
                        J_grid=cfg.J_grid,
                        lambda_grid=cfg.lambda_grid,
                        p=p, seed=cfg.base_seed,
                        train_kwargs=bic_train_kwargs,
                        n_inits=getattr(cfg, "bic_n_inits", cfg.n_inits),
                        verbose=False,
                    )

                    bic_selected[(example, error_type, tau)] = (
                        result["best_J"], result["best_lambda"]
                    )

                    msg = (f"  -> J={result['best_J']}, "
                           f"lambda={result['best_lambda']:.2f}, "
                           f"BIC={result['best_bic']:.4f}")
                    print(msg)
                    log_file.write(f"BIC {example}/{error_type}/{tau}: {msg}\n")

                    bic_writer.writerow({
                        "example": example, "error": error_type, "tau": tau,
                        "best_J": result["best_J"],
                        "best_lambda": result["best_lambda"],
                        "best_BIC": f"{result['best_bic']:.6f}",
                    })
                    bic_file.flush()

        bic_file.close()
        print(f"\nBIC selection complete. Results saved to: {bic_path}\n")
        log_file.write("\n")

    elif cfg.use_bic and args.skip_bic:
        bic_csv = args.bic_csv or os.path.join(cfg.results_dir,
                                               "bic_selected.csv")
        print(f"\n{'=' * 60}")
        print(f"Skipping BIC selection; loading cached selections")
        print(f"{'=' * 60}")
        bic_selected = load_bic_selected(bic_csv, cfg)
        log_file.write(f"Loaded {len(bic_selected)} BIC selections from "
                       f"{bic_csv} (--skip-bic)\n\n")
        print()

    # ---- Main experiment loop ----
    total_settings = (len(cfg.examples) * len(cfg.errors) * len(cfg.strategies)
                      * len(cfg.tau_list) * len(cfg.M_list))

    if cfg.use_bic:
        print(f"{'=' * 60}")
        print(f"Using BIC-selected hyperparameters per (example, error, tau)")
    else:
        print(f"{'=' * 60}")
        print(f"Using fixed hyperparameters: J={cfg.J}, lambda={cfg.lambda_}")

    print(f"Starting experiments: {total_settings} settings × "
          f"{cfg.n_replications} replications")
    print(f"Examples: {cfg.examples}, Errors: {cfg.errors}")
    print(f"Strategies: {cfg.strategies}, Tau: {cfg.tau_list}")
    print(f"N_train={cfg.N_train}, N_test={cfg.N_test}, M={cfg.M_list}")
    print(f"pilot_ratio={cfg.default_pilot_ratio}")
    print(f"{'=' * 60}\n")

    t_start = time.time()
    setting_count = 0

    for example in cfg.examples:
        for error_type in cfg.errors:
            for strategy in cfg.strategies:
                for tau in cfg.tau_list:
                    # Look up BIC-selected hyperparameters if available
                    if (example, error_type, tau) in bic_selected:
                        J_sel, lam_sel = bic_selected[(example, error_type, tau)]
                    else:
                        J_sel, lam_sel = cfg.J, cfg.lambda_

                    for M in cfg.M_list:
                        setting_count += 1
                        header = (f"[{setting_count}/{total_settings}] "
                                  f"Ex{example} err={error_type} "
                                  f"S{strategy} tau={tau} M={M} "
                                  f"J={J_sel} lam={lam_sel}")
                        print(header)
                        log_file.write(f"{header}\n")

                        for rep in range(cfg.n_replications):
                            seed = cfg.base_seed + rep * 1000 + example * 100
                            try:
                                run_single_replication(
                                    example, error_type, strategy, tau, M,
                                    cfg.default_pilot_ratio, cfg, rep, seed,
                                    writer, log_file,
                                    J_override=J_sel,
                                    lambda_override=lam_sel,
                                )
                                csv_file.flush()
                            except Exception as e:
                                msg = f"  Rep {rep} FAILED: {e}"
                                print(msg)
                                log_file.write(msg + "\n")
                                traceback.print_exc()

                            if rep == 0 or (rep + 1) % 10 == 0:
                                elapsed = time.time() - t_start
                                print(f"  rep={rep+1}/{cfg.n_replications} "
                                      f"({elapsed:.0f}s elapsed)")

    csv_file.close()
    total_time = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"Experiments complete! Total time: {total_time:.1f}s")
    print(f"Results saved to: {raw_path}")
    print(f"Log saved to: {log_path}")
    print(f"{'=' * 60}")

    log_file.write(f"\nTotal time: {total_time:.1f}s\n")
    log_file.write(f"End: {datetime.now().isoformat()}\n")
    log_file.close()


if __name__ == "__main__":
    main()
