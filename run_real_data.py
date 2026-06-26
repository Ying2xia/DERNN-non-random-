"""
Real-data experiments framework (two modes, both over multiple expectile levels).

Datasets
--------
Application 1: UCI Household Electric Power Consumption
    - Chronological train/test split (80/20); NOT random (time series).
Application 2: Beijing Multi-Site Air Quality
    - Station-based partitioning (each monitoring station is one worker):
      a natural form of non-random storage.

Modes
-----
--mode compare  (default): the head-to-head. All five methods (Centralized,
    Pilot, OS-ERNN, CSL-ERNN, DPS-ERNN) at each expectile level. For household,
    all three storage strategies; for air quality, the station partition. This
    is where OS/CSL degrade under non-random storage while DPS tracks the
    centralized benchmark.

--mode sweep: pilot-ratio sensitivity. Only Centralized (a pilot-independent
    reference, run once per tau) and DPS-ERNN (swept over pilot ratios r) are
    evaluated -- OS/CSL do not use a pilot sample, and Pilot is the baseline DPS
    replaces, so neither belongs in an r-sweep. Shows DPS approaches the
    full-data oracle even with small pilots, across expectile levels.

All pipeline fixes apply (prox_rho passed to the surrogate methods,
adam_lr=0.01 / adam_epochs=2000, device threading). Because the conditional
tau-expectile is unknown for real data, MAE/RMSE are computed against the
held-out responses y. RACT = time(Centralized) / time(method).

Usage:
    python run_real_data.py --mode compare --dataset both \
        --household-path data/household_power_consumption.txt --airquality-path air/
    python run_real_data.py --mode sweep --dataset both \
        --household-path data/household_power_consumption.txt --airquality-path air/
    (add --device cuda for GPU)
"""
import argparse
import os
import numpy as np
import torch
import csv

from methods import run_centralized, run_pilot, run_os_ernn, run_csl, run_dps
from metrics import evaluate_model
from partition import partition_data, poisson_pilot_sample

ALL_METHODS = ["Centralized", "Pilot", "OS-ERNN", "CSL-ERNN", "DPS-ERNN"]


def load_household_data(path: str):
    import pandas as pd
    df = pd.read_csv(path, sep=";", low_memory=False, na_values=["?"])
    df = df.dropna().reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                    format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    feature_cols = ["Global_reactive_power", "Voltage", "Global_intensity",
                    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Global_active_power"].values.astype(np.float32)
    X_mean, X_std = X.mean(0), X.std(0); X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std
    split = int(0.8 * len(X))
    print(f"  Household: N_train={split}, N_test={len(X)-split}, p={X.shape[1]}")
    return X[:split], y[:split], X[split:], y[split:], None


def load_airquality_data(path: str):
    import pandas as pd
    import glob
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {path}")
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    feature_cols = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
    available = [c for c in feature_cols if c in data.columns]
    data = data.dropna(subset=available + ["PM2.5"]).reset_index(drop=True)
    X = data[available].values.astype(np.float32)
    y = data["PM2.5"].values.astype(np.float32)
    station = (data["station"].values if "station" in data.columns else np.zeros(len(X)))
    X_mean, X_std = X.mean(0), X.std(0); X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    n_test = min(20000, len(X) // 5)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    print(f"  AirQuality: N_train={len(train_idx)}, N_test={len(test_idx)}, "
          f"p={X.shape[1]}, stations={len(np.unique(station))}")
    return (X[train_idx], y[train_idx], X[test_idx], y[test_idx], station[train_idx])


def station_partition(station_labels):
    widx = []
    for s in sorted(set(station_labels.tolist())):
        idx = np.where(station_labels == s)[0]
        if len(idx) > 0:
            widx.append(idx)
    return widx


def make_worker_indices(dataset, X_train, station, strategy, M, seed):
    """Household -> requested storage strategy; air quality -> station-based."""
    if dataset == "airquality" and station is not None:
        return station_partition(station)
    rng = np.random.default_rng(seed)
    return partition_data(X_train, M, strategy=strategy, rng=rng)


def fit_one(name, X_tr, y_tr, X_te, y_te, worker_data, pilot_X, pilot_y,
            p, J, lam, tau, seed, N, tk, prox_rho):
    runners = {
        "Centralized": lambda: run_centralized(X_tr, y_tr, tau, lam, p, J, seed, tk),
        "Pilot": lambda: run_pilot(pilot_X, pilot_y, tau, lam, p, J, seed, tk),
        "OS-ERNN": lambda: run_os_ernn(worker_data, tau, lam, p, J, seed, N, tk),
        "CSL-ERNN": lambda: run_csl(worker_data, tau, lam, p, J, seed, N, tk, prox_rho=prox_rho),
        "DPS-ERNN": lambda: run_dps(worker_data, pilot_X, pilot_y, tau, lam, p, J, seed, N, tk, prox_rho=prox_rho),
    }
    res = runners[name]()
    ev = evaluate_model(res["model"], X_te, y_te)
    return ev["MAE"], ev["RMSE"], res["info"]["time_seconds"]


def main():
    parser = argparse.ArgumentParser(description="DPS-ERNN Real Data Experiments")
    parser.add_argument("--mode", choices=["compare", "sweep"], default="compare",
                        help="compare = all 5 methods; sweep = Centralized + DPS over pilot ratios")
    parser.add_argument("--dataset", choices=["household", "airquality", "both"], default="both")
    parser.add_argument("--household-path", default="data/household_power_consumption.txt")
    parser.add_argument("--airquality-path", default="air/")
    parser.add_argument("--tau-list", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--pilot-ratios", type=float, nargs="+",
                        default=[0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
                        help="pilot ratios swept in --mode sweep")
    parser.add_argument("--pilot-ratio", type=float, default=0.05,
                        help="single pilot ratio used in --mode compare")
    parser.add_argument("--household-strategy", type=int, default=2,
                        help="storage strategy for household in --mode sweep")
    parser.add_argument("--M", type=int, default=20, help="workers for household")
    parser.add_argument("--J", type=int, default=10)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.01)
    parser.add_argument("--prox-rho", type=float, default=1.0)
    parser.add_argument("--adam-lr", type=float, default=0.01)
    parser.add_argument("--adam-epochs", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: device='{device}' requested but CUDA unavailable; using CPU.")
        device = "cpu"
    print(f"Using device: {device} | mode: {args.mode}")

    tk = dict(adam_epochs=args.adam_epochs, adam_lr=args.adam_lr, lbfgs_max_iter=50,
              grad_clip=10.0, early_stop_patience=200, device=device)
    J, lam, seed = args.J, args.lambda_, 42
    output = args.output or (f"results/real_data.csv" if args.mode == "compare"
                             else "results/real_data_sensitivity.csv")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    datasets = []
    if args.dataset in ["household", "both"]:
        datasets.append(("household", args.household_path, load_household_data))
    if args.dataset in ["airquality", "both"]:
        datasets.append(("airquality", args.airquality_path, load_airquality_data))

    rows = []
    for dsname, path, loader in datasets:
        print("\n" + "=" * 52 + f"\n{dsname}  ({args.mode})\n" + "=" * 52)
        if not os.path.exists(path):
            print(f"  Data not found: {path}")
            continue
        X_tr, y_tr, X_te, y_te, station = loader(path)
        p = X_tr.shape[1]
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
        y_te_t = torch.tensor(y_te, dtype=torch.float32).to(device)
        N = len(X_tr)

        if args.mode == "compare":
            strategies = [1, 2, 3] if dsname == "household" else ["station"]
            for strat in strategies:
                widx = make_worker_indices(dsname, X_tr, station,
                                           strat if isinstance(strat, int) else 1,
                                           args.M, seed)
                worker_data = [{"X": X_tr_t[i], "y": y_tr_t[i], "N_m": len(i)} for i in widx]
                rng_p = np.random.default_rng(seed + 1)
                pidx = poisson_pilot_sample(N, len(widx), widx, args.pilot_ratio, rng_p)
                pilot_X, pilot_y = X_tr_t[pidx], y_tr_t[pidx]
                for tau in args.tau_list:
                    print(f"\n  -- {dsname} strategy={strat} tau={tau} (M={len(widx)}) --")
                    times = {}
                    for name in ALL_METHODS:
                        try:
                            mae, rmse, t = fit_one(name, X_tr_t, y_tr_t, X_te_t, y_te_t,
                                                   worker_data, pilot_X, pilot_y,
                                                   p, J, lam, tau, seed, N, tk, args.prox_rho)
                            times[name] = t
                            print(f"    {name:12s} MAE={mae:.4f} RMSE={rmse:.4f} time={t:.1f}s")
                            rows.append({"dataset": dsname, "strategy": strat, "tau": tau,
                                         "pilot_ratio": args.pilot_ratio, "method": name,
                                         "MAE": mae, "RMSE": rmse, "time": t})
                        except Exception as e:
                            print(f"    {name:12s} FAILED: {e}")
                            rows.append({"dataset": dsname, "strategy": strat, "tau": tau,
                                         "pilot_ratio": args.pilot_ratio, "method": name,
                                         "MAE": float("nan"), "RMSE": float("nan"), "time": float("nan")})
                    tc = times.get("Centralized", float("nan"))
                    for r in rows[-len(ALL_METHODS):]:
                        t = r["time"]
                        r["RACT"] = (tc / t) if (t and t == t and t > 0) else float("nan")

        else:  # sweep: Centralized (once per tau) + DPS over pilot ratios
            strat = args.household_strategy if dsname == "household" else "station"
            widx = make_worker_indices(dsname, X_tr, station,
                                       args.household_strategy, args.M, seed)
            worker_data = [{"X": X_tr_t[i], "y": y_tr_t[i], "N_m": len(i)} for i in widx]
            for tau in args.tau_list:
                print(f"\n  -- {dsname} strategy={strat} tau={tau} (M={len(widx)}) --")
                # Centralized reference (pilot-independent): run once per tau
                try:
                    mae, rmse, t_cen = fit_one("Centralized", X_tr_t, y_tr_t, X_te_t, y_te_t,
                                               worker_data, None, None, p, J, lam, tau,
                                               seed, N, tk, args.prox_rho)
                    print(f"    {'Centralized':12s} (ref)      MAE={mae:.4f} RMSE={rmse:.4f} time={t_cen:.1f}s")
                    rows.append({"dataset": dsname, "strategy": strat, "tau": tau,
                                 "pilot_ratio": "full", "method": "Centralized",
                                 "MAE": mae, "RMSE": rmse, "time": t_cen, "RACT": 1.0})
                except Exception as e:
                    print(f"    Centralized FAILED: {e}"); t_cen = float("nan")
                for r in args.pilot_ratios:
                    rng_p = np.random.default_rng(seed + 1 + int(r * 100000))
                    pidx = poisson_pilot_sample(N, len(widx), widx, r, rng_p)
                    pilot_X, pilot_y = X_tr_t[pidx], y_tr_t[pidx]
                    try:
                        mae, rmse, t = fit_one("DPS-ERNN", X_tr_t, y_tr_t, X_te_t, y_te_t,
                                               worker_data, pilot_X, pilot_y,
                                               p, J, lam, tau, seed, N, tk, args.prox_rho)
                        ract = (t_cen / t) if (t and t > 0 and t_cen == t_cen) else float("nan")
                        print(f"    DPS-ERNN r={r:6.1%} n={len(pidx):6d} MAE={mae:.4f} RMSE={rmse:.4f} time={t:.1f}s")
                        rows.append({"dataset": dsname, "strategy": strat, "tau": tau,
                                     "pilot_ratio": r, "method": "DPS-ERNN",
                                     "MAE": mae, "RMSE": rmse, "time": t, "RACT": ract})
                    except Exception as e:
                        print(f"    DPS r={r} FAILED: {e}")

    if rows:
        with open(output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "strategy", "tau", "pilot_ratio",
                                              "method", "MAE", "RMSE", "time", "RACT"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in w.fieldnames})
        print(f"\nResults written to: {output}")


if __name__ == "__main__":
    main()
