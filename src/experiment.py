"""
Experiment runner: trains one MEB per class, for every solver, and
collects convergence metrics (iterations, time, duality-gap trace).
"""

import numpy as np
import pandas as pd

from .data import build_meb_problem
from .solvers import MAXIT, MAXTIME, EPS, SEED


def run_experiments_per_class(X_train, y_train, solvers,
                               maxit=MAXIT, maxtime=MAXTIME, eps=EPS, seed=SEED):
    """Trains solvers['name'](Q_c, c_c, ...) independently for every class
    in y_train. Returns {solver_name: {class_label: metrics_dict}}.
    """
    classes = np.unique(y_train)
    results = {name: {} for name in solvers}

    print("-" * 110)
    print(f"{'Class':<7} | {'Solver':<22} | {'Samples':<8} | {'Iter':<8} | "
          f"{'Time (s)':<9} | {'Final Gap':<12} | {'Status'}")
    print("-" * 110)

    for label in classes:
        X_c = X_train[y_train == label]
        P_c, Q_c, c_c = build_meb_problem(X_c)

        for name, solver_fn in solvers.items():
            x_opt, iters, f_val, t_exec, time_vec, gap_vec = solver_fn(
                Q_c, c_c, maxit=maxit, maxtime=maxtime, eps=eps, seed=seed
            )

            center = P_c @ x_opt
            radius = np.sqrt(max(0.0, -f_val))
            final_gap = gap_vec[-1] if len(gap_vec) > 0 else np.inf

            gap_converged = final_gap <= eps
            reached_maxit = iters >= maxit

            if gap_converged:
                status = "Converged (Gap)"
            elif reached_maxit:
                status = "MaxIter"
            elif t_exec >= maxtime - 0.01:
                status = "Timeout"
            else:
                status = f"Stopped (gap={final_gap:.2e})"

            results[name][str(label)] = {
                "weights": x_opt, "center": center, "radius": radius,
                "num_samples": P_c.shape[1], "iters": iters, "time": t_exec,
                "time_vec": time_vec, "gap_vec": gap_vec,
                "final_gap": final_gap, "f_final": f_val,
            }

            print(f"{str(label):<7} | {name:<22} | {P_c.shape[1]:<8} | {iters:<8} | "
                  f"{t_exec:<9.2f} | {final_gap:<12.4e} | {status}")

        del Q_c, P_c

    print("-" * 110)
    return results


def build_summary_table(all_results, eps=EPS):
    """Aggregates per-class results into one row per (dataset, solver):
    convergence rate, average time/iterations/final gap over converged runs.
    """
    rows = []
    for ds_name, ds_data in all_results.items():
        for solver_name, models in ds_data["models"].items():
            n_total = len(models)
            converged = [m for m in models.values() if m["final_gap"] <= eps]
            n_conv = len(converged)

            if n_conv > 0:
                avg_time = np.mean([m["time"] for m in converged])
                avg_iters = np.mean([m["iters"] for m in converged])
                avg_gap = np.mean([m["final_gap"] for m in converged])
            else:
                avg_time = avg_iters = avg_gap = np.nan

            rows.append({
                "Dataset": ds_name.upper(),
                "Solver": solver_name,
                "Converged": f"{n_conv}/{n_total}",
                "Avg Time (s)": avg_time,
                "Avg Iterations": avg_iters,
                "Avg Final Gap": avg_gap,
            })

    df = pd.DataFrame(rows)
    df["Avg Time (s)"] = df["Avg Time (s)"].round(2)
    df["Avg Iterations"] = df["Avg Iterations"].round(0)
    df["Avg Final Gap"] = df["Avg Final Gap"].map(
        lambda v: f"{v:.4e}" if pd.notna(v) else "N/A"
    )
    return df
