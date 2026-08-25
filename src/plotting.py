"""
Visualization helpers: per-class convergence plots, aggregate convergence
bands, the cross-dataset time-to-convergence heatmap, and the anomaly
detection image grid.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SOLVER_ORDER = ["Standard FW", "Pairwise FW", "Fully-Corrective FW"]
SOLVER_COLORS = {
    "Standard FW": "tab:blue",
    "Pairwise FW": "tab:orange",
    "Fully-Corrective FW": "tab:green",
}


def smooth_gap(gap_array, window=100):
    """Rolling geometric mean of the duality gap, in log space, to filter
    out the high-frequency zig-zagging noise of vanilla FW."""
    log_g = np.log10(np.clip(gap_array, 1e-16, None))
    log_smoothed = pd.Series(log_g).rolling(window=window, min_periods=1).mean().values
    return 10 ** log_smoothed


def plot_convergence_comparison(models_by_solver, dataset_name):
    """One row per class, duality gap vs. iterations (left) and vs. time
    (right), with all solvers overlaid."""
    classes = sorted(next(iter(models_by_solver.values())).keys())
    n_classes = len(classes)
    name_to_window = {"Standard FW": 100, "Pairwise FW": 10, "Fully-Corrective FW": 1}

    fig, axes = plt.subplots(n_classes, 2, figsize=(16, 4 * n_classes), squeeze=False)

    for i, label in enumerate(classes):
        ax_it, ax_t = axes[i]
        for name, models in models_by_solver.items():
            data = models[label]
            g = smooth_gap(np.array(data["gap_vec"]), window=name_to_window[name])
            t = np.array(data["time_vec"])[:len(g)]
            color = SOLVER_COLORS.get(name)

            ax_it.loglog(np.arange(1, len(g) + 1), g, label=name, color=color, lw=1.8)
            ax_t.semilogy(t, g, label=name, color=color, lw=1.8)

        ax_it.set_title(f"Class {label} — Duality Gap vs Iterations", fontweight="bold")
        ax_it.set_xlabel("Iterations (log)")
        ax_t.set_title(f"Class {label} — Duality Gap vs Time", fontweight="bold")
        ax_t.set_xlabel("Time (s)")
        for ax in (ax_it, ax_t):
            ax.set_ylabel("Duality Gap (rolling geo-mean)")
            ax.grid(True, which="both", alpha=0.5, linestyle="--")
            ax.legend(fontsize=9)

    fig.suptitle(f"Convergence Analysis per Class — {dataset_name}",
                 fontsize=18, fontweight="bold", y=1.001)
    plt.tight_layout()
    plt.show()


def _aggregate_curve(models, solver, x_key, n_grid=200):
    """Interpolates every class's gap trace onto a common grid (log-spaced
    for iterations, linear for time) and returns the geometric mean plus
    the 25-75% band across classes."""
    per_class, x_max = [], 0.0
    for m in models[solver].values():
        g = np.clip(np.asarray(m["gap_vec"], float), 1e-16, None)
        if x_key == "iter":
            x = np.arange(1, len(g) + 1, dtype=float)
        else:
            x = np.asarray(m["time_vec"], float)[:len(g)]
        ok = np.isfinite(x) & np.isfinite(g)
        x, g = x[ok], g[ok]
        if len(x) < 2:
            continue
        order = np.argsort(x)
        x, g = x[order], g[order]
        per_class.append((x, g))
        x_max = max(x_max, x[-1])

    if not per_class:
        return None, None, None, None

    if x_key == "iter":
        grid = np.logspace(0, np.log10(max(x_max, 1.0)), n_grid)
    else:
        grid = np.linspace(0.0, x_max, n_grid)

    logs = [np.interp(grid, x, np.log10(g), left=np.log10(g[0]), right=np.log10(g[-1]))
            for x, g in per_class]
    logs = np.vstack(logs)
    mean = 10 ** np.mean(logs, axis=0)
    lo = 10 ** np.percentile(logs, 25, axis=0)
    hi = 10 ** np.percentile(logs, 75, axis=0)
    return grid, mean, lo, hi


def plot_dataset_convergence(all_results, ds_name, save=None, band=True):
    """Aggregate (mean over the 10 classes) duality gap vs. iterations and
    vs. time, with a 25-75% inter-class shaded band."""
    models = all_results[ds_name]["models"]
    fig, (ax_it, ax_t) = plt.subplots(1, 2, figsize=(15, 5))

    for solver in models:
        color = SOLVER_COLORS.get(solver)
        xi, gi, lo_i, hi_i = _aggregate_curve(models, solver, "iter")
        if xi is not None:
            ax_it.loglog(xi, gi, color=color, lw=2.2, label=solver)
            if band:
                ax_it.fill_between(xi, lo_i, hi_i, color=color, alpha=0.12)
        xt, gt, lo_t, hi_t = _aggregate_curve(models, solver, "time")
        if xt is not None:
            ax_t.semilogy(xt, gt, color=color, lw=2.2, label=solver)
            if band:
                ax_t.fill_between(xt, lo_t, hi_t, color=color, alpha=0.12)

    ax_it.set_title(f"{ds_name.upper()} — Avg Duality Gap vs Iterations", fontweight="bold")
    ax_it.set_xlabel("Iterations (log)")
    ax_t.set_title(f"{ds_name.upper()} — Avg Duality Gap vs Time", fontweight="bold")
    ax_t.set_xlabel("Time [s]")
    for ax in (ax_it, ax_t):
        ax.set_ylabel("Duality Gap (geo-mean, log)")
        ax.grid(True, which="both", alpha=0.4, linestyle="--")
        ax.legend(fontsize=9)

    fig.suptitle(f"Aggregate Convergence Analysis — {ds_name.upper()} "
                 "(mean over the 10 classes)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


def convergence_time_frame(all_results, ds_name):
    models = all_results[ds_name]["models"]
    solvers = [s for s in SOLVER_ORDER if s in models] or list(models.keys())
    classes = sorted(next(iter(models.values())).keys(), key=lambda c: int(c))
    data = {s: [models[s][c]["time"] for c in classes] for s in solvers}
    df = pd.DataFrame(data, index=[f"Class {c}" for c in classes])
    df.index.name = ds_name.upper()
    return df


def combined_time_heatmap(all_results, datasets=("mnist", "fashion_mnist", "cifar10"),
                           save="pivot_tables_time_heatmap.png",
                           shared_scale=True, log_color=True):
    """Cross-dataset, cross-solver time-to-convergence heatmap (green=fast,
    red=slow), on a shared color scale so datasets are directly comparable."""
    datasets = [d for d in datasets if d in all_results]
    frames = {d: convergence_time_frame(all_results, d) for d in datasets}

    color_vals = {}
    for d, df in frames.items():
        color_vals[d] = np.log10(df.values + 1e-9) if log_color else df.values

    if shared_scale:
        all_c = np.concatenate([c.ravel() for c in color_vals.values()])
        vmin, vmax = all_c.min(), all_c.max()
    else:
        vmin = vmax = None

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.6), squeeze=False)
    axes = axes[0]
    last_im = None

    for ax, d in zip(axes, datasets):
        df = frames[d]
        vals = df.values
        cval = color_vals[d]
        im = ax.imshow(cval, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)
        last_im = im

        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns, rotation=15, ha="right")
        ax.set_yticks(range(df.shape[0]))
        ax.set_yticklabels(df.index)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                ax.text(j, i, f"{vals[i, j]:.2f}s", ha="center", va="center",
                        fontsize=8.5, color="black")

        ax.set_title(d.upper(), fontweight="bold", fontsize=13)

    fig.suptitle("Time-to-Convergence per Class — all datasets\n"
                 "green = fast · red = slow"
                 + ("  (shared color scale)" if shared_scale else ""),
                 fontweight="bold", fontsize=14, y=1.04)

    cb = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cb.set_label("log10(time [s])" if log_color else "time [s]")

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    return frames


def report_extreme_samples(meb_models, X_test, y_test, ratios,
                            dataset_name, solver_name, k=3):
    """Displays the k most and k least anomalous test samples per class,
    as a grid of images with their ratio scores."""
    n_features = X_test.shape[1]
    if n_features == 784:
        img_shape, cmap = (28, 28), "gray"
    elif n_features == 3072:
        img_shape, cmap = (32, 32, 3), None
    else:
        print(f"Unsupported image format ({n_features} features).")
        return

    classes = sorted(meb_models.keys())
    fig, axes = plt.subplots(2 * k, len(classes),
                              figsize=(2 * len(classes), 2.1 * 2 * k))

    print(f"\n[{dataset_name} | {solver_name}] "
          f"Top-{k} most / least anomalous samples per class (ratio = dist / R):")

    for col, label in enumerate(classes):
        idx = np.where(y_test == label)[0]
        order = np.argsort(ratios[idx])
        least = idx[order[:k]]
        most = idx[order[::-1][:k]]

        most_str = ", ".join(f"#{i} (r={ratios[i]:.2f})" for i in most)
        least_str = ", ".join(f"#{i} (r={ratios[i]:.2f})" for i in least)
        print(f"  Class {label} | MOST:  {most_str}")
        print(f"          | LEAST: {least_str}")

        for row, i in enumerate(most):
            ax = axes[row, col]
            ax.imshow(X_test[i].reshape(img_shape), cmap=cmap, interpolation="nearest")
            ax.set_title(f"r={ratios[i]:.2f}", fontsize=9, color="darkred")
            ax.axis("off")
        for row, i in enumerate(least):
            ax = axes[k + row, col]
            ax.imshow(X_test[i].reshape(img_shape), cmap=cmap, interpolation="nearest")
            ax.set_title(f"r={ratios[i]:.2f}", fontsize=9, color="darkgreen")
            ax.axis("off")

        axes[0, col].text(0.5, 1.45, f"Class {label}", transform=axes[0, col].transAxes,
                           ha="center", fontsize=11, fontweight="bold")

    axes[0, 0].set_ylabel("MOST\nanomalous", fontsize=10)
    axes[k, 0].set_ylabel("LEAST\nanomalous", fontsize=10)

    fig.suptitle(f"Anomaly Detection — {dataset_name} ({solver_name})\n"
                 f"Top rows: {k} most anomalous | Bottom rows: {k} least anomalous",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()
