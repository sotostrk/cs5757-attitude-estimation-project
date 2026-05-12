import numpy as np
import jax.numpy as jnp
from jaxlie import SO3
import matplotlib.pyplot as plt
import os


# ── Error computation 

def compute_geodesic_errors(R_est, R_gt):
    errors = jnp.array([
        jnp.linalg.norm(SO3.from_matrix(R_est[t].T @ R_gt[t]).log())
        for t in range(len(R_est))
    ])
    return errors


def compute_metrics(errors, method_name):
    rmse = float(jnp.sqrt(jnp.mean(errors**2)))
    peak = float(jnp.max(errors))
    print(f"\n{method_name}:")
    print(f"  RMSE:       {rmse:.4f} rad ({np.degrees(rmse):.4f} deg)")
    print(f"  Peak error: {peak:.4f} rad ({np.degrees(peak):.4f} deg)")
    return rmse, peak


# ── Individual plot helpers

def plot_cost(costs, save_path="results/cost_vs_iteration.png"):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(costs, color="darkorange", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Batch Smoother Cost vs Iteration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


# ── Main combined plots  

def plot_error_grid(timestamps_easy, ekf_errors_easy, batch_errors_easy,
                    timestamps_diff, ekf_errors_diff, batch_errors_diff,
                    save_path="results/error_grid.png"):
    """
    2x2 grid: rows = easy / difficult, cols = EKF / Batch Smoother.
    """
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey=False)
    fig.suptitle("Geodesic Attitude Error over Time", fontsize=13)

    data = [
        (timestamps_easy, ekf_errors_easy,   "EKF — V1_01_easy",       "steelblue"),
        (timestamps_easy, batch_errors_easy,  "Batch — V1_01_easy",     "darkorange"),
        (timestamps_diff, ekf_errors_diff,    "EKF — V1_03_difficult",  "steelblue"),
        (timestamps_diff, batch_errors_diff,  "Batch — V1_03_difficult","darkorange"),
    ]

    for ax, (ts, errs, title, color) in zip(axes.flat, data):
        t = np.array(ts) - ts[0]
        ax.plot(t, np.degrees(np.array(errs)), color=color, linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (deg)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_cost_comparison(costs_easy, costs_diff,
                         save_path="results/cost_comparison.png"):
    """
    Both sequences' batch smoother cost on the same axes.
    """
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(np.array(costs_easy), color="steelblue",   linewidth=1.5, label="V1_01_easy")
    plt.plot(np.array(costs_diff), color="darkorange",  linewidth=1.5, label="V1_01_difficult")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Batch Smoother Cost vs Iteration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_runtime_bars(ekf_time, batch_time,
                      save_path="results/runtime_bars.png"):
    os.makedirs("results", exist_ok=True)
    labels = ["EKF\n(online)", "Batch Smoother\n(offline)"]
    times  = [ekf_time, batch_time]
    colors = ["steelblue", "darkorange"]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(labels, times, color=colors, width=0.4)
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{t:.2f}s", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Wall-clock time (s)")
    plt.title("Runtime Comparison (averaged over both sequences)")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_sigma_bounds(timestamps, errors, P_est,
                      title="EKF Attitude Error with ±2σ Bounds",
                      save_path="results/sigma_bounds.png"):
    os.makedirs("results", exist_ok=True)
    t = np.array(timestamps) - timestamps[0]
    err_deg = np.degrees(np.array(errors))

    # Per-axis 1σ from diagonal of P, then norm for the combined bound
    sigma = np.sqrt(np.array(
        jnp.diagonal(P_est, axis1=1, axis2=2)  # (T, 3)
    ))
    sigma_norm_deg = np.degrees(2 * np.linalg.norm(sigma, axis=1))  # 2σ combined

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(t,  sigma_norm_deg, -sigma_norm_deg,
                    alpha=0.25, color="steelblue", label="±2σ bound")
    ax.plot(t, err_deg,  color="purple", linewidth=1.0, label="EKF error")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Degrees")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


# ── Legacy wrapper 

def evaluate(timestamps, R_gt, R_ekf, R_smoother=None,
             smoother_costs=None, save_path="results/attitude_error.png"):
    ekf_errors = compute_geodesic_errors(R_ekf, R_gt)
    smoother_errors = (compute_geodesic_errors(R_smoother, R_gt)
                       if R_smoother is not None else None)
    compute_metrics(ekf_errors, "EKF")
    if smoother_errors is not None:
        compute_metrics(smoother_errors, "Batch Smoother")
    if smoother_costs is not None:
        plot_cost(smoother_costs)
    return ekf_errors, smoother_errors