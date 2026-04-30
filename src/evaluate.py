import jax.numpy as jnp
from jaxlie import SO3
import matplotlib.pyplot as plt
import os


def compute_geodesic_errors(R_est, R_gt):
    """
    Compute geodesic error at every timestep.
    """
    errors = jnp.array([
        jnp.linalg.norm(SO3.from_matrix(R_est[t].T @ R_gt[t]).log())
        for t in range(len(R_est))
    ])
    return errors


def compute_metrics(errors, method_name):
    """
    Compute and print RMSE and peak error.
    """
    rmse = jnp.sqrt(jnp.mean(errors**2))
    peak = jnp.max(errors)
    print(f"\n{method_name}:")
    print(f"  RMSE:       {float(rmse):.4f} rad ({float(jnp.degrees(rmse)):.4f} deg)")
    print(f"  Peak error: {float(peak):.4f} rad ({float(jnp.degrees(peak)):.4f} deg)")
    return float(rmse), float(peak)


def plot_errors(timestamps, ekf_errors, smoother_errors=None,
                save_path="results/attitude_error.png"):
    """
    Plot geodesic error over time for EKF and optionally batch smoother.
    """
    os.makedirs("results", exist_ok=True)
    t = timestamps - timestamps[0]

    plt.figure(figsize=(12, 5))
    plt.plot(t, jnp.degrees(ekf_errors),
             label="EKF", color="steelblue", linewidth=1.0)

    if smoother_errors is not None:
        plt.plot(t, jnp.degrees(smoother_errors),
                 label="Batch Smoother", color="darkorange", linewidth=1.0)

    plt.xlabel("Time (s)")
    plt.ylabel("Geodesic Error (degrees)")
    plt.title("Attitude Estimation Error over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_sigma_bounds(timestamps, errors, P_est,
                      save_path="results/sigma_bounds.png"):
    """
    Plot geodesic error with 2-sigma bounds from filter covariance.
    """
    os.makedirs("results", exist_ok=True)
    t = timestamps - timestamps[0]

    sigma = jnp.sqrt(jnp.diagonal(P_est, axis1=1, axis2=2))  # (N, 3)
    sigma_norm = 2 * jnp.linalg.norm(sigma, axis=1)           # (N,)

    plt.figure(figsize=(12, 5))
    plt.plot(t, jnp.degrees(errors),
             label="EKF error", color="steelblue", linewidth=1.0)
    plt.plot(t, jnp.degrees(sigma_norm),
             label="2σ bound", color="red",
             linewidth=1.0, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Degrees")
    plt.title("EKF Attitude Error with 2σ Bounds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_cost(costs, save_path="results/cost_vs_iteration.png"):
    """
    Plot batch smoother cost vs iteration.
    """
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


def evaluate(timestamps, R_gt, R_ekf, R_smoother=None,
             smoother_costs=None, save_path="results/attitude_error.png"):
    """
    Main evaluation function.
    """
    ekf_errors = compute_geodesic_errors(R_ekf, R_gt)
    smoother_errors = (compute_geodesic_errors(R_smoother, R_gt)
                       if R_smoother is not None else None)

    compute_metrics(ekf_errors, "EKF")
    if smoother_errors is not None:
        compute_metrics(smoother_errors, "Batch Smoother")

    plot_errors(timestamps, ekf_errors, smoother_errors,
                save_path=save_path)

    if smoother_costs is not None:
        plot_cost(smoother_costs)