import time
import numpy as np
from src.data_loader import load_euroc, truncate
from src.ekf import run_ekf
from src.batch_smoother import run_batch_smoother
from src.evaluate import (
    compute_geodesic_errors, compute_metrics,
    plot_error_grid, plot_cost_comparison,
    plot_runtime_bars, plot_sigma_bounds,
)

# ── Easy sequence ─────────────────────────────────────────────────────────────
print("\n=== V1_01_easy ===")
timestamps_e, gyro_e, R_gt_e, dt_e = load_euroc("data/mav0")
timestamps_e, gyro_e, R_gt_e = truncate(
    timestamps_e, gyro_e, R_gt_e, seconds=30)
print(f"Truncated to {len(timestamps_e)} samples")

print("\nRunning EKF...")
t0 = time.time()
R_ekf_e, P_ekf_e = run_ekf(gyro_e, R_gt_e, dt_e)
ekf_time_e = time.time() - t0
print(f"EKF runtime: {ekf_time_e:.3f}s")

print("\nRunning Batch Smoother...")
t0 = time.time()
R_batch_e, costs_e = run_batch_smoother(gyro_e, R_gt_e, dt_e, n_iter=50, lr=1e-2)
batch_time_e = time.time() - t0
print(f"Batch smoother runtime: {batch_time_e:.3f}s")

ekf_errors_e   = compute_geodesic_errors(R_ekf_e,   R_gt_e)
batch_errors_e = compute_geodesic_errors(R_batch_e, R_gt_e)
compute_metrics(ekf_errors_e,   "EKF (easy)")
compute_metrics(batch_errors_e, "Batch (easy)")

# ── Difficult sequence ────────────────────────────────────────────────────────
print("\n=== V1_03_difficult ===")
timestamps_d, gyro_d, R_gt_d, dt_d = load_euroc("data/mav0_diff")
timestamps_d, gyro_d, R_gt_d = truncate(
    timestamps_d, gyro_d, R_gt_d, seconds=30, offset=10)
print(f"Truncated to {len(timestamps_d)} samples")

print("\nRunning EKF...")
t0 = time.time()
R_ekf_d, P_ekf_d = run_ekf(gyro_d, R_gt_d, dt_d)
ekf_time_d = time.time() - t0
print(f"EKF runtime: {ekf_time_d:.3f}s")

print("\nRunning Batch Smoother...")
t0 = time.time()
R_batch_d, costs_d = run_batch_smoother(gyro_d, R_gt_d, dt_d, n_iter=50, lr=1e-2)
batch_time_d = time.time() - t0
print(f"Batch smoother runtime: {batch_time_d:.3f}s")

ekf_errors_d   = compute_geodesic_errors(R_ekf_d,   R_gt_d)
batch_errors_d = compute_geodesic_errors(R_batch_d, R_gt_d)
compute_metrics(ekf_errors_d,   "EKF (difficult)")
compute_metrics(batch_errors_d, "Batch (difficult)")

# ── Combined plots ────────────────────────────────────────────────────────────
print("\nGenerating plots...")

plot_error_grid(
    timestamps_e, ekf_errors_e, batch_errors_e,
    timestamps_d, ekf_errors_d, batch_errors_d,
    save_path="results/error_grid.png",
)

plot_cost_comparison(costs_e, costs_d,
                     save_path="results/cost_comparison.png")

plot_runtime_bars(
    (ekf_time_e + ekf_time_d) / 2,
    (batch_time_e + batch_time_d) / 2,
    save_path="results/runtime_bars.png"
)

plot_sigma_bounds(timestamps_e, ekf_errors_e, P_ekf_e,
                  title="EKF ±2σ Consistency — V1_01_easy",
                  save_path="results/sigma_easy.png")

plot_sigma_bounds(timestamps_d, ekf_errors_d, P_ekf_d,
                  title="EKF ±2σ Consistency — V1_03_difficult",
                  save_path="results/sigma_diff.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Runtime Summary ===")
print(f"Easy      — EKF: {ekf_time_e:.3f}s | Batch: {batch_time_e:.3f}s")
print(f"Difficult — EKF: {ekf_time_d:.3f}s | Batch: {batch_time_d:.3f}s")