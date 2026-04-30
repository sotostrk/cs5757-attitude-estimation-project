import time
from src.data_loader import load_euroc, truncate
from src.ekf import run_ekf
from src.evaluate import evaluate, plot_sigma_bounds, compute_geodesic_errors

# ── Easy sequence ─────────────────────────────────────────────────────
print("\n=== V1_01_easy ===")
timestamps, gyro, R_gt, dt = load_euroc("data/mav0")
timestamps, gyro, R_gt = truncate(timestamps, gyro, R_gt, seconds=30)
print(f"Truncated to {len(timestamps)} samples ({30} seconds)")

print("\nRunning EKF...")
t_start = time.time()
R_ekf, P_ekf = run_ekf(gyro, R_gt, dt)
ekf_time_easy = time.time() - t_start
print(f"EKF runtime: {ekf_time_easy:.3f} seconds")

evaluate(timestamps, R_gt, R_ekf, save_path="results/easy_error.png")
ekf_errors = compute_geodesic_errors(R_ekf, R_gt)
plot_sigma_bounds(timestamps, ekf_errors, P_ekf,
                  save_path="results/easy_sigma.png")

# ── Difficult sequence ────────────────────────────────────────────────
print("\n=== V1_03_difficult ===")
timestamps_d, gyro_d, R_gt_d, dt_d = load_euroc("data/mav0_diff")
timestamps_d, gyro_d, R_gt_d = truncate(timestamps_d, gyro_d, 
                                         R_gt_d, seconds=30)
print(f"Truncated to {len(timestamps_d)} samples ({30} seconds)")

print("\nRunning EKF...")
t_start = time.time()
R_ekf_d, P_ekf_d = run_ekf(gyro_d, R_gt_d, dt_d)
ekf_time_diff = time.time() - t_start
print(f"EKF runtime: {ekf_time_diff:.3f} seconds")

evaluate(timestamps_d, R_gt_d, R_ekf_d, 
         save_path="results/diff_error.png")
ekf_errors_d = compute_geodesic_errors(R_ekf_d, R_gt_d)
plot_sigma_bounds(timestamps_d, ekf_errors_d, P_ekf_d,
                  save_path="results/diff_sigma.png")