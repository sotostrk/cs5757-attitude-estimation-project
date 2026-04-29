import jax.numpy as jnp
from src.data_loader import load_euroc
from src.ekf import run_ekf
from jaxlie import SO3


timestamps, gyro, R_gt, dt = load_euroc("data/mav0")

R_est = run_ekf(gyro, R_gt, dt)

print("R_est shape:", R_est.shape)
print("First estimated rotation:\n", R_est[0])
print("First ground truth rotation:\n", R_gt[0])

# compute geodesic error at a few timesteps
for t in [0, 100, 1000, 5000, 10000, 29119]:
    err = jnp.linalg.norm(
        SO3.from_matrix(R_est[t].T @ R_gt[t]).log()
    )
    print(f"t={t:6d} | geodesic error: {err:.4f} rad")