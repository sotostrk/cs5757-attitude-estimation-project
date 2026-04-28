from src.data_loader import load_euroc

timestamps, gyro, R_gt, dt = load_euroc("data/mav0")

print("gyro shape:", gyro.shape)
print("R_gt shape:", R_gt.shape)
print("First gyro reading:", gyro[0])
print("First rotation matrix:\n", R_gt[0])

# Sanity check: rotation matrices should be orthogonal
import jax.numpy as jnp
err = jnp.mean(jnp.abs(R_gt[0] @ R_gt[0].T - jnp.eye(3)))
print("Orthogonality error (should be ~0):", err)