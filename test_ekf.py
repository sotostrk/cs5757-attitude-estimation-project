import matplotlib.pyplot as plt
from src.data_loader import load_euroc
import jax.numpy as jnp

timestamps, gyro, R_gt, dt = load_euroc("data/mav0_diff")

t = timestamps - timestamps[0]
gyro_norm = jnp.linalg.norm(gyro, axis=1)

plt.figure(figsize=(12, 4))
plt.plot(t, gyro_norm)
plt.xlabel("Time (s)")
plt.ylabel("||omega|| (rad/s)")
plt.title("Gyro magnitude over time — V1_03_difficult")
plt.grid(True, alpha=0.3)
plt.savefig("results/gyro_profile_diff.png", dpi=150)
plt.show()

print(f"Max gyro magnitude: {float(jnp.max(gyro_norm)):.3f} rad/s")
print(f"Mean gyro magnitude: {float(jnp.mean(gyro_norm)):.3f} rad/s")
print(f"Duration: {float(t[-1]):.1f} seconds")