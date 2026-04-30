import jax
import jax.numpy as jnp
from jaxlie import SO3


def run_ekf(gyro, R_gt, dt):
    Q = jnp.diag(jnp.array([0.01, 0.01, 0.01]))
    V = jnp.diag(jnp.array([0.01, 0.01, 0.01]))

    def ekf_step(carry, inputs):
        R, P = carry
        omega, R_ground = inputs

        # --- Predict ---
        F = SO3.exp(-omega * dt).as_matrix()
        R_pred = (SO3.from_matrix(R) @ SO3.exp(omega * dt)).as_matrix()
        P_pred = F @ P @ F.T + Q

        # --- Innovation ---
        z = SO3.from_matrix(R_pred.T @ R_ground).log()

        # --- Kalman Gain ---
        K = P_pred @ jnp.linalg.inv(P_pred + V)

        # --- Update ---
        R_new = (SO3.from_matrix(R_pred) @ SO3.exp(K @ z)).as_matrix()
        P_new = (jnp.eye(3) - K) @ P_pred

        # return both R and P as outputs
        return (R_new, P_new), (R_new, P_new)

    R0 = R_gt[0]
    P0 = jnp.eye(3) * 0.1

    inputs = (gyro[:-1], R_gt[1:])
    _, (R_est, P_est) = jax.lax.scan(ekf_step, (R0, P0), inputs)

    R_est = jnp.concatenate([R0[None], R_est], axis=0)
    P_est = jnp.concatenate([P0[None], P_est], axis=0)

    return R_est, P_est