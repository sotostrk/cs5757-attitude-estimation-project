import jax
import jax.numpy as jnp
from jaxlie import SO3


def run_ekf(gyro, R_gt, dt, sigma=0.3):
    Q = jnp.diag(jnp.array([1e-5, 1e-5, 1e-5]))
    #V = jnp.eye(3) * (sigma ** 2)

    def ekf_step(carry, inputs):
        R, P, key = carry
        omega, R_gt_t = inputs

        # ── Predict ───────────────────────────────────────────────────────────
        F      = SO3.exp(-omega * dt).as_matrix()
        R_pred = (SO3.from_matrix(R) @ SO3.exp(omega * dt)).as_matrix()
        P_pred = F @ P @ F.T + Q

        # ── Update ────────────────────────────────────────────────────────────
        # scale measurement noise with angular velocity
        omega_norm = jnp.linalg.norm(omega)
        sigma_t    = sigma * (1.0 + 5.0 * omega_norm)
        V_t        = jnp.eye(3) * (sigma_t ** 2)

        # synthetic noisy attitude measurement
        key, subkey = jax.random.split(key)
        eta    = sigma_t * jax.random.normal(subkey, shape=(3,))
        R_meas = (SO3.from_matrix(R_gt_t) @ SO3.exp(eta)).as_matrix()

        # innovation
        z   = SO3.from_matrix(R_pred.T @ R_meas).log()
        H   = jnp.eye(3)
        S   = H @ P_pred @ H.T + V_t
        K   = P_pred @ H.T @ jnp.linalg.inv(S)

        R_new = (SO3.from_matrix(R_pred) @ SO3.exp(K @ z)).as_matrix()
        IKH   = jnp.eye(3) - K @ H
        P_new = IKH @ P_pred @ IKH.T + K @ V_t @ K.T

        return (R_new, P_new, key), (R_new, P_new)

    R0  = R_gt[0]
    P0  = jnp.eye(3) * 1e-6
    key = jax.random.PRNGKey(0)

    _, (R_est, P_est) = jax.lax.scan(
        ekf_step, (R0, P0, key), (gyro[:-1], R_gt[1:])
    )

    R_est = jnp.concatenate([R0[None], R_est], axis=0)
    P_est = jnp.concatenate([P0[None], P_est], axis=0)

    return R_est, P_est