import jax
import jax.numpy as jnp
from jaxlie import SO3
from src.shampoo import mat_inv_fourth_root


def cost_fn(delta_phi, R_current, gyro, dt):
    def retract_single(R, dp):
        return (SO3.from_matrix(R) @ SO3.exp(dp)).as_matrix()

    R_retracted = jax.vmap(retract_single)(R_current, delta_phi)

    R_t  = R_retracted[:-1]
    R_t1 = R_retracted[1:]

    def single_residual(R_t_i, R_t1_i, omega_i):
        R_pred = SO3.from_matrix(R_t_i) @ SO3.exp(omega_i * dt)
        residual = (SO3.from_matrix(R_t1_i).inverse() @ R_pred).log()
        return residual

    residuals = jax.vmap(single_residual)(R_t, R_t1, gyro)
    return 0.5 * jnp.sum(residuals**2)


def run_batch_smoother(gyro, R_gt, dt, n_iter=50, lr=1e-3):
    T = len(gyro)

    R_current = jnp.array(R_gt)
    delta_phi = jnp.zeros((T, 3))

    # diagonal approximation for L — shape (T,) instead of (T,T)
    L_diag = jnp.ones(T) * 1e-6   # small init for numerical safety
    # full R matrix true Shampoo on rotation axes
    R_mat  = jnp.eye(3) * 1e-6

    grad_fn = jax.jit(jax.grad(cost_fn))

    costs = []

    print(f"  Running Shampoo for {n_iter} iterations...")
    for i in range(n_iter):

        c = cost_fn(delta_phi, R_current, gyro[:-1], dt)
        G = grad_fn(delta_phi, R_current, gyro[:-1], dt)  # (T, 3)

        costs.append(float(c))
        if i % 10 == 0:
            print(f"    iter {i:3d} | cost: {float(c):.6f}")

        # update preconditioners
        # diagonal of G @ G.T is just sum of squared rows
        L_diag = L_diag + jnp.sum(G**2, axis=1)        # (T,)
        R_mat  = R_mat  + G.T @ G                        # (3, 3)

        # inverse fourth roots
        # diagonal case
        L_inv4 = (L_diag**(-0.25))[:, None]             # (T, 1)
        R_inv4 = mat_inv_fourth_root(R_mat)              # (3, 3)

        #shampoo step 
        step = (L_inv4 * G) @ R_inv4                    # (T, 3)
        delta_phi = delta_phi - lr * step

        # retract and reset
        R_current = jax.vmap(
            lambda R, dp: (SO3.from_matrix(R) @ SO3.exp(dp)).as_matrix()
        )(R_current, delta_phi)

        delta_phi = jnp.zeros((T, 3))

    return R_current, jnp.array(costs)