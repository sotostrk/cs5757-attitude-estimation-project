import jax.numpy as jnp

def mat_inv_fourth_root(A):
    """
    Compute A^(-1/4) via eigendecomposition.
    A: (n, n) symmetric positive definite matrix
    returns: (n, n) matrix
    """
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-6)  # numerical safety
    return eigvecs @ jnp.diag(eigvals**(-0.25)) @ eigvecs.T