import jax.numpy as jnp

def skew(w):
    """
    Convert a 3-vector to a 3x3 skew-symmetric matrix.
    w: (3,) array
    returns: (3, 3) skew-symmetric matrix
    """
    wx, wy, wz = w
    return jnp.array([
        [ 0.0, -wz,  wy],
        [ wz,  0.0, -wx],
        [-wy,  wx,  0.0]
    ])