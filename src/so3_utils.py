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


def exp_so3(w):
    """
    Exponential map: R3 -> SO(3) via Rodrigues formula.
    w: (3,) rotation vector, direction = axis, magnitude = angle
    returns: (3, 3) rotation matrix
    """
    angle = jnp.linalg.norm(w)
    # when angle is very small use first order approximation
    # to avoid division by zero
    W = skew(w)
    R = (jnp.eye(3)
         + jnp.sinc(angle / jnp.pi) * W
         + 0.5 * (jnp.sinc(angle / (2 * jnp.pi)))**2 * W @ W)
    return R


def log_so3(R):
    """
    Logarithmic map: SO(3) -> R3.
    R: (3, 3) rotation matrix
    returns: (3,) rotation vector
    """
    # angle of rotation from trace
    cos_angle = jnp.clip((jnp.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = jnp.arccos(cos_angle)
    
    # when angle is very small use first order approximation
    coeff = jnp.where(
        angle < 1e-6,
        0.5,
        angle / (2.0 * jnp.sin(angle))
    )
    
    # extract skew symmetric part
    log_R = coeff * (R - R.T)
    
    # extract the 3-vector from skew matrix
    w = jnp.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]])
    return w


def retract(R, dw):
    """
    Retraction on SO(3): apply a tangent vector correction to a rotation.
    R: (3, 3) rotation matrix
    dw: (3,) tangent vector
    returns: (3, 3) updated rotation matrix
    """
    return R @ exp_so3(dw)