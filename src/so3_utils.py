import jax.numpy as jnp

def skew(w):
   
    wx, wy, wz = w
    return jnp.array([
        [ 0.0, -wz,  wy],
        [ wz,  0.0, -wx],
        [-wy,  wx,  0.0]
    ])


