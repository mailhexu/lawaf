import numpy as np
from math import sin, cos


#@numba.njit('float64[:](float64[:],float64[:])', fastmath=True)
def cross(a, b):
    ret = np.empty(shape=3, dtype=numba.float64)
    ret[0] = a[1] * b[2] - a[2] * b[1]
    ret[1] = a[2] * b[0] - a[0] * b[2]
    ret[2] = a[0] * b[1] - a[1] * b[0]
    return ret


#@numba.njit('float64[:](float64, float64[:], float64[:])', fastmath=True)
def rotate_vec(angle, axis, vec):
    half_angle = 0.5 * angle
    r = axis * sin(half_angle)
    w = cos(half_angle)
    return vec + 2.0 * cross(r, (cross(r, vec) + w * vec))
