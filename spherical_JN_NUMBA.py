import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, jv
from scipy.integrate import quad
from scipy.interpolate import interp1d


@njit
def spherical_jn_numba(l, x):
    if x == 0:
        return 1 if l == 0 else 0

    return jv(l + 1/2, x) * np.sqrt(np.pi / (2*x))


x = np.arange(0, 10, 0.1)
l = 10

a = spherical_jn_numba(l, x)



