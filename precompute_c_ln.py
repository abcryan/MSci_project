# %%
import numpy as np
from scipy.special import jv
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

zeros = loadSphericalBesselZeros("zeros.csv")

# This is all for r_max = 1.

def get_c_ln(l, n):
    
    x_ln = zeros[l][n]

    c_ln = np.power((np.pi / (4 * x_ln)) * ((jv(l+1/2, x_ln))**2 - jv(l-1/2, x_ln) * jv(l+3/2, x_ln)), -1/2)

    return c_ln


def precompute_c_ln_values(l_max, n_max, saveFileName):
    c_ln_values = np.zeros((l_max + 1, n_max))

    for l in range(l_max + 1):
        for n in range(n_max):
            c_ln_values[l][n] = get_c_ln(l, n)


    # Save to a .csv file
    np.savetxt(saveFileName, c_ln_values, delimiter=",")

    return c_ln_values


def get_c_ln_values_without_r_max(fileName):
    return np.loadtxt(fileName, delimiter=",")


def get_c_ln_values_with_r_max(fileName, r_max):
    return np.power(r_max, -3/2) * np.loadtxt(fileName, delimiter=",")


# %%

# c_ln_values = precompute_c_ln_values(100, 1000, "c_ln.csv")

# %%
