# %%
import numpy as np
from numba import jit
from utils import calc_n_max_l
from precompute_sph_bessel_zeros import loadSphericalBesselZeros


sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def create_power_spectrum(k_max, N_k_bins, k_bin_heights):
    k_bin_edges = np.linspace(0, k_max, N_k_bins + 1)

    # Remove the k=0 edge since it's unnecessary
    k_bin_edges = k_bin_edges[1:]

    return k_bin_edges, k_bin_heights


# Parametrised power spectrum
@jit(nopython=True)
def P_parametrised(k, k_bin_edges, k_bin_heights):

    if k <= 0:
        return 0

    for i in range(len(k_bin_edges)):
        if k <= k_bin_edges[i]:
            return k_bin_heights[i]

    return 0


# Top hat power spectrum
def P_top_hat(k, k_max=300):
    if k < k_max:
        return 1
    else:
        return 0


def generate_f_lmn(l_max, r_max, k_max, P):
    n_max = calc_n_max_l(0, k_max, r_max)
    f_lmn_values = np.zeros((l_max + 1, l_max + 1, n_max + 1), dtype=complex)

    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max)

        for m in range(l + 1):
            for n in range(n_max_l + 1):
                k_ln = sphericalBesselZeros[l][n] / r_max

                if m == 0:
                    f_lmn_values[l][m][n] = np.random.normal(0, np.sqrt(P(k_ln)))
                else:
                    f_lmn_values[l][m][n] = np.random.normal(0, np.sqrt(P(k_ln)/2)) + np.random.normal(0, np.sqrt(P(k_ln)/2)) * 1j


    return f_lmn_values



# %%

# f_lmn = generate_f_lmn(100, 100)


# %%

# f_lmn

# %%
