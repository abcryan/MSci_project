# %%
import numpy as np
from utils import calc_n_max_l
from precompute_sph_bessel_zeros import loadSphericalBesselZeros


sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def p(k, k_max=300):
    if k < k_max:
        return 1
    else:
        return 0


def generate_f_lmn(l_max, r_max, k_max):
    n_max = calc_n_max_l(0, k_max, r_max)
    f_lmn_values = np.zeros((l_max + 1, l_max + 1, n_max + 1), dtype=complex)

    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max)

        for m in range(l + 1):
            for n in range(n_max_l + 1):
                k_ln = sphericalBesselZeros[l][n] / r_max

                if m == 0:
                    f_lmn_values[l][m][n] = np.random.normal(0, np.sqrt(p(k_ln)))
                else:
                    f_lmn_values[l][m][n] = np.random.normal(0, np.sqrt(p(k_ln)/2)) + np.random.normal(0, np.sqrt(p(k_ln)/2)) * 1j


    return f_lmn_values



# %%

# f_lmn = generate_f_lmn(100, 100)


# %%

# f_lmn

# %%
