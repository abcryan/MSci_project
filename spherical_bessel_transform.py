# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn
from scipy.interpolate import interp1d
from utils import calc_n_max_l, computeIntegralSplit


from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def get_interpolated_a_lms(radii_fiducial, all_fiducial_coeffs, l_max):
    a_lm_real_interps = []
    a_lm_imag_interps = []

    for l in range(l_max + 1):
        a_l_real_interps = []
        a_l_imag_interps = []

        for m in range(l + 1):
            real_parts = []
            imag_parts = []

            for i in range(len(radii_fiducial)):
                coeffs = all_fiducial_coeffs[i].coeffs

                real_parts.append(coeffs[0][l][m])
                imag_parts.append(coeffs[1][l][m])

            a_lm_interp_real = interp1d(radii_fiducial, real_parts)
            a_lm_interp_imag = interp1d(radii_fiducial, imag_parts)

            a_l_real_interps.append(a_lm_interp_real)
            a_l_imag_interps.append(a_lm_interp_imag)

        a_lm_real_interps.append(a_l_real_interps)
        a_lm_imag_interps.append(a_l_imag_interps)

    return (a_lm_real_interps, a_lm_imag_interps)


def plot_a_lm(radii_fiducial, a_lm_real_interps, a_lm_imag_interps, l, m):
    plt.plot(radii_fiducial, a_lm_real_interps[l][m](radii_fiducial), label="real")
    plt.plot(radii_fiducial, a_lm_imag_interps[l][m](radii_fiducial), label="imag")
    plt.xlabel("r_0")
    plt.title("a_%d,%d(r_0)" % (l, m))
    plt.legend()
    plt.show()


def calc_f_lmn_0(radii_fiducial, all_observed_grids, l_max, k_max, n_max):

    r_max_0 = radii_fiducial[-1]

    # Expand the grid using pyshtools. This outputs spherical harmonic coefficients
    all_fiducial_coeffs = [observed_grid.expand() for observed_grid in all_observed_grids]


    # Compute the a_lm(r)'s
    a_lm_real_interps, a_lm_imag_interps = get_interpolated_a_lms(radii_fiducial, all_fiducial_coeffs, l_max)


    # Finally, compute f_lmn^0
    f_lmn_0 = np.zeros((l_max + 1, l_max + 1, n_max + 1), dtype=complex)


    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max_0) # Will using r_max_0 instead of r_max change the number of modes?

        print("l = %d" % l)

        for m in range(l + 1):
            for n in range(n_max_l + 1):
                k_ln = sphericalBesselZeros[l][n] / r_max_0
                c_ln = ((r_max_0)**(-3/2)) * c_ln_values_without_r_max[l][n]

                def real_integrand(r0):
                    return spherical_jn(l, k_ln * r0) * r0*r0 * a_lm_real_interps[l][m](r0)

                def imag_integrand(r0):
                    return spherical_jn(l, k_ln * r0) * r0*r0 * a_lm_imag_interps[l][m](r0)


                # real_integral, error = quad(real_integrand, 0, r_max_0)
                # imag_integral, error = quad(imag_integrand, 0, r_max_0)

                real_integral = computeIntegralSplit(real_integrand, 10, r_max_0)
                imag_integral = computeIntegralSplit(imag_integrand, 10, r_max_0)

                total_integral = real_integral + (1j * imag_integral)

                f_lmn_0[l][m][n] = c_ln * total_integral


    return f_lmn_0