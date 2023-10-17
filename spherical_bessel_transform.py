# %%

import numpy as np
from numba import jit
from scipy.special import jv, spherical_jn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
from utils import calc_n_max_l, computeIntegralSplit, getZerosOfJ_lUpToBoundary


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


# ----------------------------
# Numba version
# Uses np.interp for interpolation instead of scipy.interpolate.interp1d

@jit(nopython=True)
def spherical_jn_numba(l, x):
    if x == 0:
        return 1 if l == 0 else 0

    return jv(l + 1/2, x) * np.sqrt(np.pi / (2*x))


def get_interpolated_a_lms_numba(radii_fiducial, all_fiducial_coeffs, l_max):

    a_lm_real_interps = np.zeros((l_max + 1, l_max + 1, len(radii_fiducial)))
    a_lm_imag_interps = np.zeros((l_max + 1, l_max + 1, len(radii_fiducial)))

    for l in range(l_max + 1):
        for m in range(l + 1):
            for i in range(len(radii_fiducial)):
                coeffs = all_fiducial_coeffs[i].coeffs

                a_lm_real_interps[l][m][i] = coeffs[0][l][m]
                a_lm_imag_interps[l][m][i] = coeffs[1][l][m]

    return (a_lm_real_interps, a_lm_imag_interps)


@jit(nopython=True)
def real_integrand(r0, l, k_ln, radii_fiducial, a_lm_real_vals):
    a_lm = np.interp(r0, radii_fiducial, a_lm_real_vals)
    return spherical_jn_numba(l, k_ln * r0) * r0*r0 * a_lm

@jit(nopython=True)
def imag_integrand(r0, l, k_ln, radii_fiducial, a_lm_imag_vals):
    a_lm = np.interp(r0, radii_fiducial, a_lm_imag_vals)
    return spherical_jn_numba(l, k_ln * r0) * r0*r0 * a_lm


def calc_f_lmn_0_numba(radii_fiducial, all_observed_grids, l_max, k_max, n_max, plot=False):

    r_max_0 = radii_fiducial[-1]

    # Expand the grid using pyshtools. This outputs spherical harmonic coefficients
    all_fiducial_coeffs = [observed_grid.expand() for observed_grid in all_observed_grids]


    # Compute the a_lm(r)'s
    a_lm_real_interps, a_lm_imag_interps = get_interpolated_a_lms_numba(radii_fiducial, all_fiducial_coeffs, l_max)


    # Finally, compute f_lmn^0
    f_lmn_0 = np.zeros((l_max + 1, l_max + 1, n_max + 1), dtype=complex)


    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max_0) # Will using r_max_0 instead of r_max change the number of modes?

        print("l = %d" % l)

        for m in range(l + 1):
            for n in range(n_max_l + 1):
                # print("l = %d, m = %d, n = %d" % (l, m, n))

                k_ln = sphericalBesselZeros[l][n] / r_max_0
                c_ln = ((r_max_0)**(-3/2)) * c_ln_values_without_r_max[l][n]

                # Divide the integral into chunks,
                # according to zeros of the spherical Bessel function
                # First, locate the zeros of j_l(k_ln * r) up to r_max_0

                r_boundary = k_ln * r_max_0
                zeros = getZerosOfJ_lUpToBoundary(l, r_boundary) / k_ln
                zeros = np.append(zeros, [r_max_0])
                zeros = np.insert(zeros, 0, 0)


                # If there are fewer than 15 chunks, divide the integral into 15 chunks
                if np.size(zeros) - 1 < 15:
                    zeros = np.linspace(0, r_max_0, 15 + 1)


                # Now, compute the integral over each chunk
                real_integral = 0
                imag_integral = 0

                if plot:
                    x = radii_fiducial
                    y1 = [real_integrand(r_i, l, k_ln, x, a_lm_real_interps[l][m]) for r_i in x]
                    y2 = [imag_integrand(r_i, l, k_ln, x, a_lm_imag_interps[l][m]) for r_i in x]

                    plt.plot(x, y1)
                    plt.plot(x, y2)
                    plt.vlines(zeros, np.min(y1), np.max(y1), "r", "dotted")
                    plt.show()


                for i in range(0, np.size(zeros) - 1):

                    real_chunk, error = quad(real_integrand, zeros[i], zeros[i+1],  args=(l, k_ln, radii_fiducial, a_lm_real_interps[l][m]))
                    real_integral += real_chunk

                    imag_chunk, error = quad(imag_integrand, zeros[i], zeros[i+1], args=(l, k_ln, radii_fiducial, a_lm_imag_interps[l][m]))
                    imag_integral += imag_chunk

                # real_integral, error = quad(real_integrand, 0, r_max_0, args=(l, k_ln, radii_fiducial, a_lm_real_interps[l][m]))
                # imag_integral, error = quad(imag_integrand, 0, r_max_0, args=(l, k_ln, radii_fiducial, a_lm_imag_interps[l][m]))

                total_integral = real_integral + (1j * imag_integral)

                f_lmn_0[l][m][n] = c_ln * total_integral


    return f_lmn_0

# %%
