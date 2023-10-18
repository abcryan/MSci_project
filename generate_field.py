# %%

import numpy as np
import pyshtools as pysh
from scipy.special import spherical_jn

from generate_f_lmn import generate_f_lmn
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from utils import calc_n_max_l
from distance_redshift_relation import *

c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def a_lm(r_i, l, m, k_max, r_max, f_lmn):
    s = 0

    n_max_l = calc_n_max_l(l, k_max, r_max)

    for n in range(n_max_l + 1):
        k_ln = sphericalBesselZeros[l][n] / r_max

        s += ((r_max)**(-3/2)) * c_ln_values_without_r_max[l][n] * spherical_jn(l, k_ln * r_i) * f_lmn[l][m][n]


    return s


def calcSphHarmCoeffs(r_i, l_max, k_max, r_max, f_lmn):
    cilm = np.zeros((2, l_max + 1, l_max + 1))

    for l in range(l_max + 1):

        # We don't need to supply the -ve m
        # Since we are working with real fields
        for m in range(l + 1):
            coeff = a_lm(r_i, l, m, k_max, r_max, f_lmn)

            cilm[0][l][m] = np.real(coeff)
            cilm[1][l][m] = np.imag(coeff)


    return cilm


def generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max):
    """
    Generates a field f(z, theta, phi).
    """

    f_lmn_true = generate_f_lmn(l_max, r_max_true, k_max)

    true_z_of_r = getInterpolatedZofR(omega_matter_true)
    z_true = true_z_of_r(radii_true)


    # Calculate the spherical harmonic coefficients for each shell
    all_coeffs = []

    for i in range(len(radii_true)):
        r_true = radii_true[i]

        cilm = calcSphHarmCoeffs(r_true, l_max, k_max, r_max_true, f_lmn_true)
        coeffs = pysh.SHCoeffs.from_array(cilm)

        all_coeffs.append(coeffs)


    # Expand the coefficients & evaluate the field on a grid
    all_grids = []

    for i in range(len(radii_true)):
        grid = all_coeffs[i].expand()

        all_grids.append(grid)


    # Return the field we've generated
    return (z_true, all_grids)


def multiplyFieldBySelectionFunction(radii_true, all_grids, phiOfR):
    """
    Multiply a field by the provided selection function, to produce the observed field.
    """

    all_observed_grids = []

    for i in range(len(radii_true)):
        grid = all_grids[i]

        all_observed_grids.append(grid * float(phiOfR(radii_true[i])))


    return (radii_true, all_observed_grids)