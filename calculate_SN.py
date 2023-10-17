# %%
import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import quad

from utils import calc_n_max_l, getZerosOfJ_lUpToBoundary, computeIntegralSimpson
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")



def calc_all_SN(l_max, k_max, r_max_0, phiOfR0):
    # The maximum number of modes is when l=0
    n_max_0 = calc_n_max_l(0, k_max, r_max_0)

    SN_lnn_prime = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))


    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max_0)

        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):

                SN_lnn_prime[l][n1][n2] = calculate_SN(n1, n2, l, r_max_0, phiOfR0, simpson=True, simpsonNpts=1000)

    return SN_lnn_prime


def calculate_SN(n, n_prime, l, r_max_0, phiOfR0, simpson=False, simpsonNpts=None):
    """
    Calculate the shot noise, splitting the integral into chunks based on zeros of the integrand.
    """

    k_ln = sphericalBesselZeros[l][n] / r_max_0
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max_0


    def SN_integrand(r0):

        return phiOfR0(r0) * spherical_jn(l, k_ln_prime*r0) * spherical_jn(l, k_ln*r0) * r0*r0

    r0_boundary_1 = k_ln_prime * r_max_0
    r0_boundary_2 = k_ln * r_max_0

    r0_zeros_1 = getZerosOfJ_lUpToBoundary(l, r0_boundary_1) / k_ln_prime
    r0_zeros_2 = getZerosOfJ_lUpToBoundary(l, r0_boundary_2) / k_ln


    # Combine and sort the zeros
    zeros = np.sort(np.append(r0_zeros_1, r0_zeros_2))

    # Remove any duplicate zeros (which occur in the case n=n')
    zeros = np.unique(zeros)

    zeros = np.append(zeros, [r_max_0])
    zeros = np.insert(zeros, 0, 0)


    integral = 0

    if simpson:
        for i in range(0, np.size(zeros) - 1):
            integral += computeIntegralSimpson(SN_integrand, zeros[i], zeros[i+1], simpsonNpts)
    else:
        for i in range(0, np.size(zeros) - 1):
            integralChunk, error = quad(SN_integrand, zeros[i], zeros[i+1])
            integral += integralChunk


    return np.power(r_max_0, -3) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * integral
