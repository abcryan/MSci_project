# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn
from scipy.integrate import quad

from utils import calc_n_max_l, getZerosOfJ_lUpToBoundary, computeIntegralSimpson
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")



def calc_all_W(l_max, k_max, r_max, r0OfR, rOfR0, phiOfR0):
    # The maximum number of modes is when l=0
    n_max_0 = calc_n_max_l(0, k_max, r_max)

    W_lnn_prime = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))


    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max)

        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):

                W_lnn_prime[l][n1][n2] = calculate_W(n1, n2, l, r_max, r0OfR, rOfR0, phiOfR0, simpson=True, simpsonNpts=1000)

    return W_lnn_prime


def calculate_W(n, n_prime, l, r_max, r0OfR, rOfR0, phiOfR0, simpson=False, simpsonNpts=None, plot=False):
    """
    Calculate W_nn'^l, splitting the integral into chunks based on zeros of the integrand.

    Note: Technically the upper limit of the integral is r_max, but we should pass in r_max_0 in order to keep the k_ln modes the same.

    This should not change the integral as long as the selection function goes to zero well within the outer boundary.
    """

    k_ln = sphericalBesselZeros[l][n] / r_max
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max


    def W_integrand(r):
        r0 = r0OfR(r)

        return phiOfR0(r0) * spherical_jn(l, k_ln_prime*r) * spherical_jn(l, k_ln*r0) * r*r

    r_boundary = k_ln_prime * r_max
    r0_boundary = k_ln * r0OfR(r_max)

    r_zeros = getZerosOfJ_lUpToBoundary(l, r_boundary) / k_ln_prime
    r0_zeros = getZerosOfJ_lUpToBoundary(l, r0_boundary) / k_ln

    # Convert r0 values to r values
    r0_zeros = rOfR0(r0_zeros)

    # Combine and sort the zeros
    zeros = np.sort(np.append(r_zeros, r0_zeros))

    # Remove any duplicate zeros (which occur in the case r = r0)
    zeros = np.unique(zeros)


    zeros = np.append(zeros, [r_max])
    zeros = np.insert(zeros, 0, 0)


    if plot:
        x = np.linspace(0, r_max, 2000)
        y = W_integrand(x)

        plt.figure(dpi=200)
        plt.plot(x, y)
        plt.vlines(zeros, np.min(y), np.max(y), "r", "dotted")
        plt.show()


    integral = 0

    if simpson:
        for i in range(0, np.size(zeros) - 1):
            integral += computeIntegralSimpson(W_integrand, zeros[i], zeros[i+1], simpsonNpts)
    else:
        for i in range(0, np.size(zeros) - 1):
            integralChunk, error = quad(W_integrand, zeros[i], zeros[i+1])
            integral += integralChunk


    return np.power(r_max, -3) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * integral



# Old code: Calculate W by splitting the integral into Nsplit chunks
# def calc_W(n, n_prime, l, r_max, R, r0OfR, Nsplit=10, epsabs=1.49e-8):
#     k_ln = sphericalBesselZeros[l][n] / r_max
#     k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max


#     def W_integrand(r):
#         r0 = r0OfR(r)

#         return r*r * np.exp(-r0*r0/(2*R*R)) * spherical_jn(l, k_ln_prime*r) * spherical_jn(l, k_ln*r0)


#     integral = computeIntegralSplit(W_integrand, Nsplit, r_max, epsabs)

#     return np.power(r_max, -3) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * integral