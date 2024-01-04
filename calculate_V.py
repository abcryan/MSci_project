# %%
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, jv, jvp
from scipy.integrate import quad
from scipy.interpolate import interp1d


from utils import calc_n_max_l, getZerosOfJ_lUpToBoundary, computeIntegralSimpson
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def interpolate_WandV_values(l_max, n_max_ls, omega_matters, Ws, Vs, step=0.00001, plot=False, plotIndex=None):
    # The maximum number of modes is when l=0
    n_max_0 = n_max_ls[0]

    # Number of interpolation outputs
    N = int(((np.max(omega_matters) - np.min(omega_matters)) / step) + 1)
    
    omega_matters_output = np.linspace(np.min(omega_matters), np.max(omega_matters), N)

    Ws_output = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1, np.size(omega_matters_output)))
    Vs_output = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1, np.size(omega_matters_output)))


    for l in range(l_max + 1):
        n_max_l = n_max_ls[l]

        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):

                # Interpolate V^l_nn (Ωₘ)

                omega_matters = omega_matters
                W_of_omega_matters = [Ws[i][l][n1][n2] for i in range(len(omega_matters))]
                V_of_omega_matters = [Vs[i][l][n1][n2] for i in range(len(omega_matters))]


                Ws_output[l][n1][n2] = interp1d(omega_matters, W_of_omega_matters, kind="quadratic")(omega_matters_output)
                Vs_output[l][n1][n2] = interp1d(omega_matters, V_of_omega_matters, kind="quadratic")(omega_matters_output)

                # only plotting one of them
                if plot:
                    l_plot, n1_plot, n2_plot = plotIndex

                    if (l == l_plot) and (n1 == n1_plot) and (n2 == n2_plot):
                        plt.figure(dpi=200)
                        # plt.plot(omega_matters, V_of_omega_matters, ".")
                        plt.plot(omega_matters_output, Vs_output[l][n1][n2])
                        plt.title(f"l={l}, n1={n1}, n2={n2}")
                        plt.show()

    return (omega_matters_output, Ws_output, Vs_output)




# Numba version
# Uses np.interp for interpolation instead of scipy.interpolate.interp1d


#@jit(nopython=True)
def spherical_jn_numba(l, x):
    if x == 0:
        return 1 if l == 0 else 0

    return jv(l + 1/2, x) * np.sqrt(np.pi / (2*x))

# calculate the derivative of the spherical bessel function
#@jit(nopython=True)
def spherical_prime_jn_numba(l, x):
    if x == 0:
        return 1 if l == 0 else 0

    return jvp(l + 1/2, x) * np.sqrt(np.pi / (2*x))



######
# THIS IS THE FUNCTION THAT IS CALLED FOR THE ANALYSIS
def calc_all_V_numba(l_max, k_max, r_max, r0_vals, r_vals, V_integrand_numba):
    # The maximum number of modes is when l=0
    n_max_0 = calc_n_max_l(0, k_max, r_max)

    V_lnn_prime = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))


    for l in range(l_max + 1):
        print("l = %d" % l)
        n_max_l = calc_n_max_l(l, k_max, r_max)

        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):

                V_lnn_prime[l][n1][n2] = calculate_V_numba(n1, n2, l, r_max, r0_vals, r_vals, V_integrand_numba)

    return V_lnn_prime
######


# r0_vals and r_vals are used for interpolation

# Use a function factory for the selection function
# (https://stackoverflow.com/questions/59573365/using-a-function-object-as-an-argument-for-numba-njit-function)


#this is what is inside the INTEGRAL
def make_V_integrand_numba(phiOfR0):

    #@jit(nopython=True)
    def V_integrand_numba(r, l, k_ln, k_ln_prime, r0_vals, r_vals):
        r0 = np.interp(r, r_vals, r0_vals)

        return phiOfR0(r0) * spherical_prime_jn_numba(l, k_ln_prime*r) * spherical_prime_jn_numba(l, k_ln*r0) * r*r
    
    return V_integrand_numba


def calculate_V_numba(n, n_prime, l, r_max, r0_vals, r_vals, V_integrand_numba):

    k_ln = sphericalBesselZeros[l][n] / r_max
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max

    r0_max = np.interp(r_max, r_vals, r0_vals)
    r_boundary = k_ln_prime * r_max
    r0_boundary = k_ln * r0_max

    r_zeros = getZerosOfJ_lUpToBoundary(l, r_boundary) / k_ln_prime
    r0_zeros = getZerosOfJ_lUpToBoundary(l, r0_boundary) / k_ln

    # Convert r0 values to r values
    r0_zeros = np.interp(r0_zeros, r0_vals, r_vals)

    # Combine and sort the zeros
    zeros = np.sort(np.append(r_zeros, r0_zeros))

    # Remove any duplicate zeros (which occur in the case r = r0)
    zeros = np.unique(zeros)


    zeros = np.append(zeros, [r_max])
    zeros = np.insert(zeros, 0, 0)


    integral = 0

    for i in range(0, np.size(zeros) - 1):
        integralChunk, error = quad(V_integrand_numba, zeros[i], zeros[i+1], args=(l, k_ln, k_ln_prime, r0_vals, r_vals))
        integral += integralChunk


    return np.power(r_max, -3) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * k_ln * np.power(k_ln_prime,-1) * integral


# %%
