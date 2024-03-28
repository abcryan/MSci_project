# #%% 
import time
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, jv
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d


from utils import calc_n_max_l, getZerosOfJ_lUpToBoundary, computeIntegralSimpson
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


# THIS IS THE FUNCTION THAT IS CALLED FOR THE ANALYSIS

# use for loop to do the sum since more precise
def calculate_all_GorH(l_max, k_max, r_max, F, WorV):
    # The maximum number of modes is when l=0
    n_max_0 = calc_n_max_l(0, k_max, r_max)
    GorH_lnn_prime = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))
    # GorH_lnn_prime_METHOD2 = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))

    for l in range(l_max + 1):
        print("l = %d" % l)
        n_max_l = calc_n_max_l(l, k_max, r_max)
        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):
                GorH_lnn_prime[l, n1, n2] = np.sum(F[l, n1, :] * WorV[l, :, n2])
                # for n in range(n_max_l + 1):
                #     GorH_lnn_prime_METHOD2[l][n1][n2] += (F[l, n1, n]*WorV[l, n, n2])

    # return GorH_lnn_prime, GorH_lnn_prime_METHOD2
    return GorH_lnn_prime



# #%%
# l_max = 15
# k_max = 200
# r_max = 0.75


# n_max_0 = calc_n_max_l(0, k_max, r_max)

# B = np.random.rand(l_max + 1, n_max_0 + 1, n_max_0 + 1)
# A = np.random.rand(l_max + 1, n_max_0 + 1, n_max_0 + 1)

# print(n_max_0)

# X,Y = calculate_all_GorH(l_max, k_max, r_max, A, B)

# print(np.not_equal(X, Y))
# print((X[-2,5,0]))

# # print(np.sum(A[1,:,3]*B[1,4,:]))
# # M= 0
# # for n in range(n_max_0 + 1):
# #     M+=(A[1, n, 3]*B[1, 4, n])
# # print(M)
# # print(np.where(np.not_equal(X, Y)))

# #%%