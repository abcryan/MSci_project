#%%
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


# Spherical Bessel function
def spherical_jn_numba(l, x):
    if x == 0:
        return 1 if l == 0 else 0
    return jv(l + 1/2, x) * np.sqrt(np.pi / (2*x))


# define integrand
def integrand(y, r, l, k_ln, k_ln_prime, r0_vals, r_vals, sigma):
    # r0 = np.interp(r, r_vals, r0_vals)  # dont' know if needed...
    # y0 = np.interp(y, r_vals, r0_vals)  # dont' know if needed...
    gaussian = np.exp(-(r - y)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

    return gaussian * spherical_jn_numba(l, k_ln_prime*y) * spherical_jn_numba(l, k_ln*r) * r*y



# Perform the outer integration over y
def integrate_over_y(r, l, k_ln, k_ln_prime, r0_vals, r_vals, r_max, sigma):
    # Define your range for y here (this might be problem-specific)
    if (r > 4*sigma):
        y_min = r - 4*sigma
    else:
        y_min = 0
    if (r + 4*sigma < r_max):
        y_max = r + 4*sigma
    else:
        y_max = r_max
    
    result, error = quad(integrand, y_min, y_max, args=( r, l, k_ln, k_ln_prime, r0_vals, r_vals, sigma))
    return result


def calculate_F(n, n_prime, l, r_max, r0_vals, r_vals, sigma):

    k_ln = sphericalBesselZeros[l][n] / r_max
    k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max

    r0_max = np.interp(r_max, r_vals, r0_vals) #use r0_max instead of r_max. This is in order to keep the modes kln same, which, recall, are defined as the roots of the spherical Bessel functions jl, divided by the outer boundary r0_max , in order to satisfy the boundary condition.
    
    integral1, error1, t1, integral2, error2, t2, integral3, error3, t3 = 0, 0, 0, 0, 0, 0, 0, 0, 0

    # # METHOD 1: Simply Integrate over r and y
    # s1 = time.perf_counter()
    # integral1, error1 = dblquad(integrand, 0, r_max, 0, r_max, args=(l, k_ln, k_ln_prime, r0_vals, r_vals, sigma))
    # e1 = time.perf_counter()
    # # Calculate time
    # t1 = e1 - s1


    # Method 2: Integrate over y in interval sigma around r, then integrate over r
    s2 = time.perf_counter()    
    integral2, error2 = quad(integrate_over_y, 0, r_max, args=(l, k_ln, k_ln_prime, r0_vals, r_vals,r_max, sigma))
    e2 = time.perf_counter()
    # Calculate time
    t2 = e2 - s2

    # # METHOD 3: Split the integral into chunks based on zeros of the integrand, then use Method 1
    # s3 = time.perf_counter()  
    # r_boundary = k_ln * r_max
    # y_boundary = k_ln_prime * r_max
    # r_zeros = getZerosOfJ_lUpToBoundary(l, r_boundary) / k_ln
    # y_zeros = getZerosOfJ_lUpToBoundary(l, y_boundary) / k_ln_prime
    # r_zeros = np.sort(r_zeros)
    # y_zeros = np.sort(y_zeros)
    # r_zeros = np.append(r_zeros, [r_max])
    # r_zeros = np.insert(r_zeros, 0, 0)
    # y_zeros = np.append(y_zeros, [r_max])
    # y_zeros = np.insert(y_zeros, 0, 0)

    # for i in range(0, np.size(r_zeros) - 1):
    #     for j in range(0, np.size(y_zeros) - 1):
    #         integralChunk, error3 = dblquad(integrand, y_zeros[j], y_zeros[j+1], r_zeros[i], r_zeros[i+1], args=(l, k_ln, k_ln_prime, r0_vals, r_vals, sigma))
    #         integral3 += integralChunk
    # e3 = time.perf_counter()
    # # Calculate time
    # t3 = e3 - s3

    # integral2 *= np.power(r_max, -3) * np.power(np.pi, -1) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * integral 

    return integral1, error1, t1, integral2, error2,t2, integral3, error3, t3



# THIS IS THE FUNCTION THAT IS CALLED FOR THE ANALYSIS
def calculate_all_F(l_max, k_max, r_max, r0_vals, r_vals, Vol, sigma):
    # The maximum number of modes is when l=0
    n_max_0 = calc_n_max_l(0, k_max, r_max)
    F_lnn_prime = np.zeros((l_max + 1, n_max_0 + 1, n_max_0 + 1))

    for l in range(l_max + 1):
        print("l = %d" % l)
        n_max_l = calc_n_max_l(l, k_max, r_max)
        
        for n1 in range(n_max_l + 1):
            for n2 in range(n_max_l + 1):
                F_lnn_prime[l][n1][n2] = calculate_F(n1, n2, l, r_max, r0_vals, r_vals, Vol, sigma)

    return F_lnn_prime


l =15
n=2
n_prime = 2
r_max = 0.75

#%%


integral1, error1,t1, integral2, error2,t2, integral3, error3,t3 = calculate_F(n, n_prime, l, r_max, r0_vals=2, r_vals=4, sigma=0.003)
print("Method 1: ",integral1, error1)
print("Method 2: ",integral2, error2)
print("Method 3: ",integral3, error3)


print("Elapsed time Method 1: ", t1)
print("Elapsed time Method 2: ", t2)
print("Elapsed time Method 3: ", t3)

# print(spherical_jn_numba(15, -3), spherical_jn(15, -3))
# %%

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

k_ln = sphericalBesselZeros[l][n] / r_max
k_ln_prime = sphericalBesselZeros[l][n_prime] / r_max


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 20))

# Make data.
X = np.arange(0, 1, .005)
Y = np.arange(0, 1, .005)
XX, YY = np.meshgrid(X, Y)

Z = np.zeros_like(XX)
print(Z.shape)
print(X.shape, Y.shape)
for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
        Z[i][j] = integrand(X[i], Y[j], l, k_ln, k_ln_prime, r0_vals=0, r_vals=0, sigma=0.003)
        print(j)

# Plot the surface.
surf = ax.plot_surface(XX, YY, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# %%
