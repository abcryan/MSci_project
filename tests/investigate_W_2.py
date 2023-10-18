# %%
import numpy as np
import matplotlib.pyplot as plt
from utils import calc_n_max_l

# %%

omega_matter_true = 0.315
omega_matter_0 = 0.315
l_max = 15
k_max = 200
r_max_true = 0.75
r_max_0 = 0.75
R = 0.25    

# saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_with_phi.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R)

# f_lmn_0 = np.load(saveFileName)


# %%

W_1st_terms_saveFileName = "W_1st_terms_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (l_max, k_max, r_max_0, R)
W_2nd_terms_without_delta_omega_m_saveFileName = "W_2nd_terms_without_delta_omega_m_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (l_max, k_max, r_max_0, R)


W_1st_terms = np.load(W_1st_terms_saveFileName)
W_2nd_terms_without_delta_omega_m = np.load(W_2nd_terms_without_delta_omega_m_saveFileName)

# %%

# n, n_prime, l = 37, 39, 10
n, n_prime, l = 30, 1, 1
omega_matters = np.linspace(omega_matter_0 - 0.01, omega_matter_0 + 0.01, 11)

W_full_integrals_quad = []
W_full_integrals_zeros = []
W_taylor_expansions = []


for omega_matter in omega_matters:
    W_quad_saveFileName = "data/W_no_tayl_exp_quad_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    W_zeros_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)

    # Full integral, using quad and splitting into 10 chunks
    W_full_int_quad = np.load(W_quad_saveFileName)[l][n][n_prime]

    # Full integral, splitting integral into chunks according to zeros of integrand
    W_full_int_zeros = np.load(W_zeros_saveFileName)[l][n][n_prime]


    # Taylor expansion
    delta_omega_matter = (omega_matter_0 - omega_matter)

    W_tayl_exp = W_1st_terms[l][n][n_prime] + (W_2nd_terms_without_delta_omega_m[l][n][n_prime] * delta_omega_matter)


    W_full_integrals_quad.append(W_full_int_quad)
    W_full_integrals_zeros.append(W_full_int_zeros)
    W_taylor_expansions.append(W_tayl_exp)



plt.figure(dpi=200)
plt.plot(omega_matters, W_taylor_expansions, label="Taylor expansion")
plt.plot(omega_matters, W_full_integrals_quad, label="Full integral, quad")
plt.plot(omega_matters, W_full_integrals_zeros, label="Full integral, zeros")
plt.legend()
plt.xlabel("$\Omega_m$")
plt.title("$W_{%d,%d}^{%d}$" % (n, n_prime, l))
plt.show()

# %%

calc_n_max_l(15, k_max, r_max_true)

# %%


omega_matters = np.linspace(omega_matter_0 - 0.01, omega_matter_0 + 0.01, 21)

for omega_matter in omega_matters:
    W_saveFileName = "W_tayl_exp_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)

    # Taylor expansion
    delta_omega_matter = (omega_matter_0 - omega_matter)

    W_tayl_exp = W_1st_terms + (W_2nd_terms_without_delta_omega_m * delta_omega_matter)

    np.save(W_saveFileName, W_tayl_exp)


# %%

omega_matters = np.linspace(omega_matter_0 - 0.01, omega_matter_0 + 0.01, 11)

# %%
# omega_matters = [0.305]

for omega_matter in omega_matters:
    W_quad_saveFileName = "W_no_tayl_exp_quad_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    W_zeros_saveFileName = "W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)

    # Full integrals
    W_full_int_quad = np.load(W_quad_saveFileName)


    # Taylor expansions
    delta_omega_matter = (omega_matter_0 - omega_matter)

    W_tayl_exp = W_1st_terms + (W_2nd_terms_without_delta_omega_m * delta_omega_matter)


    # Print the difference
    # print(W_full_int - W_tayl_exp)
    print("omega_m = %.3f," % omega_matter, "maximum difference:", np.max(W_full_int_quad - W_tayl_exp))
# %%

from scipy.special import spherical_jn
from scipy.integrate import quad, simpson
from distance_redshift_relation import *
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from utils import gaussianPhi

c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def computeIntegralSplit(integrand, N, upperLimit, epsabs=1.49e-8):
    answer = 0
    step = upperLimit / N

    for i in range(N):
        print("Integrating from %.4e to %.4e." % (i*step, (i+1)*step))
        integral, error = quad(integrand, i*step, (i+1)*step, epsabs=epsabs)
        answer += integral
        # print(error)

    return answer


def computeIntegralSimpson(integrand, lowerLimit, upperLimit, Npts):

    x = np.linspace(lowerLimit, upperLimit, Npts)
    y = integrand(x)

    integral = simpson(y, dx=x[1] - x[0])

    return integral



def getZerosOfJ_lUpToBoundary(l, upperLimit):
    n = 0
    root = sphericalBesselZeros[l][0]

    if root < upperLimit:
        while root < upperLimit:
            n += 1
            root = sphericalBesselZeros[l][n]

        n_max = n - 1

        return sphericalBesselZeros[l][:n_max + 1]
        
    else:
        return []



def integrateWSplitByZeros(n, n_prime, l, r_max, r0OfR, rOfR0, phiOfR0, simpson=False, simpsonNpts=None, split=False, plot=False):
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

    zeros = zeros[::2]


    # zeros = np.linspace(0, r_max, 101)
    zeros = np.append(zeros, [r_max])
    zeros = np.insert(zeros, 0, 0)
    print(zeros)


    if plot:
        x = np.linspace(0, r_max, 2000)
        y = W_integrand(x)

        plt.figure(dpi=200)
        plt.plot(x, y)
        plt.vlines(zeros, np.min(y), np.max(y), "r", "dotted")
        plt.title("$W_{%d,%d}^{%d}$ integrand" % (n, n_prime, l))
        plt.show()


    integral = 0

    if simpson:
        for i in range(0, np.size(zeros) - 1):
            # print("Integrating from %.4e to %.4e." % (zeros[i], zeros[i+1]))
            integral += computeIntegralSimpson(W_integrand, zeros[i], zeros[i+1], simpsonNpts)
    elif split:
        integral = computeIntegralSplit(W_integrand, 1000, r_max)
    else:
        for i in range(0, np.size(zeros) - 1):
            # print("Integrating from %.4e to %.4e." % (zeros[i], zeros[i+1]))
            integralChunk, error = quad(W_integrand, zeros[i], zeros[i+1])
            integral += integralChunk


    return np.power(r_max, -3) * c_ln_values_without_r_max[l][n] * c_ln_values_without_r_max[l][n_prime] * integral


# %%

omega_matter = 0.315
print("Computing W's for Ωₘ = %.3f." % omega_matter)
r0OfR = getInterpolatedR0ofR(omega_matter_0, omega_matter)
rOfR0 = getInterpolatedR0ofR(omega_matter, omega_matter_0)

def phi(r0):
    return gaussianPhi(r0, R)

# %%

integrateWSplitByZeros(n, n_prime, l, r_max_0, r0OfR, rOfR0, phi, simpson=True, simpsonNpts=1000, plot=True)
# %%
integrateWSplitByZeros(n, n_prime, l, r_max_0, r0OfR, rOfR0, phi, split=True)
# %%
integrateWSplitByZeros(n, n_prime, l, r_max_0, r0OfR, rOfR0, phi)
# %%


W_quad_saveFileName = "W_no_tayl_exp_quad_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)

# Full integrals
W_full_int_quad = np.load(W_quad_saveFileName)

# %%

W_full_int_quad[l][n][n_prime]

# %%
