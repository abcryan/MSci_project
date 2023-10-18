# %%
import numpy as np
from scipy.special import spherical_jn

from utils import calc_n_max_l, computeIntegralSplit
from generate_f_lmn import generate_f_lmn
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

from distance_redshift_relation import *


l_max = 10
k_max = 50
r_max_true = 0.8
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%

omega_matter_true = 0.5
f_lmn_true = generate_f_lmn(l_max, r_max_true, k_max)

# %%

radii_true = np.linspace(0, r_max_true, 1000)
true_z_of_r = getInterpolatedZofR(omega_matter_true)
z_true = true_z_of_r(radii_true)


# %%


def a_lm(r_i, l, m, k_max, r_max, f_lmn):
    s = 0

    # n = 0
    # k_ln = sphericalBesselZeros[l][0] / r_max

    # while k_ln < k_max:

    #     s += ((r_max)**(-3/2)) * c_ln_values[l][n] * spherical_jn(l, k_ln * r_i) * f_lmn[l][m][n]

    #     n += 1
    #     k_ln = sphericalBesselZeros[l][n] / r_max

    n_max_l = calc_n_max_l(l, k_max, r_max_true)

    for n in range(n_max_l + 1):
        k_ln = sphericalBesselZeros[l][n] / r_max

        s += ((r_max)**(-3/2)) * c_ln_values_without_r_max[l][n] * spherical_jn(l, k_ln * r_i) * f_lmn[l][m][n]


    return s


a_lms = []

for i in range(len(radii_true)):
    r_true = radii_true[i]

    a_lms.append([[a_lm(r_true, l, m, k_max, r_max_true, f_lmn_true) for m in range(l + 1)] for l in range(l_max + 1)])


# ----- OBSERVED

# Try expanding in the case where the fiducial cosmology equals the true one
omega_matter_0 = omega_matter_true

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]


# %%

a_lm_real_interps = []
a_lm_imag_interps = []

for l in range(l_max + 1):
    a_l_real_interps = []
    a_l_imag_interps = []

    for m in range(l + 1):
        real_parts = []
        imag_parts = []

        for i in range(len(radii_fiducial)):

            real_parts.append(np.real(a_lms[i][l][m]))
            imag_parts.append(np.imag(a_lms[i][l][m]))

        a_lm_interp_real = interp1d(radii_fiducial, real_parts)
        a_lm_interp_imag = interp1d(radii_fiducial, imag_parts)

        a_l_real_interps.append(a_lm_interp_real)
        a_l_imag_interps.append(a_lm_interp_imag)

    a_lm_real_interps.append(a_l_real_interps)
    a_lm_imag_interps.append(a_l_imag_interps)

# %%


# Plot some example a_lm(r)'s

# l_test, m_test = 0, 0
# l_test, m_test = 1, 0
# l_test, m_test = 1, 1
# l_test, m_test = 2, 0
# l_test, m_test = 2, 1
# l_test, m_test = 2, 2
# l_test, m_test = 3, 0
# l_test, m_test = 3, 1
# l_test, m_test = 3, 2
# l_test, m_test = 3, 3

l_test, m_test = 10, 10

plt.plot(radii_fiducial, a_lm_real_interps[l_test][m_test](radii_fiducial), label="real")
plt.plot(radii_fiducial, a_lm_imag_interps[l_test][m_test](radii_fiducial), label="imag")
plt.xlabel("r_0")
plt.title("a_%d,%d(r_0)" % (l_test, m_test))
plt.legend()
plt.show()


# %%

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

            real_integral = computeIntegralSplit(real_integrand, 100, r_max_0)
            imag_integral = computeIntegralSplit(imag_integrand, 100, r_max_0)

            total_integral = real_integral + (1j * imag_integral)

            f_lmn_0[l][m][n] = c_ln * total_integral


print(f_lmn_0)

# %%

saveFileName = "f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true)
true_f_saveFileName = "f_lmn_true-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f" % (omega_matter_true, l_max, k_max, r_max_true)

np.save(saveFileName, f_lmn_0)
np.save(true_f_saveFileName, f_lmn_true)

# %%

print(f_lmn_true - f_lmn_0)

# %%

print(np.max(f_lmn_true - f_lmn_0))

# %%