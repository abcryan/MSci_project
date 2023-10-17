# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from distance_redshift_relation import getInterpolatedRofZ, calculate_dr0Bydz

# Fit taken from Alonso et al. 2018
# https://arxiv.org/pdf/1809.01669.pdf
def n_eff(z):
    # Photometric sample
    return z*z*np.exp(-np.power(z/0.28, 0.90))


def calcNormalisationAndTotalExpNo():
    # Normalise n_eff(z), turning it into a PDF

    # Compute the normalisation factor
    normalisation, error = quad(n_eff, 0, np.inf)

    def p(z):
        return n_eff(z) / normalisation

    # Y10 photometric sample
    # Number density = 48 arcmin^-2
    # 148510800 square arcminutes in a sphere
    # (= 4pi steradians * (180/pi)^2 * (60*60) )

    total_expected_number = 48 * 148510660

    return normalisation, total_expected_number


def calc_n_of_z():
    
    normalisation, total_expected_number = calcNormalisationAndTotalExpNo()

    def n(z):
        return total_expected_number * n_eff(z) / normalisation

    return n

# -----

def calc_nbar():
    normalisation, total_expected_number = calcNormalisationAndTotalExpNo()

    A = total_expected_number / normalisation

    nbar_code_units = A / (4*np.pi)
    # nbar = nbar_code_units / (3000**3) # nbar in Mpc^-1 h

    return nbar_code_units


def calc_phi_of_r0(omega_matter_0):
    dr0Bydz = calculate_dr0Bydz(omega_matter_0)

    z = np.linspace(0, 4, 500)
    r0ofZ = getInterpolatedRofZ(omega_matter_0)
    r0 = r0ofZ(z)

    n = calc_n_of_z()
    nbar = calc_nbar()

    phi = n(z) / (4*np.pi * r0*r0 * nbar * dr0Bydz(z))
    phi[0] = 1 # Solve division by 0 error at r = 0 (manually set phi = 1)

    phiOfR0 = interp1d(r0, phi)

    return phiOfR0

# %%

# # Fit taken from Alonso et al. 2018
# # https://arxiv.org/pdf/1809.01669.pdf

# def n_eff(z):
#     # Photometric sample
#     return z*z*np.exp(-np.power(z/0.28, 0.90))

#     # Source sample
#     # return z*z*np.exp(-np.power(z/0.11, 0.68))

# # %%

# # Plot n_eff

# x = np.linspace(0, 4, 500)
# y = n_eff(x)
# plt.plot(x, y)
# plt.xlabel("z")
# plt.title("$n_{eff}(z)$")
# plt.show()

# # %%

# # Normalise n_eff(z), turning it into a PDF

# # Compute the normalisation factor
# normalisation, error = quad(n_eff, 0, np.inf)

# def p(z):
#     return n_eff(z) / normalisation

# # %%

# # Plot the PDF

# y = p(x)
# plt.plot(x, y)
# plt.xlabel("z")
# plt.title("$p(z)$")
# plt.show()

# # %%

# # Y10 photometric sample
# # Number density = 48 arcmin^-2
# # 148510800 square arcminutes in a sphere
# # (= 4pi steradians * (180/pi)^2 * (60*60) )

# total_expected_number = 48 * 148510660

# def n(z):
#     return total_expected_number * p(z)

# y = n(x)
# plt.figure(dpi=200)
# plt.plot(x, y)
# plt.xlabel("z")
# plt.title("$N*p(z) = n(z)$")
# plt.show()

# # %%

# # Compute nbar(z)

# A = total_expected_number / normalisation

# nbar_code_units = A / (4*np.pi)
# nbar = nbar_code_units / (3000**3) # nbar in Mpc^-1 h

# print("n̅ = %.3e (code units)" % nbar_code_units)
# print("n̅ = %.3e (Mpc^-1 h)^3" % nbar)

# # %%

# # Calculate phi
# omega_matter_0 = 0.315

# dr0Bydz = calculate_dr0Bydz(omega_matter_0)

# z = np.linspace(0, 4, 500)
# r0ofZ = getInterpolatedRofZ(omega_matter_0)
# r0 = r0ofZ(z)


# y = n(z) / (4*np.pi * r0*r0 * nbar_code_units * dr0Bydz(z))
# y[0] = 1

# plt.plot(z, y)
# plt.xlabel("z")
# plt.ylabel("$\phi(z)$")
# plt.show()


# # %%
