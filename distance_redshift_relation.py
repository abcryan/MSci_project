# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline 


# H_0 = 2.26e-18
# c = 3e8
H_0 = 1
c = 1


def H(z, omega_matter):
    # assume flatness
    omega_lambda = 1 - omega_matter


    return (H_0) * ((omega_matter * (1+z)**3) + omega_lambda)**(0.5)


def r(z, omega_matter):

    def integrand(z):
        return c / H(z, omega_matter)

    integral, error = quad(integrand, 0, z)

    return integral


def plotRofZ(omega_matter):
    z_vals = np.linspace(0, 5, 2001)
    r_vals = [r(z, omega_matter) for z in z_vals]

    plt.plot(z_vals, r_vals)
    plt.xlabel("z")
    plt.ylabel("r(z)")
    plt.title("$\Omega_m$=%.2f" % omega_matter)
    plt.show()



def plotZofR(omega_matter):
    z_vals = np.linspace(0, 5, 2001)
    r_vals = [r(z, omega_matter) for z in z_vals]

    plt.plot(r_vals, z_vals)
    plt.xlabel("r")
    plt.ylabel("z(r)")
    plt.title("$\Omega_m$=%.2f" % omega_matter)
    plt.show()



def getInterpolatedRofZ(omega_matter):
    z_vals = np.linspace(0, 5, 2001)
    r_vals = [r(z, omega_matter) for z in z_vals]

    r_interp = interp1d(z_vals, r_vals)
    return r_interp



def getInterpolatedZofR(omega_matter):
    z_vals = np.linspace(0, 5, 2001)
    r_vals = [r(z, omega_matter) for z in z_vals]

    z_interp = interp1d(r_vals, z_vals)
    return z_interp



def getInterpolatedR0ofR(omega_matter_0, omega_matter):
    z_vals = np.linspace(0, 5, 2001)
    r_vals = [r(z, omega_matter) for z in z_vals]
    r0_vals = [r(z, omega_matter_0) for z in z_vals]

    r0OfR = interp1d(r_vals, r0_vals)
    return r0OfR


# %%

# omega_matter = 0.5
# plotRofZ(omega_matter)
# plotZofR(omega_matter)


# %%

def calculate_dr0Bydz(omega_matter_0):
    z_vals = np.linspace(0, 5, 2001)
    r0_vals = [r(z, omega_matter_0) for z in z_vals]

    u = InterpolatedUnivariateSpline(z_vals, r0_vals, k=2)
    dr0Bydz = u.derivative()

    # plt.plot(z_vals, dr0Bydz(z_vals))
    # plt.show()

    return dr0Bydz


# %%

# Calculate the Jacobian


# import numpy as np
# import matplotlib.pyplot as plt


# u = InterpolatedUnivariateSpline(r_0_vals, r_vals, k=2)
# u_der = u.derivative()

# plt.plot(r_0_vals, r_vals, 'go')
# plt.plot(r_0_vals, u(r_0_vals), 'b--')
# plt.plot(r_0_vals, u_der(r_0_vals), 'k')

# # %%

# jacobian = r_vals**2 / r_0_vals**2 * u_der(r_0_vals)

# # %%

# plt.plot(r_0_vals, jacobian)

# %%

def calculateJacobian(r_vals, r_0_vals):
    u = InterpolatedUnivariateSpline(r_0_vals, r_vals, k=2)
    u_der = u.derivative()

    # plt.plot(r_0_vals, r_vals, 'go')
    # plt.plot(r_0_vals, u(r_0_vals), 'b--', label="interpolated r(r_0)")
    # plt.plot(r_0_vals, u_der(r_0_vals), 'k', label="dr/dr_0")
    # plt.xlabel("$r_0$")
    # plt.ylabel("r")
    # plt.legend()
    # plt.show()


    jacobian_vals = r_vals**2 / r_0_vals**2 * u_der(r_0_vals)

    jacobian = interp1d(r_0_vals, jacobian_vals)

    return jacobian

# %%


def partialRbyOmegaMatter(omega_matter, zOfR0):

    def integrand(z, omega_matter):
        return ((1 + z)**3 - 1) / ((omega_matter*(1 + z)**3 + 1 - omega_matter)**(3/2))

    integral, error = quad(integrand, 0, zOfR0, (omega_matter))

    return (c / H_0) * (-1/2) * integral


def plotPartialRbyOmegaMatter(omega_matter):
    z_vals = np.linspace(0, 5, 2001)
    r_vals = [r(z, omega_matter) for z in z_vals]
    partial_r_vals = [partialRbyOmegaMatter(omega_matter, z_vals[i]) for i in range(len(z_vals))]


    plt.plot(r_vals, partial_r_vals)
    plt.xlabel("$r_0$")
    plt.ylabel("$\partial r / \partial \Omega_m$ at $r_0$")
    plt.title("$\Omega_m$=%.2f" % omega_matter)
    plt.show()


def getPartialRbyOmegaMatterInterp(omega_matter):
    z_vals = np.linspace(0, 5, 2001)
    r_vals = [r(z, omega_matter) for z in z_vals]

    partial_r_vals = [partialRbyOmegaMatter(omega_matter, z_vals[i]) for i in range(len(z_vals))]

    # Interpolate
    dr_domega_interp = interp1d(r_vals, partial_r_vals)

    return dr_domega_interp



# %%

# omega_m = 0.5

# plotRofZ(omega_m)
# plotZofR(omega_m)
# plotPartialRbyOmegaMatter(omega_m)

# # %%

# plotPartialRbyOmegaMatter(omega_m)

# %%

# omega_matters = [0.3, 0.4, 0.5, 0.6, 0.7]
# z_vals = np.linspace(0, 1, 1000)

# for omega_m in omega_matters:
#     r_of_z_interp = getInterpolatedRofZ(omega_m)

#     plt.plot(z_vals, r_of_z_interp(z_vals), label="Ωₘ = %.3f" % omega_m)

# plt.legend()
# plt.xlabel("z")
# plt.ylabel("r(z)")
# plt.show()

# # %%

# r_vals = np.linspace(0, 0.8, 1000)

# for omega_m in omega_matters:
#     z_of_r_interp = getInterpolatedZofR(omega_m)

#     plt.plot(r_vals, z_of_r_interp(r_vals), label="Ωₘ = %.3f" % omega_m)

# plt.legend()
# plt.xlabel("r")
# plt.ylabel("z(r)")
# plt.show()

# # %%

# for omega_m in omega_matters:
#     dr_domega_interp = getPartialRbyOmegaMatterInterp(omega_m)

#     plt.plot(r_vals, dr_domega_interp(r_vals), label="Ωₘ = %.3f" % omega_m)

# plt.legend()
# plt.xlabel("r")
# plt.ylabel("$\partial r / \partial \Omega_m$ at r")
# plt.show()

# %%
