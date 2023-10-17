# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

import matplotlib as mpl
from cartopy import crs as ccrs


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def calc_n_max_l(l, k_max, r_max):
    n = 0
    k_ln = sphericalBesselZeros[l][0] / r_max

    if k_ln < k_max:

        while k_ln < k_max:
            n += 1
            k_ln = sphericalBesselZeros[l][n] / r_max

        return n - 1
        
    else:
        return 0


# Selection function
def gaussianPhi(r, R):
    return np.exp(-r*r/(2*R*R))


# The default error tolerance used by scipy.quad is epsabs=1.49e-8
def computeIntegralSplit(integrand, N, upperLimit, epsabs=1.49e-8):
    answer = 0
    step = upperLimit / N

    for i in range(N):
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
    # Used to compute integrals for W and SN
    # by splitting them into chunks based on zeros of integrand
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



# Plotting fields

def plotFieldOld(grid, r_i, r_max, k_max, l_max, lmax_calc):

    title = "r_i = %.2f, r_max = %.2f, k_max = %.2f, l_max = %d, lmax_calc = %d" % (r_i, r_max, k_max, l_max, lmax_calc)

    # Plot the field using the Mollweide projection

    fig = grid.plotgmt(projection='mollweide', colorbar='right', title=title)
    fig.show()


def plotField(grid, title="", colorbarLabel=r'$\delta(r, \theta, \phi)$', saveFileName=None):
    mpl.rcParams.update({"axes.grid" : True, "grid.color": "#333333"})

    # i = 500
    # title = r"$\delta(\mathbf{r})$ at $r$=%.2f" % radii_true[i] + "\n" + "$r_{max}$=%.2f, $k_{max}$=%d, $l_{max}$=%d" % (r_max_true, k_max, l_max)

    fig, ax = grid.plot(
        projection=ccrs.Mollweide(),
        colorbar='right',
        cb_label=colorbarLabel,
        title=title,
        grid=True,
        show=False)
    
    if saveFileName:
        plt.savefig("field.svg", transparent=True, dpi=300)

    plt.show()