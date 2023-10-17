# Code adapted from
# https://gist.github.com/timothydmorton/33ed23d99e2663df4004cb236f1b8ba5

# %%

import numpy as np
from scipy.optimize import root
from scipy.special import spherical_jn
 

def spherical_jn_sensible_grid(n, m, ngrid=100000):
    """
    Returns a grid of x values that should contain the first m zeros, but not too many.
    """
    return np.linspace(n, n + 2*m*(np.pi * (np.log(n)+1)), ngrid)



def spherical_jn_zeros(n, m, ngrid=100000):
    """
    Returns first m zeros of spherical bessel function of order n
    """

    # Handle the case of n=0 separately

    if n == 0:
        # When n=0, the roots are just the roots of sin(x)

        # Skip the first root, which is just 0
        # and does not produce a valid basis function
        # Otherwise we run into errors later

        return [(i+1)*np.pi for i in range(m)]

    else:

        # Calculate on a sensible grid
        x = spherical_jn_sensible_grid(n, m, ngrid=ngrid)
        y = spherical_jn(n, x)
        
        # Find m good initial guesses from where y switches sign
        diffs = np.sign(y)[1:] - np.sign(y)[:-1]
        ind0s = np.where(diffs)[0][:m]  # First m times sign of y changes
        x0s = x[ind0s]
        
        def fn(x):
            return spherical_jn(n, x)
        
        return [root(fn, x0).x[0] for x0 in x0s]



# Note: need to make sure ngrid is high enough (so we have a high enough resolution) to capture all the zeros
# Double check the length of the array is the number of zeros we wanted

def precompute_sph_bessel_zeros(l_max, N_zeros, saveFileName):
    sphericalBesselZeros = []

    for l in range(0, l_max + 1):
        zeros = spherical_jn_zeros(l, N_zeros)

        if (len(zeros) != N_zeros):
            print("Need higher resolution for l =", l)
            break

        sphericalBesselZeros.append(zeros)


    # Save the zeros to a .csv file
    np.savetxt(saveFileName, sphericalBesselZeros, delimiter=",")


    return sphericalBesselZeros


def loadSphericalBesselZeros(fileName):
    return np.loadtxt(fileName, delimiter=",")


# precompute_sph_bessel_zeros(100, 1000, "zeros.csv")

# sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%

# sphericalBesselZeros[l][n]
# print(sphericalBesselZeros[1][100])

# %%
