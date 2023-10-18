# %%
from scipy.special import spherical_jn
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros
from scipy.integrate import quad

c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")


def integrand(r, l, n, r_max):
    k = sphericalBesselZeros[l][n] / r_max

    return (((r_max)**(-3/2)) * c_ln_values_without_r_max[l][n])**2 * (spherical_jn(l, k*r))**2 * r*r

def checkOrthonormality(l, n, r_max):
    integral = quad(integrand, 0, r_max, args=(l, n, r_max))

    # If the basis functions are orthonormal,
    # we expect the integral to evaluate to 1
    return integral


# %%

result = checkOrthonormality(5, 1, 10)

print(result)

# %%

# Should be close to 0
1 - result[0]

# %%

result = checkOrthonormality(100, 50, 1)

print(result)

# %%
