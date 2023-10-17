
# %%
import numpy as np
import matplotlib.pyplot as plt
from generate_f_lmn import generate_f_lmn
from utils import calc_n_max_l
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

l_max = 15
r_max = 1.5
k_max = 200

# %%

f_lmn = generate_f_lmn(l_max, r_max, k_max)

# %%

Nbins = 16
k_bin_dividers = np.linspace(0, k_max, Nbins + 1)
k_bin_contents = [[] for _ in range(Nbins)]


for l in range(l_max + 1):
    n_max_l = calc_n_max_l(l, k_max, r_max)

    for n in range(n_max_l + 1):
        k_ln = sphericalBesselZeros[l][n] / r_max

        # Loop through all k bins to see which k bin this (l,n) mode belongs in

        for i in range(Nbins):

            if k_ln >= k_bin_dividers[i] and k_ln < k_bin_dividers[i+1]:

                # Add |f_lmn|^2 for all m
                for m in range(l + 1):
                    f_lmn_squared = f_lmn[l][m][n] * np.conj(f_lmn[l][m][n])

                    k_bin_contents[i].append(f_lmn_squared)

                break

# Divide the contents of each k bin by the number of modes, so that we're taking an average

for i in range(Nbins):
    k_bin_contents_i = np.array(k_bin_contents[i])
    N_modes_in_bin = np.size(k_bin_contents_i)

    k_bin_contents[i] = np.sum(k_bin_contents_i) / N_modes_in_bin

# %%

plt.figure(dpi=300)
plt.bar(k_bin_dividers[:-1], k_bin_contents, align="edge", width=k_bin_dividers[1]-k_bin_dividers[0])
plt.xlabel("k")
plt.ylabel("$\\frac{1}{N_{\\mathrm{modes}}} \sum_{lmn} |f_{lmn}|^2$")
plt.show()

# %%
