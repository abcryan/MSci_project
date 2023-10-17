
# %%
import numpy as np
import matplotlib.pyplot as plt
from generate_f_lmn import generate_f_lmn
from utils import calc_n_max_l
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

l_max = 15
r_max = 0.75
k_max = 200

# %%


def generate_f_lmn(l_max, r_max, k_max, p):
    n_max = calc_n_max_l(0, k_max, r_max)
    f_lmn_values = np.zeros((l_max + 1, l_max + 1, n_max + 1), dtype=complex)

    for l in range(l_max + 1):
        n_max_l = calc_n_max_l(l, k_max, r_max)

        for m in range(l + 1):
            for n in range(n_max_l + 1):
                k_ln = sphericalBesselZeros[l][n] / r_max

                if m == 0:
                    f_lmn_values[l][m][n] = np.random.normal(0, np.sqrt(p(k_ln)))
                else:
                    f_lmn_values[l][m][n] = np.random.normal(0, np.sqrt(p(k_ln)/2)) + np.random.normal(0, np.sqrt(p(k_ln)/2)) * 1j


    return f_lmn_values



def get_k_bin_boundaries_and_heights(f_lmn):
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

    return k_bin_dividers, k_bin_contents

# %%

amplitude_of_top_hat = 1

def p(k, k_max=300):
    if k < k_max:
        return amplitude_of_top_hat
    else:
        return 0
    
# f_lmn = generate_f_lmn(l_max, r_max, k_max, p)


# Or, load f_lmn_0 from a file
omega_matter_true = 0.315
omega_matter_0 = 0.315
l_max = 15
k_max = 200
r_max_true = 0.75
phi_r_max = 0.65

saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_phi_r_max-%.3f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, phi_r_max)

f_lmn = np.load(saveFileName)

# %%

k_bin_dividers, k_bin_contents = get_k_bin_boundaries_and_heights(f_lmn)

plt.figure(dpi=300)
plt.bar(k_bin_dividers[:-1], k_bin_contents, align="edge", width=k_bin_dividers[1]-k_bin_dividers[0])
plt.xlabel("k")
plt.ylabel("$\\frac{1}{N_{\\mathrm{modes}}} \sum_{lmn} |f_{lmn}|^2$")
plt.title("P(k) = %d" % amplitude_of_top_hat)
plt.show()

# %%

# Compare the variance of the modes to the theoretical value




# %%

# Investigate how the variance of the modes scales with the amplitude of the top hat in P(k)

# P_heights = [1, 2, 3, 4, 5]
P_heights = np.linspace(1, 10, 21)
avg_variance_of_modes = []

for P_height in P_heights:

    def p(k, k_max=300):
        if k < k_max:
            return P_height
        else:
            return 0
        
    f_lmn = generate_f_lmn(l_max, r_max, k_max, p)

    k_bin_dividers, k_bin_contents = get_k_bin_boundaries_and_heights(f_lmn)

    avg_variance = np.mean(k_bin_contents[1:]) # Use [1:] to exclude the first bin, because it contains few modes and therefore fluctuates a lot

    avg_variance_of_modes.append(avg_variance)

plt.figure(dpi=200)
plt.plot(P_heights, avg_variance_of_modes, ".")
plt.show()

# %%

coeffs = np.polyfit(P_heights, avg_variance_of_modes, 2)
x = np.linspace(0, 10, 1000)

plt.figure(dpi=200)
plt.plot(P_heights, avg_variance_of_modes, ".", label="Data")
plt.plot(x, np.poly1d(coeffs)(x), label="Quadratic fit")
plt.xlabel("Amplitude of top hat function for P(k)")
plt.ylabel("Mean of $\\frac{1}{N_{\\mathrm{modes}}} \sum_{lmn} |f_{lmn}|^2$")
plt.legend()
plt.show()

print(coeffs)

# %%
