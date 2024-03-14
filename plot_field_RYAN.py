import numpy as np
from numba import jit
from os import path
from utils import *

from generate_f_lmn import create_power_spectrum, P_parametrised, generate_f_lmn
from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from generate_field import generateGeneralField_given_delta_lmn
from distance_redshift_relation import *
from spherical_bessel_transform import calc_f_lmn_0_numba, calc_f_lmn_0
from calculate_W import calc_all_W_numba, make_W_integrand_numba, interpolate_W_values
from calculate_V import calc_all_V_numba, make_V_integrand_numba, interpolate_WandV_values
from calculate_F import calculate_all_F, interpolate_WVandF_values
from calculate_SN import calc_all_SN
from compute_likelihood import computeLikelihoodParametrised
from compute_likelihood_WandV import computeLikelihoodParametrised_WandV, computeLikelihood
from compute_likelihood_WVandF import computeLikelihoodParametrised_WVandF
from analyse_likelihood import plotContour, plotPosterior
from utils import calc_n_max_l, gaussianPhi
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

from multiprocessing import Pool

import emcee
import corner
import arviz as az


# Simple power spectrum
def P_Top_Hat(k, k_max=200):
    if k < k_max:
        return 1
    else:
        return 0

# %%

#########################
### SET UP PARAMETERS ###

l_max = 25 #15 #70 # 40 # 25 # 15
k_max = 200 
r_max_true = 0.75
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0
n_max_ls = np.array([calc_n_max_l(l, k_max, r_max_true) for l in range(l_max + 1)])
R = 0.25    # Selection function scale length
sigma = 0.001 # velocity dispersion

omega_matter_true = 0.315
omega_matter_0 = 0.315    # fiducial

P_amp = 1

# RSD parameter
b_true = 1.0   # galaxy bias parameter, 1.0 <= b <= 1.5 usually in RSD Surveys
beta_true = omega_matter_true**0.6 / b_true
# beta_true = 0.0
#########################
#########################

# More sophisticated power specturm

k_bin_edges, k_bin_heights = create_power_spectrum(k_max, 10, np.array([0.1, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]))
# k_bin_edges, k_bin_heights = create_power_spectrum(k_max, 2, np.array([0.35, 0.8]))
k_vals = np.linspace(0, 400, 1000)
P_vals = [P_parametrised(k, k_bin_edges, k_bin_heights) for k in k_vals]

# #1071E5
plt.plot(k_vals, P_vals, c="k", lw=1.25)
plt.xlim(0)
plt.ylim(0)
plt.xlabel("$k$")
plt.title("$P(k)$")
plt.tight_layout()
# plt.savefig("thesis/plots/power_spectrum_10_bins.svg")
plt.show()
    
def P_para(k, k_max=200):
    return P_parametrised(k, k_bin_edges, k_bin_heights)

# %%
# Calculate c_ln coefficients of true SBT with infinite r
c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")

# Calculate spherical Bessel zeros
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# Generate true field
radii_true = np.linspace(0, r_max_true, 1001)  
z_true, all_grids, f_lmn_true = generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max, P_para)



# %%

print(np.shape(all_grids), np.shape(z_true))
print(all_grids)

fig, ax = plt.subplots(figsize=(12, 8))

# Your custom plot call, I'm assuming this creates a plot on the existing axes 'ax'
# If all_grids[50].plot() returns a collection or contour set, capture it
# For demonstration, I'll assume it returns a QuadMesh or similar object
c = all_grids[50].plot(ax=ax)  # You may need to adjust this call based on what all_grids[50].plot() actually does

# Add a colorbar to the figure, based on the plot created
fig.colorbar(c, ax=ax)

# fig.savefig("PlotRadius50.png", dpi=500)

# %%

# plotField(all_grids[1000], title=r"$\delta(\mathbf{r})$ at $r$=%.2f" % radii_true[50] + "\n" + "$r_{max}$=%.2f, $k_{max}$=%d, $l_{max}$=%d" % (r_max_true, k_max, l_max), saveFileName="field.svg")
plotField(all_grids[10], title=r"$r$ = %.3f" % radii_true[10], saveFileName="field.png")

# %%


