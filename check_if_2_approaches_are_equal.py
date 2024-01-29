# %%
import numpy as np
from numba import jit
from os import path

from generate_f_lmn import create_power_spectrum, P_parametrised, generate_f_lmn
from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from generate_field import generateGeneralField_given_delta_lmn
from distance_redshift_relation import *
from spherical_bessel_transform import calc_f_lmn_0_numba, calc_f_lmn_0
from calculate_W import calc_all_W_numba, make_W_integrand_numba, interpolate_W_values
from calculate_V import calc_all_V_numba, make_V_integrand_numba, interpolate_WandV_values
from calculate_SN import calc_all_SN
from compute_likelihood import computeLikelihoodParametrised
from compute_likelihood_WandV import computeLikelihoodParametrised_WandV
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

l_max = 15
k_max = 200 
r_max_true = 0.75
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0
n_max_ls = np.array([calc_n_max_l(l, k_max, r_max_true) for l in range(l_max + 1)])
R = 0.25    # Selection function scale length

omega_matter_true = 0.315
omega_matter_0 = 0.315      # observed

P_amp = 1

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
z_true, all_grids = generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max, P_para)

#%%
# Add the effect of the selection function
@jit(nopython=True)
def phiOfR0(r0):
    return np.exp(-r0*r0 / (2*R*R))


radii_true, all_observed_grids = multiplyFieldBySelectionFunction(radii_true, all_grids, phiOfR0)

#%%
#########################
### Observed Quantities ###

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]

# Perform the spherical Bessel transform to obtain the coefficients

### IMPORTANT NEED TO RECOMPUTE EVERYTIME YOU UPDATE THE POWER SPECTRUM ###

#f_lmn_0_saveFileName = "data_Ryan/data_F_lmn_0/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_P-amp_%.2f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R, P_amp)
#f_lmn_0_saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_P-amp_%.2f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R, P_amp)
f_lmn_0_saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_P-parametrised-2023-11-27-10-bins.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R)
if path.exists(f_lmn_0_saveFileName):
    f_lmn_0 = np.load(f_lmn_0_saveFileName)
else:
    print('calculating observed f_lmn coefficients ...')
    f_lmn_0 = calc_f_lmn_0(radii_fiducial, all_observed_grids, l_max, k_max, n_max)
    #f_lmn_0 = calc_f_lmn_0_numba(radii_fiducial, all_observed_grids, l_max, k_max, n_max)
    # Save coefficients to a file for future use
    np.save(f_lmn_0_saveFileName, f_lmn_0)
    print("Done! File saved to", f_lmn_0_saveFileName)

#########################

#%%

# Calculate W matrix for which omega_matter_true = omega_matter_0
W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_0, R)
if path.exists(W_saveFileName):
    W = np.load(W_saveFileName)
else:
    print("Computing W's for Ωₘ = %.4f." % omega_matter_true)
    r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter_true)
    W_integrand_numba = make_W_integrand_numba(phiOfR0)     #attention, NOT NUMBA ANYMORE
    W = calc_all_W_numba(l_max, k_max, r_max_0, r0_vals, r_vals, W_integrand_numba)
    # r0OfR = getInterpolatedR0ofR(omega_matter_0, omega_matter)
    # rOfR0 = getInterpolatedR0ofR(omega_matter, omega_matter_0)
    # W = calc_all_W(l_max, k_max, r_max_0, r0OfR, rOfR0, phiOfR0)
    np.save(W_saveFileName, W)
W = np.load(W_saveFileName)


# TODO
# Calculate roh_0 integral:
# roh_lmn_0 = calculate_roh_lmn_0(l_max, k_max, ...)

# # Calculate overall coefficients:

f_lmn_true = generate_f_lmn(l_max, r_max_true, k_max, P_para)
print(f_lmn_true.shape)
print(W.shape)

roh_lmn = np.zeros(f_lmn_true.shape, dtype=complex)

for l in range(l_max + 1):
    for m in range(l + 1):
        for n in range(n_max_ls[l] + 1):
            roh_lmn[l][m][n] = np.sum(W[l][n] * f_lmn_true[l][m])


# Generate observed field via a different way
radii_true = np.linspace(0, r_max_true, 1001)  
radii_observed_METHOD2, all_observed_grids_METHOD2 = generateGeneralField_given_delta_lmn(radii_true, omega_matter_true, r_max_true, l_max, k_max, P_para, roh_lmn)

# %%
print(np.size(all_observed_grids))
print(np.size(radii_true))
print(all_observed_grids[0].shape)
# %%
observed_grid = all_observed_grids[20]
observed_grid.plot()

observed_grid_METHOD2 = all_observed_grids_METHOD2[20]
observed_grid_METHOD2.plot()

# %%
