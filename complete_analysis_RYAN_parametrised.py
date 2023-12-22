import numpy as np
from numba import jit
from os import path

from generate_f_lmn import create_power_spectrum, P_parametrised
from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from distance_redshift_relation import *
from spherical_bessel_transform import calc_f_lmn_0_numba, calc_f_lmn_0
from calculate_W import calc_all_W_numba, make_W_integrand_numba, interpolate_W_values
from calculate_SN import calc_all_SN
from compute_likelihood import computeLikelihoodParametrised
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
#########################

#%%
#########################
### Likelihood Calculation ###

# Initialize
omega_matters = np.linspace(omega_matter_0 - 0.008, omega_matter_0 + 0.005, 14)
omega_matters_inference = np.linspace(omega_matter_0 - 0.007, omega_matter_0 + 0.005, 97)
P_amps = np.linspace(0.15, 1.05, 51)
#P_amps = np.linspace(0.95, 1.05, 51)
likelihoods = np.zeros((np.size(omega_matters_inference), np.size(P_amps)))

#%%
# Compute W's
Ws = []
for omega_matter in omega_matters:
    #W_saveFileName = "data_Ryan/data_W/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    if path.exists(W_saveFileName):
        W = np.load(W_saveFileName)
    else:
        print("Computing W's for Ωₘ = %.4f." % omega_matter)
        r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter)
        W_integrand_numba = make_W_integrand_numba(phiOfR0)     #attention, NOT NUMBA ANYMORE
        W = calc_all_W_numba(l_max, k_max, r_max_0, r0_vals, r_vals, W_integrand_numba)
        # r0OfR = getInterpolatedR0ofR(omega_matter_0, omega_matter)
        # rOfR0 = getInterpolatedR0ofR(omega_matter, omega_matter_0)
        # W = calc_all_W(l_max, k_max, r_max_0, r0OfR, rOfR0, phiOfR0)
        np.save(W_saveFileName, W)
    
    W = np.load(W_saveFileName)
    Ws.append(W)

# Compute shot noise
#SN_saveFileName = "data_Ryan/data_SN/SN_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
SN_saveFileName = "data/SN_no_tayl_exp_zeros_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter_0, l_max, k_max, r_max_0, R)
if path.exists(SN_saveFileName):
    SN = np.load(SN_saveFileName)
else:
    print("Computing SN for Ωₘ⁰ = %.4f." % omega_matter_0)
    SN = calc_all_SN(l_max, k_max, r_max_0, phiOfR0)
    np.save(SN_saveFileName, SN)
    
#%%

# Use MCMC to perform likelihood analysis

#MCMC requires us to be able to evaluate the likelihood for arbitrary values of Ωₘ, so interpolate W^l_nn' (Ωₘ)    
step = 0.00001
omega_matters_interp, Ws_interp = interpolate_W_values(l_max, n_max_ls, omega_matters, Ws, step=step)
omega_matter_min, omega_matter_max = omega_matters_interp[0], omega_matters_interp[-1]



# Define the probability function as likelihood * prior.
def log_prior(theta):
    omega_matter, *k_bin_heights = theta
    k_bin_heights = np.array(k_bin_heights)
    if omega_matter_min < omega_matter < omega_matter_max and np.all(0 < k_bin_heights) and np.all(k_bin_heights < 2):
        return 0.0
    return -np.inf

def log_likelihood(theta):
    omega_matter, *k_bin_heights = theta
    k_bin_heights = np.array(k_bin_heights)
    nbar = 1e9
    return computeLikelihoodParametrised(f_lmn_0, n_max_ls, r_max_0, omega_matter, k_bin_edges, k_bin_heights, omega_matters_interp, Ws_interp, SN, nbar)

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


steps = 10000
n_walkers = 32

# %%
# calculate Monte Carlo Markov Chain

pos = np.array([0.315, *k_bin_heights]) + 1e-4 * np.random.randn(n_walkers, 11)
# pos = np.array([0.315, *k_bin_heights]) + 1e-4 * np.random.randn(n_walkers, 3)
nwalkers, ndim = pos.shape      #nwalkers = number of walkers, ndim = number of dimensions in parameter space
print("number of walkers: ", nwalkers)

# filename = "data_Ryan/EMCEE_data/parametrised_power_spectrum_10_bins.h5"
# backend = emcee.backends.HDFBackend(filename)
# backend.reset(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability
)

# Burn in
pos, prob, state = sampler.run_mcmc(pos, 100) 
sampler.reset()

# Production run
sampler.run_mcmc(pos, steps, progress=True);
print("length of mcmc: ", steps)


# %%
# fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
fig, axes = plt.subplots(11, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
# labels = ["$\Omega_m$", *["$P_%d$" % (i+1) for i in range(2)]]
labels = ["$\Omega_m$", *["$P_%d$" % (i+1) for i in range(10)]]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number");
plt.show()

# get autocorrelation time
# tau = sampler.get_autocorr_time()
# print(tau)

# %%
# corner plot

# flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
flat_samples = sampler.get_chain(discard=100, flat=True)
print(flat_samples.shape)

fig = corner.corner(
    # flat_samples, labels=labels, truths=[0.315, *[0.35, 0.8]]
    # flat_samples, labels=labels, truths=[0.315, *[0.35, 0.8]]
    flat_samples, labels=labels, truths=[0.315, *[0.1, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]]
);

# %%

# Galman Rubin convergence diagnostic

def gelman_rubin(chain):
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    θb = np.mean(chain, axis=1)
    θbb = np.mean(θb, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1) * np.sum((θbb - θb)**2, axis=0)
    var_θ = (n - 1) / n * W + 1 / n * B
    R̂ = np.sqrt(var_θ / W)
    return R̂

chain = sampler.chain

print("Dim\tMean\t\tStd.Dev\t\tR̂")
for i in range(chain.shape[-1]):
    print("{0:3d}\t{1: 5.4f}\t\t{2:5.4f}\t\t{3:3.2f}".format(
            i, 
            chain.reshape(-1, chain.shape[-1]).mean(axis=0)[i],
            chain.reshape(-1, chain.shape[-1]).std(axis=0)[i],
            gelman_rubin(chain)[i]))

if np.mean(gelman_rubin(chain)) < 1.1:
    print(" ")
    print("Converged!")
else:
    print(" ")
    print("Not converged.")




# if __name__ == "__main__":
#     main_function()
# %%


