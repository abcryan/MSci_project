# %%
import numpy as np
from numba import jit
from os import path
from multiprocessing import Pool

from generate_f_lmn import create_power_spectrum, P_parametrised
from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from distance_redshift_relation import *
from spherical_bessel_transform import calc_f_lmn_0_numba
from calculate_W import calc_all_W_numba, make_W_integrand_numba, interpolate_W_values
from calculate_SN import calc_all_SN
from compute_likelihood import computeLikelihoodParametrised
from analyse_likelihood import plotContour, marginaliseOverP
from utils import calc_n_max_l
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros


l_max = 15
k_max = 200
r_max_true = 0.75
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0
n_max_ls = np.array([calc_n_max_l(l, k_max, r_max_true) for l in range(l_max + 1)])
R = 0.25 # Selection function scale length


c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%

# First, generate a true field

omega_matter_true = 0.315
radii_true = np.linspace(0, r_max_true, 1001)

true_z_of_r = getInterpolatedZofR(omega_matter_true)
z_true = true_z_of_r(radii_true)

# %%

k_bin_edges, k_bin_heights = create_power_spectrum(200, 10, np.array([0.1, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]))

k_vals = np.linspace(0, 275, 5000)
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

def P(k):
    return P_parametrised(k, k_bin_edges, k_bin_heights)


# %%

z_true, all_grids = generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max, P)

# %%

# Add the effect of the selection function

@jit(nopython=True)
def phiOfR0(r0):
    return np.exp(-r0*r0 / (2*R*R))

# %%

radii_true, all_observed_grids = multiplyFieldBySelectionFunction(radii_true, all_grids, phiOfR0)

# %%

# --------------- OBSERVED

omega_matter_0 = 0.315

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]

# %%

# Perform the spherical Bessel transform to obtain the coefficients

# Use numba to speed up the calculation

f_lmn_0 = calc_f_lmn_0_numba(radii_fiducial, all_observed_grids, l_max, k_max, n_max)


# Save coefficients to a file for future use
saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_P-parametrised-2023-11-27-10-bins.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R)
np.save(saveFileName, f_lmn_0)
print("Done! File saved to", saveFileName)

# %%

# Or, load f_lmn_0 from a file
# omega_matter_true = 0.315
# omega_matter_0 = 0.315
# l_max = 15
# k_max = 200
# r_max_true = 0.75
# R = 0.25
# P_amp = 1

# saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_P-amp_%.2f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R, P_amp)

saveFileName = "data/f_lmn_0_true-0.315_fiducial-0.315_l_max-15_k_max-200.00_r_max_true-0.750_R-0.250_P-parametrised-2023-11-27-10-bins.npy"

f_lmn_0 = np.load(saveFileName)


# %%

# Calculate likelihood

omega_matters = np.linspace(omega_matter_0 - 0.008, omega_matter_0 + 0.005, 14)

# %%

# Compute W's
for omega_matter in omega_matters:

    W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)

    if path.exists(W_saveFileName):
        W = np.load(W_saveFileName)
    else:
        print("Computing W's for Ωₘ = %.4f." % omega_matter)

        r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter)
        W_integrand_numba = make_W_integrand_numba(phiOfR0)
        W = calc_all_W_numba(l_max, k_max, r_max_0, r0_vals, r_vals, W_integrand_numba)

        # r0OfR = getInterpolatedR0ofR(omega_matter_0, omega_matter)
        # rOfR0 = getInterpolatedR0ofR(omega_matter, omega_matter_0)
        # W = calc_all_W(l_max, k_max, r_max_0, r0OfR, rOfR0, phiOfR0)
        np.save(W_saveFileName, W)


# Compute shot noise
SN_saveFileName = "data/SN_no_tayl_exp_zeros_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter_0, l_max, k_max, r_max_0, R)

if path.exists(SN_saveFileName):
    SN = np.load(SN_saveFileName)
else:
    print("Computing SN for Ωₘ⁰ = %.4f." % omega_matter_0)

    SN = calc_all_SN(l_max, k_max, r_max_0, phiOfR0)
    np.save(SN_saveFileName, SN)


# %%

# MCMC requires us to be able to evaluate the likelihood for arbitrary values of Ωₘ
# so interpolate W^l_nn' (Ωₘ)

Ws = []

for i, omega_matter in enumerate(omega_matters):
    W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    W = np.load(W_saveFileName)

    Ws.append(W)

step = 0.00001
omega_matters_interp, Ws_interp = interpolate_W_values(l_max, n_max_ls, omega_matters, Ws, step=step)

omega_matter_min, omega_matter_max = omega_matters_interp[0], omega_matters_interp[-1]


# %%

# Use MCMC to perform likelihood analysis

import emcee

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

# %%

with Pool() as pool:    
    pos = np.array([0.315, *k_bin_heights]) + 1e-4 * np.random.randn(32, 11)
    nwalkers, ndim = pos.shape

    # Set up the backend
    # Clear it in case the file already exists
    filename = "data/parametrised_power_spectrum_10_bins.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    # To use MH, use moves=emcee.moves.GaussianMove(0.00005)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, pool=pool, backend=backend
    )
    sampler.run_mcmc(pos, 10000, progress=True)


# %%

fig, axes = plt.subplots(11, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["$\Omega_m$", *["$P_{%d}$" % (i+1) for i in range(10)]]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

# %%

tau = sampler.get_autocorr_time()
print(tau)

# %%

import corner

# flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
flat_samples = sampler.get_chain(discard=100, flat=True)
# flat_samples = sampler.get_chain(flat=True)
print(flat_samples.shape)

fig = corner.corner(
    flat_samples, labels=labels, truths=[0.315, *[0.1, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]]
    # flat_samples, labels=labels, truths=[0.315, *[0.35, 0.85, 1, 0.85, 0.35]]
)


# %%

# Or, load samples from .h5 file
filename = "data/parametrised_power_spectrum_10_bins.h5"
reader = emcee.backends.HDFBackend(filename)

tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
# samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
samples = reader.get_chain(discard=10000, flat=True)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))

# %%

# # Violin plot

# powerSpectrumSamples = samples[:, 1:] # Ignore first column, which is omega_matter

# plt.figure(dpi=200, figsize=(8, 6))
# violin_plot = plt.violinplot(powerSpectrumSamples, showmeans=False)

# plt.plot([i+1 for i in range(10)], [0.1, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3], "o", label="Truth", c="k")

# k_vals = np.linspace(0, 201, 5000)
# P_vals = [P_parametrised(k, k_bin_edges, k_bin_heights) for k in k_vals]
# plt.plot(k_vals * (10/200) + 0.5, P_vals, c="#003a66", zorder=0, lw=1, alpha=0.3)


# color = "#20A285"
# color_fill = "#1071E5"

# for violin in violin_plot['bodies']:
#     violin.set_color(color_fill)
#     violin.set_alpha(0.6)

# # parts = ["cmins", "cmaxes", "cmeans", "cbars"]
# # for part in parts:
# #     violin_plot[part].set_color(color)

# handles, labels = plt.gca().get_legend_handles_labels()
# handles.insert(0, violin_plot["bodies"][0])
# labels.insert(0, "Samples")
# plt.legend(handles, labels)
# plt.xticks([i+1 for i in range(10)], [i+1 for i in range(10)])
# plt.xlabel("$i$")
# plt.ylabel("$P_{i}$")
# plt.ylim(0)
# plt.xlim(0.5)
# # plt.savefig("thesis/plots/violin_plot.svg")
# plt.show()

# %%

# import pymc
# # from scipy.integrate import simpson
# # simpson(counts, bin_mids)

# edgecolor = "#1071E5"
# facecolor = "#1071E5"
# counts, bins, bars = plt.hist(np.transpose(samples)[0], bins=50, density=True, color=facecolor, edgecolor=edgecolor, linewidth=0.8, alpha=0.4)
# bin_mids = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

# hpdi = pymc.stats.hdi(np.transpose(samples)[0], hdi_prob=0.68)
# plt.vlines(hpdi, 0, np.max(counts), color="k", linestyle="dashed", linewidth=1, label="68% HPDI")

# plt.xlabel("$\Omega_m$")
# plt.title("$p(\Omega_m)$")
# # plt.savefig("thesis/plots/omega_matter_posterior_10_bins.svg")
# plt.show()

# peak_omega_matter = bin_mids[np.argmax(counts)]
# print("Ωₘ = %.5f +%.5f -%.5f" % (peak_omega_matter, hpdi[1] - peak_omega_matter, peak_omega_matter - hpdi[0]))

# %%
