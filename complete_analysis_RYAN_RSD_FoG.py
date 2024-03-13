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

l_max = 15 #70 # 40 # 25 # 15
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

#%%
#selection function
@jit(nopython=True)
def phiOfR0(r0):
    return np.exp(-r0*r0 / (2*R*R))

#########################
### Observed Quantities ###

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]
print("r_max_0:", r_max_0)

r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter_true)
r0_max = np.interp(r_max_0, r_vals, r0_vals)
print("r0_max:", r0_max)

#%%

# Old way of computing the observed field
# radii_true, all_observed_grids = multiplyFieldBySelectionFunction(radii_true, all_grids, phiOfR0)


# For the NEW WAY we first need to calculate the W, V and F matrices but only where omega_true = omega_0, to get the TRUE field coefficients: 

# Calculate W, V  and F matrix for which omega_matter_true = omega_matter_0
W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_0, R)
if path.exists(W_saveFileName):
    W = np.load(W_saveFileName)
else:
    print("Computing W's for Ωₘ = %.4f." % omega_matter_true)
    r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter_true)
    W_integrand_numba = make_W_integrand_numba(phiOfR0)     
    W = calc_all_W_numba(l_max, k_max, r_max_0, r0_vals, r_vals, W_integrand_numba)
    np.save(W_saveFileName, W)

V_saveFileName = "data/V_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_0, R)
if path.exists(V_saveFileName):
    V = np.load(V_saveFileName)
else:
    print("Computing V's for Ωₘ = %.4f." % omega_matter_true)
    r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter_true)
    V_integrand_numba = make_V_integrand_numba(phiOfR0)     
    V = calc_all_V_numba(l_max, k_max, r_max_0, r0_vals, r_vals, V_integrand_numba)
    np.save(V_saveFileName, V)

F_saveFileName = "data/F_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f_sigma-%.4f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_0, R, sigma)
if path.exists(F_saveFileName):
    F = np.load(F_saveFileName)
else:
    print("Computing F's for Ωₘ = %.4f." % omega_matter_true)
    r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter_true) # not needed I think
    F = calculate_all_F(l_max, k_max, r_max_0, r0_vals, r_vals, sigma)
    np.save(F_saveFileName, F)

W_true = np.load(W_saveFileName)
V_true = np.load(V_saveFileName)
F_true = np.load(F_saveFileName)

# Calculate new coefficients of the TRUE field:

# roh_lmn = np.zeros(f_lmn_true.shape, dtype=complex)
# for l in range(l_max + 1):
#     for m in range(l + 1):
#         for n in range(n_max_ls[l] + 1):
#             roh_lmn[l][m][n] = np.sum((W_true[l][n] + beta_true * V_true[l][n]) * f_lmn_true[l][m])

# dm = np.zeros(f_lmn_true.shape, dtype=complex)
# for l in range(l_max + 1):
#     for m in range(l + 1):
#         for n in range(n_max_ls[l] + 1):
#             sum = 0
#             for n_prime in range(n_max_ls[l] + 1):
#                 sum +=  (W_true[l][n][n_prime] + beta_true * V_true[l][n][n_prime]) * f_lmn_true[l][m][n_prime]
                
#             dm[l][m][n] = sum

# print("difference is: ", np.sum(roh_lmn - dm))


# With F matrix new:
roh_lmn = np.zeros(f_lmn_true.shape, dtype=complex)
for l in range(l_max + 1):
    for m in range(l + 1):
        for n in range(n_max_ls[l] + 1):
            sum_n_prime = 0
            for n_prime in range(n_max_ls[l] + 1):
                sum_n_prime_prime = 0
                for n_prime_prime in range(n_max_ls[l] + 1):
                    sum_n_prime_prime +=  (W_true[l][n_prime][n_prime_prime] + beta_true * V_true[l][n_prime][n_prime_prime]) * f_lmn_true[l][m][n_prime_prime]
                sum_n_prime += sum_n_prime_prime * F_true[l][n][n_prime]
            roh_lmn[l][m][n] = sum_n_prime

# print("difference is: ", np.sum(roh_lmn - dm))

# %%
# Generate observed field via a different way
radii_true = np.linspace(0, r_max_true, 1001)  
radii_observed_METHOD2, all_observed_grids_METHOD2 = generateGeneralField_given_delta_lmn(radii_true, omega_matter_true, r_max_true, l_max, k_max, P_para, roh_lmn)

#%%

# Perform the spherical Bessel transform to obtain the coefficients
# IMPORTANT NEED TO RECOMPUTE EVERYTIME YOU UPDATE THE POWER SPECTRUM ###

f_lmn_0_saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_P-parametrised-2023-11-27-10-bins_%.3f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R, beta_true)
if path.exists(f_lmn_0_saveFileName):
    f_lmn_0 = np.load(f_lmn_0_saveFileName)
else:
    print('calculating observed f_lmn coefficients ...')
    f_lmn_0 = calc_f_lmn_0(radii_fiducial, all_observed_grids_METHOD2, l_max, k_max, n_max)
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
betas = np.linspace(0.0, 0.7, 51)           # RSD parameter
# P_amps = np.linspace(0.95, 1.05, 51)
# likelihoods = np.zeros((np.size(omega_matters_inference), np.size(P_amps)))
likelihoods = np.zeros((np.size(omega_matters_inference), np.size(P_amps), np.size(betas)))

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
# Compute V's
Vs = []
for omega_matter in omega_matters:
    #V_saveFileName = "data_Ryan/data_V/V_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    V_saveFileName = "data/V_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    if path.exists(V_saveFileName):
        V = np.load(V_saveFileName)
    else:
        print("Computing V's for Ωₘ = %.4f." % omega_matter)
        r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter)
        V_integrand_numba = make_V_integrand_numba(phiOfR0)     #attention, NOT NUMBA ANYMORE
        V = calc_all_V_numba(l_max, k_max, r_max_0, r0_vals, r_vals, V_integrand_numba)
        # r0OfR = getInterpolatedR0ofR(omega_matter_0, omega_matter)
        # rOfR0 = getInterpolatedR0ofR(omega_matter, omega_matter_0)
        # V = calc_all_V(l_max, k_max, r_max_0, r0OfR, rOfR0, phiOfR0)
        np.save(V_saveFileName, V)
    
    V = np.load(V_saveFileName)
    Vs.append(V)

#%%
# Compute F's
Fs = []
for omega_matter in omega_matters:
    # F_saveFileName = "data/F_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f_sigma-%.4f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R, sigma)
    # if path.exists(F_saveFileName):
    #     F = np.load(F_saveFileName)
    # else:
    #     print("Computing F's for Ωₘ = %.4f." % omega_matter)
    #     r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter)
    #     F = calculate_all_F(l_max, k_max, r_max_0, r0_vals, r_vals, sigma)

    #     np.save(F_saveFileName, F)
    
    # F = np.load(F_saveFileName)
    F = np.load(F_saveFileName) # just copy all the F's from the previous cell
    Fs.append(F)
#%%

# Use MCMC to perform likelihood analysis

#MCMC requires us to be able to evaluate the likelihood for arbitrary values of Ωₘ, so interpolate W^l_nn' (Ωₘ)    
step = 0.00001
# omega_matters_interp, Ws_interp = interpolate_W_values(l_max, n_max_ls, omega_matters, Ws, step=step)
# omega_matters_interp, Ws_interp, Vs_interp = interpolate_WandV_values(l_max, n_max_ls, omega_matters, Ws, Vs, step=step)
omega_matters_interp, Ws_interp, Vs_interp, Fs_interp = interpolate_WVandF_values(l_max, n_max_ls, omega_matters, Ws, Vs, Fs, step=step)
omega_matter_min, omega_matter_max = omega_matters_interp[0], omega_matters_interp[-1]

beta_min = 0.0
beta_max = 0.7

# I don't need interpolated F values!
# USE the usual F matrix. 

#%%


# Define the probability function as likelihood * prior.
def log_prior(theta):
    omega_matter, beta, *k_bin_heights = theta
    k_bin_heights = np.array(k_bin_heights)
    if omega_matter_min < omega_matter < omega_matter_max and np.all(0 < k_bin_heights) and np.all(k_bin_heights < 2) and beta_min < beta < beta_max:
        return 0.0
    return -np.inf

#TODOOO change this to the new likelihood and update the compute_likekihood_WandVfile
def log_likelihood(theta):
    omega_matter, beta, *k_bin_heights = theta
    k_bin_heights = np.array(k_bin_heights)
    nbar = 1e9
    return computeLikelihoodParametrised_WVandF(f_lmn_0, n_max_ls, r_max_0, omega_matter, beta, k_bin_edges, k_bin_heights, omega_matters_interp, Ws_interp, Vs_interp, F, SN, nbar)
########

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


# %%
# calculate Monte Carlo Markov Chain

steps = 4000
n_walkers = 32
burnin = 200

pos = np.array([0.315, 0.5, *k_bin_heights]) + 1e-4 * np.random.randn(n_walkers, 12)
nwalkers, ndim = pos.shape      #nwalkers = number of walkers, ndim = number of dimensions in parameter space
print("number of walkers: ", nwalkers)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability
)
# Burn in
print("Burn-in:")
pos, prob, state = sampler.run_mcmc(pos, burnin, progress=True) 
sampler.reset()

# Production run
print("Production run:")
sampler.run_mcmc(pos, steps, progress=True);
print("length of mcmc: ", steps)


# %%
flat_samples = sampler.get_chain(discard=burnin, flat=True)
fig, axes = plt.subplots(12, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["$\Omega_{m}$", "$\\beta$", *["$P_%d$" % (i+1) for i in range(10)]]
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

# beta = omega_0**0.6 / b, where b is the galaxy bias parameter and is estimated to be within the range 1.0 and 1.5
# best might be to use b = 1.23 which gives beta = 0.4

# flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
flat_samples = sampler.get_chain(discard=burnin, flat=True)
print(flat_samples.shape)

fig = corner.corner(

    # flat_samples, labels=labels, truths=[0.315, *[0.35, 0.8]]
    # flat_samples, labels=labels, truths=[0.315, *[0.35, 0.8]]
    # flat_samples, labels=labels, truths=[0.315, *[0.1, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]]
    flat_samples, 
    title_fmt='.5f',
    bins=30,
    show_titles=True,
    labels=labels, 
    truths=[0.315, 0.5, *[0.1, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]],
    plot_density=True,
    plot_datapoints=True,
    fill_contours=False,
    smooth=True,
    levels=(0.6827, 0.90, 0.9545),
    quantiles=[0.16, 0.5, 0.84],
    title_kwargs={"fontsize": 10},
    truth_color='cornflowerblue',
        
);
fig.savefig("corner_plot.png", dpi=1000)

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



# %%
print(flat_samples.shape)
fg = plt.figure(figsize=(6, 6))
fig = corner.corner(
    # flat_samples, labels=labels, truths=[0.315, *[0.35, 0.8]]
    # flat_samples, labels=labels, truths=[0.315, *[0.35, 0.8]]
    # flat_samples, labels=labels, truths=[0.315, *[0.1, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]]
    flat_samples[:,:2], 
    title_fmt='.5f',
    bins=30,
    show_titles=True,
    labels=labels[:2], 
    truths=[0.315, 0.5],
    plot_density=True,
    plot_datapoints=True,
    fill_contours=False,
    smooth=True,
    levels=(0.6827, 0.90, 0.9545),
    quantiles=[0.16, 0.5, 0.84],
    title_kwargs={"fontsize": 10},
    truth_color='cornflowerblue',
    fig=fg,
    titles = ["$\Omega_{m}^{median}$", "$\\beta^{median}$"]
            );

for ax in fig.get_axes():
    for line in ax.get_lines():
        line.set_linewidth(1)  # Set to desired thickness

line1 = plt.Line2D([0], [0], color='cornflowerblue', linewidth=0.8, linestyle='-', marker='s')
line2 = plt.Line2D([0], [0], color='black', linewidth=1, linestyle='--')
# You can adjust the line colors and styles to match your plot's elements

# Choose an appropriate axis for the legend
# For a corner plot, the top right axis is usually empty, so we can use it for the legend
fig.legend([line1, line2], 
           ['$\Omega_{m}^{True}$ = %.3f \n$\\beta^{True}$ = %.1f' % (omega_matter_true, beta_true), 
            '1-$\sigma$ interval \naround median'], 
            loc=(0.63,0.75),
            prop={'size': 10},
            # shadow=True,
            )

ax = fig.axes[2]
ax.annotate('68%', xy=(0.5, 0.305), xycoords='axes fraction', fontsize=6)
ax.annotate('90%', xy=(0.49, 0.22), xycoords='axes fraction', fontsize=6)
ax.annotate('95%', xy=(0.48, 0.17), xycoords='axes fraction', fontsize=6)


# fig.gca().annotate(
#     "",
#     xy=(1.0, 1.0),
#     xycoords="figure fraction",
#     xytext=(-20, -10),
#     textcoords="offset points",
#     ha="right",
#     va="top",
# )
# axes = np.array(fig.axes).reshape((2, 2))
# for a in axes[np.triu_indices(2)]:
#     a.remove()
fig.savefig("demo.png", dpi=1000)

#  %%
if hasattr(sampler, 'get_log_prob'):
    log_prob = sampler.get_log_prob(flat=True)
else:
    # Compute log likelihood for each sample (this could be computationally expensive)
    log_prob = np.array([log_likelihood(params) for params in samples])

# Find the index of the sample with the maximum log likelihood
max_log_prob_index = np.argmax(log_prob)

# Extract the parameter values and the maximum log likelihood
max_likelihood_parameters = flat_samples[max_log_prob_index]
max_log_likelihood_value = log_prob[max_log_prob_index]

print("Maximum Log Likelihood Parameters:", max_likelihood_parameters)
print("Maximum Log Likelihood Value:", max_log_likelihood_value)

# def getDeltaLnL(likelihoods):
#     # Subtract the maximum
#     maximum = np.max(likelihoods)
#     delta_lnL = likelihoods - maximum
#     return delta_lnL

# # Selecting the parameters for the contour plot
# # Here we use the first two parameters as an example
# x_samples = flat_samples[:, 0]
# y_samples = flat_samples[:, 1]

# # Creating a 2D histogram of samples
# hist, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=30, density=True)

# hist = getDeltaLnL(hist)
# # Convert bin edges to centers
# xcenters = 0.5 * (xedges[1:] + xedges[:-1])
# ycenters = 0.5 * (yedges[1:] + yedges[:-1])

# # Plotting the contour
# plt.contourf(xcenters, ycenters, hist, cmap='viridis')
# plt.colorbar(label='Posterior density')
# plt.xlabel('Parameter 1')
# plt.ylabel('Parameter 2')
# plt.title('Posterior distribution contour plot')
# plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.ndimage

from scipy.ndimage.filters import gaussian_filter

def sigma_to_percentile(sigma):
    """Convert sigma to percentile of the Gaussian distribution."""
    return norm.cdf(sigma) - norm.cdf(-sigma)

def getDeltaLnL(likelihoods):
    # Subtract the maximum
    maximum = np.max(likelihoods)
    delta_lnL = likelihoods - maximum
    return delta_lnL


def plot_2d_likelihood(samples, x_index, y_index, labels, truths=None, sigmas=[0.5, 1, 2, 3, 4]):
    """
    Plots a 2D likelihood contour plot for two parameters.
    
    :param samples: Flat MCMC samples
    :param x_index: Index of the parameter to plot on the x-axis
    :param y_index: Index of the parameter to plot on the y-axis
    :param labels: List of parameter names for labeling the axes
    :param truths: True values of the parameters for plotting
    :param sigmas: List of sigma levels for the contours
    """
    # Define the percentile levels for the contours
    levels = [sigma_to_percentile(s) for s in sigmas]
    
    # Extract the samples for the parameters we are plotting
    x = samples[:, x_index]
    y = samples[:, y_index]

    # Estimate the 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=10, density=True)
    

    hist = getDeltaLnL(hist)
    # Convert histogram to contour levels
    hist_sorted = np.sort(hist.flatten())[::-1]
    cum_hist = np.cumsum(hist_sorted)
    cum_hist /= cum_hist[-1]
    contour_levels = [hist_sorted[np.argmax(cum_hist > level)] for level in levels]
    print(contour_levels)
    
    # Plot the 2D histogram as contours
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    # plt.contour(X.T, Y.T, hist,  cmap = 'viridis', linewidths=1.5)
    # plt.colorbar()

    X,Y,hist = scipy.ndimage.zoom(X.T,Y.T,hist.T, 3)

    # Basic contour plot
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, hist,  cmap = 'viridis', linewidths=1.5)
    ax.clabel(CS, CS.levels, inline=True,  fontsize=10)
    
    # Plot the truth values if provided
    if truths is not None:
        ax.axvline(truths[x_index], color='r', linestyle='--')
        ax.axhline(truths[y_index], color='r', linestyle='--')
    
    
    # Label the plot
    ax.xlabel(labels[x_index])
    ax.ylabel(labels[y_index])
    ax.title('2D Likelihood Contour Plot')
    ax.show()

# Example usage:
labels = ['Ωm', 'β'] # Replace with your parameter names
plot_2d_likelihood(flat_samples, x_index=0, y_index=1, labels=labels, truths=[0.315, 0.5])

# %%