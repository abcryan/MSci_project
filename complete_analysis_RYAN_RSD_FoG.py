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
from calculate_GandH import calculate_all_GorH
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

l_max = 100 #70 # 40 # 25 # 15
k_max = 200 
r_max_true = 0.75
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0
n_max_ls = np.array([calc_n_max_l(l, k_max, r_max_true) for l in range(l_max + 1)])
R = 0.25    # Selection function scale length
sigma = 0.001 # velocity dispersion

omega_matter_true = 0.315
omega_matter_0 = 0.310      # fiducial

P_amp = 1

# RSD parameter
b_true = 1.0   # galaxy bias parameter, 1.0 <= b <= 1.5 usually in RSD Surveys
beta_true = omega_matter_true**0.6 / b_true
# beta_true = 0.0

#########################
#########################

# More sophisticated power specturm
k_bin_edges, k_bin_heights = create_power_spectrum(k_max, 10, np.array([0.15, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]))
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

#selection function
@jit(nopython=True)
def phiOfR0(r0):
    return np.exp(-r0*r0 / (2*R*R))


# %%
# Calculate c_ln coefficients of true SBT with infinite r
c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")

# Calculate spherical Bessel zeros
sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

# %%
# Generate true field
radii_true = np.linspace(0, r_max_true, 1001)  
z_true, all_grids, f_lmn_true = generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max, P_para)

#%%
### Generate GALAXY SURVEY DATA = Observed field ###

r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
radii_fiducial = r_of_z_fiducial(z_true)
r_max_0 = radii_fiducial[-1]

# NEW WAY OF COMPUTING OBSERVED FIELD COEFFICIENTS
# For the NEW WAY we first need to calculate the W and V matrices but only where omega = omega_matter_0

# Calculate W, V and F matrix for which omega = omega_matter_0
W_saveFileName = "data/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_0, R)
if path.exists(W_saveFileName):
    W = np.load(W_saveFileName)
else:
    print("Computing W's for Ωₘ = %.4f." % omega_matter_0)
    r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter_true)
    W_integrand_numba = make_W_integrand_numba(phiOfR0)     
    W = calc_all_W_numba(l_max, k_max, r_max_0, r0_vals, r_vals, W_integrand_numba)
    np.save(W_saveFileName, W)

V_saveFileName = "data/V_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_0, R)
if path.exists(V_saveFileName):
    V = np.load(V_saveFileName)
else:
    print("Computing V's for Ωₘ = %.4f." % omega_matter_0)
    r0_vals, r_vals = getInterpolatedR0ofRValues(omega_matter_0, omega_matter_true)
    V_integrand_numba = make_V_integrand_numba(phiOfR0)     
    V = calc_all_V_numba(l_max, k_max, r_max_0, r0_vals, r_vals, V_integrand_numba)
    np.save(V_saveFileName, V)

F_saveFileName = "data/F_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-0.31500_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f_sigma-%.4f.npy" % (omega_matter_true, l_max, k_max, r_max_true, R, sigma)
if path.exists(F_saveFileName):
    F = np.load(F_saveFileName)
else:
    print("Computing F matrix")
    F = calculate_all_F(l_max, k_max, r_max_0, sigma)
    np.save(F_saveFileName, F)


W_observed = np.load(W_saveFileName)
V_observed = np.load(V_saveFileName)
F_matrix = np.load(F_saveFileName)


# Calculate new G AND H MATRIX: 

G_saveFileName = "data/G_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f_sigma-%.4f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_0, R, sigma)
if path.exists(G_saveFileName):
    G = np.load(G_saveFileName)
else:
    print("Computing G's for Ωₘ = %.4f." % omega_matter_true)
    G = calculate_all_GorH(l_max, k_max, r_max_0, F_matrix, W_observed)
    np.save(G_saveFileName, G)

H_saveFileName = "data/H_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f_sigma-%.4f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_0, R, sigma)
if path.exists(H_saveFileName):
    H = np.load(H_saveFileName)
else:
    print("Computing H's for Ωₘ = %.4f." % omega_matter_true)
    H = calculate_all_GorH(l_max, k_max, r_max_0, F_matrix, V_observed)
    np.save(H_saveFileName, H)

G_observed = np.load(G_saveFileName)
H_observed = np.load(H_saveFileName)

# Calculate observed field coefficients: n_lmn
n_lmn = np.zeros(f_lmn_true.shape, dtype=complex)
for l in range(l_max + 1):
    for m in range(l + 1):
        for n in range(n_max_ls[l] + 1):
            n_lmn[l][m][n] = np.sum((G_observed[l][n] + beta_true * H_observed[l][n]) * f_lmn_true[l][m])

## %%
# Old way of computing the observed field
# radii_true, all_observed_grids = multiplyFieldBySelectionFunction(radii_true, all_grids, phiOfR0)

# # Generate observed field via a different way (to check if coefficients were generated correctly with the NEW METHOD)
# radii_true = np.linspace(0, r_max_true, 1001)  
# radii_observed_METHOD2, all_observed_grids_METHOD2 = generateGeneralField_given_delta_lmn(radii_true, omega_matter_true, r_max_true, l_max, k_max, P_para, n_lmn)

# # Perform the spherical Bessel transform to obtain the coefficients
# f_lmn_0_saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_P-parametrised-2023-11-27-10-bins_%.3f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R, beta_true)
# if path.exists(f_lmn_0_saveFileName):
#     f_lmn_0 = np.load(f_lmn_0_saveFileName)
# else:
#     print('calculating observed f_lmn coefficients ...')
#     f_lmn_0 = calc_f_lmn_0(radii_fiducial, all_observed_grids_METHOD2, l_max, k_max, n_max)
#     # Save coefficients to a file for future use
#     np.save(f_lmn_0_saveFileName, f_lmn_0)
#     print("Done! File saved to", f_lmn_0_saveFileName)

#########################
#########################

#%%
#########################
### Likelihood Calculation ###

print(F_saveFileName)
# print(W_observed[:,20,29])
# Initialize
# omega_matters = np.linspace(omega_matter_0 - 0.008, omega_matter_0 + 0.005, 14)
# omega_matters = np.linspace(omega_matter_0 - 0.006, omega_matter_0 + 0.001, 8)
# omega_matters = np.linspace(omega_matter_0, omega_matter_0 + 0.008, 9)
omega_matters = np.linspace(omega_matter_0 - 0.010, omega_matter_0 + 0.010, 21)
# omega_matters = np.linspace(omega_matter_0 - 0.010, omega_matter_0 + 0.003, 14)
# omega_matters = np.linspace(omega_matter_0 - 0.015, omega_matter_0 + 0.05, 21)

# omega_matters = np.linspace(omega_matter_0 - 0.012, omega_matter_0 + 0.012, 18)
# P_amps = np.linspace(0.05, 1.05, 51)
# P_amps = np.linspace(0.95, 1.05, 51)
# betas = np.linspace(0.0, 0.7, 51)           # RSD parameter

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

# Compute G's and H's

Gs = []
Hs = []
i=0
for omega_matter in omega_matters:
    G_saveFileName = "data/G_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    if path.exists(G_saveFileName):
        G = np.load(G_saveFileName)
    else:
        W_matrix = Ws[i]
        G = calculate_all_GorH(l_max, k_max, r_max_0, F_matrix, W_matrix)
        np.save(G_saveFileName, G)
    
    G = np.load(G_saveFileName)
    Gs.append(G)
    i+=1

j=0
for omega_matter in omega_matters:
    H_saveFileName = "data/H_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    if path.exists(H_saveFileName):
        H = np.load(H_saveFileName)
    else:
        V_matrix = Vs[j]
        H = calculate_all_GorH(l_max, k_max, r_max_0, F_matrix, V_matrix)
        np.save(H_saveFileName, H)
    
    H = np.load(H_saveFileName)
    Hs.append(H)
    j+=1


#%%
# Use MCMC to perform likelihood analysis

#MCMC requires us to be able to evaluate the likelihood for arbitrary values of Ωₘ, so interpolate W^l_nn' (Ωₘ)    
step = 0.00001
# omega_matters_interp, Ws_interp = interpolate_W_values(l_max, n_max_ls, omega_matters, Ws, step=step)
omega_matters_interp, Gs_interp, Hs_interp = interpolate_WandV_values(l_max, n_max_ls, omega_matters, Gs, Hs, step=step)
omega_matter_min, omega_matter_max = omega_matters_interp[0], omega_matters_interp[-1]

beta_min = 0.0
beta_max = 0.7

# No need interpolated F values!
# USE the usual F matrix. 

# Define the probability function as likelihood * prior.
def log_prior(theta):
    omega_matter, beta, *k_bin_heights = theta
    k_bin_heights = np.array(k_bin_heights)
    if omega_matter_min < omega_matter < omega_matter_max and np.all(0 < k_bin_heights) and np.all(k_bin_heights < 2) and beta_min < beta < beta_max:
        return 0.0
    return -np.inf

def log_likelihood(theta):
    omega_matter, beta, *k_bin_heights = theta
    k_bin_heights = np.array(k_bin_heights)
    nbar = 1e9
    return computeLikelihoodParametrised_WandV(n_lmn, n_max_ls, r_max_0, omega_matter, beta, k_bin_edges, k_bin_heights, omega_matters_interp, Gs_interp, Hs_interp, SN, nbar)

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
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
print(flat_samples.shape)
# flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
flat_samples = sampler.get_chain(discard=burnin, flat=True)
print(flat_samples.shape)

fg = plt.figure(figsize=(24, 24))
fig = corner.corner(

    # flat_samples, labels=labels, truths=[0.315, *[0.35, 0.8]]
    # flat_samples, labels=labels, truths=[0.315, *[0.35, 0.8]]
    # flat_samples, labels=labels, truths=[0.315, *[0.1, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]]
    flat_samples, 
    title_fmt='.4f',
    bins=30,
    show_titles=True,
    labels=labels, 
    truths=[0.315, 0.5, *[0.15, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]],
    plot_density=True,
    plot_datapoints=True,
    fill_contours=False,
    smooth=True,
    fig=fg,
    levels=(0.6827, 0.90, 0.9545),
    quantiles=[0.16, 0.5, 0.84],
    title_kwargs={"fontsize": 18},
    label_kwargs={"fontsize": 18},
    truth_color='orangered',
        
);
i=0
for ax in fig.get_axes():
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.set_xlabel(labels[i],fontsize=25)
    for line in ax.get_lines():
        line.set_linewidth(1.5)  # Set to desired thickness
    i+=1

line1 = plt.Line2D([0], [0], color='orangered', linewidth=1.3, linestyle='-', marker='s')
line2 = plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--')
line3 = plt.Line2D([0], [0], color='black', linewidth=2, linestyle='-')
# You can adjust the line colors and styles to match your plot's elements

# Choose an appropriate axis for the legend
# For a corner plot, the top right axis is usually empty, so we can use it for the legend
fig.legend([line1, line2, line3], 
           ['Truth Value', 
            '$\pm 1\sigma$ Interval around Median \n (1-D Distributions)',
            'Contour Levels: 68$\mathbf{\%}$, 90$\mathbf{\%}$, 95$\mathbf{\%}$ \n (2-D Distributions)'], 
            loc=(0.63,0.75),
            prop={'size': 30},
            # shadow=True,
            )

fig.savefig("corner_plot_FoG.png", dpi=200)

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
    title_kwargs={"fontsize": 18},
    label_kwargs={"fontsize": 20},
    truth_color='dodgerblue',
    fig=fg,
    titles = ["$\Omega_{m}$", "$\\beta$"]
            );


for ax in fig.get_axes():
    ax.tick_params(axis='both', which='major', labelsize=15)
    for line in ax.get_lines():
        line.set_linewidth(1.5)  # Set to desired thickness

line1 = plt.Line2D([0], [0], color='dodgerblue', linewidth=1.5, linestyle='-', marker='s')
line2 = plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--')
line3 = plt.Line2D([0], [0], color='white', linewidth=2, linestyle='-')
# You can adjust the line colors and styles to match your plot's elements

# Choose an appropriate axis for the legend
# For a corner plot, the top right axis is usually empty, so we can use it for the legend
fig.legend([line1], 
           ['Truth' 
            # '$\pm 1\sigma$ Interval around Median \n (1D Distributions)',
            # 'Contour Levels: 68$\mathbf{\%}$, 90$\mathbf{\%}$, 95$\mathbf{\%}$ \n (2D Distributions)'
            ], 
            loc=(0.62,0.83),
            prop={'size': 18},
            # shadow=True,
            )
ax0 = fig.axes[0]
ax0.annotate('$\Omega_{m}^0 = 0.320$ ', xy=(1.25, 0.5505), xycoords='axes fraction', color='black', fontsize=20)

ax = fig.axes[2]
ax.annotate('$68\%$', xy=(0.4, 0.3705), xycoords='axes fraction', fontsize=8)
ax.annotate('$90\%$', xy=(0.4, 0.282), xycoords='axes fraction', fontsize=8)
ax.annotate('$95\%$', xy=(0.38, 0.225), xycoords='axes fraction', fontsize=8)


fig.savefig("FOG_320.png", dpi=200)



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

from matplotlib import ticker

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# import seaborn as sns

# Violin plot

powerSpectrumSamples = sampler.get_chain(discard=burnin, flat=True)[:, 2:] # Ignore first two columns, which is omega_matter and beta

plt.figure(dpi=200, figsize=(8,6))
violin_plot = plt.violinplot(powerSpectrumSamples, showmeans=False, widths=0.7)

plt.plot([i+1 for i in range(10)], [0.15, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3], "o", label="Truth", c="k",markersize=3)
# plt.grid(color='#EEEEEE', linestyle=':', linewidth=1.2)

# plot the background power spectrum
k_vals = np.linspace(0, 201, 5000)
P_vals = [P_parametrised(k, k_bin_edges, k_bin_heights) for k in k_vals]
plt.plot(k_vals * (10/200) + 0.5, P_vals, c="grey", zorder=0, lw=1, alpha=0.8)


color = "dodgerblue"
color_fill = "dodgerblue"

for violin in violin_plot['bodies']:
    violin.set_color(color_fill)
    violin.set_alpha(0.5)
    violin.set_edgecolor("dodgerblue")

parts = ["cmins", "cmaxes", "cbars"]
for part in parts:
    violin_plot[part].set_color(color)

handles, labels = plt.gca().get_legend_handles_labels()
handles.insert(0, violin_plot["bodies"][0])
labels.insert(0, "Samples")
plt.legend(handles, labels, loc="upper left", fontsize=16)
plt.xticks([i+1 for i in range(10)], [i+1 for i in range(10)], fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel(r'Bin $i$', fontsize=18)
plt.ylabel("$P_{i}$", fontsize=18)
plt.ylim(0)
plt.xlim(0.5)
plt.savefig("Plots/violin_plot_FOG_320.png")
plt.show()

# %%
