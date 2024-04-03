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
omega_matters = np.linspace(omega_matter_0 - 0.010, omega_matter_0 + 0.010, 21)
# omega_matters = np.linspace(omega_matter_0 - 0.010, omega_matter_0 + 0.009, 20)

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
omega_matters_interp_WV, Ws_interp, Vs_interp = interpolate_WandV_values(l_max, n_max_ls, omega_matters, Ws, Vs, step=step)
omega_matters_interp, Gs_interp, Hs_interp = interpolate_WandV_values(l_max, n_max_ls, omega_matters, Gs, Hs, step=step)
omega_matter_min, omega_matter_max = omega_matters_interp[0], omega_matters_interp[-1]

#%%
# print(np.shape(Gs_interp))
# print(np.shape(omega_matters_interp))
# print(np.shape(Gs))
# print(np.shape(omega_matters))
Hs = np.array(Hs)
Gs = np.array(Gs)
Ws = np.array(Ws)
Vs = np.array(Vs)

#%%
from matplotlib import ticker

import matplotlib as mpl

# eng = 'lua' # issue with lua only   <<<<<<<<
# # eng = 'pdf' # no issue here
# # eng = 'xe' # update: issue here as well

# mpl.use('pgf')
# mpl.rc('font', family='serif')
# mpl.rcParams.update({
#         "pgf.rcfonts"  : False,
#         "pgf.texsystem": eng + "latex",
#         "pgf.preamble" : '\\usepackage[utf8x]{inputenc}\\usepackage[light]{kpfonts}',
# })

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


def plotFField(l, n, n_prime, omega_matters, Hs, omega_matters_interp,Hs_interp, title="", saveFileName=None):

    fig, ax = plt.subplots(figsize=(6.3, 5.0), )

    plt.subplots_adjust(bottom=0.2, left=0.2)

    H_l_n_n_prime = Hs[:,l,n,n_prime]
    H_l_n_n_prime_interp = Hs_interp[l][n][n_prime]
    print(np.shape(H_l_n_n_prime))

    
    ax.plot(omega_matters_interp, H_l_n_n_prime_interp, label="Interpolated", color='darkorange', zorder=1, lw=2.5)
    ax.scatter(omega_matters, H_l_n_n_prime, label="Evaluated" , zorder=2, s=80, facecolors='none', edgecolors='black', linewidth = 1.5)
    ax.set_title(title, fontsize=30)
    ax.set_xlabel(r'$\Omega_m$', fontsize=25)
    # ax.legend(fontsizxe=24, loc='lower left')
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # Decrease the number of ticks on both axes
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4)) # For the x-axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5)) # For the x-axis

    ax.tick_params(axis='both', which='major', labelsize=24)

    ax.yaxis.get_offset_text().set_size(20)

    if saveFileName:
        plt.savefig("Plots/H_matrix_701205.png", transparent=False, dpi=300)

    fig.show()
#%%
    
l = 70
n = 14
n_prime = 3

plotFField(l, n, n_prime, omega_matters, Hs, omega_matters_interp, Hs_interp, title=r'$H^{%.d}_{%.d,%.d} (\Omega_m)$' % (l, n, n_prime),saveFileName="H_matrix.png")

# %%
