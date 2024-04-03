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

l_max = 100 #15 #70 # 40 # 25 # 15
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

k_bin_edges, k_bin_heights = create_power_spectrum(k_max, 10, np.array([0.15, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]))
# k_bin_edges, k_bin_heights = create_power_spectrum(k_max, 2, np.array([0.35, 0.8]))
k_vals = np.linspace(0, 350, 1000)
P_vals = [P_parametrised(k, k_bin_edges, k_bin_heights) for k in k_vals]

k_units = k_vals/3000

print(200/3000)
# #1071E5
plt.figure(figsize=(10, 7))
plt.plot(k_units, P_vals, c="k", lw=1.25)
# Add a vertical red dashed line at x=0.1 (change this to your specific x-coordinate)
plt.axvline(x=0.1, color='red', linestyle='--')
plt.text(x=0.103, y=0.8, s='Non-Linear \n Regime', color='red', fontsize=14)
# Annotate with text and an arrow pointing to the line
plt.annotate(
    '',  # Text for annotation
    xy=(0.12, 0.7),  # Point to annotate (x, y) - choose suitable y value
    xytext=(0.1025, 0.7),  # Location of text (x, y) - choose suitable coordinates
    arrowprops=dict(arrowstyle="->",
                             color='red',
                             lw=3.0,
                             ls='-'),
    color='red',  # Text color
    fontsize=12,  # Text font size
)
plt.annotate(
    '',  # Text for annotation
    xy=(0.12, 0.4),  # Point to annotate (x, y) - choose suitable y value
    xytext=(0.1025, 0.4),  # Location of text (x, y) - choose suitable coordinates
    arrowprops=dict(arrowstyle="->",
                             color='red',
                             lw=3.0,
                             ls='-'),
    color='red',  # Text color
    fontsize=12,  # Text font size
)
plt.xlim(0)
plt.ylim(0)
plt.xlabel("$k$  [$h$Mpc$^{-1}$]")
plt.legend(["$P(k)$"], loc=[0.55, 0.83], fontsize=18)
plt.savefig("Plots/power_spectrum_10_bins.png", dpi=300)
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

def plotFField(grid, title="", colorbarLabel=r'$\delta(r, \theta, \phi)$', saveFileName=None):
    mpl.rcParams.update({"axes.grid" : True, "grid.color": "#333333"})

    # i = 500
    # title = r"$\delta(\mathbf{r})$ at $r$=%.2f" % radii_true[i] + "\n" + "$r_{max}$=%.2f, $k_{max}$=%d, $l_{max}$=%d" % (r_max_true, k_max, l_max)

    # fig, ax = grid.plot(
    #     projection=ccrs.Mollweide(),
    #     colorbar='right',
    #     cb_label=colorbarLabel,
    #     title=title,
    #     grid=False,
    #     show=False)
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 50  # For the X-axis and Y-axis label
    # Specify the figure size in inches (width, height)
    fig, ax = plt.subplots(figsize=(7, 4))

    # Now use the ax object to plot your data
    grid.plot(
        ax=ax,  # Specify the Axes object to plot on
        projection=ccrs.Mollweide(),
        colorbar='right',
        cb_label=colorbarLabel,
        title=title,

        grid=False,
        show=False)
    


    if saveFileName:
        plt.savefig("field.png", transparent=False, dpi=300)

    # plt.show()

# plotField(all_grids[1000], title=r"$\delta(\mathbf{r})$ at $r$=%.2f" % radii_true[50] + "\n" + "$r_{max}$=%.2f, $k_{max}$=%d, $l_{max}$=%d" % (r_max_true, k_max, l_max), saveFileName="field.svg")
# plotField(all_grids[1000], title=r'$\delta(r, \theta, \phi)$', colorbarLabel="" ,saveFileName="field.png")
plotFField(all_grids[500], title=r'$r=1125.0$ [Mpc/h]', colorbarLabel="" ,saveFileName="field.png")

# %%

from matplotlib import ticker

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# PLOT THE PWOER SPECTRUM NICELY

# More sophisticated power specturm

k_bin_edges, k_bin_heights = create_power_spectrum(k_max, 10, np.array([0.15, 0.35, 0.6, 0.8, 0.9, 1, 0.95, 0.85, 0.7, 0.3]))
# k_bin_edges, k_bin_heights = create_power_spectrum(k_max, 2, np.array([0.35, 0.8]))
k_vals = np.linspace(0, 350, 1000)
P_vals = [P_parametrised(k, k_bin_edges, k_bin_heights) for k in k_vals]

k_units = k_vals/3000


# %%
plt.figure(figsize=(10, 7))
plt.plot(k_units, P_vals, c="k", lw=2.0)
# Add a vertical red dashed line at x=0.1 (change this to your specific x-coordinate)
plt.axvline(x=0.1, color='red', linestyle='--')
plt.text(x=0.101, y=0.8, s=r'$\textit{Non-Linear}$', color='red', fontsize=20)
plt.text(x=0.103, y=0.73, s=r'$\textit{Regime}$', color='red', fontsize=20)

# Annotate with text and an arrow pointing to the line
plt.annotate(
    '',  # Text for annotation
    xy=(0.12, 0.6),  # Point to annotate (x, y) - choose suitable y value
    xytext=(0.1025, 0.6),  # Location of text (x, y) - choose suitable coordinates
    arrowprops=dict(arrowstyle="->",
                             color='red',
                             lw=3.0,
                             ls='-'),
    color='red',  # Text color
    fontsize=12,  # Text font size
)
plt.annotate(
    '',  # Text for annotation
    xy=(0.12, 0.4),  # Point to annotate (x, y) - choose suitable y value
    xytext=(0.1025, 0.4),  # Location of text (x, y) - choose suitable coordinates
    arrowprops=dict(arrowstyle="->",
                             color='red',
                             lw=3.0,
                             ls='-'),
    color='red',  # Text color
    fontsize=12,  # Text font size
)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlim(0)
plt.ylim(0)
plt.xlabel("$\mathbf{k}$   $[\mathbf{h}$Mpc$^{-1}]$", fontsize=20)
plt.legend(["Power Spectrum"], loc=[0.41, 0.86], fontsize=20)
plt.title("$P(k)$", fontsize=30)
plt.savefig("Plots/power_spectrum_10_bins.png", dpi=300)
plt.show()
    
def P_para(k, k_max=200):
    return P_parametrised(k, k_bin_edges, k_bin_heights)

# %%
