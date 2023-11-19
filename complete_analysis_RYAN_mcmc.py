import numpy as np
from numba import jit
from os import path

from generate_field import generateTrueField, multiplyFieldBySelectionFunction
from distance_redshift_relation import *
from spherical_bessel_transform import calc_f_lmn_0_numba, calc_f_lmn_0
from calculate_W import calc_all_W_numba, make_W_integrand_numba, interpolate_W_values
from calculate_SN import calc_all_SN
from compute_likelihood import computeLikelihoodMCMC
from analyse_likelihood import plotContour, plotPosterior
from utils import calc_n_max_l, gaussianPhi
from precompute_c_ln import get_c_ln_values_without_r_max
from precompute_sph_bessel_zeros import loadSphericalBesselZeros

import emcee
import corner

def P_Top_Hat(k, k_max=200):
    if k < k_max:
        return 1
    else:
        return 0


def main_function():

    #########################
    ### SET UP PARAMETERS ###
    
    l_max = 15
    k_max = 200 #200
    r_max_true = 0.75
    n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0
    n_max_ls = np.array([calc_n_max_l(l, k_max, r_max_true) for l in range(l_max + 1)])
    R = 0.25    # Selection function scale length
    # nbar = 5

    omega_matter_true = 0.315
    omega_matter_0 = 0.315      # observed

    P_amp = 1

    #########################
    #########################
    

    # Calculate c_ln coefficients of true SBT with infinite r
    c_ln_values_without_r_max = get_c_ln_values_without_r_max("c_ln.csv")

    # Calculate spherical Bessel zeros
    sphericalBesselZeros = loadSphericalBesselZeros("zeros.csv")

    # Generate true field
    radii_true = np.linspace(0, r_max_true, 1001)    
    print('generating true field ...')
    z_true, all_grids = generateTrueField(radii_true, omega_matter_true, r_max_true, l_max, k_max, P_Top_Hat)

    # Add the effect of the selection function
    @jit(nopython=True)
    def phiOfR0(r0):
        return np.exp(-r0*r0 / (2*R*R))

    radii_true, all_observed_grids = multiplyFieldBySelectionFunction(radii_true, all_grids, phiOfR0)


    #########################
    ### Observed Quantities ###

    r_of_z_fiducial = getInterpolatedRofZ(omega_matter_0)
    radii_fiducial = r_of_z_fiducial(z_true)
    r_max_0 = radii_fiducial[-1]

    # Perform the spherical Bessel transform to obtain the coefficients

    #f_lmn_0_saveFileName = "data_Ryan/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_P-amp_%.2f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R, P_amp)
    #f_lmn_0_saveFileName = "data/f_lmn_0_true-%.3f_fiducial-%.3f_l_max-%d_k_max-%.2f_r_max_true-%.3f_R-%.3f_P-amp_%.2f.npy" % (omega_matter_true, omega_matter_0, l_max, k_max, r_max_true, R, P_amp)
    f_lmn_0_saveFileName = "data/f_lmn_0_true-0.315_fiducial-0.315_l_max-15_k_max-200.00_r_max_true-0.750_R-0.250_P-amp_1.00-2023-04-18-numba-5.npy"
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

    #########################
    ### Likelihood Calculation ###

    # Initialize
    omega_matters = np.linspace(omega_matter_0 - 0.008, omega_matter_0 + 0.005, 27)
    omega_matters_inference = np.linspace(omega_matter_0 - 0.007, omega_matter_0 + 0.005, 97)
    P_amps = np.linspace(0.95, 1.05, 51)
    likelihoods = np.zeros((np.size(omega_matters_inference), np.size(P_amps)))

    # Compute W's
    Ws = []
    for omega_matter in omega_matters:
        #W_saveFileName = "data_Ryan/W_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
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
    #SN_saveFileName = "data_Ryan/SN_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    SN_saveFileName = "data/SN_no_tayl_exp_zeros_omega_m-%.5f_omega_m_0-%.5f_l_max-%d_k_max-%.2f_r_max_0-%.4f_R-%.3f.npy" % (omega_matter, omega_matter_0, l_max, k_max, r_max_0, R)
    if path.exists(SN_saveFileName):
        SN = np.load(SN_saveFileName)
    else:
        print("Computing SN for Ωₘ⁰ = %.4f." % omega_matter_0)
        SN = calc_all_SN(l_max, k_max, r_max_0, phiOfR0)
        np.save(SN_saveFileName, SN)
        

    #MCMC requires us to be able to evaluate the likelihood for arbitrary values of Ωₘ, so interpolate W^l_nn' (Ωₘ)    
    step = 0.00001
    omega_matters_interp, Ws_interp = interpolate_W_values(l_max, n_max_ls, omega_matters, Ws, step=step)
    omega_matter_min, omega_matter_max = omega_matters_interp[0], omega_matters_interp[-1]


    # Use MCMC to perform likelihood analysis
    #TOdo tomorrow




if __name__ == "__main__":
    main_function()