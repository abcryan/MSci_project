# %%

from utils import calc_n_max_l
from compute_likelihood import calc_W_without_delta_omega_m
#, calc_all_Ws_without_delta_omega_m

from distance_redshift_relation import *


l_max = 15
k_max = 100
r_max_true = 0.8
n_max = calc_n_max_l(0, k_max, r_max_true) # There are the most modes when l=0

omega_matter_0 = 0.48

dr_domega = getPartialRbyOmegaMatterInterp(omega_matter_0)
# Ws_without_delta_omega_m = calc_all_Ws_without_delta_omega_m(l_max, k_max, r_max_0, dr_domega)

for l in range(l_max + 1):
    print("n_max_%d is %d" % (l, calc_n_max_l(l, k_max, r_max_true)))

# %%

n1, n2, l = 25, 20, 0

for epsabs in [1.49e-8, 1e-12, 1e-16]:
    print("epsabs = %s, W^%d_%d,%d =" % (epsabs, l, n1, n2), calc_W_without_delta_omega_m(n1, n2, l, r_max_true, dr_domega, Nsplit=10, epsabs=epsabs))

# %%

for Nsplit in [10, 100, 1000]:
    print("Nsplit = %s, W^%d_%d,%d =" % (Nsplit, l, n1, n2), calc_W_without_delta_omega_m(n1, n2, l, r_max_true, dr_domega, Nsplit=Nsplit))

# %%

# Try plotting the integrand of W

print("W^%d_%d,%d =" % (l, n1, n2), calc_W_without_delta_omega_m(n1, n2, l, r_max_true, dr_domega, Nsplit=10, epsabs=1e-16, plotIntegrand=True))

# %%
