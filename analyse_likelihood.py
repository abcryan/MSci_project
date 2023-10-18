# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def getDeltaLnL(likelihoods):
    # Subtract the maximum
    maximum = np.max(likelihoods)
    delta_lnL = likelihoods - maximum

    return delta_lnL


def plotContour(omega_matters, P_amps, likelihoods, title="", truth=None):
    X, Y = np.meshgrid(omega_matters, P_amps)
    delta_lnLs = getDeltaLnL(likelihoods)
    Z = np.transpose(delta_lnLs)

    fig, ax = plt.subplots(dpi=500)
    CS = ax.contour(X, Y, Z, np.array([-18.40, -11.80, -9.21, -6.18, -4.61, -2.30, 0])/2)
    ax.clabel(CS, inline=True, fontsize=10)


    # Plot the locations of the truth and the peak
    max_index = np.unravel_index(np.argmax(likelihoods), likelihoods.shape)
    max_omega_m, max_P_amp = omega_matters[max_index[0]], P_amps[max_index[1]]


    yOffset = 7

    if truth:
        # plt.plot(truth[0], truth[1], "o", ms=4, c="#1071E5")
        plt.plot(truth[0], truth[1], "o", ms=4)
        plt.annotate("truth", truth, c="tab:blue", ha="center", xytext=(0, yOffset), textcoords='offset points')

    # plt.plot(max_omega_m, max_P_amp, "o", ms=4,c="#E81313")
    plt.plot(max_omega_m, max_P_amp, "o", ms=4)
    plt.annotate("peak", (max_omega_m, max_P_amp), c="tab:orange", ha="center", xytext=(0, yOffset), textcoords='offset points')


    ax.set_title('$\\Delta \\ln L$\n' + title)
    plt.xlabel(r"$\Omega_m$")
    plt.ylabel(r"$P_{amp}$")
    # plt.savefig("contour_plot.png", dpi=500)


def marginaliseOverP(omega_matters, P_amps, likelihoods):
    delta_lnLs = getDeltaLnL(likelihoods)
    Ls = np.exp(delta_lnLs)

    # Marginalise over P_amp
    results = np.zeros(len(omega_matters))

    for i in range(len(omega_matters)):
        total = 0

        for j in range(len(P_amps) - 1):
            total += Ls[i][j] * (P_amps[j+1] - P_amps[j])
        
        results[i] = total

    # Plot the results
    plt.figure(dpi=300)
    plt.plot(omega_matters, results, ".")
    plt.xlabel(r"$\Omega_m$")
    plt.title(r"$L(\Omega_m)$, marginalised over $P_{amp}$")
    # plt.ylabel("L")
    plt.show()


def analyseLikelihood(omega_matters, likelihoods, omega_matter_true, title):
    # Find the maximum
    peak_index = np.argmax(likelihoods)
    delta_lnL = getDeltaLnL(likelihoods)


    # Plot the log likelihood function
    plt.figure(dpi=200)
    plt.plot(omega_matters, likelihoods)
    # plt.plot(omega_matters, likelihoods, '.')
    plt.xlabel("$\Omega_m$")
    plt.ylabel("ln L")
    plt.title("ln L($\Omega_m$)\n%s" % (title))
    plt.show()



    # Find the maximum
    peak_index = np.argmax(likelihoods)
    omega_m_peak = omega_matters[peak_index]
    print("Peak is at Ωₘ = %.4f" % omega_m_peak)

    # Find the index of the true Ωₘ
    true_index = np.argmin(np.abs(omega_matters - omega_matter_true))

    print("ln L(true Ωₘ) = %.3f" % np.real(likelihoods[true_index]))
    print("ln L(peak Ωₘ) = %.3f" % np.real(likelihoods[peak_index]))
    print("ln L(true Ωₘ) - ln L(peak Ωₘ) = %.3f" % np.real(likelihoods[true_index] - likelihoods[peak_index]))
    print("L(true Ωₘ) / L(peak Ωₘ) = %.3e" % np.exp(np.real(likelihoods[true_index] - likelihoods[peak_index])))



    # Plot the likelihood
    # lnL_peak = likelihoods[peak_index]
    # delta_lnL = likelihoods - lnL_peak

    # plt.figure(dpi=200)
    # plt.plot(omega_matters, np.exp(delta_lnL))
    # plt.xlabel("$\Omega_m$")
    # plt.ylabel("L/L$_{peak}$")
    # plt.title("L($\Omega_m$)/L$_{peak}$\n%s" % (title))
    # plt.show()


    # Estimate the width, sigma
    def quadratic(x, mean, sigma):
        return -1/2 * ((x - mean)/sigma)**2

    p0 = [omega_m_peak, 0.001]
    params, cov = curve_fit(quadratic, omega_matters, delta_lnL, p0)
    sigma = np.abs(params[1])

    print("σ = %.5f" % sigma)



    plt.figure(dpi=400)
    plt.plot(omega_matters, delta_lnL, ".", label="$\Delta$ ln L", c="#000000")

    x = np.linspace(np.min(omega_matters), np.max(omega_matters), 100)
    plt.plot(x, quadratic(x, *params), label="Gaussian fit", c="#73CF4F", zorder=0)


    # plt.vlines(params[0], np.min(delta_lnL), np.max(delta_lnL), "b", "dotted")
    # plt.text(params[0], (np.min(delta_lnL) + np.max(delta_lnL))/2, "peak", c="b")
    # plt.text(params[0], 10, "$\Omega_m^{peak}$ = %.4f" % params[0], c="b")
    # plt.text(params[0] - 0.001, 10, "$\Omega_m^{peak}$ = %.4f" % params[0], c="b")


    ylim = -3.8
    # plt.vlines(omega_matter_true, np.min(delta_lnL), quadratic(omega_matter_true, *params), "r", "dotted")
    plt.vlines(omega_matter_true, ylim, 0, "#314ff7", "dashed")
    plt.ylim(ylim)
    # plt.text(omega_matter_true - 0.006, 3, "$\Omega_m^{true}$ = 0.3150", c="r")


    # plt.ylim(top=30)
    plt.xlabel("$\Omega_m$", fontsize=14)
    # plt.ylabel("$\Delta$ ln L")
    # plt.title("$\Delta$ ln L($\Omega_m$)\n%s" % (title))
    plt.title("$\Delta$ ln L($\Omega_m$)\n%s" % title, fontsize=16)
    plt.legend(loc="lower left")
    # plt.savefig("lnL_1.svg")
    plt.show()


    print("Result: Ωₘ = %.5f +/- %.5f" % (params[0], sigma))

    return
# %%
