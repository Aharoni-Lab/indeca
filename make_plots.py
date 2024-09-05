# %% import and definition
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from routine.simulation import AR2exp, AR2tau, ar_pulse, eval_exp, find_dhm, tau2AR
from routine.update_AR import fit_sumexp, fit_sumexp_split, solve_fit_h

FIG_PATH = "figs/presentation"

os.makedirs(FIG_PATH, exist_ok=True)
sns.set_theme(style="darkgrid")

# %% plots for presentation
end = 100
theta1, theta2 = tau2AR(10, 3)
ar, t, pulse = ar_pulse(theta1, theta2, end)
fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(ar, lw=2)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.set_xlabel("time")
fig.savefig(os.path.join(FIG_PATH, "ar2.svg"), bbox_inches="tight")

# %% plots of two real and complex biexp
end = 50
fig, axs = plt.subplots(2, figsize=(3, 3.5))
for iplt, (theta1, theta2) in enumerate([(1.6, -0.62), (1.6, -0.7)]):
    # ar process
    ar, t, pulse = ar_pulse(theta1, theta2, end)
    t_plt = np.linspace(0, end, 1000)
    # exp form
    tau1, tau2 = AR2tau(theta1, theta2)
    is_biexp, tconst, coefs = AR2exp(theta1, theta2)
    exp_form = eval_exp(t, is_biexp, tconst, coefs)
    exp_plt = eval_exp(t_plt, is_biexp, tconst, coefs)
    (t_r, t_d), t_hat = find_dhm(is_biexp, tconst, coefs)
    assert np.isclose(ar, exp_form).all()
    # plotting
    axs[iplt].plot(t_plt, exp_plt, label="exp", lw=2)
    axs[iplt].set_title(
        r"$\theta_1$ = {:.2f}, $\theta_2$ = {:.2f}".format(theta1, theta2)
        + "\n"
        + r"$\lambda_1$ = {:.1f}, $\lambda_2$ = {:.1f}".format(-1 / tau1, -1 / tau2)
    )
    axs[iplt].axes.xaxis.set_ticklabels([])
    axs[iplt].axes.yaxis.set_ticklabels([])
fig.tight_layout()
plt.subplots_adjust(hspace=0.5)
fig.savefig(os.path.join(FIG_PATH, "ar_exp.svg"), bbox_inches="tight")

# %% plot half-max metrics
end = 50
fig, axs = plt.subplots(2, figsize=(4, 5))
for iplt, (theta1, theta2) in enumerate([(1.6, -0.62), (1.6, -0.7)]):
    # ar process
    ar, t, pulse = ar_pulse(theta1, theta2, end)
    t_plt = np.linspace(0, end, 1000)
    # exp form
    is_biexp, tconst, coefs = AR2exp(theta1, theta2)
    exp_form = eval_exp(t, is_biexp, tconst, coefs)
    exp_plt = eval_exp(t_plt, is_biexp, tconst, coefs)
    (t_r, t_d), t_hat = find_dhm(is_biexp, tconst, coefs)
    assert np.isclose(ar, exp_form).all()
    # plotting
    axs[iplt].plot(t_plt, exp_plt, lw=2)
    axs[iplt].axvline(t_hat, lw=1.5, ls=":", color="grey", label="max")
    axs[iplt].axvline(t_r, lw=1.5, ls=":", color="red", label="r0")
    axs[iplt].axvline(t_d, lw=1.5, ls=":", color="blue", label="r1")
    axs[iplt].set_title(
        r"$\theta_1$ = {:.2f}, $\theta_2$ = {:.2f}".format(theta1, theta2)
    )
    axs[iplt].set_xlabel("time")
    axs[iplt].axes.yaxis.set_ticklabels([])
    axs[iplt].legend()
fig.tight_layout()
plt.subplots_adjust(hspace=0.7)
fig.savefig(os.path.join(FIG_PATH, "ar_metric.svg"), bbox_inches="tight")
