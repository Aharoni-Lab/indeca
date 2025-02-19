# %% imports
import matplotlib.pyplot as plt
import numpy as np

from minian_bin.simulation import (
    AR2exp,
    AR2tau,
    apply_arcoef,
    apply_exp,
    ar_pulse,
    eval_exp,
    exp_pulse,
    find_dhm,
    markov_fire,
    tau2AR,
)
from minian_bin.AR_kernel import construct_G, fit_sumexp, fit_sumexp_split, solve_fit_h

# %% verify AR2tau
end = 60
for theta1, theta2 in [(1.6, -0.62), (1.6, -0.7)]:
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
    fig, ax = plt.subplots()
    ax.plot(t_plt, exp_plt, label="exp", lw=2)
    ax.plot(t, ar, label="ar", lw=3, ls=":")
    ax.axvline(t_hat, lw=1.5, ls=":", color="grey")
    ax.axvline(t_r, lw=1.5, ls=":", color="red")
    ax.axvline(t_d, lw=1.5, ls=":", color="blue")
    ax.legend()

# %% validate biexp2AR
end = 60
for tau_d, tau_r, p in zip(
    np.random.uniform(6, 10, 5),
    np.random.uniform(1, 4, 5),
    np.random.uniform(0.5, 2, 5),
):
    biexp, t, _ = exp_pulse(tau_d, tau_r, end, p_d=p, p_r=-p)
    theta1, theta2, scl = tau2AR(tau_d, tau_r, p=p, return_scl=True)
    ar, t, _ = ar_pulse(theta1, theta2, end)
    biexp_scl = (biexp / scl)[1:]
    assert np.isclose(biexp_scl, ar[:-1]).all()
    # plotting
    fig, ax = plt.subplots()
    ax.plot(t, biexp, label="exp", lw=2)
    ax.plot(t[:-1], biexp_scl, label="exp_scal", lw=2)
    ax.plot(t, ar, label="ar", lw=3, ls=":")
    ax.legend()

# %% verify biexp fit
end = 100
theta1, theta2 = tau2AR(10, 3)
ar, t, pulse = ar_pulse(theta1, theta2, end)
for ns in [0, 0.5, 1]:
    np.random.seed(0)
    ar_in = ar + ns * (np.random.random(end) - 0.5)
    lams, ps, ar_fit = fit_sumexp(ar_in, 2)
    print("noise: {}\nlams: {}\nps: {}".format(ns, lams, ps))
    fig, ax = plt.subplots()
    ax.plot(ar, label="true", lw=2)
    ax.plot(ar_in, label="input", lw=1.5)
    ax.plot(ar_fit, label="fit", lw=3, ls=":")
    ax.legend()

# %% verify AR and biexp
end = 100
tau_d, tau_r = 10, 3
# ar
theta1, theta2 = tau2AR(tau_d, tau_r)
ar, t_ar, pulse = ar_pulse(theta1, theta2, end)
# biexp
v = np.exp(-t_ar / tau_d) - np.exp(-t_ar / tau_r)
biexp = np.convolve(v, pulse, mode="full")[:end]
# kernel fit
lams_biexp, ps_biexp, h_biexp, h_fit_biexp, _, _ = solve_fit_h(
    biexp, pulse, fit_method="solve"
)
lams_ar, ps_ar, h_ar, h_fit_ar, _, _ = solve_fit_h(ar, pulse, fit_method="solve")
print("biexp fit: taus: {}, coefs: {}".format(1 / -lams_biexp, ps_biexp))
print("biexp fit: taus: {}, coefs: {}".format(1 / -lams_ar, ps_ar))
# numerical_fit
lams_biexp_num, ps_biexp_num, h_biexp_num, h_fit_biexp_num, _, _ = solve_fit_h(
    biexp, pulse, fit_method="numerical"
)
lams_ar_num, ps_ar_num, h_ar_num, h_fit_ar_num, _, _ = solve_fit_h(
    ar, pulse, fit_method="numerical"
)
print(
    "numerical biexp fit: taus: {}, coefs: {}".format(1 / -lams_biexp_num, ps_biexp_num)
)
print("numerical biexp fit: taus: {}, coefs: {}".format(1 / -lams_ar_num, ps_ar_num))
# plotting
fig, axs = plt.subplots(3, sharey=True, figsize=(5, 8))
axs[0].plot(biexp, label="biexp", lw=2)
axs[0].plot(ar, label="ar", lw=3, ls=":")
axs[0].plot(pulse, label="pulse", lw=1.5)
axs[0].legend()
axs[1].plot(h_biexp, label="biexp_kernel", lw=1.5)
axs[1].plot(h_fit_biexp, label="biexp_kernel_fit", lw=3, ls=":")
axs[1].plot(h_ar, label="ar_kernel", lw=1.5)
axs[1].plot(h_fit_ar, label="ar_kernel_fit", lw=3, ls=":")
axs[1].legend()
axs[2].plot(h_biexp_num, label="biexp_kernel", lw=1.5)
axs[2].plot(h_fit_biexp_num, label="biexp_kernel_fit", lw=3, ls=":")
axs[2].plot(h_ar_num, label="ar_kernel", lw=1.5)
axs[2].plot(h_fit_ar_num, label="ar_kernel_fit", lw=3, ls=":")
axs[2].legend()

# %% verify equivalence of shifted AR(2) and constrained biexp
s = markov_fire(1000, np.array([[0.98, 0.02], [0.75, 0.25]]))
for tau_d_org, tau_r_org in [(6, 1), (10, 3)]:
    # ar coefficients
    theta1, theta2 = tau2AR(tau_d_org, tau_r_org)
    tau_d, tau_r, p = AR2tau(theta1, theta2, solve_amp=True)
    assert np.isclose(tau_d, tau_d_org)
    assert np.isclose(tau_r, tau_r_org)
    # pulse
    ar, t, pulse = ar_pulse(theta1, theta2, nsamp=100, shifted=True)
    exp, t, pulse = exp_pulse(tau_d, tau_r, nsamp=100, p_d=p, p_r=-p)
    assert np.isclose(ar, exp).all()
    fig, ax = plt.subplots()
    ax.plot(exp, label="exp", lw=2)
    ax.plot(ar, label="ar", lw=3, ls=":")
    ax.legend()
    # trace
    ar = apply_arcoef(s, (theta1, theta2), shifted=True)
    exp = apply_exp(s, tau_d, tau_r, p, -p)
    assert np.isclose(ar, exp).all()
    fig, ax = plt.subplots()
    ax.plot(exp, label="exp", lw=2)
    ax.plot(ar, label="ar", lw=3, ls=":")
    ax.legend()
