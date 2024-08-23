# %% imports
import matplotlib.pyplot as plt
import numpy as np

from routine.simulation import AR2exp, AR2tau, ar_pulse, eval_exp, find_dhm, tau2AR

# %% verify AR2tau
end = 60
for theta1, theta2 in [(1.6, -0.62), (1.6, -0.7)]:
    # ar process
    ar, t = ar_pulse(theta1, theta2, end)
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
