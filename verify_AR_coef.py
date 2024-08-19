# %% imports
import matplotlib.pyplot as plt
import numpy as np

from routine.update_bin import construct_G
from routine.simulation import AR2tau, tau2AR
from routine.utilities import norm

# %% verify AR process
# parameters
tau_d = 6
tau_r = 1
t = np.arange(100).astype(float)
# biexponential model
L = np.array([[1, 1], [-1 / tau_d, -1 / tau_r]])
coef = np.linalg.inv(L) @ np.array([1, 0.5]).reshape((-1, 1))
biexp = coef[0] * np.exp(-t / tau_d) + coef[1] * np.exp(-t / tau_r)
# AR model
theta1, theta2 = tau2AR(tau_d, tau_r)
pulse = np.zeros_like(t)
pulse[0] = 1
ar = np.zeros_like(t)
for i in range(len(t)):
    if i > 1:
        ar[i] = pulse[i] + theta1 * ar[i - 1] + theta2 * ar[i - 2]
    elif i > 0:
        ar[i] = pulse[i] + theta1 * ar[i - 1]  # implies ar[i-2] == 0, i.e. derivative 1
    else:
        ar[i] = pulse[i]
# verify AR model with linalg
G = construct_G((tau_d, tau_r), len(t), fromTau=True).todense()
Gi = np.linalg.inv(G)
assert np.isclose(Gi @ pulse, ar).all()
assert np.isclose(G @ ar, pulse).all()
# plotting
fig, ax = plt.subplots()
ax.plot(biexp, label="biexp")
ax.plot(ar, label="ar")
ax.plot(Gi[:, 0], label="Gi")
ax.legend()
print(theta1, theta2)
print(AR2tau(theta1, theta2))
