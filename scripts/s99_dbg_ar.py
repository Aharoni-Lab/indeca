# %% import and definition
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from minian_bin.simulation import exp_pulse
from minian_bin.update_AR import fit_sumexp_gd, fit_sumexp_iter

# %% test different fitting
nsamp = 100
tr, t, _ = exp_pulse(6, 1, nsamp)
metrics = []
trs = []
for ns in np.linspace(0, 0.3, 100):
    np.random.seed(18)
    y = tr + np.random.uniform(-ns, ns, nsamp)
    y_norm = y / y.sum()
    for fit_amp in [True, False, "norm"]:
        if fit_amp == False:
            lams, ps, h_fit, _ = fit_sumexp_iter(y_norm, ar_mode=False)
        else:
            lams, ps, h_fit = fit_sumexp_gd(y_norm, ar_mode=False, fit_amp=fit_amp)
        taus = -1 / lams
        metrics.append(
            pd.DataFrame(
                [
                    {
                        "ns": ns,
                        "fit_amp": fit_amp,
                        "tau_d": taus[0],
                        "tau_r": taus[1],
                        "p": ps[0],
                        "err": np.abs(y_norm - h_fit).sum(),
                    }
                ]
            )
        )
        trs.extend(
            [
                pd.DataFrame(
                    {"ns": ns, "fit_amp": fit_amp, "tr": "y_norm", "x": t, "y": y_norm}
                ),
                pd.DataFrame(
                    {"ns": ns, "fit_amp": fit_amp, "tr": "h_fit", "x": t, "y": h_fit}
                ),
            ]
        )
metrics = pd.concat(metrics, ignore_index=True)
trs = pd.concat(trs, ignore_index=True)
g = sns.FacetGrid(
    metrics.melt(id_vars=["ns", "fit_amp"]),
    row="variable",
    col="fit_amp",
    margin_titles=True,
    hue="variable",
    legend_out=True,
    sharey="row",
    aspect=2,
)
g.map_dataframe(sns.lineplot, x="ns", y="value")
