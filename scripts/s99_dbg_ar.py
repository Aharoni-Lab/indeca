# %% import and definition
import itertools as itt
import os

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import xarray as xr
from tqdm.auto import tqdm

from indeca.simulation import exp_pulse
from indeca.AR_kernel import fit_sumexp_gd, fit_sumexp_iter, solve_fit_h_num
from indeca.utils import norm, scal_lstsq

FIG_PATH = "./figs/dbg_ar"
INT_PATH = "./intermediate/dbg_ar"
os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(INT_PATH, exist_ok=True)

# %% test different fitting
nsamp = 100
tr, t, _ = exp_pulse(6, 1, nsamp)
metrics = []
trs = []
for ns in np.linspace(0, 0.3, 100):
    np.random.seed(18)
    y = tr + np.random.uniform(-ns, ns, nsamp)
    y_norm = y / y.sum()
    for fit_amp in [True, False]:
        if fit_amp == False:
            lams, ps, h_fit, _ = fit_sumexp_iter(y_norm)
        else:
            lams, ps, h_fit = fit_sumexp_gd(y_norm, fit_amp=True)
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
    height=1.5,
    aspect=2,
)
g.map_dataframe(sns.lineplot, x="ns", y="value")
g.figure.savefig(os.path.join(FIG_PATH, "ar_fit.svg"), bbox_inches="tight")

# %% explore error space of different ar coefs
coefs = [(6, 1), (10, 3), (15, 5)]
err_spc = np.linspace(-4, 4, 201)
ns_lv = np.linspace(0, 0.3, 31)
nsamp = 100
err_df = []
for (tau_d, tau_r), dd, dr, ns in tqdm(
    list(itt.product(coefs, err_spc, err_spc, ns_lv))
):
    t_d, t_r = tau_d + dd, tau_r + dr
    if t_d > t_r and t_r > 0:
        p_true = 1 / (np.exp(-1 / tau_d) - np.exp(-1 / tau_r))
        p_denom = np.exp(-1 / t_d) - np.exp(-1 / t_r)
        if p_denom > 0:
            p = 1 / p_denom
        else:
            continue
        tr, t, _ = exp_pulse(tau_d, tau_r, nsamp, p_d=p_true, p_r=-p_true)
        y = tr + np.random.uniform(-ns, ns, nsamp)
        y_var = np.linalg.norm(y)
        tr_fit, t, _ = exp_pulse(t_d, t_r, nsamp, p_d=p, p_r=-p)
        scal = scal_lstsq(tr_fit, y)
        err = np.linalg.norm(y - tr_fit).clip(0, y_var)
        err_scal = np.linalg.norm(y - tr_fit * scal).clip(0, y_var)
        edf = pd.DataFrame(
            [
                {
                    "gt": "tau_d={}, tau_r={}".format(tau_d, tau_r),
                    "ns": ns,
                    "tau_d": t_d,
                    "tau_r": t_r,
                    "scal": p,
                    "err": err,
                    "err_scal": err_scal,
                }
            ]
        )
        err_df.append(edf)
err_df = pd.concat(err_df, ignore_index=True)
err_df.to_feather(os.path.join(INT_PATH, "err_df.feat"))


# %% plot errors
def norm_err(err):
    return norm(np.log(err + 1e-10))


err_df = pd.read_feather(os.path.join(INT_PATH, "err_df.feat"))
for gt, edf in err_df.groupby("gt"):
    for zvar in ["scal", "err", "err_scal"]:
        earr = edf.set_index(["ns", "tau_d", "tau_r"])[zvar].to_xarray()
        earr = xr.apply_ufunc(
            norm_err,
            earr,
            input_core_dims=[["tau_d", "tau_r"]],
            output_core_dims=[["tau_d", "tau_r"]],
            vectorize=True,
        )
        fig = px.imshow(
            earr,
            facet_col="ns",
            facet_col_wrap=8,
            aspect="equal",
            facet_col_spacing=5e-3,
            facet_row_spacing=2e-2,
        )
        fig.update_layout(height=2000)
        fig.write_html(os.path.join(FIG_PATH, "{}-{}.html".format(zvar, gt)))

# %% debug fitting
import plotly.express as px
from scipy.signal import medfilt
from statsmodels.robust import norms

from indeca.simulation import exp_pulse

h = np.load("h.npy")[1:]
p = 1 / (np.exp(-1 / 6) - np.exp(-1))
h_true, _, _ = exp_pulse(6, 1, len(h), p_d=p, p_r=-p)
scal_true = scal_lstsq(h_true, h)
w = np.ones_like(h)
for i in range(10):
    lams, p, scal, h_fit = fit_sumexp_gd(h, y_weight=w, fit_amp="scale")
    taus = -1 / lams
    print(taus)
    # w = -norms.AndrewWave().weights(h - h_fit) + 2
    # w = ((h - h_fit) ** 2).clip(1e-3, None)

fig = px.line(h)
fig.add_scatter(y=h_fit)
fig.add_scatter(y=h_true * scal_true)
fig.show()
