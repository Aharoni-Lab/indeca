# %% imports and definition
from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from testing_utils.compose import GridSpec
from testing_utils.misc import load_agg_result
from testing_utils.plotting import (
    plot_agg_boxswarm,
    plot_met_ROC_thres,
    plot_pipeline_iter,
)

from minian_bin.simulation import AR2exp, AR2tau, ar_pulse, eval_exp, find_dhm

IN_RES_PATH = Path(__file__).parent / "output" / "data" / "agg"
FIG_PATH_PN = Path(__file__).parent / "output" / "figs" / "print" / "panels"
FIG_PATH_FIG = Path(__file__).parent / "output" / "figs" / "print" / "figures"
COLORS = {"annotation": "#566573"}
PNLAB_PARAM = {"size": 11, "weight": "bold"}
sns.set_theme(
    context="paper",
    style="darkgrid",
    rc={
        "xtick.major.pad": -2,
        "ytick.major.pad": -2,
        "text.latex.preamble": r"\usepackage{amsmath}",
    },
)
FIG_PATH_PN.mkdir(parents=True, exist_ok=True)
FIG_PATH_FIG.mkdir(parents=True, exist_ok=True)


# %% deconv-thres
fig_w, fig_h = 5, 2.2
fig_path = FIG_PATH_PN / "deconv-thres.svg"
resdf = load_agg_result(IN_RES_PATH / "test_demo_solve_thres")
ressub = resdf.query(
    "upsamp==1 & ns_lev==0.5 & rand_seed==2 & taus=='(6, 1)'"
).drop_duplicates()
fig = plt.figure(figsize=(fig_w, fig_h))
fig = plot_met_ROC_thres(
    ressub,
    fig=fig,
    grid_kws={"width_ratios": [1.6, 1]},
    log_err=False,
    annt_color=COLORS["annotation"],
    annt_lw=2,
)
fig.tight_layout(h_pad=0.2, w_pad=0.8)
fig.savefig(fig_path, bbox_inches="tight")

# %% deconv-upsamp
fig_path = FIG_PATH_PN / "deconv-upsamp.svg"
resdf = load_agg_result(IN_RES_PATH / "test_solve_thres").drop_duplicates()
ressub = resdf.query("taus=='(6, 1)'")
g = plot_agg_boxswarm(
    ressub,
    row="upsamp",
    col="upsamp_y",
    x="ns_lev",
    y="f1",
    facet_kws={"margin_titles": True, "height": 2, "aspect": 1},
    swarm_kws={"size": 3.5, "linewidth": 1},
    box_kws={"fill": False},
)
g.set_xlabels("Noise Level")
g.set_ylabels("f1 Score")
g.set_titles(
    row_template="Upsampling $k$ = {row_name}",
    col_template="Data downsampling: {col_name}",
)
g.set(ylim=(0.75, 1.02))
g.figure.savefig(fig_path, bbox_inches="tight")


# %% deconv-full
def sel_thres(resdf, th_idx, label, met_cols):
    res = resdf.iloc[th_idx, :]
    return pd.DataFrame([{"label": label} | res[met_cols].to_dict()])


def agg_result(
    resdf,
    samp_thres=[0.25, 0.5, 0.75],
    met_cols=["mdist", "f1", "prec", "recall", "scals", "objs"],
):
    res_agg = []
    res_raw = resdf[resdf["group"] == "CNMF"]
    res_nopn = resdf[resdf["group"] == "No Penalty"]
    # raw threshold results
    for th in samp_thres:
        th_idx = np.argmin((res_raw["thres"] - th).abs())
        res_agg.append(sel_thres(res_raw, th_idx, "thres {:.2f}".format(th), met_cols))
    # opt threshold with scaling
    opt_idx_scl = res_nopn["opt_idx"].unique().item()
    res_agg.append(sel_thres(res_nopn, opt_idx_scl, "InDeCa", met_cols))
    res_agg = pd.concat(res_agg, ignore_index=True)
    return res_agg.set_index("label")


fig_path = FIG_PATH_PN / "deconv-full.svg"
grp_dim = ["tau_d", "tau_r", "ns_lev", "upsamp", "rand_seed"]
resdf = load_agg_result(IN_RES_PATH / "test_demo_solve_penal").drop_duplicates()
resagg = resdf.groupby(grp_dim).apply(agg_result).reset_index().drop_duplicates()
ressub = resagg.query("tau_d == 6 & tau_r == 1")
g = plot_agg_boxswarm(
    ressub,
    row="upsamp",
    col="ns_lev",
    x="label",
    y="f1",
    facet_kws={"height": 1.8, "aspect": 1.3, "margin_titles": True},
    swarm_kws={"size": 3.5, "linewidth": 1},
    box_kws={"fill": False},
)
g.tick_params(axis="x", rotation=45)
g.set_xlabels("")
g.set_ylabels("f1 Score")
g.set_titles(
    row_template="Upsampling $k$ = {row_name}",
    col_template="Noise level: {col_name}",
)
g.figure.savefig(fig_path, bbox_inches="tight")

# %% make deconv figure
pns = {
    "A": (FIG_PATH_PN / "deconv-thres.svg", (0, 0)),
    "B": (FIG_PATH_PN / "deconv-upsamp.svg", (0, 1), (3, 1)),
    "C": (FIG_PATH_PN / "deconv-full.svg", (1, 0), (2, 1)),
}
fig = GridSpec(
    param_text=PNLAB_PARAM, wsep=5, hsep=0, halign="center", valign="top", **pns
)
fig.tile()
fig.save(FIG_PATH_FIG / "deconv.svg")

# %% ar-dhm
fig_path = FIG_PATH_PN / "ar-dhm.svg"
fig_w, fig_h = 4, 3.8
with sns.axes_style("white"):
    fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h))
end = 65
for iplt, (theta1, theta2) in enumerate([(1.6, -0.62), (1.6, -0.7)]):
    # ar process
    ar, t, pulse = ar_pulse(theta1, theta2, end)
    t_plt = np.linspace(0, end, 1000)
    # exp form
    is_biexp, tconst, coefs = AR2exp(theta1, theta2)
    tau_d, tau_r = AR2tau(theta1, theta2)
    exp_form = eval_exp(t, is_biexp, tconst, coefs)
    exp_plt = eval_exp(t_plt, is_biexp, tconst, coefs)
    (dhm0, dhm1), t_max = find_dhm(is_biexp, tconst, coefs)
    assert np.isclose(ar, exp_form).all()
    # plotting
    axs[iplt].plot(t_plt, exp_plt, lw=2, color="grey")
    axs[iplt].axvline(
        t_max, lw=1.5, ls="--", color="grey", label="Maximum" if iplt == 0 else None
    )
    axs[iplt].axvline(
        dhm0,
        lw=1.5,
        ls=":",
        color="red",
        label=r"$\text{DHM}_r$" if iplt == 0 else None,
    )
    axs[iplt].axvline(
        dhm1,
        lw=1.5,
        ls=":",
        color="blue",
        label=r"$\text{DHM}_d$" if iplt == 0 else None,
    )
    axs[iplt].yaxis.set_visible(False)
    axs[iplt].text(
        0.6 if iplt == 0 else 0.5,
        0.93,
        "$AR(2)$\n"
        "coefficients:\n"
        r"$"
        r"\begin{aligned}"
        rf"\gamma_1 &= {theta1:.2f}\\"
        rf"\gamma_2 &= {theta2:.2f}\\"
        r"\end{aligned}"
        r"$",
        ha="center",
        va="top",
        transform=axs[iplt].transAxes,
        usetex=True,
    )
    axs[iplt].text(
        0.86 if iplt == 0 else 0.82,
        0.93,
        "bi-exponential\n"
        "coefficients:\n"
        r"$"
        r"\begin{aligned}"
        rf"\tau_d &= {tau_d:.2f}\\"
        rf"\tau_r &= {tau_r:.2f}\\"
        r"\end{aligned}"
        r"$",
        ha="center",
        va="top",
        transform=axs[iplt].transAxes,
        usetex=True,
    )
    if iplt == 0:
        axs[iplt].tick_params(axis="x", labelbottom=False)
    else:
        axs[iplt].set_xlabel("Timesteps")
fig.legend(loc="upper center", ncol=3)
fig.savefig(fig_path, bbox_inches="tight")


# %% ar-full
def AR_scatter(
    data, color, x, y, palette, zorder, annt_color="gray", annt_lw=1, **kwargs
):
    ax = plt.gca()
    data = data.copy()
    res_gt = data[data["method"] == "truth"]
    x_gt = res_gt[x].unique().item()
    y_gt = res_gt[y].unique().item()
    ax.axhline(y_gt, c=annt_color, lw=annt_lw, ls=":", zorder=0)
    ax.axvline(x_gt, c=annt_color, lw=annt_lw, ls=":", zorder=0)
    data["method"] = data["method"] + data["unit"].map(
        lambda u: "-all" if u == "all" else ""
    )
    for (mthd, isreal), subdf in data[data["method"] != "truth-all"].groupby(
        ["method", "isreal"], observed=True
    ):
        mk_kws = (
            {"ec": None, "fc": palette[mthd]}
            if isreal
            else {"ec": palette[mthd], "fc": "none"}
        )
        ax.scatter(subdf[x], subdf[y], label=mthd, **mk_kws, **kwargs)


fig_path = FIG_PATH_PN / "ar-full.svg"
resdf = load_agg_result(IN_RES_PATH / "test_demo_solve_fit_h_num")
ressub = resdf.query("taus == '(6, 1)' & upsamp < 5").astype({"upsamp": int})
cmap = plt.get_cmap("tab10").colors
palette = {
    "cnmf_smth": cmap[0],
    "cnmf_raw": cmap[1],
    "solve_fit": cmap[2],
    "solve_fit-all": cmap[3],
}
lab_map = {
    "cnmf_smth": "Direct /w \nSmoothing",
    "cnmf_raw": "Direct",
    "solve_fit": "InDeCa",
    "solve_fit-all": "InDeCa /w \nshared kernel",
}
g = sns.FacetGrid(ressub, row="upsamp", col="ns_lev", height=2, margin_titles=True)
g.map_dataframe(
    AR_scatter,
    x="dhm0",
    y="dhm1",
    zorder={"cnmf_smth": 1, "cnmf_raw": 1, "solve_fit": 2, "solve_fit-all": 3},
    palette=palette,
    lw=0.5,
    s=8,
    annt_color=COLORS["annotation"],
)
g.add_legend()
g.set_xlabels(r"$\text{DHM}_r$")
g.set_ylabels(r"$\text{DHM}_d$")
g.set_titles(
    row_template="Upsampling $k$ = {row_name}",
    col_template="Noise level: {col_name}",
)
for lab in g._legend.texts:
    lab.set_text(lab_map[lab.get_text()])
g.figure.savefig(fig_path, bbox_inches="tight")

# %% make ar figure
pns = {
    "A": (FIG_PATH_PN / "ar-dhm.svg", (0, 0)),
    "B": (FIG_PATH_PN / "ar-full.svg", (0, 1)),
}
fig = GridSpec(
    param_text=PNLAB_PARAM, wsep=7, hsep=0, halign="center", valign="top", **pns
)
fig.tile()
fig.save(FIG_PATH_FIG / "ar.svg")
