# %% imports and definition
from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle

from indeca.AR_kernel import estimate_coefs, solve_fit_h_num
from indeca.simulation import AR2exp, AR2tau, ar_pulse, eval_exp, find_dhm
from tests.conftest import fixt_deconv
from tests.testing_utils.compose import GridSpec
from tests.testing_utils.misc import load_agg_result
from tests.testing_utils.plotting import (
    agg_annot_group,
    plot_agg_boxswarm,
    plot_met_ROC_thres,
    plot_pipeline_iter,
)

tab20c = plt.get_cmap("tab20c").colors
tab20b = plt.get_cmap("tab20b").colors
dark2 = plt.get_cmap("Dark2").colors

IN_EXT_RES_PATH = Path(__file__).parent / "tests" / "output" / "data" / "external"
IN_RES_PATH = Path(__file__).parent / "tests" / "output" / "data" / "agg"
FIG_PATH_PN = Path(__file__).parent / "tests" / "output" / "figs" / "print" / "panels"
FIG_PATH_FIG = Path(__file__).parent / "tests" / "output" / "figs" / "print" / "figures"
COLORS = {
    "background": "#e1eaf3",
    "annotation": "#566573",
    "annotation_maj": dark2[0],
    "annotation_min": dark2[2],
    "indeca_maj": tab20c[4],
    "indeca_min": tab20c[6],
    "indeca0": tab20c[4],
    "indeca1": tab20c[5],
    "indeca2": tab20c[6],
    "indeca3": tab20c[7],
    "cnmf_maj": tab20c[0],
    "cnmf_min": tab20c[1],
    "cnmf0": tab20c[0],
    "cnmf1": tab20c[1],
    "cnmf2": tab20c[2],
    "cnmf3": tab20c[3],
    "genericA": tab20b[0],
    "genericA0": tab20b[0],
    "genericA1": tab20b[1],
    "genericA2": tab20b[2],
    "genericA3": tab20b[3],
    "genericB": tab20b[12],
    "genericB0": tab20b[12],
    "genericB1": tab20b[13],
    "genericB2": tab20b[14],
    "genericB3": tab20b[15],
    "genericC": tab20b[4],
    "genericC0": tab20b[4],
    "genericC1": tab20b[5],
    "genericC2": tab20b[6],
    "genericC3": tab20b[7],
}
PNLAB_PARAM = {"size": 11, "weight": "bold"}
RC_PARAM = {
    "xtick.major.pad": -2,
    "ytick.major.pad": -2,
    "axes.labelpad": 1,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.framealpha": 0.8,
    "legend.facecolor": (
        0.9176470588235294,
        0.9176470588235294,
        0.9490196078431372,
    ),
    "legend.edgecolor": (0.8, 0.8, 0.8),
    "legend.borderaxespad": 0.5,
    "legend.handletextpad": 0.8,
}
sns.set_theme(context="paper", style="darkgrid", rc=RC_PARAM)
FIG_PATH_PN.mkdir(parents=True, exist_ok=True)
FIG_PATH_FIG.mkdir(parents=True, exist_ok=True)


# %% flow chart
def add_vl_legend(vls, ax, label, **kwargs):
    cl = vls.get_color()
    ax.add_line(
        Line2D(
            [],
            [],
            marker="|",
            color=cl,
            linestyle="None",
            markersize=8,
            markeredgewidth=1.5,
            label=label,
        )
    )


def pad_ylim(ax, mul):
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[0] + (ylim[1] - ylim[0]) * mul)


fig_path = FIG_PATH_FIG / "flow_chart.svg"
# Simulated data
T = 200
thres_idx = [100, 500, 900]
deconv, y, c, _, s, _, scl = fixt_deconv(taus=(6, 2), y_len=T, ns_lev=0.5)
g, _ = estimate_coefs(y, p=2, noise_freq=None, use_smooth=False, add_lag=5)
h_init, _, _ = ar_pulse(g[0], g[1], nsamp=T, shifted=True)
opt_s, opt_c, opt_scl, _, intm = deconv.solve_thres(return_intm=True)
s_slv, thres, svals, cvals, yfvals, _, _, opt_idx = intm
thres_sub, svals_sub, cvals_sub, yfvals_sub = (
    [thres[i] for i in thres_idx],
    [svals[i] for i in thres_idx],
    [cvals[i] for i in thres_idx],
    [yfvals[i] for i in thres_idx],
)
_, _, _, h_free, h_fit = solve_fit_h_num(y, opt_s, scal=opt_scl)
# Setup figure
padding = {"spikes": 0.2}
fig, axs = plt.subplots(
    4, 2, figsize=(6.8, 6.8), gridspec_kw={"wspace": 0.32, "hspace": 0.35}
)
axs_dict = {
    "p1": axs[0, 0],
    "p2": axs[0, 1],
    "p3": axs[1, 1],
    "p4": axs[2, 1],
    "p5": axs[3, 1],
    "p6": axs[3, 0],
    "p7": axs[2, 0],
    "p8": axs[1, 0],
}
# Plotting
## p1
lns = axs_dict["p1"].plot(y, color=COLORS["genericA"], label="Input Signal")
vls = axs_dict["p1"].vlines(
    np.where(s)[0],
    ymin=y.min() - np.ptp(y) * 0.4 - padding["spikes"],
    ymax=y.min() - padding["spikes"],
    color=COLORS["genericB"],
)
add_vl_legend(vls, axs_dict["p1"], "True Spikes")
axs_dict["p1"].legend(
    loc="upper center", ncol=2, columnspacing=1.5, handlelength=1, handletextpad=0.4
)
pad_ylim(axs_dict["p1"], 1.4)
## p2
axs_dict["p2"].plot(h_init, color=COLORS["genericC"], label="Initial Kernel")
axs_dict["p2"].legend()
## p3
axs_dict["p3"].plot(s_slv, color=COLORS["genericA"], label="Deconvolved Signal")
axs_dict["p3"].legend(loc="upper right")
pad_ylim(axs_dict["p3"], 1.3)
## p4
axs_dict["p4"].plot([], [], label="Threshold:", alpha=0)
for i, bs in enumerate(svals_sub):
    vls = axs_dict["p4"].vlines(
        np.where(bs)[0], ymin=i, ymax=i + 0.8, color=COLORS["genericB{}".format(i + 1)]
    )
    add_vl_legend(vls, axs_dict["p4"], "{:.1f}".format(thres_sub[i]))
leg = axs_dict["p4"].legend(
    loc="upper center",
    ncol=4,
    columnspacing=0.8,
    handlelength=0.8,
    handletextpad=0.2,
)
pad_ylim(axs_dict["p4"], 1.35)
## p5
axs_dict["p5"].plot([], [], label="Threshold:", alpha=0)
for i, rec in enumerate(yfvals_sub):
    axs_dict["p5"].plot(
        rec + i * 2,
        color=COLORS["genericA{}".format(i + 1)],
        label="{:.1f}".format(thres_sub[i]),
    )
leg = axs_dict["p5"].legend(
    loc="upper center",
    ncol=4,
    columnspacing=0.8,
    handlelength=0.8,
    handletextpad=0.2,
)
pad_ylim(axs_dict["p5"], 1.35)
## p6
lns = axs_dict["p6"].plot(opt_c, color=COLORS["genericA"], label="Model Calcium")
vls = axs_dict["p6"].vlines(
    np.where(opt_s)[0],
    ymin=y.min() - np.ptp(y) * 0.4 - padding["spikes"],
    ymax=y.min() - padding["spikes"],
    color=COLORS["genericB"],
)
add_vl_legend(vls, axs_dict["p6"], "Model Spikes")
axs_dict["p6"].legend(
    loc="upper center", ncol=2, columnspacing=1.5, handlelength=1, handletextpad=0.4
)
pad_ylim(axs_dict["p6"], 1.4)
## p7
axs_dict["p7"].plot(h_free, color=COLORS["genericC"], label="Free Kernel")
axs_dict["p7"].legend()
## p8
axs_dict["p8"].plot(h_fit, color=COLORS["genericC"], label="Bi-exponential Kernel")
axs_dict["p8"].legend()

# Clean axes
for ax in axs_dict.values():
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Arrow style
arrowstyle = dict(
    arrowstyle="simple, head_length=1, head_width=2, tail_width=1",
    facecolor=COLORS["background"],
    edgecolor="black",
    linewidth=1.2,
)

# Draw arrows using ConnectionPatch
connections = [
    ("p1", "p2", (1, 0.5), (0, 0.5), "Estimate", "arc", (0, -0.05)),
    ("p2", "p3", (1, 0.3), (1, 0.7), "Deconvolve", "arc", (-0.06, 0)),
    ("p3", "p4", (1, 0.3), (1, 0.7), "Threshold", "arc", (-0.06, 0)),
    ("p4", "p5", (1, 0.3), (1, 0.7), "Reconvolve", "arc", (-0.06, 0)),
    ("p5", "p6", (0, 0.5), (1, 0.5), "Select", "arc", (0, 0.04)),
    ("p6", "p7", (0, 0.7), (0, 0.3), "Extract", "arc", (0.04, 0)),
    ("p7", "p8", (0, 0.7), (0, 0.3), "Fit Exponential", "arc", (0.08, 0)),
    ("p8", "p3", (1, 0.5), (0, 0.5), "Update", "arc", (0, -0.04)),
]

for a_from, a_to, xyA, xyB, label, conn_sty, txt_offset in connections:
    if conn_sty == "arc":
        conn_sty = "arc3, rad=-0.4"
    elif conn_sty == "straight":
        conn_sty = None
    con = ConnectionPatch(
        xyA=xyA,
        coordsA="axes fraction",
        axesA=axs_dict[a_from],
        xyB=xyB,
        coordsB="axes fraction",
        axesB=axs_dict[a_to],
        connectionstyle=conn_sty,
        **arrowstyle,
    )
    fig.add_artist(con)
    # Label midpoint
    bboxA, bboxB = axs_dict[a_from].get_position(), axs_dict[a_to].get_position()
    ptA = (bboxA.x0 + xyA[0] * bboxA.width, bboxA.y0 + xyA[1] * bboxA.height)
    ptB = (bboxB.x0 + xyB[0] * bboxB.width, bboxB.y0 + xyB[1] * bboxB.height)
    txt = fig.text(
        (ptA[0] + ptB[0]) / 2 + txt_offset[0],
        (ptA[1] + ptB[1]) / 2 + txt_offset[1],
        label,
        ha="center",
        va="center",
        fontsize=10,
        backgroundcolor=COLORS["background"],
    )
    txt.set_bbox(
        {
            "facecolor": COLORS["background"],
            "alpha": 0.9,
            "edgecolor": "black",
            "capstyle": "round",
            "joinstyle": "round",
        }
    )
fig.savefig(fig_path, bbox_inches="tight")


# %% deconv-thres
fig_w, fig_h = 5.8, 2.2
fig_path = FIG_PATH_PN / "deconv-thres.svg"
resdf = load_agg_result(IN_RES_PATH / "test_demo_solve_thres")
ressub = (
    resdf.query("upsamp==1 & ns_lev==0.5 & rand_seed==2 & taus=='(6, 1)'")
    .drop_duplicates()
    .copy()
)
fig = plt.figure(figsize=(fig_w, fig_h))
fig = plot_met_ROC_thres(
    ressub,
    fig=fig,
    grid_kws={"width_ratios": [2, 1]},
    log_err=False,
    annt_color=COLORS["annotation"],
    annt_lw=2,
)
fig.align_ylabels()
fig.tight_layout(h_pad=0.2, w_pad=0.8)
fig.savefig(fig_path, bbox_inches="tight")


# %% deconv-upsamp
def upsamp_heatmap(data, color, **kwargs):
    ax = plt.gca()
    data_pvt = (
        data.groupby(["upsamp_y", "upsamp"])["f1"]
        .mean()
        .reset_index()
        .pivot(index="upsamp_y", columns="upsamp", values="f1")
    )
    sns.heatmap(data_pvt, ax=ax, **kwargs)
    for i in range(len(data_pvt)):
        rect = Rectangle(
            (i, i), 1, 1, fill=False, edgecolor=COLORS["annotation"], linewidth=1
        )
        ax.add_patch(rect)


fig_path = FIG_PATH_PN / "deconv-upsamp.svg"
resdf = load_agg_result(IN_RES_PATH / "test_solve_thres").drop_duplicates()
ressub = resdf.query("taus=='(6, 1)'").copy()
vmin, vmax = 0.48, 1.02
g = sns.FacetGrid(ressub, col="ns_lev", margin_titles=True, height=2, aspect=0.9)
g.map_dataframe(
    upsamp_heatmap,
    vmin=vmin,
    vmax=vmax,
    square=True,
    cbar=False,
    linewidths=0.1,
    linecolor=COLORS["annotation"],
)
g.set_xlabels("Upsampling $k$")
g.set_ylabels("Data downsampling")
g.set_titles(col_template="Noise (A.U.): {col_name}")
fig = g.figure
cbar_ax = fig.add_axes([0.95, 0.25, 0.02, 0.6])
cm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax))
cbar = fig.colorbar(cm, cax=cbar_ax, ticks=[0.5, 0.75, 1.0])
cbar.set_label("F1 score", rotation=270, va="bottom")
cbar_ax.tick_params(size=0, pad=2)
fig.tight_layout(rect=[0, 0, 0.98, 1])
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
        res_agg.append(sel_thres(res_raw, th_idx, "Thres {:.2f}".format(th), met_cols))
    # opt threshold with scaling
    opt_idx_scl = res_nopn["opt_idx"].unique().item()
    res_agg.append(sel_thres(res_nopn, opt_idx_scl, "InDeCa", met_cols))
    res_agg = pd.concat(res_agg, ignore_index=True)
    return res_agg.set_index("label")


fig_path = FIG_PATH_PN / "deconv-full.svg"
grp_dim = ["tau_d", "tau_r", "ns_lev", "upsamp", "rand_seed"]
resdf = load_agg_result(IN_RES_PATH / "test_demo_solve_penal").drop_duplicates()
resagg = resdf.groupby(grp_dim).apply(agg_result).reset_index().drop_duplicates()
ressub = resagg.query("tau_d == 6 & tau_r == 1").copy()
palette = {
    "Thres 0.25": COLORS["cnmf0"],
    "Thres 0.50": COLORS["cnmf1"],
    "Thres 0.75": COLORS["cnmf2"],
    "InDeCa": COLORS["indeca_maj"],
}
g = plot_agg_boxswarm(
    ressub,
    row="upsamp",
    col="ns_lev",
    x="label",
    y="f1",
    facet_kws={"height": 1.5, "aspect": 1.3, "margin_titles": True},
    swarm_kws={"size": 3, "linewidth": 0.8, "palette": palette},
    box_kws={"width": 0.5, "fill": False, "palette": palette},
    annt_group={"InDeCa": ["Thres 0.25", "Thres 0.50", "Thres 0.75"]},
)
g.tick_params(axis="x", rotation=45)
g.set_xlabels("")
g.set_ylabels("F1 score")
g.set_titles(
    row_template="Upsampling $k$: {row_name}",
    col_template="Noise (A.U.): {col_name}",
)
g.figure.tight_layout(h_pad=0.3, w_pad=0.4)
g.figure.savefig(fig_path, bbox_inches="tight")

# %% make deconv figure
pns = {
    "A": (FIG_PATH_PN / "deconv-thres.svg", (0, 0)),
    "B": (FIG_PATH_PN / "deconv-upsamp.svg", (1, 0)),
    "C": (FIG_PATH_PN / "deconv-full.svg", (2, 0)),
}
fig = GridSpec(
    param_text=PNLAB_PARAM, wsep=0, hsep=0, halign="left", valign="top", **pns
)
fig.tile()
fig.save(FIG_PATH_FIG / "deconv.svg")

# %% ar-dhm
fig_path = FIG_PATH_PN / "ar-dhm.svg"
fig_w, fig_h = 6.8, 2
with sns.axes_style("white"):
    fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h))
end = 60
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
        t_max,
        lw=1.5,
        ls="--",
        color=COLORS["annotation"],
        label="Maximum" if iplt == 0 else None,
    )
    axs[iplt].axvline(
        dhm0,
        lw=1.5,
        ls=":",
        color=COLORS["annotation_maj"],
        label=r"$\text{DHM}_r$" if iplt == 0 else None,
    )
    axs[iplt].axvline(
        dhm1,
        lw=1.5,
        ls=":",
        color=COLORS["annotation_min"],
        label=r"$\text{DHM}_d$" if iplt == 0 else None,
    )
    axs[iplt].yaxis.set_visible(False)
    axs[iplt].text(
        0.54 if iplt == 0 else 0.42,
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
        0.84 if iplt == 0 else 0.78,
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
    axs[iplt].set_xlabel("Timesteps")
fig.tight_layout()
fig.legend(
    loc="center left", bbox_to_anchor=(1.01, 0.5), bbox_transform=axs[-1].transAxes
)
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
ressub = (
    resdf.query("taus == '(6, 1)' & upsamp < 5 & rand_seed == 2")
    .astype({"upsamp": int})
    .copy()
)
palette = {
    "cnmf_smth": COLORS["cnmf_min"],
    "cnmf_raw": COLORS["cnmf_maj"],
    "solve_fit": COLORS["indeca_min"],
    "solve_fit-all": COLORS["indeca_maj"],
}
lab_map = {
    "cnmf_smth": "Direct /w \nsmoothing",
    "cnmf_raw": "Direct",
    "solve_fit": "InDeCa",
    "solve_fit-all": "InDeCa /w \nshared kernel",
}
g = sns.FacetGrid(
    ressub, row="upsamp", col="ns_lev", height=1.5, aspect=1.15, margin_titles=True
)
g.map_dataframe(
    AR_scatter,
    x="dhm0",
    y="dhm1",
    zorder={"cnmf_smth": 1, "cnmf_raw": 1, "solve_fit": 2, "solve_fit-all": 3},
    palette=palette,
    lw=0.4,
    s=4.5,
    annt_color=COLORS["annotation"],
)
g.add_legend(
    handletextpad=RC_PARAM["legend.handletextpad"],
    borderaxespad=RC_PARAM["legend.borderaxespad"],
    handlelength=0.2,
    facecolor=RC_PARAM["legend.facecolor"],
    edgecolor=RC_PARAM["legend.edgecolor"],
    frameon=RC_PARAM["legend.frameon"],
    fancybox=RC_PARAM["legend.fancybox"],
    framealpha=RC_PARAM["legend.framealpha"],
    bbox_to_anchor=(1.02, 0.5),
)
g.set_xlabels(r"$\text{DHM}_r$ (timesteps)")
g.set_ylabels(r"$\text{DHM}_d$" + "\n(timesteps)")
g.set_titles(
    row_template="Upsampling $k$: {row_name}",
    col_template="Noise level (A.U.): {col_name}",
)
for lab in g._legend.texts:
    lab.set_text(lab_map[lab.get_text()])
g.figure.savefig(fig_path, bbox_inches="tight")

# %% make ar figure
pns = {
    "A": (FIG_PATH_PN / "ar-dhm.svg", (0, 0), (1, 1), "left"),
    "B": (FIG_PATH_PN / "ar-full.svg", (1, 0)),
}
fig = GridSpec(
    param_text=PNLAB_PARAM, wsep=0, hsep=0, halign="left", valign="top", **pns
)
fig.tile()
fig.save(FIG_PATH_FIG / "ar.svg")


# %% make pipeline-iter figure
def plot_iter(
    data, color, label, swarm_kws=dict(), line_kws=dict(), box_kws=dict(), palette=None
):
    ax = plt.gca()
    mthd = data["method"].unique().item()
    met = data["metric"].unique().item()
    if mthd == "indeca":
        data = data.astype({"iter": int})
        dat_all = data[data["use_all"]]
        dat_ind = data[~data["use_all"]]
        swm = sns.swarmplot(
            dat_ind,
            x="iter",
            y="value",
            ax=ax,
            color=color if palette is None else palette["indeca-ind"],
            edgecolor="auto",
            warn_thresh=0.9,
            **swarm_kws,
        )
        lns = sns.lineplot(
            dat_all,
            x="iter",
            y="value",
            ax=ax,
            color=color if palette is None else palette["indeca-shared"],
            estimator=None,
            errorbar=None,
            units="unit_id",
            zorder=3,
            **line_kws,
        )
        leg_handles["Independent kernel"] = swm.collections[0]
        leg_handles["Shared kernel"] = lns.lines[0]
        ax.set_xlabel("Iteration")
    elif mthd == "cnmf":
        data = data.astype({"qthres": float})
        data["value"] = data["value"].where(data["qthres"] == 0.5)
        sns.swarmplot(
            data,
            y="value",
            ax=ax,
            color=color if palette is None else palette["cnmf"],
            edgecolor="auto",
            warn_thresh=0.9,
            **swarm_kws,
        )
        sns.boxplot(
            data,
            y="value",
            color=color if palette is None else palette["cnmf"],
            ax=ax,
            fill=False,
            showfliers=False,
            **box_kws,
        )
        ax.set_xlabel("Final output")
        ax.set_ylabel(
            {"dhm0": r"$\text{DHM}_r$ (sec)", "dhm1": r"$\text{DHM}_d$ (sec)"}[met]
        )


fig_path = FIG_PATH_PN / "pipeline-iter.svg"
res_bin = load_agg_result(IN_RES_PATH / "test_demo_pipeline_realds")
res_cnmf = load_agg_result(IN_RES_PATH / "test_demo_pipeline_realds_cnmf")
id_vars = [
    "dsname",
    "ncell",
    "method",
    "use_all",
    "tau_init",
    "unit_id",
    "iter",
    "qthres",
    "test_id",
    "upsamp",
]
val_vals = ["f1", "dhm0", "dhm1"]
resdf = (
    pd.concat([res_bin, res_cnmf], ignore_index=True)
    .drop_duplicates()
    .fillna("None")
    .melt(
        id_vars=id_vars,
        value_vars=val_vals,
        var_name="metric",
        value_name="value",
    )
    .drop_duplicates()
)
ressub = resdf.query(
    "ncell == 'None' & method != 'gt'"
    "& tau_init == 'None' & iter != 10 & metric != 'f1'"
).copy()
row_lab_map = {
    "f1": "f1 Score",
    "dhm0": r"$\text{DHM}_r$",
    "dhm1": r"$\text{DHM}_d$",
}
col_lab_map = {"cnmf": "CNMF", "indeca": "InDeCa"}
leg_handles = dict()
ressub["row_lab"] = ressub["metric"].map(row_lab_map)
ressub["col_lab"] = ressub["method"].map(col_lab_map)
row_ord = [row_lab_map[r] for r in ["dhm0", "dhm1"]]
col_ord = [col_lab_map[c] for c in ["cnmf", "indeca"]]
palette = {
    "cnmf": COLORS["cnmf_maj"],
    "indeca-ind": COLORS["indeca_min"],
    "indeca-shared": COLORS["indeca_maj"],
}
g = sns.FacetGrid(
    ressub,
    height=1.3,
    aspect=3 / 1.3,
    row="row_lab",
    col="col_lab",
    sharey="row",
    sharex="col",
    hue="col_lab",
    row_order=row_ord,
    col_order=col_ord,
    hue_order=col_ord,
    margin_titles=True,
    gridspec_kws={"width_ratios": [1, 4]},
)
g.map_dataframe(
    plot_iter,
    swarm_kws={"s": 3, "linewidth": 0.6},
    box_kws={"width": 0.4},
    palette=palette,
)
g.set_titles(row_template="", col_template="{col_name}")
for ax in g.axes.flat:
    tt = ax.get_title()
    if tt == "InDeCa":
        ax.set_title(tt, pad=25)
    else:
        ax.set_title(tt, pad=15)
fig = g.figure
fig.align_ylabels()
fig.legend(
    handles=list(leg_handles.values()),
    labels=list(leg_handles.keys()),
    title="",
    loc="upper center",
    bbox_to_anchor=(0.6, 1.01),
    ncol=2,
)
fig.subplots_adjust(hspace=0.08, wspace=0.02)
fig.savefig(fig_path, bbox_inches="tight")


# %% make pipeline-comp figure
def xlab(row):
    if row["method"] in ["cnmf", "oasis"]:
        if row["metric"] in ["f1", "prec", "rec", "mdist"]:
            return "{}\nthreshold\n{}".format(row["method"], row["qthres"])
        else:
            return "{}".format(row["method"])
    elif row["method"] == "mlspike":
        return "{}".format(row["method"])
    else:
        lab = "InDeCa"
        if row["use_all"]:
            lab += " /w\nshared\nkernel"
        else:
            lab += " /w\nindependent\nkernel"
        if row["tau_init"] != "None":
            lab += "\n+\ninitial\nconstants"
        return lab


def plot_ds(data, palette=None, color=None):
    ax = plt.gca()
    ax = sns.boxplot(
        data,
        x="xlab",
        y="value",
        hue="xlab",
        width=0.5,
        fill=False,
        palette=palette,
        showfliers=False,
    )
    ax = sns.swarmplot(
        data,
        x="xlab",
        y="value",
        hue="xlab",
        edgecolor="auto",
        palette=palette,
        linewidth=1,
        s=4,
    )
    # agg_annot_group(
    #     data,
    #     x="xlab",
    #     y="value",
    #     group={
    #         "InDeCa /w\nindependent\nkernel": [
    #             "CNMF\nthreshold\n0.25",
    #             "CNMF\nthreshold\n0.5",
    #             "CNMF\nthreshold\n0.75",
    #         ],
    #         "InDeCa /w\nshared\nkernel": [
    #             "CNMF\nthreshold\n0.25",
    #             "CNMF\nthreshold\n0.5",
    #             "CNMF\nthreshold\n0.75",
    #         ],
    #     },
    # )
    ax.set_xlabel("")
    ax.set_ylabel("F1 score")


def sel_iter(df):
    its = df["iter"].unique()
    if len(its) == 1 and its.item() == "None":
        return df
    else:
        iter_last = its.max()
        return df[df["iter"] == iter_last]


res_bin = load_agg_result(IN_RES_PATH / "test_demo_pipeline_realds")
res_cnmf = load_agg_result(IN_RES_PATH / "test_demo_pipeline_realds_cnmf")
res_oasis = pd.read_feather(IN_EXT_RES_PATH / "caiman" / "metrics.feat")
res_mlspike = pd.read_feather(IN_EXT_RES_PATH / "mlspike" / "metrics.feat")
id_vars = [
    "dsname",
    "ncell",
    "method",
    "use_all",
    "tau_init",
    "unit_id",
    "iter",
    "qthres",
    "test_id",
    "upsamp",
]
val_vals = ["f1", "corr_raw", "corr_gs", "corr_dtw"]
resdf = (
    pd.concat([res_bin, res_cnmf, res_oasis, res_mlspike], ignore_index=True)
    .drop_duplicates()
    .fillna("None")
    .melt(
        id_vars=id_vars,
        value_vars=val_vals,
        var_name="metric",
        value_name="value",
    )
    .drop_duplicates()
)
ressub = (
    resdf.query("ncell == 'None' & method != 'gt' & tau_init == 'None'")
    .groupby(["dsname", "method", "use_all", "metric"])
    .apply(sel_iter, include_groups=False)
    .reset_index()
    .copy()
)
ressub["xlab"] = ressub.apply(xlab, axis="columns")
ressub = ressub.sort_values("xlab").drop_duplicates(
    subset=["method", "dsname", "xlab", "unit_id", "metric", "value"]
)
palette = {
    "CNMF\nthreshold\n0.25": COLORS["cnmf0"],
    "CNMF\nthreshold\n0.5": COLORS["cnmf1"],
    "CNMF\nthreshold\n0.75": COLORS["cnmf2"],
    "mlspike": COLORS["genericA"],
    "InDeCa /w\nindependent\nkernel": COLORS["indeca_min"],
    "InDeCa /w\nshared\nkernel": COLORS["indeca_maj"],
}
for met, met_df in ressub.groupby("metric"):
    fig_path = FIG_PATH_PN / "pipeline-comp-{}.svg".format(met)
    if met == "f1":
        met_df = met_df[met_df["method"].isin(["indeca", "mlspike"])]
        asp = 1.6
    else:
        met_df = met_df[met_df["method"] != "mlspike"]
        asp = 1.6
    g = sns.FacetGrid(
        met_df.replace({"None": np.nan}),
        col="dsname",
        col_wrap=5,
        height=2.5,
        aspect=asp,
    )
    # g.map_dataframe(plot_ds, palette=palette)
    g.map_dataframe(plot_ds)
    g.figure.savefig(fig_path, bbox_inches="tight")

# %% make pipeline figure
pns = {
    "A": (FIG_PATH_PN / "pipeline-iter.svg", (0, 0)),
    "B": (FIG_PATH_PN / "pipeline-comp.svg", (1, 0)),
}
fig = GridSpec(
    param_text=PNLAB_PARAM, wsep=0, hsep=0, halign="left", valign="top", **pns
)
fig.tile()
fig.save(FIG_PATH_FIG / "pipeline.svg")
