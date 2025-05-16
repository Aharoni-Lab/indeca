# %% imports and definition
from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from testing_utils.compose import GridSpec
from testing_utils.misc import load_agg_result
from testing_utils.plotting import (
    agg_annot_group,
    plot_agg_boxswarm,
    plot_met_ROC_thres,
    plot_pipeline_iter,
)

from minian_bin.simulation import AR2exp, AR2tau, ar_pulse, eval_exp, find_dhm

tab20c = plt.get_cmap("tab20c").colors
dark2 = plt.get_cmap("Dark2").colors

IN_RES_PATH = Path(__file__).parent / "output" / "data" / "agg"
FIG_PATH_PN = Path(__file__).parent / "output" / "figs" / "print" / "panels"
FIG_PATH_FIG = Path(__file__).parent / "output" / "figs" / "print" / "figures"
COLORS = {
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
    if mthd == "minian-bin":
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
col_lab_map = {"cnmf": "CNMF", "minian-bin": "InDeCa"}
leg_handles = dict()
ressub["row_lab"] = ressub["metric"].map(row_lab_map)
ressub["col_lab"] = ressub["method"].map(col_lab_map)
row_ord = [row_lab_map[r] for r in ["dhm0", "dhm1"]]
col_ord = [col_lab_map[c] for c in ["cnmf", "minian-bin"]]
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
    if row["method"] == "cnmf":
        return "CNMF\nthreshold\n{}".format(row["qthres"])
    else:
        lab = "InDeCa"
        if row["use_all"]:
            lab += " /w\nshared\nkernel"
        else:
            lab += " /w\nindependent\nkernel"
        if row["tau_init"] != "None":
            lab += "\n+\ninitial\nconstants"
        return lab


fig_path = FIG_PATH_PN / "pipeline-comp.svg"
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
    "ncell == 'None' & method != 'gt' & iter in (10, 'None') & metric == 'f1' & tau_init == 'None'"
).copy()
ressub["xlab"] = ressub.apply(xlab, axis="columns")
ressub = ressub.sort_values("xlab")
palette = {
    "CNMF\nthreshold\n0.25": COLORS["cnmf0"],
    "CNMF\nthreshold\n0.5": COLORS["cnmf1"],
    "CNMF\nthreshold\n0.75": COLORS["cnmf2"],
    "InDeCa /w\nindependent\nkernel": COLORS["indeca_min"],
    "InDeCa /w\nshared\nkernel": COLORS["indeca_maj"],
}
fig, ax = plt.subplots(figsize=(6.4, 2))
ax = sns.boxplot(
    ressub,
    x="xlab",
    y="value",
    hue="xlab",
    width=0.5,
    fill=False,
    palette=palette,
    showfliers=False,
)
ax = sns.swarmplot(
    ressub,
    x="xlab",
    y="value",
    hue="xlab",
    edgecolor="auto",
    palette=palette,
    linewidth=1,
    s=4,
)
agg_annot_group(
    ressub,
    x="xlab",
    y="value",
    group={
        "InDeCa /w\nindependent\nkernel": [
            "CNMF\nthreshold\n0.25",
            "CNMF\nthreshold\n0.5",
            "CNMF\nthreshold\n0.75",
        ],
        "InDeCa /w\nshared\nkernel": [
            "CNMF\nthreshold\n0.25",
            "CNMF\nthreshold\n0.5",
            "CNMF\nthreshold\n0.75",
        ],
    },
)
ax.set_xlabel("")
ax.set_ylabel("F1 score")
fig.savefig(fig_path, bbox_inches="tight")

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
