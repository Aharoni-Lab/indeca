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

IN_RES_PATH = Path(__file__).parent / "output" / "data" / "agg"
FIG_PATH_PN = Path(__file__).parent / "output" / "figs" / "print" / "panels"
FIG_PATH_FIG = Path(__file__).parent / "output" / "figs" / "print" / "figures"
COLORS = {"annotation": "#566573"}
PNLAB_PARAM = {"size": 11, "weight": "bold"}
sns.set_theme(
    context="paper", style="darkgrid", rc={"xtick.major.pad": -2, "ytick.major.pad": -2}
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
