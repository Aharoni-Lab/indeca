# %% imports and definition
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from testing_utils.misc import load_agg_result
from testing_utils.plotting import plot_agg_boxswarm

IN_RES_PATH = Path(__file__).parent / "output" / "data" / "agg_results"
FIG_PATH = Path(__file__).parent / "output" / "figs" / "agg_results"


# %% plot pipeline results
fig_path = FIG_PATH / "demo_pipeline"
fig_path.mkdir(parents=True, exist_ok=True)
result = load_agg_result(IN_RES_PATH / "test_demo_pipeline")
id_vars = ["method", "use_all", "unit_id", "iter", "param_upsamp_param"]
val_vals = ["mdist", "f1"]
resdf = pd.melt(
    result,
    id_vars=id_vars,
    value_vars=val_vals,
    var_name="metric",
    value_name="value",
)
resdf["row_lab"] = resdf["metric"]
resdf["col_lab"] = (
    resdf["use_all"].map(lambda u: "all_cell" if u else "individual")
    + "-"
    + resdf["param_upsamp_param"].astype(str)
)
g = sns.FacetGrid(resdf, row="row_lab", col="col_lab", sharey="row", margin_titles=True)
g.map_dataframe(sns.lineplot, x="iter", y="value")


# %% plot AR results
def AR_scatter(data, color, x, y, palette, zorder, **kwargs):
    ax = plt.gca()
    data = data.copy()
    res_gt = data[data["method"] == "truth"]
    x_gt = res_gt[x].unique().item()
    y_gt = res_gt[y].unique().item()
    ax.axhline(y_gt, c="gray", ls=":", zorder=0)
    ax.axvline(x_gt, c="gray", ls=":", zorder=0)
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


fig_path = FIG_PATH / "demo_solve_fit_h"
fig_path.mkdir(parents=True, exist_ok=True)
result = load_agg_result(IN_RES_PATH / "test_demo_solve_fit_h_num")
if result is not None:
    result = result.rename(columns=lambda c: c.removesuffix("_param"))
    result["param_taus"] = result["param_taus"].map(lambda t: tuple(t.tolist()))
    cmap = plt.get_cmap("tab10").colors
    palette = {
        "cnmf_smth": cmap[0],
        "cnmf_raw": cmap[1],
        "solve_fit": cmap[2],
        "solve_fit-all": cmap[3],
    }
    for (td, tr), res_sub in result.groupby("param_taus"):
        g = sns.FacetGrid(
            res_sub, row="param_upsamp", col="param_ns_level", margin_titles=True
        )
        g.map_dataframe(
            AR_scatter,
            x="dhm0",
            y="dhm1",
            zorder={"cnmf_smth": 1, "cnmf_raw": 1, "solve_fit": 2, "solve_fit-all": 3},
            palette=palette,
            lw=0.6,
            s=6,
        )
        g.add_legend()
        g.figure.savefig(
            fig_path / "tau({},{}).svg".format(td, tr), bbox_inches="tight"
        )
        plt.close(g.figure)


# %% plot penalty results
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
    res_pn = resdf[resdf["group"] == "Penalty"]
    # raw threshold results
    for th in samp_thres:
        th_idx = np.argmin((res_raw["thres"] - th).abs())
        res_agg.append(sel_thres(res_raw, th_idx, "thres{:.2f}".format(th), met_cols))
    # opt threshold with scaling
    opt_idx_scl = res_nopn["opt_idx"].unique().item()
    res_agg.append(
        sel_thres(res_nopn, opt_idx_scl, "optimal thres\n/w scaling", met_cols)
    )
    # opt penalty
    opt_idx_pn = res_pn["opt_idx"].unique().item()
    res_agg.append(
        sel_thres(res_pn, opt_idx_pn, "optimal /w\nscaling & penalty", met_cols)
    )
    res_agg = pd.concat(res_agg, ignore_index=True)
    return res_agg.set_index("label")


fig_path = FIG_PATH / "demo_solve_penal"
fig_path.mkdir(parents=True, exist_ok=True)
result = load_agg_result(IN_RES_PATH / "test_demo_solve_penal")
if result is not None:
    result = result[result["param_y_scaling_param"]]
    grp_dim = ["tau_d", "tau_r", "ns_lev", "upsamp", "param_rand_seed_param"]
    res_agg = result.groupby(grp_dim).apply(agg_result).reset_index()
    for (td, tr), res_sub in res_agg.groupby(["tau_d", "tau_r"]):
        for met in ["mdist", "f1", "prec", "recall"]:
            g = plot_agg_boxswarm(
                res_sub,
                row="upsamp",
                col="ns_lev",
                x="label",
                y=met,
                facet_kws={"height": 3.5},
            )
            g.tick_params(rotation=45)
            g.set(yscale="log")
            g.figure.savefig(
                fig_path / "tau({},{})-{}.svg".format(td, tr, met), bbox_inches="tight"
            )
            plt.close(g.figure)


# %% plot thresholds
fig_path = FIG_PATH / "solve_thres"
fig_path.mkdir(parents=True, exist_ok=True)
result = load_agg_result(IN_RES_PATH / "test_solve_thres")
if result is not None:
    for (td, tr), res_sub in result.groupby(["tau_d", "tau_r"]):
        for met in ["mdist", "f1", "precs", "recall"]:
            g = plot_agg_boxswarm(
                res_sub, row="upsamp", col="upsamp_y", x="ns_lev", y=met
            )
            g.figure.savefig(
                fig_path / "tau({},{})-{}.svg".format(td, tr, met), bbox_inches="tight"
            )
            plt.close(g.figure)
