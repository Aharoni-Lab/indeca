# %% imports and definition
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from testing_utils.plotting import plot_agg_boxswarm

IN_RES_PATH = Path(__file__).parent / "output" / "data" / "test_results"
FIG_PATH = Path(__file__).parent / "output" / "figs" / "test_results"


# %% plot penalty results
def sel_thres(resdf, th_idx, label, met_cols):
    res = resdf.iloc[th_idx, :]
    return pd.DataFrame([{"label": label} | res[met_cols].to_dict()])


def agg_result(
    resdf,
    samp_thres=[0.2, 0.4, 0.6, 0.8],
    met_cols=["mdist", "f1", "prec", "recall", "scals", "objs"],
):
    res_agg = []
    res_scl, res_noscl = resdf[resdf["thres_scaling"]], resdf[~resdf["thres_scaling"]]
    res_raw = res_noscl[res_noscl["penal"] == 0]
    res_scl_nopn = res_scl[res_scl["penal"] == 0]
    res_scl_pn = res_scl[res_scl["penal"] != 0]
    # raw threshold results
    for th in samp_thres:
        th_idx = np.argmin((res_raw["thres"] - th).abs())
        res_agg.append(sel_thres(res_raw, th_idx, "thres{:.1f}".format(th), met_cols))
    # opt threshold results
    opt_idx_raw = res_raw["opt_idx"].unique().item()
    res_agg.append(sel_thres(res_raw, opt_idx_raw, "optimal thres", met_cols))
    # opt threshold with scaling
    opt_idx_scl = res_scl_nopn["opt_idx"].unique().item()
    res_agg.append(
        sel_thres(res_scl_nopn, opt_idx_scl, "optimal thres\n/w scaling", met_cols)
    )
    # opt penalty
    opt_idx_pn = res_scl_pn["opt_idx"].unique().item()
    res_agg.append(
        sel_thres(res_scl_pn, opt_idx_pn, "optimal /w\nscaling & penalty", met_cols)
    )
    res_agg = pd.concat(res_agg, ignore_index=True)
    return res_agg.set_index("label")


fig_path = FIG_PATH / "demo_solve_penal"
fig_path.mkdir(parents=True, exist_ok=True)
result = pd.read_feather(IN_RES_PATH / "test_demo_solve_penal.feat")
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
        g.figure.savefig(
            fig_path / "tau({},{})-{}.svg".format(td, tr, met), bbox_inches="tight"
        )
        plt.close(g.figure)


# %% plot thresholds
fig_path = FIG_PATH / "solve_thres"
fig_path.mkdir(parents=True, exist_ok=True)
result = pd.read_feather(IN_RES_PATH / "test_solve_thres.feat")
for (td, tr), res_sub in result.groupby(["tau_d", "tau_r"]):
    for met in ["mdist", "f1", "precs", "recall"]:
        g = plot_agg_boxswarm(res_sub, row="upsamp", col="upsamp_y", x="ns_lev", y=met)
        g.figure.savefig(
            fig_path / "tau({},{})-{}.svg".format(td, tr, met), bbox_inches="tight"
        )
        plt.close(g.figure)
