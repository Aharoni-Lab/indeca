# %% imports and definition
from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
sns.set_theme(
    context="paper", style="darkgrid", rc={"xtick.major.pad": -2, "ytick.major.pad": -2}
)
FIG_PATH_PN.mkdir(parents=True, exist_ok=True)
FIG_PATH_FIG.mkdir(parents=True, exist_ok=True)


# %% deconv-thres
fig_w, fig_h = 5, 2.5
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
fig.savefig(str(fig_path), bbox_inches="tight")
