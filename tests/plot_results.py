# %% imports and definition
from pathlib import Path

import pandas as pd
import seaborn as sns

IN_RES_PATH = Path(__file__).parent / "output" / "data" / "test_results"
FIG_PATH = Path(__file__).parent / "output" / "figs" / "test_results"

# %% plot thresholds
fig_path = FIG_PATH / "solve_thres"
fig_path.mkdir(parents=True, exist_ok=True)
result = pd.read_feather(IN_RES_PATH / "test_solve_thres.feat")
for (td, tr), res_sub in result.groupby(["tau_d", "tau_r"]):
    for met in ["mdist", "f1", "precs", "recall"]:
        g = sns.FacetGrid(res_sub, row="upsamp", col="upsamp_y")
        g.map_dataframe(
            sns.boxplot,
            x="ns_lev",
            y=met,
            hue="ns_lev",
            saturation=0.5,
            showfliers=False,
            palette="tab10",
        )
        g.map_dataframe(
            sns.swarmplot,
            x="ns_lev",
            y=met,
            hue="ns_lev",
            edgecolor="auto",
            palette="tab10",
            size=5,
            linewidth=1.2,
        )
        g.tight_layout()
        g.figure.savefig(
            fig_path / "tau({},{})-{}.svg".format(td, tr, met), bbox_inches="tight"
        )
