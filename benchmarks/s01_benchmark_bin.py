# %% import and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import xarray as xr
from distributed import LocalCluster
from routine.cnmf import solve_deconv
from routine.utils import (
    compute_ROC,
    norm_per_cell,
    plot_corr,
    plot_ROC,
    plot_ROC_scatter,
)
from tqdm.auto import tqdm

from minian_bin.AR_kernel import estimate_coefs
from minian_bin.deconv import construct_G, construct_R, max_thres

IN_PATH = {
    "org": "./intermediate/simulated/simulated-ar-samp.nc",
    "upsamp": "./intermediate/simulated/simulated-ar-upsamp.nc",
}
INT_PATH = "./intermediate/benchmark_bin"
FIG_PATH = "./figs/benchmark_bin"
PARAM_TAU_D = 6
PARAM_TAU_R = 1
PARAM_UPSAMP = 10
PARAM_EST_AR = False
PARAM_SIG_LEV = (1, 10)

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)
if __name__ == "__main__":
    cluster = LocalCluster(
        n_workers=32, dashboard_address="localhost:8787", threads_per_worker=1
    )
    client = cluster.get_client()


# %% temporal update
if __name__ == "__main__":
    sps_penal = 1
    max_iters = 50
    for up_type, up_factor in {"org": 1, "upsamp": PARAM_UPSAMP}.items():
        # get data
        sim_ds = xr.open_dataset(IN_PATH[up_type])
        C_gt = sim_ds["C"].dropna("frame", how="all")
        C_gt = norm_per_cell(C_gt)
        subset = C_gt.coords["unit_id"]
        np.random.seed(42)
        sig_lev = xr.DataArray(
            np.sort(
                np.random.uniform(
                    low=PARAM_SIG_LEV[0],
                    high=PARAM_SIG_LEV[1],
                    size=C_gt.sizes["unit_id"],
                )
            ),
            dims=["unit_id"],
            coords={"unit_id": C_gt.coords["unit_id"]},
            name="sig_lev",
        )
        noise = np.random.normal(loc=0, scale=1, size=C_gt.shape)
        Y_solve = (C_gt * sig_lev + noise).sel(unit_id=subset)
        updt_ds = [Y_solve.rename("Y_solve"), sig_lev.sel(unit_id=subset)]
        iter_df = []
        # update
        res = {
            "C": [],
            "S": [],
            "b": [],
            "C-bin": [],
            "S-bin": [],
            "b-bin": [],
            "S-bin-scal": [],
            "scal": [],
        }
        for y in tqdm(
            Y_solve.transpose("unit_id", "frame"), total=Y_solve.sizes["unit_id"]
        ):
            # parameters
            y_norm = np.array(y)
            T = len(y_norm)
            g, tn = estimate_coefs(
                y_norm, p=2, noise_freq=0.4, use_smooth=False, add_lag=0
            )
            if PARAM_EST_AR:
                G = construct_G(g, T * up_factor, fromTau=False)
            else:
                G = construct_G(
                    (PARAM_TAU_D * up_factor, PARAM_TAU_R * up_factor),
                    T * up_factor,
                    fromTau=True,
                )
            R = construct_R(T, up_factor)
            # org algo
            c, s, b = solve_deconv(y_norm, G, l1_penal=sps_penal * tn, R=R)
            res["C"].append(c.squeeze())
            res["S"].append(s.squeeze())
            res["b"].append(b)
            # bin algo
            c_bin, s_bin, b_bin, scale, s_scal, it_df = solve_deconv_bin(y_norm, G, R)
            it_df["unit_id"] = y.coords["unit_id"].item()
            it_df["up_type"] = up_type
            iter_df.append(it_df)
            res["C-bin"].append(c_bin.squeeze())
            res["S-bin"].append(s_bin.squeeze())
            res["S-bin-scal"].append(s_scal.squeeze())
            res["b-bin"].append(b_bin)
            res["scal"].append(scale)
        # save variables
        for vname, dat in res.items():
            dat = np.stack(dat)
            if dat.ndim == 1:
                updt_ds.append(
                    xr.DataArray(
                        dat,
                        dims="unit_id",
                        coords={"unit_id": Y_solve.coords["unit_id"]},
                        name=vname,
                    )
                )
            elif dat.ndim == 2:
                updt_ds.append(
                    xr.DataArray(
                        dat,
                        dims=["unit_id", "frame"],
                        coords={
                            "unit_id": Y_solve.coords["unit_id"],
                            "frame": (
                                sim_ds.coords["frame"]
                                if up_type == "upsamp"
                                else Y_solve.coords["frame"]
                            ),
                        },
                        name=vname,
                    )
                )
            else:
                raise ValueError
        updt_ds = xr.merge(updt_ds)
        updt_ds.to_netcdf(os.path.join(INT_PATH, "updt_ds-{}.nc".format(up_type)))
        iter_df = pd.concat(iter_df, ignore_index=True)
        iter_df.to_feather(os.path.join(INT_PATH, "iter_df-{}.feat".format(up_type)))


# %% compute metrics
if __name__ == "__main__":
    th_range = (0.01, 0.99)
    nthres = int(980 / 5 + 1)
    met_df = []
    for up_type, true_ds in IN_PATH.items():
        updt_ds = xr.open_dataset(
            os.path.join(INT_PATH, "updt_ds-{}.nc".format(up_type))
        )
        true_ds = xr.open_dataset(true_ds)
        if up_type == "upsamp":
            metS = compute_ROC(
                updt_ds["S"],
                true_ds["S"].dropna("frame", how="all"),
                nthres=nthres,
                th_min=th_range[0],
                th_max=th_range[1],
                ds=PARAM_UPSAMP,
                metadata={"method": "CNMF", "dataset": "upsamp-down"},
                use_warp=True,
            )
            metS_bin_scal = compute_ROC(
                updt_ds["S-bin-scal"],
                true_ds["S"].dropna("frame", how="all"),
                nthres=nthres,
                th_min=th_range[0],
                th_max=th_range[1],
                ds=PARAM_UPSAMP,
                metadata={"method": "minian-bin-scal", "dataset": "upsamp-down"},
                use_warp=True,
            )
            metS_bin = compute_ROC(
                updt_ds["S-bin"].coarsen({"frame": PARAM_UPSAMP}).sum(),
                true_ds["S"].dropna("frame", how="all"),
                metadata={"method": "minian-bin", "dataset": "upsamp-down"},
                use_warp=True,
            )
            metS_up = compute_ROC(
                updt_ds["S"],
                true_ds["S_true"],
                nthres=nthres,
                th_min=th_range[0],
                th_max=th_range[1],
                metadata={"method": "CNMF", "dataset": "upsamp"},
            )
            metS_up_bin_scal = compute_ROC(
                updt_ds["S-bin-scal"],
                true_ds["S_true"],
                nthres=nthres,
                th_min=th_range[0],
                th_max=th_range[1],
                metadata={"method": "minian-bin-scal", "dataset": "upsamp"},
            )
            metS_up_bin = compute_ROC(
                updt_ds["S-bin"],
                true_ds["S_true"],
                metadata={"method": "minian-bin", "dataset": "upsamp"},
            )
            met_df.extend(
                [metS, metS_bin, metS_up, metS_up_bin, metS_bin_scal, metS_up_bin_scal]
            )
        else:
            metS = compute_ROC(
                updt_ds["S"],
                true_ds["S"],
                nthres=nthres,
                th_min=th_range[0],
                th_max=th_range[1],
                metadata={"method": "CNMF", "dataset": "org"},
            )
            metS_bin_scal = compute_ROC(
                updt_ds["S-bin-scal"],
                true_ds["S"],
                nthres=nthres,
                th_min=th_range[0],
                th_max=th_range[1],
                metadata={"method": "minian-bin-scal", "dataset": "org"},
            )
            metS_bin = compute_ROC(
                updt_ds["S-bin"],
                true_ds["S"],
                metadata={"method": "minian-bin", "dataset": "org"},
            )
            met_df.extend([metS, metS_bin_scal, metS_bin])
    met_df = pd.concat(met_df, ignore_index=True)
    met_df["thres"] = met_df["thres"].round(5)
    met_df.to_feather(os.path.join(INT_PATH, "metrics.feat"))


# %% plot metrics
if __name__ == "__main__":
    met_df = pd.read_feather(os.path.join(INT_PATH, "metrics.feat"))
    np.random.seed(42)
    plt_uids = np.random.choice(met_df["unit_id"].unique(), 10)
    # ROC plot
    met_sub = met_df[met_df["unit_id"].isin(plt_uids)].astype({"unit_id": str})
    g = sns.FacetGrid(met_sub, row="unit_id", col="dataset", sharex=False, sharey=False)
    g.map_dataframe(plot_ROC)
    g.add_legend()
    g.figure.savefig(os.path.join(FIG_PATH, "ROC.svg"), bbox_inches="tight")
    # corr plot
    met_sub = met_df[met_df["unit_id"].isin(plt_uids)].astype({"unit_id": str})
    g = sns.FacetGrid(met_sub, row="unit_id", col="dataset", sharex=False, sharey=False)
    g.map_dataframe(plot_corr)
    g.add_legend()
    g.figure.savefig(os.path.join(FIG_PATH, "corr.svg"), bbox_inches="tight")
    # ROC scatter plot
    met_sub = met_df[
        np.logical_or(
            met_df["thres"].isin(np.linspace(0.1, 0.9, 9)),
            met_df["method"] == "minian-bin",
        )
    ].astype({"thres": str})
    g = sns.FacetGrid(met_sub, col="dataset", sharex=False, sharey=False)
    g.map_dataframe(plot_ROC_scatter)
    g.add_legend()
    g.figure.savefig(os.path.join(FIG_PATH, "ROC_scatter.svg"), bbox_inches="tight")
    # corr violin plot
    met_sub = met_df[
        np.logical_or(
            met_df["thres"].isin(np.linspace(0.1, 0.9, 9)),
            met_df["method"] == "minian-bin",
        )
    ].astype({"thres": str})
    g = sns.catplot(
        met_sub,
        row="dataset",
        x="thres",
        y="corr",
        hue="method",
        kind="violin",
        aspect=4,
        height=2.5,
        sharey=False,
        sharex=False,
    )
    g.figure.savefig(os.path.join(FIG_PATH, "corr_violin.svg"), bbox_inches="tight")

# %% plot alpha correlations
if __name__ == "__main__":
    met_df = (
        pd.read_feather("./intermediate/benchmark_bin/metrics.feat")
        .sort_values(["method", "dataset"])
        .set_index(["method", "dataset"])
    )
    for up_type, true_ds in IN_PATH.items():
        updt_ds = xr.open_dataset(
            os.path.join(INT_PATH, "updt_ds-{}.nc".format(up_type))
        )
        true_ds = xr.open_dataset(true_ds)
        sig_scal = updt_ds[["sig_lev", "scal"]].to_dataframe()
        sig_scal["S_gt"] = true_ds["S"].mean("frame").to_series()
        sig_scal["S"] = updt_ds["S"].mean("frame").to_series()
        sig_scal["S-bin"] = updt_ds["S-bin"].mean("frame").to_series()
        x_dat = sig_scal.reset_index().melt(
            id_vars=["unit_id"],
            value_vars=["sig_lev", "S_gt"],
            var_name="x_var",
            value_name="x",
        )
        y_dat = sig_scal.reset_index().melt(
            id_vars=["unit_id"],
            value_vars=["scal", "S", "S-bin"],
            var_name="y_var",
            value_name="y",
        )
        scal_df = x_dat.merge(y_dat, on="unit_id")
        g = sns.FacetGrid(scal_df, row="x_var", col="y_var", sharex=False, sharey=False)
        g.map_dataframe(sns.regplot, x="x", y="y", scatter_kws={"s": 10})
        g.figure.savefig(os.path.join(FIG_PATH, "alpha-{}.svg".format(up_type)))


# %% plot examples
if __name__ == "__main__":
    met_df = (
        pd.read_feather("./intermediate/benchmark_bin/metrics.feat")
        .sort_values(["method", "dataset"])
        .set_index(["method", "dataset"])
    )
    for up_type, true_ds in IN_PATH.items():
        updt_ds = xr.open_dataset(
            os.path.join(INT_PATH, "updt_ds-{}.nc".format(up_type))
        )
        true_ds = xr.open_dataset(true_ds)
        nsamp = min(5, updt_ds.sizes["unit_id"])
        C_gt, S_gt = true_ds["C"].dropna("frame", how="all"), true_ds["S"].dropna(
            "frame", how="all"
        ).rename("S_gt")
        Y_solve, S, S_bin, S_bin_scal = (
            updt_ds["Y_solve"].dropna("frame", how="all"),
            updt_ds["S"].dropna("frame", how="all"),
            updt_ds["S-bin"].dropna("frame", how="all"),
            updt_ds["S-bin-scal"].dropna("frame", how="all"),
        )
        if up_type == "upsamp":
            plt_trs = [
                s.coarsen({"frame": PARAM_UPSAMP})
                .sum()
                .assign_coords(frame=Y_solve.coords["frame"])
                .rename("S_th{:.1f}".format(th))
                for s, th in zip(*max_thres(S, 3, return_thres=True))
            ]
            S = (
                S.coarsen({"frame": PARAM_UPSAMP})
                .sum()
                .assign_coords(frame=Y_solve.coords["frame"])
            )
            S_bin = (
                S_bin.coarsen({"frame": PARAM_UPSAMP})
                .sum()
                .assign_coords(frame=Y_solve.coords["frame"])
            )
            S_bin_scal = (
                S_bin_scal.coarsen({"frame": PARAM_UPSAMP})
                .sum()
                .assign_coords(frame=Y_solve.coords["frame"])
            )
        else:
            plt_trs = [
                s.rename("S_th{:.1f}".format(th))
                for s, th in zip(*max_thres(S, 3, return_thres=True))
            ]
        met_sub = met_df.loc["minian-bin", up_type]
        cur_uids = met_sub.sort_values("f1", ascending=False)["unit_id"]
        plt_trs.extend([Y_solve, C_gt, S_gt, S, S_bin, S_bin_scal])
        for met_grp, exp_set in {
            "best": cur_uids[:nsamp],
            "worst": cur_uids[-nsamp:],
            "fair": cur_uids[int(len(cur_uids) / 2) : int(len(cur_uids) / 2) + nsamp],
        }.items():
            plt_dat = pd.concat(
                [
                    # norm_per_cell(tr.sel(unit_id=np.array(exp_set))).to_dataframe()
                    tr.transpose("unit_id", "frame")
                    .sel(unit_id=np.array(exp_set))
                    .to_dataframe()
                    for tr in plt_trs
                ],
                axis="columns",
            ).reset_index()
            plt_dat = plt_dat.melt(id_vars=["frame", "unit_id"]).sort_values(
                ["unit_id", "variable", "frame"]
            )
            fig = px.line(
                plt_dat, facet_row="unit_id", x="frame", y="value", color="variable"
            )
            fig.update_layout(height=nsamp * 200)
            fig.write_html(
                os.path.join(FIG_PATH, "exp-{}-{}.html".format(up_type, met_grp))
            )
