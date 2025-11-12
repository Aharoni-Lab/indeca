import numpy as np
import pandas as pd
from dtw import dtw
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import zscore


def is_boolean(arr):
    if arr.dtype == bool:
        return True
    return np.all((arr == 0) | (arr == 1))


def assignment_distance(
    s_ref: np.ndarray = None,
    s_slv: np.ndarray = None,
    t_ref: np.ndarray = None,
    t_slv: np.ndarray = None,
    samp_ratio: float = None,
    tdist_thres: float = None,
    tdist_agg: str = "median",
    include_range: float = None,
):
    if s_ref is not None:
        s_ref = np.nan_to_num(s_ref)
    if s_slv is not None:
        s_slv = np.nan_to_num(s_slv)
    if t_ref is None:
        assert s_ref is not None
        s_ref = s_ref.astype(float)
        assert is_boolean(s_ref)
        t_ref = np.where(s_ref)[0]
    if t_slv is None:
        assert s_slv is not None
        s_slv = s_slv.astype(float)
        assert is_boolean(s_slv)
        t_slv = np.where(s_slv)[0]
    if samp_ratio is None:
        if s_ref is not None and s_slv is not None:
            samp_ratio = len(s_ref) / len(s_slv)
        else:
            samp_ratio = 1
    t_slv = t_slv * samp_ratio
    if include_range is not None:
        t0, t1 = include_range
        t_ref = t_ref[np.logical_and(t_ref >= t0, t_ref <= t1)]
        t_slv = t_slv[np.logical_and(t_slv >= t0, t_slv <= t1)]
    dist_mat = cdist(t_ref.reshape((-1, 1)), t_slv.reshape((-1, 1)))
    if tdist_thres is not None:
        dist_mat_mask = dist_mat <= tdist_thres
        dist_mat = np.where(dist_mat_mask, dist_mat, dist_mat.max() * 1e16)
        feas_idx_ref = dist_mat_mask.sum(axis=1).astype(bool)
        feas_idx_slv = dist_mat_mask.sum(axis=0).astype(bool)
        dist_mat = dist_mat[feas_idx_ref, :][:, feas_idx_slv]
    idx_ref, idx_slv = linear_sum_assignment(dist_mat)
    tdists = dist_mat[idx_ref, idx_slv]
    if tdist_thres is not None:
        idx_mask = tdists <= tdist_thres
        idx_ref, idx_slv, tdists = (
            idx_ref[idx_mask],
            idx_slv[idx_mask],
            tdists[idx_mask],
        )
    tp = len(idx_ref)
    if len(t_slv) > 0:
        precision = tp / len(t_slv)
    else:
        precision = 0
    if len(t_ref) > 0:
        recall = tp / len(t_ref)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    if len(tdists) > 0:
        if tdist_agg == "median":
            mdist = np.median(tdists)
        elif tdist_agg == "mean":
            mdist = np.mean(tdists)
        else:
            raise NotImplementedError("Aggregation method must be 'median' or 'mean'")
    else:
        mdist = np.nan
    return mdist, f1, precision, recall


def compute_f1_metrics(s_ref, svals, add_met, **kwargs):
    mets = [assignment_distance(s_ref, ss, **kwargs) for ss in svals]
    metdf = pd.DataFrame(
        {
            "mdist": np.array([d[0] for d in mets]),
            "f1": np.array([d[1] for d in mets]),
            "prec": np.array([d[2] for d in mets]),
            "recall": np.array([d[3] for d in mets]),
        }
    )
    for met_name, mets in add_met.items():
        metdf[met_name] = mets
    return metdf


def compute_metrics(
    s_slv,
    s_ref,
    ap_df=None,
    fluo_df=None,
    pre_scaling: bool = True,
    rolling_window: int = 30,
    smoothing_sigma: int = 0.5,
    tdist_thres: float = 5,
    compute_f1: bool = True,
    compute_corr: bool = True,
):
    s_slv = np.nan_to_num(np.array(s_slv)).astype(float)
    s_ref = np.nan_to_num(np.array(s_ref)).astype(float)
    met_dict = dict()
    if compute_f1 and ap_df is not None and s_slv.sum() > 0:
        assert fluo_df is not None
        if pre_scaling:
            scl = max(s_ref.sum() / s_slv.sum(), 1)
            sb = np.around(s_slv * scl)
        else:
            sb = s_slv
        sh_res = []
        for sh_idx in range(-rolling_window, rolling_window):
            sb_sh = np.roll(sb, sh_idx).astype(int)
            sb_idx = nzidx_int(sb_sh)
            if len(sb_idx) > 0:
                t_sb = np.interp(sb_idx, fluo_df["frame"], fluo_df["fluo_time"])
                t_ap = ap_df["ap_time"]
                mdist, f1, prec, rec = assignment_distance(
                    t_ref=np.atleast_1d(t_ap),
                    t_slv=np.atleast_1d(t_sb),
                    tdist_thres=fluo_df["fluo_time"].diff().median() * tdist_thres,
                )
                sh_res.append(
                    {
                        "sh_f1": sh_idx,
                        "mdist": mdist,
                        "f1": f1,
                        "prec": prec,
                        "rec": rec,
                    }
                )
        sh_res = pd.DataFrame(sh_res)
        try:
            opt_idx = sh_res["f1"].argmax()
            f1_met = sh_res.loc[opt_idx].to_dict()
        except KeyError:
            f1_met = dict()
        met_dict = met_dict | f1_met
    if compute_corr and s_slv.sum() > 0:
        corr_met = dict()
        sh_res = []
        s_slv_smth = gaussian_filter1d(s_slv, smoothing_sigma)
        s_ref_smth = gaussian_filter1d(s_ref, smoothing_sigma)
        for sh_idx in range(-rolling_window, rolling_window):
            s_sh = np.roll(s_slv, sh_idx)
            s_sh_smth = np.roll(s_slv_smth, sh_idx)
            craw = np.corrcoef(s_sh, s_ref)[0, 1]
            cgs = np.corrcoef(s_sh_smth, s_ref_smth)[0, 1]
            sh_res.append({"sh": sh_idx, "corr_raw": craw, "corr_gs": cgs})
        sh_res = pd.DataFrame(sh_res)
        opt_idx_raw = sh_res["corr_raw"].argmax()
        opt_idx_smth = sh_res["corr_gs"].argmax()
        opt_sh = sh_res.loc[opt_idx_smth, "sh"]
        s_slv_sh = np.roll(s_slv_smth, opt_sh)
        corr_met = (
            corr_met
            | sh_res.loc[opt_idx_raw, ["sh", "corr_raw"]]
            .rename({"sh": "sh_raw"})
            .to_dict()
        )
        corr_met = (
            corr_met
            | sh_res.loc[opt_idx_smth, ["sh", "corr_gs"]]
            .rename({"sh": "sh_gs"})
            .to_dict()
        )
        corr_met = corr_met | {
            "corr_dtw": dtw_corr(s_ref_smth, s_slv_sh, window_size=tdist_thres)
        }
        met_dict = met_dict | corr_met
    return met_dict


def df_assign_metadata(df, meta_dict):
    for dname, dval in meta_dict.items():
        df[dname] = dval
    return df


def nzidx_int(arr):
    idxs = []
    for idx in np.where(arr)[0]:
        val = arr[idx]
        idxs.extend([idx] * val)
    return idxs


def dtw_corr(s_ref: np.ndarray = None, s_slv: np.ndarray = None, window_size: int = 5):
    if s_ref is not None:
        s_ref = np.nan_to_num(s_ref)
    if s_slv is not None:
        s_slv = np.nan_to_num(s_slv)
    s_ref_z = zscore(s_ref)
    s_slv_z = zscore(s_slv)
    algn = dtw(
        s_slv_z,
        s_ref_z,
        step_pattern="asymmetric",
        window_type="sakoechiba",
        window_args={"window_size": window_size},
    )
    return np.corrcoef(s_ref_z[algn.index2], s_slv_z[algn.index1])[0, 1]
