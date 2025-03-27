import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


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
    dist_mat = cdist(t_ref.reshape((-1, 1)), t_slv.reshape((-1, 1)))
    if tdist_thres is not None:
        dist_mat_mask = dist_mat <= tdist_thres
        dist_mat = np.where(dist_mat_mask, dist_mat, tdist_thres * 1e16)
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
    if tdist_agg == "median":
        mdist = np.median(tdists)
    elif tdist_agg == "mean":
        mdist = np.mean(tdists)
    else:
        raise NotImplementedError("Aggregation method must be 'median' or 'mean'")
    return mdist, f1, precision, recall


def compute_metrics(s_ref, svals, add_met, **kwargs):
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


def df_assign_metadata(df, meta_dict):
    for dname, dval in meta_dict.items():
        df[dname] = dval
    return df
