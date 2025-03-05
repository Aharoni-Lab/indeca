import numpy as np
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
    idx_mask = tdists <= tdist_thres
    idx_ref, idx_slv, tdists = idx_ref[idx_mask], idx_slv[idx_mask], tdists[idx_mask]
    tp = len(idx_ref)
    precision = tp / len(t_slv)
    recall = tp / len(t_ref)
    f1 = 2 * (precision * recall) / (precision + recall)
    if tdist_agg == "median":
        mdist = np.median(tdists)
    elif tdist_agg == "mean":
        mdist = np.mean(tdists)
    else:
        raise NotImplementedError("Aggregation method must be 'median' or 'mean'")
    return mdist, f1, precision, recall
