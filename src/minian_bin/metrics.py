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
    idx_ref, idx_slv = linear_sum_assignment(dist_mat)
    tp = len(idx_ref)
    precision = tp / len(t_slv)
    recall = tp / len(t_ref)
    f1 = 2 * (precision * recall) / (precision + recall)
    med_dist = np.median(dist_mat[idx_ref, idx_slv])
    return med_dist, f1, precision, recall
