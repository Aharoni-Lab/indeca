import itertools as itt
import warnings
from typing import Tuple

import cvxpy as cp
import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sps
import xarray as xr
from scipy.ndimage import label
from scipy.optimize import direct
from scipy.signal import ShortTimeFFT, find_peaks
from scipy.special import huber

from indeca.logging_config import get_module_logger
from indeca.simulation import AR2tau, ar_pulse, exp_pulse, solve_p, tau2AR
from indeca.utils import scal_lstsq

# Initialize logger for this module
logger = get_module_logger("deconv")
logger.info("Deconv module initialized")

try:
    import cuosqp
except ImportError:
    logger.warning("No GPU solver support")


def construct_R(T: int, up_factor: int):
    if up_factor > 1:
        return sps.csc_matrix(
            (
                np.ones(T * up_factor),
                (np.repeat(np.arange(T), up_factor), np.arange(T * up_factor)),
            ),
            shape=(T, T * up_factor),
        )
    else:
        return sps.eye(T, format="csc")


def sum_downsample(a, factor):
    return np.convolve(a, np.ones(factor), mode="full")[factor - 1 :: factor]


def construct_G(fac: np.ndarray, T: int, fromTau=False):
    fac = np.array(fac)
    assert fac.shape == (2,)
    if fromTau:
        fac = np.array(tau2AR(*fac))
    return sps.dia_matrix(
        (
            np.tile(np.concatenate(([1], -fac)), (T, 1)).T,
            -np.arange(len(fac) + 1),
        ),
        shape=(T, T),
    ).tocsc()


def max_thres(
    a: xr.DataArray,
    nthres: int,
    th_min=0.1,
    th_max=0.9,
    ds=None,
    return_thres=False,
    th_amplitude=False,
    delta=1e-6,
    reverse_thres=False,
    nz_only: bool = False,
):
    amax = a.max()
    if reverse_thres:
        thres = np.linspace(th_max, th_min, nthres)
    else:
        thres = np.linspace(th_min, th_max, nthres)
    if th_amplitude:
        S_ls = [np.floor_divide(a, (amax * th).clip(delta, None)) for th in thres]
    else:
        S_ls = [(a > (amax * th).clip(delta, None)) for th in thres]
    if ds is not None:
        S_ls = [sum_downsample(s, ds) for s in S_ls]
    if nz_only:
        Snz = [ss.sum() > 0 for ss in S_ls]
        S_ls = [ss for ss, nz in zip(S_ls, Snz) if nz]
        thres = [th for th, nz in zip(thres, Snz) if nz]
    if return_thres:
        return S_ls, thres
    else:
        return S_ls


class DeconvBin:
    def __init__(
        self,
        y: np.array = None,
        y_len: int = None,
        theta: np.array = None,
        tau: np.array = None,
        ps: np.array = None,
        coef: np.array = None,
        coef_len: int = 100,
        scale: float = 1,
        penal: str = "l1",
        use_base: bool = False,
        upsamp: int = 1,
        norm: str = "l2",
        mixin: bool = False,
        backend: str = "osqp",
        nthres: int = 1000,
        err_weighting: str = None,
        masking_radius: int = None,
        pks_polish: bool = True,
        th_min: float = 0,
        th_max: float = 1,
        max_iter_l0: int = 30,
        max_iter_penal: int = 500,
        max_iter_scal: int = 50,
        delta_l0: float = 1e-4,
        delta_penal: float = 1e-3,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        dashboard=None,
        dashboard_uid=None,
    ) -> None:
        # book-keeping
        if y is not None:
            self.y_len = len(y)
        else:
            assert y_len is not None
            self.y_len = y_len
        if coef_len is not None and coef_len > self.y_len:
            warnings.warn("Coefficient length longer than data")
            coef_len = self.y_len
        if theta is not None:
            self.theta = np.array(theta)
            if tau is None:
                tau_d, tau_r, p = AR2tau(theta[0], theta[1], solve_amp=True)
                self.tau = np.array([tau_d, tau_r])
                self.ps = np.array([p, -p])
                coef, _, _ = exp_pulse(
                    tau_d,
                    tau_r,
                    p_d=p,
                    p_r=-p,
                    nsamp=coef_len * upsamp,
                    kn_len=coef_len * upsamp,
                    trunc_thres=atol,
                )
        if tau is not None:
            assert (
                ps is not None
            ), "exp coefficients must be provided together with time constants."
            if theta is None:
                self.theta = np.array(tau2AR(tau[0], tau[1]))
            self.tau = np.array(tau)
            self.ps = ps
            coef, _, _ = exp_pulse(
                tau[0],
                tau[1],
                p_d=ps[0],
                p_r=ps[1],
                nsamp=coef_len * upsamp,
                kn_len=coef_len * upsamp,
                trunc_thres=atol,
            )
        if coef is None:
            assert coef_len is not None
            coef = np.ones(coef_len * upsamp)
        self.coef_len = len(coef)
        self.T = self.y_len * upsamp
        l0_penal = 0
        l1_penal = 0
        self.free_kernel = False
        self.penal = penal
        self.use_base = use_base
        self.l0_penal = l0_penal
        self.w_org = np.ones(self.T)
        self.w = np.ones(self.T)
        self.upsamp = upsamp
        self.norm = norm
        self.backend = backend
        self.nthres = nthres
        self.th_min = th_min
        self.th_max = th_max
        self.max_iter_l0 = max_iter_l0
        self.max_iter_penal = max_iter_penal
        self.max_iter_scal = max_iter_scal
        self.delta_l0 = delta_l0
        self.delta_penal = delta_penal
        self.atol = atol
        self.rtol = rtol
        self.dashboard = dashboard
        self.dashboard_uid = dashboard_uid
        self.nzidx_s = np.arange(self.T)
        self.nzidx_c = np.arange(self.T)
        self.x_cache = None
        self.err_weighting = err_weighting
        self.masking_r = masking_radius
        self.pks_polish = pks_polish
        self.err_wt = np.ones(self.y_len)
        if err_weighting == "fft":
            self.stft = ShortTimeFFT(win=np.ones(self.coef_len), hop=1, fs=1)
            self.yspec = self._get_stft_spec(y)
        if y is not None:
            self.huber_k = 0.5 * np.std(y)
            self.err_total = self._res_err(y)
        else:
            self.huber_k = 0
            self.err_total = 0
        self._update_R()
        # setup cvxpy
        if self.backend == "cvxpy":
            self.R = cp.Constant(self.R, name="R")
            self.c = cp.Variable((self.T, 1), nonneg=True, name="c")
            self.s = cp.Variable((self.T, 1), nonneg=True, name="s", boolean=mixin)
            self.y = cp.Parameter(shape=(self.y_len, 1), name="y")
            self.coef = cp.Parameter(value=coef, shape=self.coef_len, name="coef")
            self.scale = cp.Parameter(value=scale, name="scale", nonneg=True)
            self.l1_penal = cp.Parameter(value=l1_penal, name="l1_penal", nonneg=True)
            self.l0_w = cp.Parameter(
                shape=self.T, value=self.l0_penal * self.w, nonneg=True, name="w_l0"
            )  # product of l0_penal * w!
            if y is not None:
                self.y.value = y.reshape((-1, 1))
            if coef is not None:
                self.coef.value = coef
            if use_base:
                self.b = cp.Variable(nonneg=True, name="b")
            else:
                self.b = cp.Constant(value=0, name="b")
            if norm == "l1":
                self.err_term = cp.sum(
                    cp.abs(self.y - self.scale * self.R @ self.c - self.b)
                )
            elif norm == "l2":
                self.err_term = cp.sum_squares(
                    self.y - self.scale * self.R @ self.c - self.b
                )
            elif norm == "huber":
                self.err_term = cp.sum(
                    cp.huber(self.y - self.scale * self.R @ self.c - self.b)
                )
            obj = cp.Minimize(
                self.err_term
                + self.l0_w.T @ cp.abs(self.s)
                + self.l1_penal * cp.sum(cp.abs(self.s))
            )
            if self.free_kernel:
                dcv_cons = [
                    self.c[:, 0] == cp.convolve(self.coef, self.s[:, 0])[: self.T]
                ]
            else:
                self.theta = cp.Parameter(
                    value=self.theta, shape=self.theta.shape, name="theta"
                )
                G_diag = sps.eye(self.T - 1) + sum(
                    [
                        cp.diag(cp.promote(-self.theta[i], (self.T - i - 2,)), -i - 1)
                        for i in range(self.theta.shape[0])
                    ]
                )  # diag part of unshifted G
                G = cp.bmat(
                    [
                        [np.zeros((self.T - 1, 1)), G_diag],
                        [np.zeros((1, 1)), np.zeros((1, self.T - 1))],
                    ]
                )
                dcv_cons = [self.s == G @ self.c]
            edge_cons = [self.c[0, 0] == 0, self.s[-1, 0] == 0]
            amp_cons = [self.s <= 1]
            self.prob_free = cp.Problem(obj, dcv_cons + edge_cons)
            self.prob = cp.Problem(obj, dcv_cons + edge_cons + amp_cons)
            self._update_HG()  # self.H and self.G not used for cvxpy problems
        elif self.backend in ["osqp", "emosqp", "cuosqp"]:
            # book-keeping
            if y is None:
                self.y = np.ones(self.y_len)
            else:
                self.y = y
            if coef is None:
                self.coef = np.ones(self.coef_len)
            else:
                self.coef = coef
            self.c = np.zeros(self.T * upsamp)
            self.s = np.zeros(self.T * upsamp)
            self.s_bin = None
            self.b = 0
            self.l1_penal = l1_penal
            self.scale = scale
            self._update_Wt()
            self._setup_prob_osqp()
        if self.dashboard is not None:
            self.dashboard.update(
                h=self.coef.value if backend == "cvxpy" else self.coef,
                uid=self.dashboard_uid,
            )
        tr_exp, _, _ = exp_pulse(
            self.tau[0],
            self.tau[1],
            p_d=self.ps[0],
            p_r=self.ps[1],
            nsamp=self.coef_len,
        )
        theta = self.theta.value if self.backend == "cvxpy" else self.theta
        tr_ar, _, _ = ar_pulse(theta[0], theta[1], nsamp=self.coef_len, shifted=True)
        assert (~np.isnan(coef)).all()
        assert np.isclose(
            tr_exp, coef, atol=self.atol
        ).all(), "exp time constant inconsistent"
        assert np.isclose(
            tr_ar, coef, atol=self.atol
        ).all(), "ar coefficients inconsistent"

    def update(
        self,
        y: np.ndarray = None,
        tau: np.ndarray = None,
        coef: np.ndarray = None,
        scale: float = None,
        scale_mul: float = None,
        l0_penal: float = None,
        l1_penal: float = None,
        w: np.ndarray = None,
        update_weighting: bool = False,
        clear_weighting: bool = False,
        scale_coef: bool = False,
    ) -> None:
        logger.debug(
            f"Updating parameters - backend: {self.backend}, tau: {tau}, scale: {scale}, scale_mul: {scale_mul}, l0_penal: {l0_penal}, l1_penal: {l1_penal}"
        )
        if self.backend == "cvxpy":
            if y is not None:
                self.y.value = y
            if tau is not None:
                theta_new = np.array(tau2AR(tau[0], tau[1]))
                p = solve_p(tau[0], tau[1])
                coef, _, _ = exp_pulse(
                    tau[0],
                    tau[1],
                    p_d=p,
                    p_r=-p,
                    nsamp=self.coef_len,
                    kn_len=self.coef_len,
                )
                self.coef.value = coef
                self.theta.value = theta_new
                self._update_HG()
            if coef is not None:
                if scale_coef:
                    scale_mul = scal_lstsq(coef, self.coef).item()
                self.coef.value = coef
                self._update_HG()
            if scale is not None:
                self.scale.value = scale
            if scale_mul is not None:
                self.scale.value = scale_mul * self.scale.value
            if l1_penal is not None:
                self.l1_penal.value = l1_penal
            if l0_penal is not None:
                self.l0_penal = l0_penal
            if w is not None:
                self._update_w(w)
            if l0_penal is not None or w is not None:
                self.l0_w.value = self.l0_penal * self.w
        elif self.backend in ["osqp", "emosqp", "cuosqp"]:
            # update input params
            if y is not None:
                self.y = y
            if tau is not None:
                theta_new = np.array(tau2AR(tau[0], tau[1]))
                p = solve_p(tau[0], tau[1])
                coef, _, _ = exp_pulse(
                    tau[0],
                    tau[1],
                    p_d=p,
                    p_r=-p,
                    nsamp=self.coef_len,
                    kn_len=self.coef_len,
                )
                self.tau = tau
                self.ps = np.array([p, -p])
                self.theta = theta_new
            if coef is not None:
                if scale_coef:
                    scale_mul = scal_lstsq(coef, self.coef).item()
                self.coef = coef
            if scale is not None:
                self.scale = scale
            if scale_mul is not None:
                self.scale = scale_mul * self.scale
            if l1_penal is not None:
                self.l1_penal = l1_penal
            if l0_penal is not None:
                self.l0_penal = l0_penal
            if w is not None:
                self._update_w(w)
            # update internal variables
            updt_HG, updt_P, updt_A, updt_q0, updt_q, updt_bounds, setup_prob = [
                False
            ] * 7
            if coef is not None:
                self._update_HG()
                updt_HG = True
            if self.err_weighting is not None and update_weighting:
                self._update_Wt(clear=clear_weighting)
                if self.err_weighting == "adaptive":
                    setup_prob = True
                else:
                    updt_P = True
                    updt_q0 = True
                    updt_q = True
            if self.norm == "huber":
                if any((scale is not None, scale_mul is not None, updt_HG)):
                    self._update_A()
                    updt_A = True
                if any(
                    (w is not None, l0_penal is not None, l1_penal is not None, updt_HG)
                ):
                    self._update_q()
                    updt_q = True
                if y is not None:
                    self._update_bounds()
                    updt_bounds = True
            else:
                if any((updt_HG, updt_A)):
                    A_before = self.A.copy()
                    self._update_A()
                    assert self.A.shape == A_before.shape
                    assert (self.A.nonzero()[0] == A_before.nonzero()[0]).all()
                    assert (self.A.nonzero()[1] == A_before.nonzero()[1]).all()
                    updt_A = True
                if any((scale is not None, scale_mul is not None, updt_HG, updt_P)):
                    P_before = self.P.copy()
                    self._update_P()
                    assert self.P.shape == P_before.shape
                    assert (self.P.nonzero()[0] == P_before.nonzero()[0]).all()
                    assert (self.P.nonzero()[1] == P_before.nonzero()[1]).all()
                    updt_P = True
                if any(
                    (
                        scale is not None,
                        scale_mul is not None,
                        y is not None,
                        updt_HG,
                        updt_q0,
                    )
                ):
                    q0_before = self.q0.copy()
                    self._update_q0()
                    assert self.q0.shape == q0_before.shape
                    updt_q0 = True
                if any(
                    (
                        w is not None,
                        l0_penal is not None,
                        l1_penal is not None,
                        updt_q0,
                        updt_q,
                    )
                ):
                    q_before = self.q.copy()
                    self._update_q()
                    assert self.q.shape == q_before.shape
                    updt_q = True
            # update prob
            logger.debug(f"Updating optimization problem with {self.backend}")
            if self.backend == "emosqp":
                if updt_P:
                    self.prob_free.update_P(self.P.data, None, 0)
                    self.prob.update_P(self.P.data, None, 0)
                if updt_q:
                    self.prob_free.update_lin_cost(self.q)
                    self.prob.update_lin_cost(self.q)
            elif self.backend in ["osqp", "cuosqp"] and any(
                (updt_P, updt_q, updt_A, updt_bounds, setup_prob)
            ):
                if setup_prob:
                    self._setup_prob_osqp()
                else:
                    self.prob_free.update(
                        Px=self.P.copy().data if updt_P else None,
                        q=self.q.copy() if updt_q else None,
                        Ax=self.A.copy().data if updt_A else None,
                        l=self.lb.copy() if updt_bounds else None,
                        u=self.ub_inf.copy() if updt_bounds else None,
                    )
                    self.prob.update(
                        Px=self.P.copy().data if updt_P else None,
                        q=self.q.copy() if updt_q else None,
                        Ax=self.A.copy().data if updt_A else None,
                        l=self.lb.copy() if updt_bounds else None,
                        u=self.ub.copy() if updt_bounds else None,
                    )
            logger.debug("Optimization problem updated")

    def _cut_pks_labs(self, s, labs, pks):
        pk_labs = np.full_like(labs, -1)
        lb = 0
        for ilab in range(labs.max() + 1):
            lb_idxs = np.where(labs == ilab)[0]
            cur_pks = [p for p in pks if p in lb_idxs]
            if len(cur_pks) > 1:
                p_start = lb_idxs[0]
                for p0, p1 in zip(cur_pks[:-1], cur_pks[1:]):
                    p_stop = p0 + np.argmin(s[p0:p1]).item()
                    pk_labs[p_start:p_stop] = lb
                    lb += 1
                    p_start = p_stop
                pk_labs[p_stop : lb_idxs[-1] + 1] = lb
                lb += 1
            else:
                pk_labs[lb_idxs] = lb
                lb += 1
        return pk_labs

    def _merge_sparse_regs(
        self, s, regs, err_rtol=0, max_len=9, constraint_sum: bool = True
    ):
        s_ret = s.copy()
        for r in range(regs.max() + 1):
            ridx = np.where(regs == r)[0]
            ridx = sorted(list(set(ridx).intersection(set(self.nzidx_s))))
            rlen = len(ridx)
            rsum = s[ridx].sum()
            ns_min = max(int(np.around(rsum)), 1)
            if rlen > max_len or ns_min > rlen or rlen <= 1:
                continue
            s_new = s_ret.copy()
            s_new[ridx] = 0
            err_before = self._compute_err(s=s_ret[self.nzidx_s])
            err_ls = []
            idx_ls = []
            if constraint_sum:
                ns_vals = [ns_min]
            else:
                ns_vals = list(range(ns_min, rlen + 1))
            for ns in ns_vals:
                for idxs in itt.combinations(ridx, ns):
                    idxs = np.array(idxs)
                    s_test = s_new.copy()
                    s_test[idxs] = rsum / ns
                    err_after = self._compute_err(s=s_test[self.nzidx_s])
                    err_ls.append(err_after)
                    idx_ls.append(idxs)
            err_min_idx = np.argmin(err_ls)
            err_min = err_ls[err_min_idx]
            if err_min - err_before < err_rtol * err_before:
                idx_min = idx_ls[err_min_idx]
                s_new[idx_min] = rsum / len(idx_min)
                s_ret = s_new
        return s_ret

    def _pad_s(self, s=None):
        if s is None:
            s = self.s
        s_ret = np.zeros(self.T)
        s_ret[self.nzidx_s] = s
        return s_ret

    def _pad_c(self, c=None):
        if c is None:
            c = self.s
        c_ret = np.zeros(self.T)
        c_ret[self.nzidx_c] = c
        return c_ret

    def solve(
        self,
        amp_constraint: bool = True,
        update_cache: bool = False,
        pks_polish: bool = None,
        pks_delta: float = 1e-5,
        pks_err_rtol: float = 10,
        pks_cut: bool = False,
    ) -> np.ndarray:
        if self.l0_penal == 0:
            opt_s, opt_b = self._solve(
                amp_constraint=amp_constraint, update_cache=update_cache
            )
        else:
            metric_df = None
            for i in range(self.max_iter_l0):
                cur_s, cur_obj = self._solve(amp_constraint, return_obj=True)
                if metric_df is None:
                    obj_best = np.inf
                    obj_last = np.inf
                else:
                    obj_best = metric_df["obj"][1:].min()
                    obj_last = np.array(metric_df["obj"])[-1]
                opt_s = np.where(cur_s > self.delta_l0, cur_s, 0)
                obj_gap = np.abs(cur_obj - obj_best)
                obj_delta = np.abs(cur_obj - obj_last)
                cur_met = pd.DataFrame(
                    [
                        {
                            "iter": i,
                            "obj": cur_obj,
                            "nnz": (opt_s > 0).sum(),
                            "obj_gap": obj_gap,
                            "obj_delta": obj_delta,
                        }
                    ]
                )
                metric_df = pd.concat([metric_df, cur_met], ignore_index=True)
                if any((obj_gap < self.rtol * np.obj_best, obj_delta < self.atol)):
                    break
                else:
                    self.update(
                        w=np.clip(
                            np.ones(self.T) / (self.delta_l0 * np.ones(self.T) + opt_s),
                            0,
                            1e5,
                        )
                    )  # clip to avoid numerical issues
            else:
                warnings.warn(
                    "l0 heuristic did not converge in {} iterations".format(
                        self.max_iter_l0
                    )
                )
        self.b = opt_b
        if pks_polish is None:
            pks_polish = amp_constraint
        if pks_polish and self.backend != "cvxpy":
            s_pad = self._pad_s(s=opt_s)
            s_ft = np.where(s_pad > pks_delta, s_pad, 0)
            labs, _ = label(s_ft)
            labs = labs - 1
            if pks_cut:
                pks_idx, _ = find_peaks(s_ft)
                labs = self._cut_pks_labs(s=s_ft, labs=labs, pks=pks_idx)
            opt_s = self._merge_sparse_regs(s=s_ft, regs=labs, err_rtol=pks_err_rtol)
            opt_s = opt_s[self.nzidx_s]
        self.s = np.abs(opt_s)
        return self.s, self.b

    def solve_thres(
        self,
        scaling: bool = True,
        amp_constraint: bool = True,
        ignore_res: bool = False,
        return_intm: bool = False,
        pks_polish: bool = None,
        obj_crit: str = None,
    ) -> Tuple[np.ndarray]:
        if self.backend == "cvxpy":
            y = np.array(self.y.value.squeeze())
        elif self.backend in ["osqp", "emosqp", "cuosqp"]:
            y = np.array(self.y)
        opt_s, opt_b = self.solve(amp_constraint=amp_constraint, pks_polish=pks_polish)
        R = self.R.value if self.backend == "cvxpy" else self.R
        if ignore_res:
            res = y - opt_b - self.scale * R @ self._compute_c(opt_s)
        else:
            res = np.zeros_like(y)
        svals, thres = max_thres(
            opt_s,
            self.nthres,
            th_min=self.th_min,
            th_max=self.th_max,
            reverse_thres=True,
            return_thres=True,
            nz_only=True,
        )
        if not len(svals) > 0:
            if return_intm:
                svals, thres = max_thres(
                    opt_s,
                    self.nthres,
                    th_min=self.th_min,
                    th_max=self.th_max,
                    reverse_thres=True,
                    return_thres=True,
                )
            else:
                return (
                    np.full(len(self.nzidx_s), np.nan),
                    np.full(len(self.nzidx_c), np.nan),
                    0,
                    np.inf,
                )
        cvals = [self._compute_c(s) for s in svals]
        yfvals = [np.array((R @ c).todense()).squeeze() for c in cvals]
        if scaling:
            scal_fit = [scal_lstsq(yf, y - res, fit_intercept=True) for yf in yfvals]
            scals = [sf[0] for sf in scal_fit]
            bs = [sf[1] for sf in scal_fit]
        else:
            scals = [self.scale] * len(yfvals)
            bs = [(y - res - scl * yf).mean() for scl, yf in zip(scals, yfvals)]
        objs = [
            self._compute_err(s=ss, y_fit=scl * yf, res=res, b=bb, obj_crit=obj_crit)
            for ss, scl, yf, bb in zip(svals, scals, yfvals, bs)
        ]
        scals = np.array(scals).clip(0, None)
        objs = np.where(scals > 0, objs, np.inf)
        if obj_crit == "spk_diff":
            err_null = self._compute_err(
                s=np.zeros_like(opt_s), res=res, b=opt_b, obj_crit=obj_crit
            )
            objs_pad = np.array([err_null, *objs])
            nspk = np.array([0] + [(ss > 0).sum() for ss in svals])
            objs_diff = np.diff(objs_pad)
            nspk_diff = np.diff(nspk)
            merr_diff = objs_diff / nspk_diff
            avg_err = (objs_pad.min() - err_null) / nspk.max()
            opt_idx = np.max(np.where(merr_diff < avg_err))
            objs = merr_diff
        else:
            opt_idx = np.argmin(objs)
        s_bin = svals[opt_idx]
        self.s_bin = s_bin
        assert len(s_bin) == len(self.nzidx_s)
        self.c_bin = np.array(cvals[opt_idx].todense()).squeeze()
        self.b = bs[opt_idx]
        if return_intm:
            return (
                self.s_bin,
                self.c_bin,
                scals[opt_idx],
                objs[opt_idx],
                (opt_s, thres, svals, cvals, yfvals, scals, objs, opt_idx),
            )
        else:
            return self.s_bin, self.c_bin, scals[opt_idx], objs[opt_idx]

    def solve_penal(
        self, masking=True, scaling=True, return_intm=False, pks_polish=None
    ) -> Tuple[np.ndarray]:
        if self.penal is None:
            opt_s, opt_c, opt_scl, opt_obj = self.solve_thres(
                scaling=scaling, return_intm=return_intm, pks_polish=pks_polish
            )
            opt_penal = 0
        elif self.penal in ["l0", "l1"]:
            pn = "{}_penal".format(self.penal)
            self.update(**{pn: 0})
            if masking:
                self._reset_cache()
                self._update_mask()
            s_nopn, _, _, err_nopn, intm = self.solve_thres(
                scaling=scaling, return_intm=True, pks_polish=pks_polish
            )
            s_min = intm[0]
            ymean = self.y.mean()
            err_full = self._compute_err(s=np.zeros(len(self.nzidx_s)), b=ymean)
            err_min = self._compute_err(s=s_min)
            ub, ub_last = err_full, err_full
            for _ in range(int(np.ceil(np.log2(ub)))):
                self.update(**{pn: ub})
                s, b = self.solve(pks_polish=pks_polish)
                cur_err = self._compute_err(s=s, b=b)
                # DIRECT finds weird solutions with high penalty and baseline,
                # so we want to eliminate those possibilities
                if np.abs(cur_err - err_min) < 0.5 * np.abs(err_full - err_min):
                    ub = ub_last
                    break
                else:
                    ub_last = ub
                    ub = ub / 2

            def opt_fn(x):
                self.update(**{pn: x.item()})
                _, _, _, obj = self.solve_thres(scaling=False, pks_polish=pks_polish)
                if self.dashboard is not None:
                    self.dashboard.update(
                        uid=self.dashboard_uid,
                        penal_err={"penal": x.item(), "scale": self.scale, "err": obj},
                    )
                if obj < err_full:
                    return obj
                else:
                    return np.inf

            res = direct(
                opt_fn,
                bounds=[(0, ub)],
                maxfun=self.max_iter_penal,
                locally_biased=False,
                vol_tol=1e-2,
            )
            direct_pn = res.x
            if not res.success:
                logger.warning(
                    "could not find optimal penalty within {} iterations".format(
                        res.nfev
                    )
                )
                opt_penal = 0
            elif err_nopn <= opt_fn(direct_pn):
                # DIRECT seem to mistakenly report high penalty when 0 penalty attains better error
                opt_penal = 0
            else:
                opt_penal = direct_pn.item()
            self.update(**{pn: opt_penal})
            if return_intm:
                opt_s, opt_c, opt_scl, opt_obj, intm = self.solve_thres(
                    scaling=scaling, return_intm=return_intm, pks_polish=pks_polish
                )
            else:
                opt_s, opt_c, opt_scl, opt_obj = self.solve_thres(
                    scaling=scaling, return_intm=return_intm, pks_polish=pks_polish
                )
            if opt_scl == 0:
                logger.warning("could not find non-zero solution")
        if return_intm:
            return opt_s, opt_c, opt_scl, opt_obj, opt_penal, intm
        else:
            return opt_s, opt_c, opt_scl, opt_obj, opt_penal

    def solve_scale(
        self,
        reset_scale: bool = True,
        concur_penal: bool = False,
        return_met: bool = False,
        obj_crit: str = None,
        early_stop: bool = True,
        masking: bool = True,
    ) -> Tuple[np.ndarray]:
        if self.penal in ["l0", "l1"]:
            pn = "{}_penal".format(self.penal)
            self.update(**{pn: 0})
        self._reset_cache()
        self._reset_mask()
        if reset_scale:
            self.update(scale=1)
            s_free, _ = self.solve(amp_constraint=False)
            self.update(scale=np.ptp(s_free))
        else:
            s_free = np.zeros(len(self.nzidx_s))
        metric_df = None
        for i in range(self.max_iter_scal):
            if concur_penal:
                cur_s, cur_c, cur_scl, cur_obj_raw, cur_penal = self.solve_penal(
                    scaling=i > 0,
                    pks_polish=self.pks_polish and (i > 1 or not reset_scale),
                )
            else:
                cur_penal = 0
                cur_s, cur_c, cur_scl, cur_obj_raw = self.solve_thres(
                    scaling=i > 0,
                    pks_polish=self.pks_polish and (i > 1 or not reset_scale),
                    obj_crit=obj_crit,
                )
            if self.dashboard is not None:
                pad_s = np.zeros(self.T)
                pad_s[self.nzidx_s] = cur_s
                self.dashboard.update(
                    uid=self.dashboard_uid,
                    c=self.R @ cur_c,
                    s=self.R_org @ pad_s,
                    scale=cur_scl,
                )
            if metric_df is None:
                prev_scals = np.array([np.inf])
                opt_obj = np.inf
                opt_scal = np.inf
                last_obj = np.inf
                last_scal = np.inf
            else:
                opt_idx = metric_df["obj"].idxmin()
                opt_obj = metric_df.loc[opt_idx, "obj"].item()
                opt_scal = metric_df.loc[opt_idx, "scale"].item()
                prev_scals = np.array(metric_df["scale"])
                last_scal = prev_scals[-1]
                last_obj = np.array(metric_df["obj"])[-1]
            y_wt = np.array(self.y * self.err_wt)
            err_tt = self._res_err(y_wt - y_wt.mean())
            cur_obj = (cur_obj_raw - err_tt) / err_tt
            cur_met = pd.DataFrame(
                [
                    {
                        "iter": i,
                        "scale": cur_scl,
                        "obj_raw": cur_obj_raw,
                        "obj": cur_obj,
                        "penal": cur_penal,
                        "nnz": (cur_s > 0).sum(),
                    }
                ]
            )
            metric_df = pd.concat([metric_df, cur_met], ignore_index=True)
            if self.err_weighting == "adaptive" and i <= 1:
                self.update(update_weighting=True)
                if masking:
                    self._update_mask(use_wt=self.masking_r is None)
            if any(
                (
                    np.abs(cur_scl - opt_scal) < self.rtol * opt_scal,
                    np.abs(cur_obj - opt_obj) < self.rtol * opt_obj,
                    np.abs(cur_scl - last_scal) < self.atol,
                    np.abs(cur_obj - last_obj) < self.atol * 1e-3,
                    early_stop and cur_obj > last_obj,
                )
            ):
                break
            elif cur_scl == 0:
                warnings.warn("exit with zero solution")
                break
            elif np.abs(cur_scl - prev_scals).min() < self.atol:
                self.update(scale=(cur_scl + last_scal) / 2)
            else:
                self.update(scale=cur_scl)
        else:
            warnings.warn("max scale iterations reached")
        opt_idx = metric_df["obj"].idxmin()
        self.update(update_weighting=True, clear_weighting=True)
        self._reset_cache()
        self._reset_mask()
        self.update(scale=metric_df.loc[opt_idx, "scale"])
        cur_s, cur_c, cur_scl, cur_obj, cur_penal = self.solve_penal(
            scaling=False, masking=False, pks_polish=self.pks_polish
        )
        opt_s, opt_c = np.zeros(self.T), np.zeros(self.T)
        opt_s[self.nzidx_s] = cur_s
        opt_c[self.nzidx_c] = cur_c
        nnz = int(opt_s.sum())
        self.update(update_weighting=True)
        y_wt = np.array(self.y * self.err_wt)
        err_tt = self._res_err(y_wt - y_wt.mean())
        err_cur = self._compute_err(s=opt_s)
        err_rel = (err_cur - err_tt) / err_tt
        self.update(update_weighting=True, clear_weighting=True)
        if self.dashboard is not None:
            self.dashboard.update(
                uid=self.dashboard_uid,
                c=self.R_org @ opt_c,
                s=self.R_org @ opt_s,
                scale=cur_scl,
            )
        self._reset_cache()
        self._reset_mask()
        if return_met:
            return opt_s, opt_c, cur_scl, cur_obj, err_rel, nnz, cur_penal, metric_df
        else:
            return opt_s, opt_c, cur_scl, cur_obj, err_rel, nnz, cur_penal

    def _setup_prob_osqp(self) -> None:
        logger.debug("Setting up OSQP problem")
        self._update_HG()
        self._update_P()
        self._update_q0()
        self._update_q()
        self._update_A()
        self._update_bounds()
        if self.backend == "emosqp":
            m = osqp.OSQP()
            m.setup(
                P=self.P,
                q=self.q,
                A=self.A,
                l=self.lb,
                u=self.ub_inf,
                check_termination=25,
                eps_abs=self.atol * 1e-4,
                eps_rel=1e-8,
            )
            m.codegen(
                "osqp-codegen-prob_free",
                parameters="matrices",
                python_ext_name="emosqp_free",
                force_rewrite=True,
            )
            m.update(u=self.ub)
            m.codegen(
                "osqp-codegen-prob",
                parameters="matrices",
                python_ext_name="emosqp",
                force_rewrite=True,
            )
            import emosqp
            import emosqp_free

            self.prob_free = emosqp_free
            self.prob = emosqp
        elif self.backend in ["osqp", "cuosqp"]:
            if self.backend == "osqp":
                self.prob_free = osqp.OSQP()
                self.prob = osqp.OSQP()
            elif self.backend == "cuosqp":
                self.prob_free = cuosqp.OSQP()
                self.prob = cuosqp.OSQP()
            P_copy = self.P.copy()
            q_copy = self.q.copy()
            A_copy = self.A.copy()
            lb_copy = self.lb.copy()
            ub_inf_copy = self.ub_inf.copy()
            self.prob_free.setup(
                P=P_copy,
                q=q_copy,
                A=A_copy,
                l=lb_copy,
                u=ub_inf_copy,
                verbose=False,
                polish=True,
                warm_start=False,
                # adaptive_rho=False,
                eps_abs=1e-6,
                eps_rel=1e-6,
                eps_prim_inf=1e-7,
                eps_dual_inf=1e-7,
                # max_iter=int(1e5) if self.backend == "osqp" else None,
                # eps_prim_inf=1e-8,
            )
            P_copy = self.P.copy()
            q_copy = self.q.copy()
            A_copy = self.A.copy()
            lb_copy = self.lb.copy()
            ub_copy = self.ub.copy()
            self.prob.setup(
                P=P_copy,
                q=q_copy,
                A=A_copy,
                l=lb_copy,
                u=ub_copy,
                verbose=False,
                polish=True,
                warm_start=False,
                # adaptive_rho=False,
                eps_abs=1e-6,
                eps_rel=1e-6,
                eps_prim_inf=1e-7,
                eps_dual_inf=1e-7,
                # max_iter=int(1e5) if self.backend == "osqp" else None,
                # eps_prim_inf=1e-8,
            )
        logger.debug(f"{self.backend} setup completed successfully")

    def _solve(
        self,
        amp_constraint: bool = True,
        return_obj: bool = False,
        update_cache: bool = False,
    ) -> np.ndarray:
        if amp_constraint:
            prob = self.prob
        else:
            prob = self.prob_free
        # if self.backend in ["osqp", "emosqp", "cuosqp"] and self.x_cache is not None:
        #     prob.warm_start(x=self.x_cache)
        res = prob.solve()
        if self.backend == "cvxpy":
            opt_s = self.s.value.squeeze()
            opt_b = 0
        elif self.backend in ["osqp", "emosqp", "cuosqp"]:
            x = res[0] if self.backend == "emosqp" else res.x
            if res.info.status not in ["solved", "solved inaccurate"]:
                warnings.warn("Problem not solved. status: {}".format(res.info.status))
                # osqp mistakenly report primal infeasibility when using masks
                # with high l1 penalty. manually set solution to zero in such cases
                if res.info.status in [
                    "primal infeasible",
                    "primal infeasible inaccurate",
                ]:
                    x = np.zeros_like(x, dtype=float)
                else:
                    x = x.astype(float)
            # if update_cache:
            #     self.x_cache = x
            #     prob.warm_start(x=x)
            if self.norm == "huber":
                xlen = len(self.nzidx_s) if self.free_kernel else len(self.nzidx_c)
                sol = x[:xlen]
            else:
                sol = x
            opt_b = sol[0]
            if self.free_kernel:
                opt_s = sol[1:]
            else:
                opt_s = self.G @ sol[1:]
        if return_obj:
            if self.backend == "cvxpy":
                opt_obj = res
            elif self.backend in ["osqp", "emosqp", "cuosqp"]:
                opt_obj = self._compute_err()
            return opt_s, opt_b, opt_obj
        else:
            return opt_s, opt_b

    def _compute_c(self, s: np.ndarray = None) -> np.ndarray:
        if s is not None:
            return self.H @ sps.csc_matrix(s.reshape(-1, 1))
        else:
            if self.backend == "cvxpy":
                return self.c.value.squeeze()
            elif self.backend in ["osqp", "emosqp", "cuosqp"]:
                return self.H @ self.s

    def _compute_err(
        self,
        y_fit: np.ndarray = None,
        b: np.ndarray = None,
        c: np.ndarray = None,
        s: np.ndarray = None,
        res: np.ndarray = None,
        obj_crit: str = None,
    ) -> float:
        if self.backend == "cvxpy":
            # TODO: add support
            raise NotImplementedError
        elif self.backend in ["osqp", "emosqp", "cuosqp"]:
            y = np.array(self.y)
        if res is not None:
            y = y - res
        if b is None:
            b = self.b
        y = y - b
        if y_fit is None:
            if c is None:
                c = self._compute_c(s)
            y_fit = np.array((self.R @ c * self.scale).todense()).squeeze()
        r = y - y_fit
        err = self._res_err(r)
        if obj_crit in [None, "spk_diff"]:
            return np.array(err).item()
        else:
            nspk = (s > 0).sum()
            if obj_crit == "mean_spk":
                err_total = self._res_err(y - y.mean())
                return np.array((err - err_total) / nspk).item()
            elif obj_crit in ["aic", "bic"]:
                noise_model = "normal"
                T = len(r)
                if noise_model == "normal":
                    mu = r.mean()
                    sigma = ((r - mu) ** 2).sum() / T
                    logL = -0.5 * (
                        T * np.log(2 * np.pi * sigma)
                        + 1 / sigma * ((r - mu) ** 2).sum()
                    )
                elif noise_model == "lognormal":
                    ymin = y.min()
                    logy = np.log(y - ymin + 1)
                    logy_hat = np.log(y_fit - ymin + 1)
                    logr = logy - logy_hat
                    mu = np.mean(logr)
                    sigma = ((logr - mu) ** 2).sum() / T
                    logL = np.sum(
                        -logy
                        - (logr - mu) ** 2 / (2 * sigma)
                        - 0.5 * np.log(2 * np.pi * sigma)
                    )
                if obj_crit == "aic":
                    return np.array(2 * (nspk - logL)).item()
                elif obj_crit == "bic":
                    return np.array(nspk * np.log(T) - 2 * logL).item()
            else:
                raise ValueError("invalid objective criterion: {}".format(obj_crit))

    def _res_err(self, r: np.ndarray):
        if self.err_wt is not None:
            r = self.err_wt * r
        if self.norm == "l1":
            return np.sum(np.abs(r))
        elif self.norm == "l2":
            return np.sum((r) ** 2)
        elif self.norm == "huber":
            err_hub = huber(self.huber_k, r)
            err_qud = r**2 / 2
            return np.sum(np.where(r >= 0, err_hub, err_qud))

    def _reset_cache(self) -> None:
        self.x_cache = None

    def _reset_mask(self) -> None:
        self.nzidx_s = np.arange(self.T)
        self.nzidx_c = np.arange(self.T)
        self._update_R()
        self._update_w()
        if self.backend in ["osqp", "emosqp", "cuosqp"]:
            self._setup_prob_osqp()

    def _update_mask(self, use_wt: bool = False, amp_constraint: bool = True) -> None:
        if self.backend in ["osqp", "emosqp", "cuosqp"]:
            if use_wt:
                nzidx_s = np.where(self.R.T @ self.err_wt)[0]
            else:
                if self.masking_r is not None:
                    mask = np.zeros(self.T)
                    for nzidx in np.where(self._pad_s(self.s_bin) > 0)[0]:
                        mask[
                            max(nzidx - self.masking_r, 0) : min(
                                nzidx + self.masking_r, self.T
                            )
                        ] = 1
                    nzidx_s = np.where(mask)[0]
                else:
                    self._reset_mask()
                    opt_s, _ = self.solve(amp_constraint)
                    nzidx_s = np.where(opt_s > self.delta_penal)[0]
            if len(nzidx_s) == 0:
                raise ValueError
            self.nzidx_s = nzidx_s
            self._update_R()
            self._update_w()
            self._setup_prob_osqp()
            if not self.free_kernel and len(self.nzidx_c) < self.T:
                res = self.prob.solve()
                # osqp mistakenly report primal infeasible in some cases
                # disable masking in such cases
                # potentially related: https://github.com/osqp/osqp/issues/485
                if res.info.status == "primal infeasible":
                    self._reset_mask()
        else:
            # TODO: add support
            raise NotImplementedError("masking not supported for cvxpy backend")

    def _update_w(self, w_new=None) -> None:
        if w_new is not None:
            self.w_org = w_new
        self.w = self.w_org[self.nzidx_s]

    def _update_R(self) -> None:
        self.R_org = construct_R(self.y_len, self.upsamp)
        self.R = self.R_org[:, self.nzidx_c]

    def _update_Wt(self, clear=False) -> None:
        coef = self.coef.value if self.backend == "cvxpy" else self.coef
        if clear:
            logger.debug("Clearing error weighting")
            self.err_wt = np.ones(self.y_len)
        elif self.err_weighting == "fft":
            logger.debug("Updating error weighting with fft")
            hspec = self._get_stft_spec(coef)[:, int(self.coef_len / 2)]
            self.err_wt = (
                (hspec.reshape(-1, 1) * self.yspec).sum(axis=0)
                / np.linalg.norm(hspec)
                / np.linalg.norm(self.yspec, axis=0)
            )
        elif self.err_weighting == "corr":
            logger.debug("Updating error weighting with corr")
            for i in range(self.y_len):
                yseg = self.y[i : i + self.coef_len]
                if len(yseg) <= 1:
                    continue
                cseg = coef[: len(yseg)]
                with np.errstate(all="ignore"):
                    self.err_wt[i] = np.corrcoef(yseg, cseg)[0, 1].clip(0, 1)
            self.err_wt = np.nan_to_num(self.err_wt)
        elif self.err_weighting == "adaptive":
            if self.s_bin is not None:
                self.err_wt = np.zeros(self.y_len)
                s_bin_R = self.R @ self._pad_s(self.s_bin)
                for nzidx in np.where(s_bin_R > 0)[0]:
                    self.err_wt[nzidx : nzidx + self.coef_len] = 1
            else:
                self.err_wt = np.ones(self.y_len)
        self.Wt = sps.diags(self.err_wt)

    def _update_HG(self) -> None:
        coef = self.coef.value if self.backend == "cvxpy" else self.coef
        self.H_org = sps.diags(
            [np.repeat(coef[i], self.T - i) for i in range(len(coef))],
            offsets=-np.arange(len(coef)),
            format="csc",
        )
        try:
            H_shape, H_nnz = self.H.shape, self.H.nnz
        except AttributeError:
            H_shape, H_nnz = None, None
        self.H = self.H_org[:, self.nzidx_s][self.nzidx_c, :]
        logger.debug(
            f"Updating H matrix - shape before: {H_shape}, shape new: {self.H.shape}, nnz before: {H_nnz}, nnz new: {self.H.nnz}"
        )
        if not self.free_kernel:
            theta = self.theta.value if self.backend == "cvxpy" else self.theta
            G_diag = sps.diags(
                [np.ones(self.T - 1)]
                + [np.repeat(-theta[i], self.T - 2 - i) for i in range(theta.shape[0])],
                offsets=np.arange(0, -theta.shape[0] - 1, -1),
                format="csc",
            )
            self.G_org = sps.bmat(
                [[None, G_diag], [np.zeros((1, 1)), None]], format="csc"
            )
            try:
                G_shape, G_nnz = self.G.shape, self.G.nnz
            except AttributeError:
                G_shape, G_nnz = None, None
            self.G = self.G_org[:, self.nzidx_c][self.nzidx_s, :]
            logger.debug(
                f"Updating G matrix - shape before: {G_shape}, shape new: {self.G.shape}, nnz before: {G_nnz}, nnz new: {self.G.nnz}"
            )
            # assert np.isclose(
            #     np.linalg.pinv(self.H.todense()), self.G.todense(), atol=self.atol
            # ).all()

    def _get_stft_spec(self, x: np.ndarray) -> np.ndarray:
        spec = np.abs(self.stft.stft(x)) ** 2
        t = self.stft.t(len(x))
        t_mask = np.logical_and(t >= 0, t < len(x))
        return spec[:, t_mask]

    def _get_M(self) -> sps.csc_matrix:
        if self.free_kernel:
            return sps.hstack(
                [
                    np.ones((self.R.shape[0], 1)),
                    self.scale * self.R @ self.H,
                ],
                format="csc",
            )
        else:
            return sps.hstack(
                [np.ones((self.R.shape[0], 1)), self.scale * self.R], format="csc"
            )

    def _update_P(self) -> None:
        if self.norm == "l1":
            # TODO: add support
            raise NotImplementedError(
                "l1 norm not yet supported with backend {}".format(self.backend)
            )
        elif self.norm == "l2":
            M = self._get_M()
            P = M.T @ self.Wt.T @ self.Wt @ M
        elif self.norm == "huber":
            lc, ls, ly = len(self.nzidx_c), len(self.nzidx_s), self.y_len
            if self.free_kernel:
                P = sps.bmat(
                    [
                        [sps.csc_matrix((ls, ls)), None, None],
                        [None, sps.csc_matrix((ly, ly)), None],
                        [None, None, sps.eye(ly, format="csc")],
                    ]
                )
            else:
                P = sps.bmat(
                    [
                        [sps.csc_matrix((lc, lc)), None, None],
                        [None, sps.csc_matrix((ly, ly)), None],
                        [None, None, sps.eye(ly, format="csc")],
                    ]
                )
        try:
            P_shape, P_nnz = self.P.shape, self.P.nnz
        except AttributeError:
            P_shape, P_nnz = None, None
        logger.debug(
            f"Updating P matrix - shape before: {P_shape}, shape new: {P.shape}, nnz before: {P_nnz}, nnz new: {P.nnz}"
        )
        self.P = sps.triu(P).tocsc()

    def _update_q0(self) -> None:
        if self.norm == "l1":
            # TODO: add support
            raise NotImplementedError(
                "l1 norm not yet supported with backend {}".format(self.backend)
            )
        elif self.norm == "l2":
            M = self._get_M()
            self.q0 = -M.T @ self.Wt.T @ self.Wt @ self.y
        elif self.norm == "huber":
            ly = self.y_len
            lx = len(self.nzidx_s) if self.free_kernel else len(self.nzidx_c)
            self.q0 = (
                np.concatenate([np.zeros(lx), np.ones(ly), np.ones(ly)]) * self.huber_k
            )

    def _update_q(self) -> None:
        if self.norm == "l1":
            # TODO: add support
            raise NotImplementedError(
                "l1 norm not yet supported with backend {}".format(self.backend)
            )
        elif self.norm == "l2":
            if self.free_kernel:
                ww = np.concatenate([np.zeros(1), self.w])
                qq = np.concatenate([np.zeros(1), np.ones_like(self.w)])
                self.q = self.q0 + self.l0_penal * ww + self.l1_penal * qq
            else:
                G_p = sps.hstack([np.zeros((self.G.shape[0], 1)), self.G], format="csc")
                self.q = (
                    self.q0
                    + self.l0_penal * self.w @ G_p
                    + self.l1_penal * np.ones(self.G.shape[0]) @ G_p
                )
        elif self.norm == "huber":
            pad_k = np.zeros(self.y_len)
            if self.free_kernel:
                self.q = (
                    self.q0
                    + self.l0_penal * np.concatenate([self.w, pad_k, pad_k])
                    + self.l1_penal
                    * np.concatenate([np.ones(len(self.nzidx_s)), pad_k, pad_k])
                )
            else:
                self.q = (
                    self.q0
                    + self.l0_penal * np.concatenate([self.w @ self.G, pad_k, pad_k])
                    + self.l1_penal
                    * np.concatenate([np.ones(self.G.shape[0]) @ self.G, pad_k, pad_k])
                )

    def _update_A(self) -> None:
        if self.free_kernel:
            Ax = sps.eye(len(self.nzidx_s), format="csc")
            Ar = self.scale * self.R @ self.H
        else:
            Ax = sps.csc_matrix(self.G_org[:, self.nzidx_c])
            # record spike terms that requires constraint
            self.nzidx_A = np.where((Ax != 0).sum(axis=1))[0]
            Ax = Ax[self.nzidx_A, :]
            Ar = self.scale * self.R
        try:
            A_shape, A_nnz = self.A.shape, self.A.nnz
        except AttributeError:
            A_shape, A_nnz = None, None
        if self.norm == "huber":
            e = sps.eye(self.y_len, format="csc")
            self.A = sps.bmat(
                [
                    [Ax, None, None],
                    [None, e, None],
                    [None, None, -e],
                    [Ar, e, e],
                ],
                format="csc",
            )
        else:
            self.A = sps.bmat([[np.ones((1, 1)), None], [None, Ax]], format="csc")
        logger.debug(
            f"Updating A matrix - shape before: {A_shape}, shape new: {self.A.shape}, nnz before: {A_nnz}, nnz new: {self.A.nnz}"
        )

    def _update_bounds(self) -> None:
        if self.norm == "huber":
            xlen = len(self.nzidx_s) if self.free_kernel else self.T
            self.lb = np.concatenate(
                [np.zeros(xlen + self.y_len * 2), self.y - self.huber_k]
            )
            self.ub = np.concatenate(
                [np.ones(xlen), np.full(self.y_len * 2, np.inf), self.y - self.huber_k]
            )
            self.ub_inf = np.concatenate(
                [np.full(xlen + self.y_len * 2, np.inf), self.y - self.huber_k]
            )
        else:
            bb = np.clip(self.y.mean(), 0, None) if self.use_base else 0
            if self.free_kernel:
                self.lb = np.zeros(len(self.nzidx_s) + 1)
                self.ub = np.concatenate([np.full(1, bb), np.ones(len(self.nzidx_s))])
                self.ub_inf = np.concatenate(
                    [np.full(1, bb), np.full(len(self.nzidx_s), np.inf)]
                )
            else:
                ub_pad, ub_inf_pad = np.zeros(self.T), np.zeros(self.T)
                ub_pad[self.nzidx_s] = 1
                ub_inf_pad[self.nzidx_s] = np.inf
                self.lb = np.zeros(len(self.nzidx_A) + 1)
                self.ub = np.concatenate([np.full(1, bb), ub_pad[self.nzidx_A]])
                self.ub_inf = np.concatenate([np.full(1, bb), ub_inf_pad[self.nzidx_A]])
        assert (self.ub >= self.lb).all()
        assert (self.ub_inf >= self.lb).all()
