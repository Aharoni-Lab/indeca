import warnings
from typing import Tuple

import cuosqp
import cvxpy as cp
import numpy as np
import osqp
import pandas as pd
import piqp
import scipy.sparse as sps
import xarray as xr
from line_profiler import profile
from scipy.linalg import convolution_matrix
from scipy.optimize import direct
from scipy.special import huber

from minian_bin.cnmf import filt_fft, get_ar_coef, noise_fft
from minian_bin.simulation import AR2tau, exp_pulse, tau2AR
from minian_bin.utilities import scal_lstsq


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


def max_thres(
    a: xr.DataArray,
    nthres: int,
    th_min=0.1,
    th_max=0.9,
    ds=None,
    return_thres=False,
    th_amplitude=False,
    delta=1e-8,
    reverse_thres=False,
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
        coef: np.array = None,
        coef_len: int = 100,
        scale: float = 1,
        penal: str = "l1",
        use_base: bool = False,
        upsamp: int = 1,
        norm: str = "l1",
        mixin: bool = False,
        backend: str = "cvxpy",
        nthres: int = 1000,
        th_min: float = 0,
        th_max: float = 1,
        max_iter_l0: int = 30,
        max_iter_penal: int = 500,
        max_iter_scal: int = 10,
        delta_l0: float = 1e-4,
        delta_penal: float = 1e-4,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> None:
        # book-keeping
        if y is not None:
            self.y_len = len(y)
        else:
            assert y_len is not None
            self.y_len = y_len
        if theta is not None:
            self.theta = np.array(theta)
            tau_d, tau_r, p = AR2tau(theta[0], theta[1], solve_amp=True)
            self.tau = np.array([tau_d, tau_r])
            coef, _, _ = exp_pulse(
                tau_d,
                tau_r,
                p_d=p,
                p_r=-p,
                nsamp=coef_len * upsamp,
                kn_len=coef_len * upsamp,
                trunc_thres=atol,
            )
        elif tau is not None:
            self.theta = np.array(tau2AR(tau[0], tau[1]))
            self.tau = np.array(tau)
            _, _, p = AR2tau(self.theta[0], self.theta[1], solve_amp=True)
            coef, _, _ = exp_pulse(
                tau[0],
                tau[1],
                p_d=p,
                p_r=-p,
                nsamp=coef_len * upsamp,
                kn_len=coef_len * upsamp,
                trunc_thres=atol,
            )
        else:
            if coef is None:
                assert coef_len is not None
                coef = np.ones(coef_len * upsamp)
        self.coef_len = len(coef)
        self.T = self.y_len * upsamp
        if penal is None:
            l0_penal = 0
            l1_penal = 0
        elif penal == "l1":
            l0_penal = 0
            l1_penal = 1
        elif penal == "l0":
            l0_penal = 1
            l1_penal = 0
        self.free_kernel = True
        self.penal = penal
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
        self.nzidx_s = np.arange(self.T)
        self.nzidx_c = np.arange(self.T)
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
            self.l1_penal = l1_penal
            self.scale = scale
            if use_base:
                # TODO: add support
                raise NotImplementedError(
                    "Baseline term not yet supported with backend {}".format(backend)
                )
            self._setup_prob_osqp()

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
    ) -> None:
        if self.backend == "cvxpy":
            if y is not None:
                self.y.value = y
            if tau is not None:
                theta_new = np.array(tau2AR(tau[0], tau[1]))
                _, _, p = AR2tau(theta_new[0], theta_new[1], solve_amp=True)
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
                _, _, p = AR2tau(theta_new[0], theta_new[1], solve_amp=True)
                coef, _, _ = exp_pulse(
                    tau[0],
                    tau[1],
                    p_d=p,
                    p_r=-p,
                    nsamp=self.coef_len,
                    kn_len=self.coef_len,
                )
                self.theta = theta_new
            if coef is not None:
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
            updt_HG, updt_P, updt_A, updt_q0, updt_q = [False] * 5
            if coef is not None:
                self._update_HG()
                updt_HG = True
            if updt_HG:
                self._update_A()
                updt_A = True
            if any((scale is not None, scale_mul is not None, updt_HG)):
                self._update_P()
                updt_P = True
            if any((scale is not None, scale_mul is not None, y is not None, updt_HG)):
                self._update_q0()
                updt_q0 = True
            if any(
                (w is not None, l0_penal is not None, l1_penal is not None, updt_q0)
            ):
                self._update_q()
                updt_q = True
            # update prob
            if self.backend == "emosqp":
                if updt_P:
                    self.prob_free.update_P(self.P.data, None, 0)
                    self.prob.update_P(self.P.data, None, 0)
                if updt_q:
                    self.prob_free.update_lin_cost(self.q)
                    self.prob.update_lin_cost(self.q)
            elif self.backend in ["osqp", "cuosqp"]:
                self.prob_free.update(
                    Px=self.P.copy().data if updt_P else None,
                    q=self.q.copy() if updt_q else None,
                    Ax=self.A.copy().data if updt_A else None,
                )
                self.prob.update(
                    Px=self.P.copy().data if updt_P else None,
                    q=self.q.copy() if updt_q else None,
                    Ax=self.A.copy().data if updt_A else None,
                )

    def solve(self, amp_constraint: bool = True) -> np.ndarray:
        if self.l0_penal == 0:
            opt_s = self._solve(amp_constraint)
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
        return np.abs(opt_s)

    def solve_thres(self) -> Tuple[np.ndarray]:
        if self.backend == "cvxpy":
            y = self.y.value.squeeze()
        elif self.backend in ["osqp", "emosqp", "cuosqp"]:
            y = self.y
        opt_s = self.solve()
        svals = max_thres(
            opt_s,
            self.nthres,
            th_min=self.th_min,
            th_max=self.th_max,
            reverse_thres=True,
        )
        cvals = [self._compute_c(s) for s in svals]
        R = self.R.value if self.backend == "cvxpy" else self.R
        yfvals = [R @ c for c in cvals]
        scals = [scal_lstsq(yf, y) for yf in yfvals]
        objs = [self._compute_err(y_fit=scl * yf) for scl, yf in zip(scals, yfvals)]
        opt_idx = np.argmin(objs)
        return svals[opt_idx], cvals[opt_idx], scals[opt_idx], objs[opt_idx]

    def solve_penal(self, masking=True) -> Tuple[np.ndarray]:
        if self.penal is None:
            opt_s, opt_c, opt_scl, opt_obj = self.solve_thres()
            opt_penal = 0
        elif self.penal in ["l0", "l1"]:
            pn = "{}_penal".format(self.penal)
            if masking:
                self.update(**{pn: 0})
                self._update_mask()
            ub = self._compute_err(s=np.zeros(len(self.nzidx_s)))
            for _ in range(int(np.ceil(np.log2(ub)))):
                self.update(**{pn: ub})
                s = self.solve()
                if (s > self.delta_penal).sum() > 0:
                    ub = ub * 2
                    break
                else:
                    ub = ub / 2
            else:
                warnings.warn("max ub iterations reached")

            def opt_fn(x):
                self.update(**{pn: x.item()})
                _, _, _, obj = self.solve_thres()
                return obj

            res = direct(
                opt_fn,
                bounds=[(0, ub)],
                maxfun=self.max_iter_penal,
                eps=self.atol,
                vol_tol=1e-2,
            )
            if not res.success:
                warnings.warn(
                    "could not find optimal penalty within {} iterations".format(
                        res.nfev
                    )
                )
            opt_penal = res.x.item()
            self.update(**{pn: opt_penal})
            opt_s, opt_c, opt_scl, opt_obj = self.solve_thres()
            if opt_scl == 0:
                warnings.warn("could not find non-zero solution")
        return opt_s, opt_c, opt_scl, opt_obj, opt_penal

    def solve_scale(self, reset_scale: bool = True) -> Tuple[np.ndarray]:
        if reset_scale:
            self.update(scale=1)
            s_free = self.solve(amp_constraint=False)
            self.update(scale=np.ptp(s_free))
        metric_df = None
        for i in range(self.max_iter_scal):
            cur_s, cur_c, cur_scl, cur_obj, cur_penal = self.solve_penal()
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
            cur_met = pd.DataFrame(
                [
                    {
                        "iter": i,
                        "scale": cur_scl,
                        "obj": cur_obj,
                        "penal": cur_penal,
                        "nnz": (cur_s > 0).sum(),
                    }
                ]
            )
            metric_df = pd.concat([metric_df, cur_met], ignore_index=True)
            if any(
                (
                    np.abs(cur_scl - opt_scal) < self.rtol * opt_scal,
                    np.abs(cur_obj - opt_obj) < self.rtol * opt_obj,
                    np.abs(cur_scl - last_scal) < self.atol,
                    np.abs(cur_obj - last_obj) < self.atol,
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
        opt_s, opt_c = np.zeros(self.T), np.zeros(self.T)
        opt_s[self.nzidx_s] = cur_s
        opt_c[self.nzidx_c] = cur_c
        return opt_s, opt_c, cur_scl, cur_obj, cur_penal

    def _setup_prob_osqp(self) -> None:
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
            self.prob_free.setup(
                P=self.P.copy(),
                q=self.q.copy(),
                A=self.A.copy(),
                l=self.lb.copy(),
                u=self.ub_inf.copy(),
                check_termination=25,
                eps_abs=1e-5 if self.backend == "osqp" else self.atol * 1e-4,
                eps_rel=1e-5 if self.backend == "osqp" else 1e-8,
                verbose=False,
                polish=True,
                warm_start=True if self.backend == "osqp" else False,
                max_iter=int(1e5) if self.backend == "osqp" else None,
            )
            self.prob.setup(
                P=self.P.copy(),
                q=self.q.copy(),
                A=self.A.copy(),
                l=self.lb.copy(),
                u=self.ub.copy(),
                check_termination=25,
                eps_abs=1e-5 if self.backend == "osqp" else self.atol * 1e-4,
                eps_rel=1e-5 if self.backend == "osqp" else 1e-8,
                verbose=False,
                polish=True,
                warm_start=True if self.backend == "osqp" else False,
                max_iter=int(1e5) if self.backend == "osqp" else None,
            )

    def _solve(
        self, amp_constraint: bool = True, return_obj: bool = False
    ) -> np.ndarray:
        if amp_constraint:
            prob = self.prob
        else:
            prob = self.prob_free
        res = prob.solve()
        if self.backend == "cvxpy":
            opt_s = self.s.value.squeeze()
        elif self.backend in ["osqp", "emosqp", "cuosqp"]:
            x = res[0] if self.backend == "emosqp" else res.x
            # osqp mistakenly report primal infeasibility when using masks with high l1 penalty
            # manually set solution to zero in such cases
            if res.info.status == "primal infeasible":
                x = np.zeros_like(x, dtype=float)
            if self.free_kernel:
                opt_s = x
            else:
                opt_s = (self.G @ x)[self.nzidx_s]
            self.s = opt_s
        if return_obj:
            if self.backend == "cvxpy":
                opt_obj = res
            elif self.backend in ["osqp", "emosqp", "cuosqp"]:
                opt_obj = self._compute_err()
            return opt_s, opt_obj
        else:
            return opt_s

    def _compute_c(self, s: np.ndarray = None) -> np.ndarray:
        if s is not None:
            return self.H @ s
        else:
            if self.backend == "cvxpy":
                return self.c.value.squeeze()
            elif self.backend in ["osqp", "emosqp", "cuosqp"]:
                return self.H @ self.s

    def _compute_err(
        self, y_fit: np.ndarray = None, c: np.ndarray = None, s: np.ndarray = None
    ) -> float:
        if self.backend == "cvxpy":
            # TODO: add support
            raise NotImplementedError
        elif self.backend in ["osqp", "emosqp", "cuosqp"]:
            y = self.y
        if y_fit is None:
            if c is None:
                c = self._compute_c(s)
            y_fit = self.R @ c
        if self.norm == "l1":
            return np.sum(np.abs(y - y_fit))
        elif self.norm == "l2":
            return np.sum((y - y_fit) ** 2)
        elif self.norm == "huber":
            return np.sum(huber(1, y - y_fit)) * 2

    def _reset_mask(self) -> None:
        self.nzidx_s = np.arange(self.T)
        self.nzidx_c = np.arange(self.T)
        self._update_R()
        self._update_w()
        if self.backend in ["osqp", "emosqp", "cuosqp"]:
            self._setup_prob_osqp()

    def _update_mask(self, amp_constraint: bool = True) -> None:
        self._reset_mask()
        if self.backend in ["osqp", "emosqp", "cuosqp"]:
            opt_s = self.solve(amp_constraint)
            opt_c = self.H @ opt_s
            self.nzidx_s = np.where(opt_s > self.delta_penal)[0]
            self.nzidx_c = np.where(opt_c > self.delta_penal)[0]
            self._update_R()
            self._update_w()
            self._setup_prob_osqp()
            if len(self.nzidx_c) < self.T:
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
        if self.free_kernel:
            self.w = self.w_org[self.nzidx_s]
        else:
            self.w = self.w_org

    def _update_R(self) -> None:
        self.R_org = construct_R(self.y_len, self.upsamp)
        self.R = self.R_org[:, self.nzidx_c]

    def _update_HG(self) -> None:
        coef = self.coef.value if self.backend == "cvxpy" else self.coef
        self.H_org = sps.diags(
            [np.repeat(coef[i], self.T - i) for i in range(len(coef))],
            offsets=-np.arange(len(coef)),
            format="csc",
        )
        self.H = self.H_org[:, self.nzidx_s][self.nzidx_c, :]
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
            self.G = self.G_org[:, self.nzidx_c]
            # assert np.isclose(
            #     np.linalg.pinv(self.H.todense()), self.G.todense(), atol=self.atol
            # ).all()

    def _update_P(self) -> None:
        if self.norm == "l1":
            # TODO: add support
            raise NotImplementedError(
                "l1 norm not yet supported with backend {}".format(self.backend)
            )
        elif self.norm == "l2":
            if self.free_kernel:
                P = self.scale**2 * self.H.T @ self.R.T @ self.R @ self.H
            else:
                P = self.scale**2 * self.R.T @ self.R
            # assert np.isclose(P.todense(), P.T.todense()).all()
            self.P = sps.triu(P).tocsc()
        elif self.norm == "huber":
            # TODO: add support
            raise NotImplementedError(
                "huber norm not yet supported with backend {}".format(self.backend)
            )

    def _update_q0(self) -> None:
        if self.norm == "l1":
            # TODO: add support
            raise NotImplementedError(
                "l1 norm not yet supported with backend {}".format(self.backend)
            )
        elif self.norm == "l2":
            if self.free_kernel:
                self.q0 = -self.scale * self.H.T @ self.R.T @ self.y
            else:
                self.q0 = -self.scale * self.R.T @ self.y
        elif self.norm == "huber":
            # TODO: add support
            raise NotImplementedError(
                "huber norm not yet supported with backend {}".format(self.backend)
            )

    def _update_q(self) -> None:
        if self.norm == "l1":
            # TODO: add support
            raise NotImplementedError(
                "l1 norm not yet supported with backend {}".format(self.backend)
            )
        elif self.norm == "l2":
            if self.free_kernel:
                self.q = (
                    self.q0
                    + self.l0_penal * self.w
                    + self.l1_penal * np.ones_like(self.q0)
                )
            else:
                self.q = (
                    self.q0
                    + self.l0_penal * self.w @ self.G
                    + self.l1_penal * np.ones(self.G.shape[0]) @ self.G
                )
        elif self.norm == "huber":
            # TODO: add support
            raise NotImplementedError(
                "huber norm not yet supported with backend {}".format(self.backend)
            )

    def _update_A(self) -> None:
        if self.free_kernel:
            self.A = sps.eye(len(self.nzidx_s), format="csc")
        else:
            G_sub = self.G_org[:, self.nzidx_c]
            zmask = np.array(
                np.logical_and(G_sub.sum(axis=1) == 0, self.G_org.sum(axis=1) != 0)
            ).squeeze()
            self.A = sps.csc_matrix(G_sub[~zmask, :])

    def _update_bounds(self) -> None:
        if self.free_kernel:
            self.lb = np.zeros(len(self.nzidx_s))
            self.ub = np.ones(len(self.nzidx_s))
            self.ub_inf = np.full(len(self.nzidx_s), np.inf)
        else:
            self.lb = np.zeros(self.A.shape[0])
            self.ub = np.ones(self.A.shape[0])
            self.ub_inf = np.full(self.A.shape[0], np.inf)
