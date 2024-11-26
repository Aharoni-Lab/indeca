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
from minian_bin.simulation import tau2AR
from minian_bin.utilities import scal_lstsq


def construct_R(T: int, up_factor: int):
    rs_vec = np.zeros(T * up_factor)
    rs_vec[:up_factor] = 1
    return sps.coo_matrix(
        np.stack([np.roll(rs_vec, up_factor * i) for i in range(T)], axis=0)
    )


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
        coef: np.array = None,
        coef_len: int = 60,
        scale: float = 1,
        penal: str = "l1",
        ar_mode: bool = False,
        use_base: bool = False,
        upsamp: int = 1,
        norm: str = "l1",
        mixin: bool = False,
        backend: str = "cvxpy",
        nthres: int = 1000,
        th_min: float = 0,
        th_max: float = 1,
        max_iter_l0: int = 30,
        max_iter_penal: int = 30,
        max_iter_scal: int = 10,
        delta_l0: float = 1e-4,
        delta_penal: float = 1e-4,
        atol=1e-3,
        rtol=1e-3,
    ) -> None:
        # book-keeping
        if y is not None:
            self.y_len = len(y)
        else:
            assert y_len is not None
            self.y_len = y_len
        if coef is not None:
            self.coef_len = len(coef)
        else:
            assert coef_len is not None
            self.coef_len = coef_len
        if upsamp > 1:
            R = construct_R(self.y_len, upsamp)
            self.T = self.y_len * upsamp
        else:
            R = sps.eye(self.y_len)
            self.T = self.y_len
        if penal is None:
            l0_penal = 0
            l1_penal = 0
        elif penal == "l1":
            l0_penal = 0
            l1_penal = 1
        elif penal == "l0":
            l0_penal = 1
            l1_penal = 0
        self.penal = penal
        self.l0_penal = l0_penal
        self.w = np.ones(self.T)
        self.upsamp = upsamp
        self.ar_mode = ar_mode
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
        # setup cvxpy
        if self.backend == "cvxpy":
            self.R = cp.Constant(R, name="R")
            self.c = cp.Variable((self.T, 1), nonneg=True, name="c")
            self.s = cp.Variable((self.T, 1), nonneg=True, name="s", boolean=mixin)
            self.y = cp.Parameter(shape=(self.y_len, 1), name="y")
            self.coef = cp.Parameter(shape=self.coef_len, name="coef")
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
            if ar_mode:
                self.G = sum(
                    [
                        cp.diag(cp.promote(-self.coef[i], (self.T - i - 1,)), -i - 1)
                        for i in range(self.coef_len)
                    ]
                ) + sps.eye(self.T)
                dcv_cons = [self.s == self.G @ self.c]
            else:
                # self.H = sum(
                #     [
                #         cp.diag(cp.promote(self.coef[i], (self.T - i,)), -i)
                #         for i in range(self.coef_len)
                #     ]
                # )
                if self.coef.value is not None:
                    self._update_H()
                dcv_cons = [
                    self.c[:, 0] == cp.convolve(self.coef, self.s[:, 0])[: self.T]
                ]
            amp_cons = [self.s <= 1]
            self.prob_free = cp.Problem(obj, dcv_cons)
            self.prob = cp.Problem(obj, dcv_cons + amp_cons)
        elif self.backend in ["osqp", "cuosqp"]:
            # book-keeping
            if y is None:
                self.y = np.ones(self.y_len)
            else:
                self.y = y
            if coef is None:
                self.coef = np.ones(self.coef_len)
            else:
                self.coef = coef
            self.c = np.zeros(self.T)
            self.s = np.zeros(self.T)
            self.l1_penal = l1_penal
            self.scale = scale
            if upsamp > 1:
                # TODO: add support
                raise NotImplementedError(
                    "Upsampling not yet supported with backend {}".format(backend)
                )
            if use_base:
                # TODO: add support
                raise NotImplementedError(
                    "Baseline term not yet supported with backend {}".format(backend)
                )
            self._update_H()
            self._update_P()
            self._update_q0()
            self._update_q()
            self.A = sps.eye(self.T, format="csc")
            if backend == "osqp":
                m = osqp.OSQP()
                m.setup(
                    P=self.P,
                    q=self.q,
                    A=self.A,
                    l=np.zeros(self.T),
                    u=np.inf,
                    check_termination=25,
                    eps_abs=self.atol * 1e-4,
                    eps_rel=1e-8,
                )
                m.codegen(
                    "osqp-codegen",
                    parameters="matrices",
                    python_ext_name="emosqp_free",
                    force_rewrite=True,
                )
                m.update(u=np.ones(self.T))
                m.codegen(
                    "osqp-codegen",
                    parameters="matrices",
                    python_ext_name="emosqp",
                    force_rewrite=True,
                )
                import emosqp
                import emosqp_free

                self.prob_free = emosqp_free
                self.prob = emosqp
            elif backend == "cuosqp":
                self.prob_free = cuosqp.OSQP()
                self.prob = cuosqp.OSQP()
                self.prob_free.setup(
                    P=self.P,
                    q=self.q,
                    A=self.A,
                    l=np.zeros(self.T),
                    u=np.inf,
                    check_termination=25,
                    eps_abs=self.atol * 1e-4,
                    eps_rel=1e-8,
                )
                self.prob.setup(
                    P=self.P,
                    q=self.q,
                    A=self.A,
                    l=np.zeros(self.T),
                    u=np.ones(self.T),
                    check_termination=25,
                    eps_abs=self.atol * 1e-4,
                    eps_rel=1e-8,
                )

    def update(
        self,
        y: np.ndarray = None,
        coef: np.ndarray = None,
        scale: float = None,
        l0_penal: float = None,
        l1_penal: float = None,
        w: np.ndarray = None,
    ) -> None:
        if self.backend == "cvxpy":
            if y is not None:
                self.y.value = y
            if coef is not None:
                self.coef.value = coef
                self._update_H()
            if scale is not None:
                self.scale.value = scale
            if l1_penal is not None:
                self.l1_penal.value = l1_penal
            if l0_penal is not None:
                self.l0_penal = l0_penal
            if w is not None:
                self.w = w
            if l0_penal is not None or w is not None:
                self.l0_w.value = self.l0_penal * self.w
        elif self.backend in ["osqp", "cuosqp"]:
            # update input params
            if y is not None:
                self.y = y
            if coef is not None:
                self.coef = coef
            if scale is not None:
                self.scale = scale
            if l1_penal is not None:
                self.l1_penal = l1_penal
            if l0_penal is not None:
                self.l0_penal = l0_penal
            if w is not None:
                self.w = w
            # update internal variables
            updt_H, updt_P, updt_q0, updt_q = [False] * 4
            if coef is not None:
                self._update_H()
                updt_H = True
            if any((scale is not None, updt_H)):
                self._update_P()
                updt_P = True
            if any((scale is not None, y is not None, updt_H)):
                self._update_q0()
                updt_q0 = True
            if any(
                (w is not None, l0_penal is not None, l1_penal is not None, updt_q0)
            ):
                self._update_q()
                updt_q = True
            # update prob
            if self.backend == "osqp":
                if updt_P:
                    self.prob_free.update_P(self.P)
                    self.prob.update_P(self.P)
                if updt_q:
                    self.prob_free.update_lin_cost(self.q)
                    self.prob.update_lin_cost(self.q)
            elif self.backend == "cuosqp":
                self.prob_free.update(
                    P=self.P if updt_P else None, q=self.q if updt_q else None
                )
                self.prob.update(
                    P=self.P if updt_P else None, q=self.q if updt_q else None
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
        elif self.backend in ["osqp", "cuosqp"]:
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
        scals = [scal_lstsq(c, y) for c in cvals]
        objs = [self._compute_err(c=scl * c) for scl, c in zip(scals, cvals)]
        opt_idx = np.argmin(objs)
        return svals[opt_idx], cvals[opt_idx], scals[opt_idx], objs[opt_idx]

    def solve_penal(self) -> Tuple[np.ndarray]:
        if self.penal is None:
            opt_s, opt_c, opt_scl, opt_obj = self.solve_thres()
            opt_penal = 0
        elif self.penal in ["l0", "l1"]:
            pn = "{}_penal".format(self.penal)
            ub = self._compute_err(s=np.zeros(self.T))
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
                vol_tol=self.atol,
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
                print("wrong")
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
            elif np.abs(cur_scl - prev_scals).min() < self.atol:
                self.update(scale=(cur_scl + last_scal) / 2)
            else:
                self.update(scale=cur_scl)
        else:
            warnings.warn("max scale iterations reached")
        return cur_s, cur_c, cur_scl, cur_obj, cur_penal

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
        elif self.backend in ["osqp", "cuosqp"]:
            opt_s = res[0] if self.backend == "osqp" else res.x
            self.s = opt_s
        if return_obj:
            if self.backend == "cvxpy":
                opt_obj = res
            elif self.backend in ["osqp", "cuosqp"]:
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
            elif self.backend in ["osqp", "cuosqp"]:
                return self.H @ self.s

    def _compute_err(self, c: np.ndarray = None, s: np.ndarray = None) -> float:
        if self.backend == "cvxpy":
            y = self.y.value.squeeze()
        elif self.backend in ["osqp", "cuosqp"]:
            y = self.y
        if c is None:
            c = self._compute_c(s)
        if self.norm == "l1":
            return np.sum(np.abs(y - c))
        elif self.norm == "l2":
            return np.sum((y - c) ** 2)
        elif self.norm == "huber":
            return np.sum(huber(1, y - c)) * 2

    def _update_H(self) -> None:
        if self.ar_mode:
            # TODO: add support
            raise NotImplementedError(
                "AR mode not yet supported with backend {}".format(self.backend)
            )
        else:
            coef = self.coef.value if self.backend == "cvxpy" else self.coef
            self.H = sps.csc_matrix(convolution_matrix(coef, self.T)[: self.T, :])

    def _update_P(self) -> None:
        if self.norm == "l1":
            # TODO: add support
            raise NotImplementedError(
                "l1 norm not yet supported with backend {}".format(self.backend)
            )
        elif self.norm == "l2":
            self.P = self.scale**2 * self.H.T @ self.H
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
            self.q0 = -self.scale * self.H.T @ self.y
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
            self.q = (
                self.q0 + self.l0_penal * self.w + self.l1_penal * np.ones_like(self.q0)
            )
        elif self.norm == "huber":
            # TODO: add support
            raise NotImplementedError(
                "huber norm not yet supported with backend {}".format(self.backend)
            )
