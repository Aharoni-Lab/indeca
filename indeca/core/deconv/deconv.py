"""Main deconvolution module."""

import itertools as itt
import math
import os
import warnings
from typing import Tuple, Any, Optional
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import pandas as pd
from scipy.optimize import direct
from scipy.signal import find_peaks
from scipy.ndimage import label

from indeca.utils.logging_config import get_module_logger
from indeca.core.simulation import AR2tau, ar_pulse, exp_pulse, solve_p, tau2AR
from indeca.utils.utils import scal_lstsq
from .config import DeconvConfig
from .solver import DeconvSolver, CVXPYSolver, OSQPSolver
from .utils import max_thres, max_consecutive, sum_downsample

# Initialize logger for this module
logger = get_module_logger("deconv")
logger.info("Deconv module initialized")

@dataclass
class InputParams:
    """Input parameters for deconv module."""
    y: np.ndarray = None
    y_len: int = None
    theta: np.ndarray = None
    tau: np.ndarray = None
    ps: np.ndarray = None
    coef: np.ndarray = None
    coef_len: int = 100
    scale: float = 1
    penal: str = "l1"
    use_base: bool = False
    upsamp: int = 1
    norm: str = "l2"
    mixin: bool = False
    backend: str = "osqp"
    free_kernel: bool = False
    nthres: int = 1000
    err_weighting: str = None
    wt_trunc_thres: float = 1e-2
    masking_radius: int = None,
    pks_polish: bool = True
    th_min: float = 0
    th_max: float = 1
    density_thres: float = None
    ncons_thres: int = None
    min_rel_scl: float = None
    max_iter_l0: int = 30
    max_iter_penal: int = 500
    max_iter_scal: int = 50
    delta_l0: float = 1e-4
    delta_penal: float = 1e-4
    atol: float = 1e-3
    rtol: float = 1e-3
    Hlim: int = 1e5
    dashboard = None
    dashboard_uid = None

@dataclass
class UpdateParams:
    """Update parameters for update function defined in DeconvBin class."""
    y: np.ndarray = None
    tau: np.ndarray = None
    coef: np.ndarray = None
    scale: float = None
    scale_mul: float = None
    l0_penal: float = None
    l1_penal: float = None
    w: np.ndarray = None
    update_weighting: bool = False
    clear_weighting: bool = False
    scale_coef: bool = False

class DeconvBin:
    """Deconvolution main class.

    This class wraps the solver backends and provides high-level methods
    for spike inference including thresholding, penalty optimization,
    and scale estimation.
    """
    def __init__(self, params: InputParams) -> None:
        self.params = params
        # Handle y input
        if params.y is not None:
            self.y_len = len(params.y)
            self.y = params.y
        else:
            assert params.y_len is not None
            self.y_len = params.y_len
            self.y = np.zeros(params.y_len)

        coef_len = params.coef_len
        if coef_len is not None and coef_len > self.y_len:
            warnings.warn("Coefficient length longer than data")
            coef_len = self.y_len

        # Store tau/theta/ps
        self.theta = None
        self.tau = None
        self.ps = None

        # Compute coefficients from theta or tau.

        if theta is not None:
            theta = np.array(self.theta)
            if self.tau is None:
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
                    trunc_thres=self.params.atol,
                )
        if self.params.tau is not None:
            assert (
                ps is not None
            ), "exp coefficients must be provided together with time constants."
            if self.params.theta is None:
                self.theta = np.array(tau2AR(self.params.tau[0], self.params.tau[1]))
            self.tau = np.array(self.params.tau)
            self.ps = self.params.ps
            coef, _, _ = exp_pulse(
                self.params.tau[0],
                self.params.tau[1],
                p_d=self.params.ps[0],
                p_r=self.params.ps[1],
                nsamp=coef_len * self.params.upsamp,
                kn_len=coef_len * self.params.upsamp,
                trunc_thres=self.params.atol,
            )
        if coef is None:
            assert coef_len is not None
            coef = np.ones(coef_len * self.params.upsamp)

        # `coef_len` (config) is the *base* kernel length; the stored `coef` is
        # already upsampled to length `coef_len * upsamp`.
        self.coef_len = len(coef)

        # Create config (note: frozen after creation)
        self.cfg = DeconvConfig(params)

        # Dashboard for visualization
        self.dashboard = params.dashboard
        self.dashboard_uid = params.dashboard_uid

        # Penalty tracking - solver tracks scale, we track penalty locally
        self._l0_penal = 0.0
        self._l1_penal = 0.0

        #Solver function for the correct backend
        class SolverFunc:
            def __init__(self):
                self._solver = None

            @property
            def solver(self) -> None:
                if self._solver is None:
                    if self.cfg.backend == "cvxpy":
                        if self.cfg.free_kernel:
                            raise NotImplementedError(
                                "CVXPY backend does not support free_kernel mode"
                            )
                        self._solver = CVXPYSolver(
                            self.cfg,
                            self.y_len,
                            y=self.y,
                            coef=self.coef,
                            theta=self.theta,
                            tau=self.tau,
                            ps=self.ps,
                        )
                    elif self.cfg.backend in ["osqp", "cuosqp"]:
                        self._solver = OSQPSolver(
                            self.cfg,
                            self.y_len,
                            y=self.y,
                            coef=self.coef,
                            theta=self.theta,
                            tau=self.tau,
                            ps=self.ps,
                        )
                    else:
                        raise ValueError(f"Unknown backend: {self.cfg.backend}")
                return self._solver

        # State
        self.T = self._solver.T
        self.s = np.zeros(self.T)
        self.b = 0
        self.c_bin = None
        self.s_bin = None

    # Update dashboard with initial kernel
        if self.dashboard is not None:
            self.dashboard.update(h=coef, uid=self.dashboard_uid)

    # Validate coefficients
        self._solver.validate_coefficients(atol=self.params.atol)

    # NOTE: do not store an "err_total" here. `_res_err` expects a residual,
    # not the raw `y`, and this value was misleading and unused.

    @property
    def scale(self) -> float:
        """Current scale value (delegated to solver)."""
        return self._solver.scale

    @property
    def H(self):
        """Convolution matrix H."""
        return self._solver.H

    @property
    def R(self):
        """Resampling matrix R."""
        return self._solver.R

    @property
    def R_org(self):
        """Original (full) resampling matrix."""
        return self._solver.R_org

    @property
    def nzidx_s(self):
        """Nonzero indices for s."""
        return self._solver.nzidx_s

    @property
    def nzidx_c(self):
        """Nonzero indices for c."""
        return self._solver.nzidx_c

    @property
    def coef(self):
        """Coefficient kernel."""
        return self._solver.coef

    @property
    def err_wt(self):
        """Error weighting vector."""
        return self._solver.err_wt

    @property
    def wgt_len(self):
        """Error weighting length."""
        return self._solver.wgt_len

    @err_wt.setter
    def err_wt(self, value):
        """Allow direct assignment (used by tests/demo code)."""
        self._solver.err_wt = np.array(value)
        self._solver.Wt = sps.diags(self._solver.err_wt)

    def update(self, update_param: UpdateParams) -> None:
        """Update parameters."""
        logger.debug(f"Updating parameters - backend: {self.cfg.backend}")

        self.theta_new = None

        if self.params.tau is not None:
            theta_new = np.array(tau2AR(self.params.tau[0], self.params.tau[1]))
            p = solve_p(self.params.tau[0], self.params.tau[1])
            coef_new, _, _ = exp_pulse(
                self.params.tau[0],
                self.params.tau[1],
                p_d=p,
                p_r=-p,
                nsamp=self.cfg.coef_len * self.cfg.upsamp,
                kn_len=self.cfg.coef_len * self.cfg.upsamp,
            )
            coef = coef_new
            self.params.s.tau = self.params.tau
            self.theta = theta_new
            self.ps = np.array([p, -p])

        if coef is not None and self.params.scale_coef:
            current_coef = (
                self._solver.coef if self._solver.coef is not None else np.ones_like(coef)
            )
            scale_mul = scal_lstsq(coef, current_coef).item()

        if self.params.l0_penal is not None:
            self._l0_penal = self.params.l0_penal
        if self.params.l1_penal is not None:
            self._l1_penal = self.params.l1_penal

        # Forward updates to solver (solver handles scale directly)
        self._solver.update(
            y=self.params.y,
            coef=coef,
            scale=self.params.scale,
            scale_mul=scale_mul,
            l1_penal=self._l1_penal if self.params.l1_penal is not None else None,
            l0_penal=self._l0_penal if self.params.l0_penal is not None else None,
            w=self.params.w,
            theta=theta_new if theta_new is not None else self.theta,
            update_weighting=self.params.update_weighting,
            clear_weighting=self.params.clear_weighting,
            scale_coef=self.params.scale_coef,
        )

        if self.params.y is not None:
            self.y = self.params.y

    def _pad_s(self, s: np.ndarray = None) -> np.ndarray:
        """Pad sparse s to full length."""
        return self._solver._pad_s(s)

    def _pad_c(self, c: np.ndarray = None) -> np.ndarray:
        """Pad sparse c to full length."""
        return self._solver._pad_c(c)

    def _reset_cache(self) -> None:
        """Reset solver cache."""
        self._solver.reset_cache()

    def _reset_mask(self) -> None:
        """Reset solver mask to full range."""
        self._solver.reset_mask()

    def _update_mask(self, use_wt: bool = False, amp_constraint: bool = True) -> None:
        """Update mask based on current solution."""
        # CVXPY doesn't support masking
        if self.cfg.backend == "cvxpy":
            return
        if self.cfg.backend in ["osqp", "cuosqp"]:
            if use_wt:
                nzidx_s = np.where(self.R.T @ self.err_wt)[0]
            else:
                if self.cfg.masking_radius is not None:
                    mask = np.zeros(self.T)
                    for nzidx in np.where(self._pad_s(self.s_bin) > 0)[0]:
                        start = max(nzidx - self.cfg.masking_radius, 0)
                        end = min(nzidx + self.cfg.masking_radius, self.T)
                        mask[start:end] = 1
                    nzidx_s = np.where(mask)[0]
                else:
                    self._reset_mask()
                    opt_s, _ = self.solve(amp_constraint)
                    nzidx_s = np.where(opt_s > self.cfg.delta_penal)[0]

            if len(nzidx_s) == 0:
                logger.warning("Empty mask, resetting")
                self._reset_mask()
                return

            self.solver.set_mask(nzidx_s)

            # Verify mask is valid
            if not self.cfg.free_kernel and len(self.nzidx_c) < self.T:
                res = self.solver.prob.solve()
                if res.info.status == "primal infeasible":
                    logger.warning("Mask caused primal infeasibility, resetting")
                    self._reset_mask()
        else:
            raise NotImplementedError("Masking not supported for cvxpy backend")

    def _cut_pks_labs(self, s, labs, pks):
        """Cut peak labels at valleys between peaks."""
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
        """Merge sparse regions to minimize error."""
        max_combos = 10_000
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
                # Defensive guard: avoid combinatorial explosion.
                if math.comb(rlen, ns) > max_combos:
                    continue
                for idxs in itt.combinations(ridx, ns):
                    idxs = np.array(idxs)
                    s_test = s_new.copy()
                    s_test[idxs] = rsum / ns
                    err_after = self._compute_err(s=s_test[self.nzidx_s])
                    err_ls.append(err_after)
                    idx_ls.append(idxs)
            if len(err_ls) > 0:
                err_min_idx = np.argmin(err_ls)
                err_min = err_ls[err_min_idx]
                if err_min - err_before < err_rtol * err_before:
                    idx_min = idx_ls[err_min_idx]
                    s_new[idx_min] = rsum / len(idx_min)
                    s_ret = s_new
        return s_ret
    
    def solve(
        self,
        amp_constraint: bool = True,
        update_cache: bool = False,
        pks_polish: bool = None,
        pks_delta: float = 1e-5,
        pks_err_rtol: float = 10,
        pks_cut: bool = False,
        ) -> Tuple[np.ndarray, float]:
        """Solve main routine (l0 heuristic wrapper)."""
        if self._l0_penal == 0:
            opt_s, opt_b, _ = self.solver.solve(amp_constraint=amp_constraint)
        else:
            # L0 heuristic via reweighted L1
            metric_df = None
            for i in range(self.cfg.max_iter_l0):
                cur_s, cur_b, _ = self.solver.solve(amp_constraint=amp_constraint)
                # Compute objective explicitly since solver returns 0
                cur_obj = self._compute_err(s=cur_s, b=cur_b)

                if metric_df is None:
                    obj_best = np.inf
                    obj_last = np.inf
                else:
                    obj_best = (
                        metric_df["obj"][1:].min() if len(metric_df) > 1 else np.inf
                    )
                    obj_last = np.array(metric_df["obj"])[-1]

                opt_s = np.where(cur_s > self.cfg.delta_l0, cur_s, 0)
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

                if any([obj_gap < self.cfg.rtol * obj_best, obj_delta < self.cfg.atol]):
                    break
                else:
                    w_new = np.clip(
                        np.ones(self.T) / (self.cfg.delta_l0 * np.ones(self.T) + opt_s),
                        0,
                        1e5,
                    )
                    self.update(w=w_new)
            else:
                warnings.warn(
                    f"l0 heuristic did not converge in {self.cfg.max_iter_l0} iterations"
                )

            opt_s, opt_b, _ = self.solver.solve(amp_constraint=amp_constraint)

        self.b = opt_b

        # Peak polishing
        if pks_polish is None:
            pks_polish = amp_constraint
        if pks_polish and self.cfg.backend != "cvxpy":
            s_pad = self._pad_s(opt_s) if len(opt_s) == len(self.nzidx_s) else opt_s
            s_ft = np.where(s_pad > pks_delta, s_pad, 0)
            labs, _ = label(s_ft)
            labs = labs - 1
            if pks_cut:
                pks_idx, _ = find_peaks(s_ft)
                labs = self._cut_pks_labs(s=s_ft, labs=labs, pks=pks_idx)
            opt_s = self._merge_sparse_regs(s=s_ft, regs=labs, err_rtol=pks_err_rtol)
            if len(opt_s) == self.T:
                opt_s = opt_s[self.nzidx_s]

        self.s = np.abs(opt_s)
        return self.s, self.b

    def _compute_c(self, s: np.ndarray = None) -> np.ndarray:
        """Compute c from s via convolution."""
        if s is not None:
            return self._solver.convolve(s)
        else:
            return self._solver.convolve(self.s)

    def _res_err(self, r: np.ndarray) -> float:
        """Compute residual error."""
        if self.err_wt is not None:
            r = self.err_wt * r
        if self.cfg.norm == "l1":
            return float(np.linalg.norm(r, ord=1))
        elif self.cfg.norm == "l2":
            return float(np.dot(r,r))
            #np.sum(r**2)
        elif self.cfg.norm == "huber":
            # True Huber loss:
            # 0.5*r^2                    if |r| <= k
            # k*(|r| - 0.5*k)            otherwise
            k = float(self._solver.huber_k)
            ar = np.abs(r)
            quad = 0.5 * (r**2)
            lin = k * (ar - 0.5 * k)
            return float(np.sum(np.where(ar <= k, quad, lin)))

# In process of refactoring this function
    def _compute_err(
        self,
        y_fit: np.ndarray = None,
        b: float = None,
        c: np.ndarray = None,
        s: np.ndarray = None,
        res: np.ndarray = None,
        obj_crit: str = None,
    ) -> float:
        """Compute error/objective value using AIC and BIC.
        
        Args:   y_fit: Fitted fluorescence trace
                b: Background
                c: Calcium trace
                s: Spike train
                res: Residual
                obj_crit: Objective criterion to compute
        """
        #y = np.array(self.y)  I am changing this to self.y because I don't think y needs to be copied. should speed things up. 
        y = self.y
        if res is not None:
            y = y - res
        if b is None:
            b = self.b
        y = y - b
        if y_fit is None:
            if c is None:
                c = self._compute_c(s)
            R = self.R
            if sps.issparse(c):
                y_fit = np.array((R @ c * self.scale).todense()).squeeze()
            else:
                y_fit = np.array(R @ c * self.scale).squeeze()

        r = y - y_fit
        err = self._res_err(r)

        if obj_crit in [None, "spk_diff"]:
            return float(err)
        else:
            nspk = (s > 0).sum() if s is not None else (self.s > 0).sum()
            if obj_crit == "mean_spk":
                err_total = self._res_err(y - y.mean())
                return float((err - err_total) / max(nspk, 1))
            elif obj_crit in ["aic", "bic"]:
                T = len(r)
                mu = r.mean()
                r_hat = r - mu
                var = np.dot(r_hat, r_hat) / T 
                sigma = max(var, 1e-10)
                #sigma = max(((r - mu) ** 2).sum() / T, 1e-10)
                #logL = -0.5 * (
                #    T * np.log(2 * np.pi * sigma) + 1 / sigma * ((r - mu) ** 2).sum()
                #)
                logL = -0.5 * T * (np.log(2 * np.pi * sigma) + 1) #Simplified this bc ((r - mu) ** 2).sum() should just be T * sigma 
                
                if obj_crit == "aic":
                    return float(2 * (nspk - logL))
                elif obj_crit == "bic":
                    return float(nspk * np.log(T) - 2 * logL)
            return float(err)

    def _max_thres(self, s, nz_only=True):
        """Apply max thresholding to solution."""
        S_ls, thres = max_thres(
            np.array(s),
            nthres=self.cfg.nthres,
            th_min=self.cfg.th_min,
            th_max=self.cfg.th_max,
            reverse_thres=True,
            return_thres=True,
            nz_only=nz_only,
        )
        # Ensure we return numpy arrays (not xarray.DataArray)
        S_ls = [np.array(ss) for ss in S_ls]

        # Apply density threshold
        if self.cfg.density_thres is not None:  
            Sden = [ss.sum() / self.T for ss in S_ls]
            S_ls = [ss for ss, den in zip(S_ls, Sden) if den < self.cfg.density_thres]
            thres = [th for th, den in zip(thres, Sden) if den < self.cfg.density_thres]

        # Apply consecutive threshold
        if self.cfg.ncons_thres is not None:
            S_pad = [self._pad_s(ss) for ss in S_ls]
            Sncons = [max_consecutive(np.array(ss)) for ss in S_pad]
            if len(Sncons) > 0 and min(Sncons) < self.cfg.ncons_thres:
                S_ls = [
                    ss
                    for ss, ncons in zip(S_ls, Sncons)
                    if ncons <= self.cfg.ncons_thres
                ]
                thres = [
                    th
                    for th, ncons in zip(thres, Sncons)
                    if ncons <= self.cfg.ncons_thres
                ]
            elif len(S_ls) > 0:
                S_ls = [S_ls[0]]
                thres = [thres[0]]

        return S_ls, thres

    def solve_thres(
        self,
        scaling: bool = True,
        amp_constraint: bool = True,
        ignore_res: bool = False,
        return_intm: bool = False,
        pks_polish: bool = None,
        obj_crit: str = None,
    ) -> Tuple[np.ndarray, ...]:
        """Solve with thresholding."""
        y = np.array(self.y)
        opt_s, opt_b = self.solve(amp_constraint=amp_constraint, pks_polish=pks_polish)
        R = self.R

        if ignore_res:
            c = self._compute_c(opt_s)
            if sps.issparse(c):
                res = y - opt_b - self.scale * np.array((R @ c).todense()).squeeze()
            else:
                res = y - opt_b - self.scale * (R @ c).squeeze()
        else:
            res = np.zeros_like(y)

        svals, thres = self._max_thres(opt_s)
        if not len(svals) > 0:
            if return_intm:
                svals, thres = self._max_thres(opt_s, nz_only=False)
            else:
                return (
                    np.full(len(self.nzidx_s), np.nan),
                    np.full(len(self.nzidx_c), np.nan),
                    0,
                    np.inf,
                )

        cvals = [self._compute_c(s) for s in svals]

        def to_arr(m):
            return (
                np.array(m.todense()).squeeze()
                if sps.issparse(m)
                else np.array(m).squeeze()
            )

        yfvals = [to_arr(R @ c) for c in cvals]

        if scaling:
            scal_fit = [scal_lstsq(yf, y - res, fit_intercept=True) for yf in yfvals]
            scals = [sf[0] for sf in scal_fit]
            bs = [sf[1] for sf in scal_fit]

            if self.cfg.min_rel_scl is not None:
                scl_thres = np.max(y) * self.cfg.min_rel_scl
                valid_idx = np.where(np.array(scals) > scl_thres)[0]
                if len(valid_idx) > 0:
                    # Optional legacy compatibility: reproduce old deconv behavior where
                    # scale filtering shrinks `scals/bs` but does NOT shrink `svals/yfvals`.
                    # This leads to zip() truncation later and can change which candidate is selected.
                    legacy_bug = (
                        os.environ.get("INDECA_LEGACY_SCALE_FILTER_BUG", "0") == "1"
                    )
                    scals = [scals[i] for i in valid_idx]
                    bs = [bs[i] for i in valid_idx]
                    if not legacy_bug:
                        svals = [svals[i] for i in valid_idx]
                        cvals = [cvals[i] for i in valid_idx]
                        yfvals = [yfvals[i] for i in valid_idx]
                else:
                    max_idx = np.argmax(scals)
                    legacy_bug = (
                        os.environ.get("INDECA_LEGACY_SCALE_FILTER_BUG", "0") == "1"
                    )
                    scals = [scals[max_idx]]
                    bs = [bs[max_idx]]
                    if not legacy_bug:
                        svals = [svals[max_idx]]
                        cvals = [cvals[max_idx]]
                        yfvals = [yfvals[max_idx]]
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
            nspk_diff = np.where(nspk_diff == 0, 1, nspk_diff)  # Avoid division by zero
            merr_diff = objs_diff / nspk_diff
            avg_err = (objs_pad.min() - err_null) / max(nspk.max(), 1)
            opt_idx = (
                int(np.max(np.where(merr_diff < avg_err)[0]))
                if np.any(merr_diff < avg_err)
                else 0
            )
            objs = merr_diff
        else:
            opt_idx = int(np.argmin(objs))

        s_bin = svals[opt_idx]
        self.s_bin = s_bin
        self.c_bin = to_arr(cvals[opt_idx])
        self.b = bs[opt_idx]
        self.solver.s_bin = s_bin  # Update solver's s_bin for adaptive weighting

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
    ) -> Tuple[np.ndarray, ...]:
        """Solve with penalty optimization via DIRECT."""
        if self.cfg.penal is None:
            opt_s, opt_c, opt_scl, opt_obj = self.solve_thres(
                scaling=scaling, return_intm=return_intm, pks_polish=pks_polish
            )
            opt_penal = 0
            if return_intm:
                return opt_s, opt_c, opt_scl, opt_obj, opt_penal, None
            return opt_s, opt_c, opt_scl, opt_obj, opt_penal

        pn = f"{self.cfg.penal}_penal"
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

        # Find upper bound for penalty
        ub, ub_last = err_full, err_full
        for _ in range(int(np.ceil(np.log2(ub + 1)))):
            self.update(**{pn: ub})
            s, b = self.solve(pks_polish=pks_polish)
            cur_err = self._compute_err(s=s, b=b)
            if np.abs(cur_err - err_min) < 0.5 * np.abs(err_full - err_min):
                ub = ub_last
                break
            else:
                ub_last = ub
                ub = ub / 2

        def opt_fn(x):
            self.update(**{pn: float(x)})
            _, _, _, obj = self.solve_thres(scaling=False, pks_polish=pks_polish)
            if self.dashboard is not None:
                self.dashboard.update(
                    uid=self.dashboard_uid,
                    penal_err={"penal": float(x), "scale": self.scale, "err": obj},
                )
            return obj if obj < err_full else np.inf

        try:
            res = direct(
                opt_fn,
                bounds=[(0, max(ub, 1e-6))],
                maxfun=self.cfg.max_iter_penal,
                locally_biased=False,
                vol_tol=1e-2,
            )
            direct_pn = res.x
            if not res.success:
                logger.warning(
                    f"Could not find optimal penalty within {res.nfev} iterations"
                )
                opt_penal = 0
            elif err_nopn <= opt_fn(direct_pn):
                opt_penal = 0
            else:
                opt_penal = float(direct_pn)
        except Exception as e:
            logger.warning(f"DIRECT optimization failed: {e}")
            opt_penal = 0

        self.update(**{pn: opt_penal})
        if return_intm:
            opt_s, opt_c, opt_scl, opt_obj, intm = self.solve_thres(
                scaling=scaling, return_intm=True, pks_polish=pks_polish
            )
            return opt_s, opt_c, opt_scl, opt_obj, opt_penal, intm
        else:
            opt_s, opt_c, opt_scl, opt_obj = self.solve_thres(
                scaling=scaling, pks_polish=pks_polish
            )
            if opt_scl == 0:
                logger.warning("Could not find non-zero solution")
            return opt_s, opt_c, opt_scl, opt_obj, opt_penal

    def solve_scale(
        self,
        reset_scale: bool = True,
        concur_penal: bool = False,
        return_met: bool = False,
        obj_crit: str = None,
        early_stop: bool = True,
        masking: bool = True,
    ) -> Tuple[np.ndarray, ...]:
        """Solve with iterative scale estimation."""
        if self.cfg.penal in ["l0", "l1"]:
            pn = f"{self.cfg.penal}_penal"
            self.update(**{pn: 0})

        self._reset_cache()
        self._reset_mask()

        if reset_scale:
            self.update(scale=1)
            s_free, _ = self.solve(amp_constraint=False)
            self.update(scale=np.ptp(s_free))

        metric_df = None
        for i in range(self.cfg.max_iter_scal):
            if concur_penal:
                cur_s, cur_c, cur_scl, cur_obj_raw, cur_penal = self.solve_penal(
                    scaling=i > 0,
                    pks_polish=self.cfg.pks_polish and (i > 1 or not reset_scale),
                )
            else:
                cur_penal = 0
                cur_s, cur_c, cur_scl, cur_obj_raw = self.solve_thres(
                    scaling=i > 0,
                    pks_polish=self.cfg.pks_polish and (i > 1 or not reset_scale),
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
                opt_obj = metric_df.loc[opt_idx, "obj"]
                opt_scal = metric_df.loc[opt_idx, "scale"]
                prev_scals = np.array(metric_df["scale"])
                last_scal = prev_scals[-1]
                last_obj = np.array(metric_df["obj"])[-1]

            y_wt = np.array(self.y * self.err_wt)
            err_tt = self._res_err(y_wt - y_wt.mean())
            cur_obj = (cur_obj_raw - err_tt) / max(err_tt, 1e-10)

            cur_met = pd.DataFrame(
                [
                    {
                        "iter": i,
                        "scale": cur_scl,
                        "obj_raw": cur_obj_raw,
                        "obj": cur_obj,
                        "penal": cur_penal,
                        "nnz": (cur_s > 0).sum(),
                        "density": (cur_s > 0).sum() / self.T,
                    }
                ]
            )
            metric_df = pd.concat([metric_df, cur_met], ignore_index=True)

            if self.cfg.err_weighting == "adaptive" and i <= 1:
                self.update(update_weighting=True)
            if masking and i >= 1:
                self._update_mask()

            if any(
                [
                    np.abs(cur_scl - opt_scal) < self.cfg.rtol * opt_scal,
                    np.abs(cur_obj - opt_obj) < self.cfg.rtol * opt_obj,
                    np.abs(cur_scl - last_scal) < self.cfg.atol,
                    np.abs(cur_obj - last_obj) < self.cfg.atol * 1e-3,
                    early_stop and cur_obj > last_obj,
                ]
            ):
                break
            elif cur_scl == 0:
                warnings.warn("Exit with zero solution")
                break
            elif np.abs(cur_scl - prev_scals).min() < self.cfg.atol:
                self.update(scale=(cur_scl + last_scal) / 2)
            else:
                self.update(scale=cur_scl)
        else:
            warnings.warn("Max scale iterations reached")

        # Final solve with optimal scale
        opt_idx = metric_df["obj"].idxmin()
        self.update(update_weighting=True, clear_weighting=True)
        self._reset_cache()
        self._reset_mask()
        self.update(scale=float(metric_df.loc[opt_idx, "scale"]))

        cur_s, cur_c, cur_scl, cur_obj, cur_penal = self.solve_penal(
            scaling=False, masking=False, pks_polish=self.cfg.pks_polish
        )

        opt_s = np.zeros(self.T)
        opt_c = np.zeros(self.T)
        opt_s[self.nzidx_s] = cur_s
        opt_c[self.nzidx_c] = (
            cur_c if not sps.issparse(cur_c) else cur_c.toarray().squeeze()
        )
        nnz = int(opt_s.sum())

        self.update(update_weighting=True)
        y_wt = np.array(self.y * self.err_wt)
        err_tt = self._res_err(y_wt - y_wt.mean())
        err_cur = self._compute_err(s=opt_s)
        err_rel = (err_cur - err_tt) / max(err_tt, 1e-10)

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
