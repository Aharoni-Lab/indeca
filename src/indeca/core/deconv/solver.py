"""Solver implementations for deconv."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import warnings

import cvxpy as cp
import numpy as np
import osqp
import scipy.sparse as sps
from scipy.special import huber
from scipy.signal import ShortTimeFFT

from indeca.utils.logging_config import get_module_logger
from indeca.core.simulation import tau2AR, solve_p, exp_pulse, ar_pulse
from indeca.utils.utils import scal_lstsq
from .config import DeconvConfig
from .utils import construct_R, bin_convolve, get_stft_spec

logger = get_module_logger("deconv_solver")

# Try to import GPU solver
try:
    import cuosqp

    HAS_CUOSQP = True
except ImportError:
    HAS_CUOSQP = False
    logger.debug("cuosqp not available")


class DeconvSolver(ABC):
    """Abstract base class for deconvolution solvers."""

    def __init__(
        self,
        config: DeconvConfig,
        y_len: int,
        y: np.ndarray | None = None,
        coef: np.ndarray | None = None,
        theta: np.ndarray | None = None,
        tau: np.ndarray | None = None,
        ps: np.ndarray | None = None,
    ):
        self.cfg = config
        self.y_len = y_len
        self.T = y_len * self.cfg.upsamp
        self.y = y if y is not None else np.zeros(y_len)
        self.coef = coef
        self.coef_len = (
            len(coef) if coef is not None else config.coef_len * config.upsamp
        )
        self.theta = theta
        self.tau = tau
        self.ps = ps

        # Scale tracking (mutable, since config is frozen)
        self.scale = config.scale

        # Penalty tracking
        self.l0_penal = 0.0
        self.l1_penal = 0.0

        # Weight vectors
        self.w_org = np.ones(self.T)
        self.w = np.ones(self.T)

        # Masking indices
        self.nzidx_s = np.arange(self.T)
        self.nzidx_c = np.arange(self.T)

        # Matrices
        self.R_org = construct_R(self.y_len, self.cfg.upsamp)
        self.R = self.R_org
        self.H = None
        self.H_org = None
        self.G = None
        self.G_org = None

        # Cache
        self.x_cache = None
        self.s_bin = None  # Binary spike solution from thresholding

        # Error weighting
        self.err_wt = np.ones(self.y_len)
        self.wgt_len = self.coef_len
        self.Wt = sps.diags(self.err_wt)
        # Precompute squared version for P matrix
        self._Wt_sq = sps.diags(self.err_wt ** 2)

        # Huber parameter
        self.huber_k = 0.5 * np.std(self.y) if y is not None else 0

    @abstractmethod
    def update(self, **kwargs):
        """Update solver parameters."""
        pass

    @abstractmethod
    def solve(self, amp_constraint: bool = True) -> Tuple[np.ndarray, float, Any]:
        """Solve the optimization problem."""
        pass

    def reset_cache(self) -> None:
        """Reset solution cache."""
        self.x_cache = None

    def reset_mask(self) -> None:
        """Reset masks to full range."""
        self.nzidx_s = np.arange(self.T)
        self.nzidx_c = np.arange(self.T)
        self._update_R()
        self._update_w()

    def set_mask(self, nzidx_s: np.ndarray, nzidx_c: np.ndarray = None):
        """Set mask indices. Override in subclasses that don't support masking."""
        self.nzidx_s = nzidx_s
        # Old behavior (from `old deconv.py`): masking is applied to spike indices (s)
        # while keeping the calcium state indices (c) unmasked unless explicitly provided.
        if nzidx_c is not None:
            self.nzidx_c = nzidx_c
        self._update_R()
        self._update_w()

    def _update_R(self) -> None:
        """Update R matrix based on mask."""
        self.R = self.R_org[:, self.nzidx_c]

    def _update_w(self, w_new: np.ndarray = None) -> None:
        """Update weight vector."""
        if w_new is not None:
            self.w_org = w_new
        self.w = self.w_org[self.nzidx_s]

    def _pad_s(self, s: np.ndarray = None) -> np.ndarray:
        """Pad sparse s to full length."""
        if s is None:
            s = np.zeros(len(self.nzidx_s))
        s_ret = np.zeros(self.T)
        s_ret[self.nzidx_s] = s
        return s_ret

    def _pad_c(self, c: np.ndarray = None) -> np.ndarray:
        """Pad sparse c to full length."""
        if c is None:
            c = np.zeros(len(self.nzidx_c))
        c_ret = np.zeros(self.T)
        c_ret[self.nzidx_c] = c
        return c_ret

    def _update_HG(self) -> None:
        """Update H (convolution) and G (AR inverse) matrices."""
        coef = self.coef
        if coef is None:
            return

        # H matrix: convolution matrix
        # IMPORTANT: in free-kernel mode the optimization uses R @ H explicitly,
        # so H must always be materialized (do not drop it based on Hlim).
        if (
            self.cfg.free_kernel
            or self.cfg.Hlim is None
            or self.T * len(coef) < self.cfg.Hlim
        ):
            self.H_org = sps.diags(
                [np.repeat(coef[i], self.T - i) for i in range(len(coef))],
                offsets=-np.arange(len(coef)),
                format="csc",
            )
            self.H = self.H_org[:, self.nzidx_s][self.nzidx_c, :]
            logger.debug(f"Updated H matrix - shape: {self.H.shape}, nnz: {self.H.nnz}")
        else:
            self.H = None
            self.H_org = None

        # G matrix: AR inverse (only if theta provided and not free_kernel)
        if not self.cfg.free_kernel and self.theta is not None:
            theta = self.theta
            G_diag = sps.diags(
                [np.ones(self.T - 1)]
                + [np.repeat(-theta[i], self.T - 2 - i) for i in range(theta.shape[0])],
                offsets=np.arange(0, -theta.shape[0] - 1, -1),
                format="csc",
            )
            self.G_org = sps.bmat(
                [[None, G_diag], [np.zeros((1, 1)), None]], format="csc"
            )
            self.G = self.G_org[:, self.nzidx_c][self.nzidx_s, :]
            logger.debug(f"Updated G matrix - shape: {self.G.shape}, nnz: {self.G.nnz}")
        else:
            self.G = None
            self.G_org = None

    def _update_wgt_len(self) -> None:
        """Update error weighting length based on coefficient truncation."""
        coef = self.coef
        if coef is None:
            return
        if self.cfg.wt_trunc_thres is not None:
            trunc_idx = np.where(coef > self.cfg.wt_trunc_thres)[0]
            if len(trunc_idx) > 0:
                trunc_len = int(np.around(trunc_idx[-1] / self.cfg.upsamp))
            else:
                trunc_len = int(np.around(np.where(coef > 0)[0][-1] / self.cfg.upsamp))
            if trunc_len == 0:
                trunc_len = 1
            self.wgt_len = max(min(self.coef_len, trunc_len), 1)
        else:
            self.wgt_len = self.coef_len

    def convolve(self, s: np.ndarray) -> sps.csc_matrix:
        """Convolve signal s with kernel. Returns sparse column matrix."""
        if self.cfg.free_kernel:
            assert (
                self.H is not None
            ), "Invariant violated: free_kernel=True requires a materialized H matrix"
        if self.H is not None:
            # Check if s is masked length or full length
            if len(s) == len(self.nzidx_s):
                result = self.H @ sps.csc_matrix(s.reshape(-1, 1))
            elif len(s) == self.T:
                result = self.H @ sps.csc_matrix(s[self.nzidx_s].reshape(-1, 1))
            else:
                logger.warning(
                    f"Shape mismatch in convolve: s={len(s)}, nzidx_s={len(self.nzidx_s)}"
                )
                result = sps.csc_matrix(np.zeros((len(self.nzidx_c), 1)))
            return result
        else:
            # Use bin_convolve for efficiency when H is not stored
            if s.dtype == np.bool_:
                out = bin_convolve(self.coef, s, nzidx_s=self.nzidx_s, s_len=self.T)
            else:
                s_pad = self._pad_s(s) if len(s) == len(self.nzidx_s) else s
                out = np.convolve(self.coef, s_pad)[: self.T]
            return sps.csc_matrix(out[self.nzidx_c].reshape(-1, 1))

    def validate_coefficients(self, atol: float = 1e-3) -> bool:
        """Validate that AR and exponential coefficients are consistent."""
        if self.tau is None or self.ps is None or self.theta is None:
            logger.debug("Skipping coefficient validation - missing tau/ps/theta")
            return True

        try:
            # Generate exponential pulse
            tr_exp, _, _ = exp_pulse(
                self.tau[0],
                self.tau[1],
                p_d=self.ps[0],
                p_r=self.ps[1],
                nsamp=self.coef_len,
            )

            # Generate AR pulse
            theta = self.theta
            tr_ar, _, _ = ar_pulse(
                theta[0], theta[1], nsamp=self.coef_len, shifted=True
            )

            # Validate
            if not (~np.isnan(self.coef)).all():
                logger.warning("Coefficient array contains NaN values")
                return False

            if not np.isclose(tr_exp, self.coef[: len(tr_exp)], atol=atol).all():
                logger.warning("Exp time constant inconsistent with coefficients")
                return False

            if not np.isclose(tr_ar, self.coef[: len(tr_ar)], atol=atol).all():
                logger.warning("AR coefficients inconsistent with coefficients")
                return False

            logger.debug("Coefficient validation passed")
            return True
        except Exception as e:
            logger.warning(f"Coefficient validation failed: {e}")
            return False


class CVXPYSolver(DeconvSolver):
    """CVXPY backend solver."""

    def __init__(self, config: DeconvConfig, y_len: int, **kwargs):
        super().__init__(config, y_len, **kwargs)
        self._update_HG()
        self._update_wgt_len()
        self._setup_problem()

    def set_mask(self, nzidx_s: np.ndarray, nzidx_c: np.ndarray = None):
        """CVXPY does not support masking - raise error."""
        if len(nzidx_s) != self.T:
            raise NotImplementedError(
                "CVXPY backend does not support masking. Use OSQP backend instead."
            )
        super().set_mask(nzidx_s, nzidx_c)

    def reset_mask(self) -> None:
        """CVXPY does not support masking - no-op since problem is already full."""
        # CVXPY builds the full problem once, no mask support
        # Just reset indices without rebuilding
        self.nzidx_s = np.arange(self.T)
        self.nzidx_c = np.arange(self.T)

    def _setup_problem(self):
        """Setup CVXPY optimization problem."""
        # NOTE: `free_kernel=True` is forbidden with CVXPY backend (see `DeconvConfig`).
        self.cp_R = cp.Constant(self.R, name="R")
        self.cp_c = cp.Variable((self.T, 1), nonneg=True, name="c")
        self.cp_s = cp.Variable(
            (self.T, 1), nonneg=True, name="s", boolean=self.cfg.mixin
        )
        self.cp_y = cp.Parameter(shape=(self.y_len, 1), name="y")
        self.cp_huber_k = cp.Parameter(
            value=float(self.huber_k), nonneg=True, name="huber_k"
        )

        self.cp_scale = cp.Parameter(value=self.scale, name="scale", nonneg=True)
        self.cp_l1_penal = cp.Parameter(value=0.0, name="l1_penal", nonneg=True)
        self.cp_l0_w = cp.Parameter(
            shape=self.T, value=np.zeros(self.T), nonneg=True, name="w_l0"
        )

        if self.y is not None:
            self.cp_y.value = self.y.reshape((-1, 1))

        if self.cfg.use_base:
            self.cp_b = cp.Variable(nonneg=True, name="b")
        else:
            self.cp_b = cp.Constant(value=0, name="b")

        # Error term based on norm
        term = self.cp_y - self.cp_scale * self.cp_R @ self.cp_c - self.cp_b
        if self.cfg.norm == "l1":
            self.err_term = cp.sum(cp.abs(term))
        elif self.cfg.norm == "l2":
            self.err_term = cp.sum_squares(term)
        elif self.cfg.norm == "huber":
            # Keep huber parameter consistent with OSQP backend's `huber_k`.
            self.err_term = cp.sum(cp.huber(term, M=self.cp_huber_k))

        # Objective
        obj_expr = (
            self.err_term
            + self.cp_l0_w.T @ cp.abs(self.cp_s)
            + self.cp_l1_penal * cp.sum(cp.abs(self.cp_s))
        )
        obj = cp.Minimize(obj_expr)

        # Constraints
        # AR constraint via G matrix
        self.cp_theta = cp.Parameter(
            value=self.theta, shape=self.theta.shape, name="theta"
        )
        G_diag = sps.eye(self.T - 1) + sum(
            [
                cp.diag(cp.promote(-self.cp_theta[i], (self.T - i - 2,)), -i - 1)
                for i in range(self.theta.shape[0])
            ]
        )
        G = cp.bmat(
            [
                [np.zeros((self.T - 1, 1)), G_diag],
                [np.zeros((1, 1)), np.zeros((1, self.T - 1))],
            ]
        )
        dcv_cons = [self.cp_s == G @ self.cp_c]

        edge_cons = [self.cp_c[0, 0] == 0, self.cp_s[-1, 0] == 0]
        amp_cons = [self.cp_s <= 1]

        self.prob_free = cp.Problem(obj, dcv_cons + edge_cons)
        self.prob = cp.Problem(obj, dcv_cons + edge_cons + amp_cons)

    def update(
        self,
        y: np.ndarray = None,
        coef: np.ndarray = None,
        scale: float = None,
        scale_mul: float = None,
        l1_penal: float = None,
        l0_penal: float = None,
        w: np.ndarray = None,
        theta: np.ndarray = None,
        **kwargs,
    ):
        """Update CVXPY parameters."""
        if y is not None:
            self.y = y
            self.cp_y.value = y.reshape((-1, 1))
            # Keep huber_k consistent with OSQP backend / deconv objective.
            self.huber_k = 0.5 * np.std(self.y)
            self.cp_huber_k.value = float(self.huber_k)
        if coef is not None:
            self.coef = coef
            self._update_HG()
            self._update_wgt_len()
        if scale is not None:
            self.scale = scale
            self.cp_scale.value = scale
        if scale_mul is not None:
            self.scale *= scale_mul
            self.cp_scale.value = self.scale
        if l1_penal is not None:
            self.l1_penal = l1_penal
            self.cp_l1_penal.value = l1_penal
        if l0_penal is not None:
            self.l0_penal = l0_penal
        if w is not None:
            self._update_w(w)
        if l0_penal is not None or w is not None:
            self.cp_l0_w.value = self.l0_penal * self.w
        if theta is not None and hasattr(self, "cp_theta"):
            self.theta = theta
            self.cp_theta.value = theta

    def solve(self, amp_constraint: bool = True) -> Tuple[np.ndarray, float, Any]:
        """Solve CVXPY problem."""
        prob = self.prob if amp_constraint else self.prob_free
        try:
            res = prob.solve()
        except cp.error.SolverError as e:
            logger.warning(f"CVXPY SolverError: {e}")
            res = np.inf

        opt_s = (
            self.cp_s.value.squeeze()
            if self.cp_s.value is not None
            else np.zeros(self.T)
        )
        opt_b = 0
        if (
            self.cfg.use_base
            and hasattr(self.cp_b, "value")
            and self.cp_b.value is not None
        ):
            opt_b = float(self.cp_b.value)

        return opt_s, opt_b, res


class OSQPSolver(DeconvSolver):
    """OSQP backend solver (also handles cuosqp for GPU)."""

    def __init__(self, config: DeconvConfig, y_len: int, **kwargs):
        super().__init__(config, y_len, **kwargs)

        # Additional state for OSQP
        self.prob = None
        self.prob_free = None
        self.P = None
        self.q = None
        self.q0 = None
        self.A = None
        self.lb = None
        self.ub = None
        self.ub_inf = None
        self.nzidx_A = None

        # STFT for FFT weighting
        if self.cfg.err_weighting == "fft":
            self.stft = ShortTimeFFT(win=np.ones(self.coef_len), hop=1, fs=1)
            self.yspec = get_stft_spec(self.y, self.stft)

        # Initialize matrices and problem
        self._update_HG()
        self._update_wgt_len()
        self._update_Wt()
        self._setup_prob_osqp()

    def reset_mask(self) -> None:
        """Reset masks to full range and rebuild OSQP problems."""
        super().reset_mask()
        self._update_HG()
        self._setup_prob_osqp()

    def set_mask(self, nzidx_s: np.ndarray, nzidx_c: np.ndarray = None):
        """Set mask and rebuild problem."""
        super().set_mask(nzidx_s, nzidx_c)
        self._update_HG()
        self._setup_prob_osqp()

    def _update_Wt(self, clear: bool = False) -> None:
        """Update error weighting matrix."""
        coef = self.coef
        if clear:
            logger.debug("Clearing error weighting")
            self.err_wt = np.ones(self.y_len)
        elif self.cfg.err_weighting == "fft" and hasattr(self, "stft"):
            logger.debug("Updating error weighting with fft")
            hspec = get_stft_spec(coef, self.stft)[:, int(len(coef) / 2)]
            self.err_wt = (
                (hspec.reshape(-1, 1) * self.yspec).sum(axis=0)
                / np.linalg.norm(hspec)
                / np.linalg.norm(self.yspec, axis=0)
            )
        elif self.cfg.err_weighting == "corr":
            logger.debug("Updating error weighting with corr")
            self.err_wt = np.ones(self.y_len)
            for i in range(self.y_len):
                yseg = self.y[i : i + len(coef)]
                if len(yseg) <= 1:
                    continue
                cseg = coef[: len(yseg)]
                with np.errstate(all="ignore"):
                    self.err_wt[i] = np.corrcoef(yseg, cseg)[0, 1].clip(0, 1)
            self.err_wt = np.nan_to_num(self.err_wt)
        elif self.cfg.err_weighting == "adaptive":
            if self.s_bin is not None:
                self.err_wt = np.zeros(self.y_len)
                s_bin_R = self.R @ self._pad_s(self.s_bin)
                for nzidx in np.where(s_bin_R > 0)[0]:
                    self.err_wt[nzidx : nzidx + self.wgt_len] = 1
            else:
                self.err_wt = np.ones(self.y_len)

        self.Wt = sps.diags(self.err_wt)
        # Precompute squared version for P matrix
        self._Wt_sq = sps.diags(self.err_wt ** 2)

    def _get_M(self) -> sps.csc_matrix:
        """Get the combined model matrix M = [1, scale*R] or [1, scale*R@H]."""
        if self.cfg.free_kernel:
            return sps.hstack(
                [np.ones((self.R.shape[0], 1)), self.scale * self.R @ self.H],
                format="csc",
            )
        else:
            return sps.hstack(
                [np.ones((self.R.shape[0], 1)), self.scale * self.R],
                format="csc",
            )

    def _update_P(self) -> None:
        """Update quadratic cost matrix P."""
        if self.cfg.norm == "l1":
            raise NotImplementedError("l1 norm not yet supported with OSQP backend")
        elif self.cfg.norm == "l2":
            M = self._get_M()
            # NOTE: This is the original code
            # P = M.T @ self.Wt.T @ self.Wt @ M
            # NOTE: This is the optimized code
            # We precompute Wt_sq each time Wt is updated
            P = M.T @ self._Wt_sq @ M


        elif self.cfg.norm == "huber":
            lc = len(self.nzidx_c)
            ls = len(self.nzidx_s)
            ly = self.y_len
            if self.cfg.free_kernel:
                P = sps.bmat(
                    [
                        [sps.csc_matrix((ls + 1, ls + 1)), None, None],
                        [None, sps.csc_matrix((ly, ly)), None],
                        [None, None, sps.eye(ly, format="csc")],
                    ]
                )
            else:
                P = sps.bmat(
                    [
                        [sps.csc_matrix((lc + 1, lc + 1)), None, None],
                        [None, sps.csc_matrix((ly, ly)), None],
                        [None, None, sps.eye(ly, format="csc")],
                    ]
                )

        self.P = sps.triu(P).tocsc()
        logger.debug(f"Updated P matrix - shape: {self.P.shape}, nnz: {self.P.nnz}")

    def _update_q0(self) -> None:
        """Update linear cost base q0."""
        if self.cfg.norm == "l1":
            raise NotImplementedError("l1 norm not yet supported with OSQP backend")
        elif self.cfg.norm == "l2":
            M = self._get_M()
            self.q0 = -M.T @ self.Wt.T @ self.Wt @ self.y
        elif self.cfg.norm == "huber":
            ly = self.y_len
            lx = (
                len(self.nzidx_s) + 1 if self.cfg.free_kernel else len(self.nzidx_c) + 1
            )
            self.q0 = (
                np.concatenate([np.zeros(lx), np.ones(ly), np.ones(ly)]) * self.huber_k
            )

    def _update_q(self) -> None:
        """Update linear cost vector q (including penalties)."""
        if self.cfg.norm == "l1":
            raise NotImplementedError("l1 norm not yet supported with OSQP backend")
        elif self.cfg.norm == "l2":
            if self.cfg.free_kernel:
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
        elif self.cfg.norm == "huber":
            pad_k = np.zeros(self.y_len)
            if self.cfg.free_kernel:
                self.q = (
                    self.q0
                    + self.l0_penal * np.concatenate([[0], self.w, pad_k, pad_k])
                    + self.l1_penal
                    * np.concatenate([[0], np.ones(len(self.nzidx_s)), pad_k, pad_k])
                )
            else:
                self.q = (
                    self.q0
                    + self.l0_penal
                    * np.concatenate([[0], self.w @ self.G, pad_k, pad_k])
                    + self.l1_penal
                    * np.concatenate(
                        [[0], np.ones(self.G.shape[0]) @ self.G, pad_k, pad_k]
                    )
                )

    def _update_A(self) -> None:
        """Update constraint matrix A."""
        if self.cfg.free_kernel:
            Ax = sps.eye(len(self.nzidx_s), format="csc")
            Ar = self.scale * self.R @ self.H
        else:
            Ax = sps.csc_matrix(self.G_org[:, self.nzidx_c])
            # Record spike terms that require constraint
            self.nzidx_A = np.where((Ax != 0).sum(axis=1))[0]
            Ax = Ax[self.nzidx_A, :]
            Ar = self.scale * self.R

        if self.cfg.norm == "huber":
            e = sps.eye(self.y_len, format="csc")
            self.A = sps.bmat(
                [
                    [sps.csc_matrix((Ax.shape[0], 1)), Ax, None, None],
                    [None, None, e, None],
                    [None, None, None, -e],
                    [np.ones((Ar.shape[0], 1)), Ar, e, e],
                ],
                format="csc",
            )
        else:
            self.A = sps.bmat([[np.ones((1, 1)), None], [None, Ax]], format="csc")

        logger.debug(f"Updated A matrix - shape: {self.A.shape}, nnz: {self.A.nnz}")

    def _update_bounds(self) -> None:
        """Update constraint bounds."""
        if self.cfg.norm == "huber":
            xlen = len(self.nzidx_s) if self.cfg.free_kernel else len(self.nzidx_A)
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
            bb = np.clip(self.y.mean(), 0, None) if self.cfg.use_base else 0
            if self.cfg.free_kernel:
                self.lb = np.zeros(len(self.nzidx_s) + 1)
                self.ub = np.concatenate([np.full(1, bb), np.ones(len(self.nzidx_s))])
                self.ub_inf = np.concatenate(
                    [np.full(1, bb), np.full(len(self.nzidx_s), np.inf)]
                )
            else:
                ub_pad = np.zeros(self.T)
                ub_inf_pad = np.zeros(self.T)
                ub_pad[self.nzidx_s] = 1
                ub_inf_pad[self.nzidx_s] = np.inf
                self.lb = np.zeros(len(self.nzidx_A) + 1)
                self.ub = np.concatenate([np.full(1, bb), ub_pad[self.nzidx_A]])
                self.ub_inf = np.concatenate([np.full(1, bb), ub_inf_pad[self.nzidx_A]])

        assert (self.ub >= self.lb).all(), "Upper bounds must be >= lower bounds"
        assert (
            self.ub_inf >= self.lb
        ).all(), "Upper bounds (inf) must be >= lower bounds"

    def _setup_prob_osqp(self) -> None:
        """Setup OSQP problem instances."""
        logger.debug("Setting up OSQP problem")

        self._update_P()
        self._update_q0()
        self._update_q()
        self._update_A()
        self._update_bounds()

        # Choose solver backend
        if self.cfg.backend == "cuosqp":
            if not HAS_CUOSQP:
                logger.warning("cuosqp not available, falling back to osqp")
                self.prob = osqp.OSQP()
                self.prob_free = osqp.OSQP()
            else:
                self.prob = cuosqp.OSQP()
                self.prob_free = cuosqp.OSQP()
        elif self.cfg.backend == "emosqp":
            # Stub: emosqp requires codegen, not supported in this refactor
            logger.warning("emosqp requires codegen, using osqp instead")
            self.prob = osqp.OSQP()
            self.prob_free = osqp.OSQP()
        else:
            self.prob = osqp.OSQP()
            self.prob_free = osqp.OSQP()

        # Setup constrained problem
        self.prob.setup(
            P=self.P.copy(),
            q=self.q.copy(),
            A=self.A.copy(),
            l=self.lb.copy(),
            u=self.ub.copy(),
            verbose=False,
            polish=True,
            warm_start=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            eps_prim_inf=1e-7,
            eps_dual_inf=1e-7,
        )

        # Setup unconstrained (free) problem
        self.prob_free.setup(
            P=self.P.copy(),
            q=self.q.copy(),
            A=self.A.copy(),
            l=self.lb.copy(),
            u=self.ub_inf.copy(),
            verbose=False,
            polish=True,
            warm_start=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            eps_prim_inf=1e-7,
            eps_dual_inf=1e-7,
        )

        logger.debug(f"{self.cfg.backend} setup completed successfully")

    def update(
        self,
        y: np.ndarray = None,
        coef: np.ndarray = None,
        tau: np.ndarray = None,
        theta: np.ndarray = None,
        scale: float = None,
        scale_mul: float = None,
        l1_penal: float = None,
        l0_penal: float = None,
        w: np.ndarray = None,
        update_weighting: bool = False,
        clear_weighting: bool = False,
        scale_coef: bool = False,
        **kwargs,
    ):
        """Update OSQP problem parameters."""
        logger.debug(f"Updating OSQP solver parameters")

        # Update input parameters
        if y is not None:
            self.y = y
            # Match legacy behavior: huber_k tracks the current y (used in q0 and bounds)
            self.huber_k = 0.5 * np.std(self.y)
        if tau is not None:
            theta_new = np.array(tau2AR(tau[0], tau[1]))
            p = solve_p(tau[0], tau[1])
            coef_new, _, _ = exp_pulse(
                tau[0], tau[1], p_d=p, p_r=-p, nsamp=self.coef_len, kn_len=self.coef_len
            )
            self.tau = tau
            self.ps = np.array([p, -p])
            self.theta = theta_new
            coef = coef_new
        if theta is not None:
            self.theta = theta
        if coef is not None:
            if scale_coef and self.coef is not None:
                scale_mul = scal_lstsq(coef, self.coef).item()
            self.coef = coef
        if scale is not None:
            self.scale = scale
        if scale_mul is not None:
            self.scale *= scale_mul
        if l1_penal is not None:
            self.l1_penal = l1_penal
        if l0_penal is not None:
            self.l0_penal = l0_penal
        if w is not None:
            self._update_w(w)

        # Track what needs updating
        updt_HG = coef is not None
        updt_P = False
        updt_q0 = False
        updt_q = False
        updt_A = False
        updt_bounds = False
        setup_prob = False

        if updt_HG:
            self._update_HG()
            self._update_wgt_len()

        if self.cfg.err_weighting is not None and update_weighting:
            self._update_Wt(clear=clear_weighting)
            if self.cfg.err_weighting == "adaptive":
                setup_prob = True
            else:
                updt_P = True
                updt_q0 = True
                updt_q = True

        if self.cfg.norm == "huber":
            # huber_k changes require recomputing q and bounds
            if y is not None:
                self._update_q0()
                updt_q0 = True
            if any([scale is not None, scale_mul is not None, updt_HG]):
                self._update_A()
                updt_A = True
            if any(
                [w is not None, l0_penal is not None, l1_penal is not None, updt_HG]
            ):
                self._update_q()
                updt_q = True
            if y is not None:
                self._update_bounds()
                updt_bounds = True
        else:
            if any([updt_HG, updt_A]):
                self._update_A()
                updt_A = True
            if any([scale is not None, scale_mul is not None, updt_HG, updt_P]):
                self._update_P()
                updt_P = True
            if any(
                [
                    scale is not None,
                    scale_mul is not None,
                    y is not None,
                    updt_HG,
                    updt_q0,
                ]
            ):
                self._update_q0()
                updt_q0 = True
            if any(
                [
                    w is not None,
                    l0_penal is not None,
                    l1_penal is not None,
                    updt_q0,
                    updt_q,
                ]
            ):
                self._update_q()
                updt_q = True

        # Apply updates to OSQP - conservative approach:
        # Only q can be updated in-place safely. For P, A, bounds, rebuild.
        if setup_prob or any([updt_P, updt_A, updt_bounds]):
            self._setup_prob_osqp()
        elif updt_q:
            self.prob.update(q=self.q)
            self.prob_free.update(q=self.q)

        logger.debug("OSQP problem updated")

    def solve(self, amp_constraint: bool = True) -> Tuple[np.ndarray, float, Any]:
        """Solve OSQP problem."""
        prob = self.prob if amp_constraint else self.prob_free
        res = prob.solve()

        if res.info.status not in ["solved", "solved inaccurate"]:
            logger.warning(f"OSQP not solved: {res.info.status}")
            if res.info.status in ["primal infeasible", "primal infeasible inaccurate"]:
                x = np.zeros(self.P.shape[0], dtype=float)
            else:
                x = (
                    res.x.astype(float)
                    if res.x is not None
                    else np.zeros(self.P.shape[0], dtype=float)
                )
        else:
            x = res.x

        if self.cfg.norm == "huber":
            xlen = (
                len(self.nzidx_s) + 1 if self.cfg.free_kernel else len(self.nzidx_c) + 1
            )
            sol = x[:xlen]
            opt_b = sol[0]
            if self.cfg.free_kernel:
                opt_s = sol[1:]
            else:
                opt_s = self.G @ sol[1:]
        else:
            opt_b = x[0]
            if self.cfg.free_kernel:
                opt_s = x[1:]
            else:
                c_sol = x[1:]
                opt_s = self.G @ c_sol

        # Return 0 for objective - caller should use _compute_err for correct objective
        return opt_s, opt_b, 0
