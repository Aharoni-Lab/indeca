from click import Tuple
import numpy as np
import scipy.sparse as sps
from numba import njit
from scipy.signal import ShortTimeFFT
from yaml import warnings
from indeca.core.deconv.utils import sum_downsample
from indeca.core.simulation import tau2AR
from scripts.s01_caiman_test import S_ls

# The following functions are defined sequentially as part of an iterative spike inference step where a solve function
# is first defined (solv), followed by an initial generation of thresholded spike trains (max_thres and _max_thres)
# and then the iterative search for the most ideal inferred spike train (solve_thres).

#This function allows us to generate multiple thresholded versions of an input array used as part 
# of iterative spike inference process

def max_thres(
    a: np.ndarray,
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
    """Threshold array a with nthres levels."""    
    # Normalization and setup step
    def _normalize_input(a):
        input_array = np.asarray(a)
        max_value = input_array.max()
        return input_array, max_value
    
    #Generating threshold level
    def _generate_thresholds(nthres,th_min, th_max, reverse_thres):
        if reverse_thres:
            thresholds = np.linspace(th_max, th_min, nthres)
        else:
            thresholds = np.linspace(th_min, th_max, nthres)
        return thresholds
    
    #Actually applying different threshold levels
        """Generate different versions of array (spike trains) using multiple thresholds 
        """
    def _apply_thresholds(array, max_value, thresholds, th_amplitude, delta):
        if th_amplitude:
            spike_trains = [np.floor_divide(array, (max_value * th).clip(delta, None)) for th in thresholds]
        else:
            spike_trains = [(array > (max_value * th).clip(delta, None)) for th in thresholds]
        return spike_trains
        
    #Downsampling (optional)
    def _downsample(spike_trains, ds):
        if ds is None:
            return spike_trains
        else:
        downsampled_spikes = [sum_downsample(input_array, ds) for input_array in spike_trains]
        return downsampled_spikes    
        
    #Filter for nonzero vals
    def _filter_nonzero(spike_trains, thresholds, nz_only):
        if not nz_only:
            return spike_trains, thresholds
        
        filtered_trains_nz = [input_array.sum() > 0 for input_array in spike_trains]
        #Compares the spike_trains, threshold arrays against the filtered_trains_nz array and keeps elements that 
        #nonzero
        spike_trains = [ss for ss, nz in zip(spike_trains, filtered_trains_nz) if nz]
        filtered_thresholds = [th for th, nz in zip(thresholds, filtered_trains_nz) if nz]
    
        return filtered_trains_nz, filtered_thresholds
    
    #Results
    def _give_results(spike_trains, thresholds, return_thres)
        if return_thres:
            return spike_trains, thresholds
        else:
            return spike_trains
    
    #max_thres function using all our modules
    input_array, max_value = _normalize_input(a)
    
    thresholds = _generate_threshold_levels(nthres, th_min, th_max, reverse_thres)
    
    spike_trains = _apply_thresholds(
        input_array, max_value, thresholds, th_amplitude, delta
    )
    
    spike_trains = _apply_downsampling(spike_trains, ds)
    
    spike_trains, thresholds = _filter_nonzero_results(
        spike_trains, thresholds, nz_only
    )
    
    return _give_results(spike_trains, thresholds, return_thres)

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
        #Helper methods
        #Note: Need to change opt_s and opt_b to optimal_signal and optimal_baseline in the main solver.py file
        # as well
        
        def _solve_without_l0_penalty(self, amp_constraint: bool) -> Tuple[np.ndarray, float]:
            """Solve optimization directly without L0 penalty."""
            optimal_signal, optimal_baseline, _ = self.solver.solve(amp_constraint=amp_constraint)
            return optimal_signal, optimal_baseline
        def _solve_with_l0_heuristic(self, amp_constraint: bool) -> Tuple[np.ndarray, float]:
            """Solve  using reweighted L1 heuristic for L0 penalty."""
            metrics_dataframe = None
            for iteration in range(self.cfg.max_iter_l0):
            # Solve current iteration
            current_signal, current_baseline, _ = self.solver.solve(amp_constraint=amp_constraint)
            current_objective = self._compute_err(s=current_signal, b=current_baseline)
        
            # Get best and last objectives from history
            best_objective, last_objective = self._get_objective_history(metrics_dataframe)
        
            # Threshold small values and compute convergence metrics
            thresholded_signal = np.where(
                current_signal > self.cfg.delta_l0, 
                current_signal, 
                0
            )
            objective_gap = np.abs(current_objective - best_objective)
            objective_delta = np.abs(current_objective - last_objective)
        
            # Record metrics
            metrics_dataframe = self._record_iteration_metrics(
                metrics_dataframe,
                iteration,
                current_objective,
                thresholded_signal,
                objective_gap,
                objective_delta,
            )
        
            # Check convergence
            if self._has_converged(objective_gap, best_objective, objective_delta):
                break
        
            # Update weights for next iteration
            self._update_weights_for_next_iteration(thresholded_signal)
            else:
            warnings.warn(
            f"L0 heuristic did not converge in {self.cfg.max_iter_l0} iterations"
            )
    
            # Final solve with updated weights
            optimal_signal, optimal_baseline, _ = self.solver.solve(amp_constraint=amp_constraint)
            return optimal_signal, optimal_baseline
            
            
       #############################     
        #No L0 penalty case
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