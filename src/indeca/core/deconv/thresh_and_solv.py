import numpy as np
import scipy.sparse as sps
from numba import njit
from scipy.signal import ShortTimeFFT
from indeca.core.deconv.utils import sum_downsample
from indeca.core.simulation import tau2AR
from scripts.s01_caiman_test import S_ls

# The following functions are defined sequentially as part of an iterative spike inference step where a solve function
# is first defined (solv), followed by an initial generation of thresholded spike trains (max_thres and _max_thres)
# and then the iterative search for the most ideal inferred spike train (solve_thres).

#This function allows us to generate multiple thresholded versions of an input array used as part 
# of iterative spike inference search

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
    # First define all helper methods
    
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
    
    return _package_results(spike_trains, thresholds, return_thres)