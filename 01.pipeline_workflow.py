# %% This lays out a potential workflow for the general use of minian-bin.
# Workflow is:
# 0. Handle imports and definitions
# 1. Generate or inport dataset at normal FPS and upsampled for calcium imaging
# 2. estimate initial guess at convolution kernel
# 3. Solve for non-binarized 's'
# 4. Binarize 's' using a single threshold across all cells and determine per cell scaling factor
# 5. Update free kernel based on binarized spikes
# 6. Optionally fit free kernel to bi-exponential and generate new kernel from this
# 7. Iterate back to step 4 and repeat until some metric is reached

# %% 0. Handle imports and definitions
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
from routine.simulation import simulate_traces



# %% 1. Generate or import dataset at normal FPS for calcium imaging
generate_new_dataset = True
save_new_dataset = True

NUM_CELLS = 20
LENGTH_IN_SEC = 30
PARAM_TAU_D = 0.2 # units of seconds
PARAM_TAU_R = 0.04 # units of seconds

OUT_PATH = "./intermediate/simulated/simulated.nc"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Check if the file already exists
if os.path.isfile(OUT_PATH) and not generate_new_dataset:
    # Load previously saved dataset
    ds = pd.read_pickle(OUT_PATH)
    print(f"Loaded existing dataset from {OUT_PATH}")
else:
    # Generate new dataset
    np.random.seed(42)
    ds = simulate_traces(
        num_cells=NUM_CELLS,
        length_in_sec=LENGTH_IN_SEC, # at the defined fps
        tmp_P=np.array([[0.998, 0.002], [0.75, 0.25]]),
        tmp_tau_d=PARAM_TAU_D,
        tmp_tau_r=PARAM_TAU_R,
        approx_fps=30,
        spike_sampling_rate=500, # This results in a 2ms resolution for the spikes (which is roughly the time for a spike and refactory period)
    )
    
    # Save the generated dataset
    if save_new_dataset:
        ds.to_pickle(OUT_PATH)
        print(f"Generated new dataset and saved to {OUT_PATH}")
# %% 1.1 Plot initial generated data

# Select a subset of cells to plot (e.g., first 5 cells)
num_cells_to_plot = 10
cells_to_plot = ds.iloc[:num_cells_to_plot]

# Create a figure with subplots for each cell
fig = make_subplots(rows=num_cells_to_plot, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=[f'Cell {i+1}' for i in range(num_cells_to_plot)])

for idx, (_, cell) in enumerate(cells_to_plot.iterrows()):
    # Calculate time arrays for both normal and upsampled data
    t = np.arange(len(cell['C'])) / cell['fps']
    t_upsampled = np.arange(len(cell['C_true'])) / (cell['fps'] * len(cell['C_true']) / len(cell['C']))
    
    # Plot C and C_upsampled
    fig.add_trace(go.Scatter(x=t, y=cell['C']/cell['upsample_factor'], name='C', line=dict(color='blue', width=2)), row=idx+1, col=1)
    fig.add_trace(go.Scatter(x=t_upsampled, y=cell['C_true'], name='C_true', line=dict(color='cyan', width=1)), row=idx+1, col=1)
    
    # Plot S and S_upsampled
    fig.add_trace(go.Scatter(x=t, y=cell['S'], name='S', line=dict(color='red', width=2, shape='hv')), row=idx+1, col=1)
    fig.add_trace(go.Scatter(x=t_upsampled, y=cell['S_true'], name='S_true', line=dict(color='orange', width=1, shape='hv')), row=idx+1, col=1)

# Update layout
fig.update_layout(height=100*num_cells_to_plot, title_text='Simulated Calcium Traces and Spike Trains',
                  showlegend=False)
fig.update_xaxes(title_text='Time (s)', row=num_cells_to_plot, col=1)

# Show the plot
fig.show()


# %% 2. estimate initial guess at convolution kernel
# Currently this just manually sets the time constant for the decay and rise
# TODO: Make this more robust by some sort of fitting to initial C data
TAU_D = 0.2 # units of seconds
TAU_R = 0.04 # units of seconds

# We are constraining the bi-exponential to be 0 at t=0.
# This results in a kernel that follows the form k(t) = A * (exp(-t/TAU_R) - exp(-t/TAU_D))
# Where A is chosen such that the area under the kernel is 1.

# %% 3.1 Upsample C to match the upsampled spike data
spike_sampling_rate = ds['fps'].iloc[0] * ds['upsample_factor'].iloc[0]  # Assuming all rows have the same spike_sampling_rate

ds['C_upsampled'] = ds.apply(lambda row: np.interp(
    np.linspace(0, len(row['C']) - 1, len(row['C']) * row['upsample_factor']),
    np.arange(len(row['C'])),
    row['C']
), axis=1)

print(f"Interpolated C to match spike sampling rate of {spike_sampling_rate} Hz")
# %% 3.2 Plot comparison of C and C_upsampled for a few cells

num_cells_to_plot = 5
cells_to_plot = ds.iloc[:num_cells_to_plot]

fig = make_subplots(rows=num_cells_to_plot, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=[f'Cell {i+1}' for i in range(num_cells_to_plot)])

for idx, (_, cell) in enumerate(cells_to_plot.iterrows()):
    # Calculate time arrays for both normal and upsampled data
    t = np.arange(len(cell['C'])) / cell['fps']
    t_upsampled = np.arange(len(cell['C_upsampled'])) / spike_sampling_rate
    
    # Plot C
    fig.add_trace(go.Scatter(x=t, y=cell['C'], name='C', line=dict(color='blue', width=2)), row=idx+1, col=1)
    
    # Plot C_upsampled
    fig.add_trace(go.Scatter(x=t_upsampled, y=cell['C_upsampled'], name='C_upsampled', line=dict(color='red', width=1)), row=idx+1, col=1)

# Update layout
fig.update_layout(height=200*num_cells_to_plot, title_text='Comparison of C and C_upsampled',
                  showlegend=True)
fig.update_xaxes(title_text='Time (s)', row=num_cells_to_plot, col=1)
fig.update_yaxes(title_text='Fluorescence', col=1)

# Show the plot
fig.show()

print(f"Plotted comparison of C and C_upsampled for {num_cells_to_plot} cells")




# %% 4. Binarize 's' using a single threshold across all cells and determine per cell scaling factor
# %% 5. Update free kernel based on binarized spikes
# %% 6. Optionally fit free kernel to bi-exponential and generate new kernel from this
# %% 7. Iterate back to step 4 and repeat until some metric is reached