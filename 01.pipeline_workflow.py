# %% This lays out a potential workflow for the general use of minian-bin.
# Workflow is:
# 0. Handle imports and definitions
# 1. Generate or inport dataset at normal FPS for calcium imaging
# 2. Upscale data to ~1KHz sampling
# 3. estimate initial guess at convolution kernel
# 4. Solve for non-binarized 's'
# 5. Binarize 's' using a single threshold across all cells and determine per cell scaling factor
# 6. Update free kernel based on binarized spikes
# 7. Optionally fit free kernel to bi-exponential and generate new kernel from this
# 8. Iterate back to step 4 and repeat until some metric is reached

# %% 0. Handle imports and definitions
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
from routine.simulation import simulate_traces



# %% 1. Generate or import dataset at normal FPS for calcium imaging
generate_new_dataset = False
OUT_PATH = "./intermediate/simulated/simulated.nc"
PARAM_TAU_D = 0.2 # units of seconds
PARAM_TAU_R = 0.04 # units of seconds

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
        num_cells=40,
        length_in_sec=30, # at the defined fps
        tmp_P=np.array([[0.998, 0.002], [0.75, 0.25]]),
        tmp_tau_d=PARAM_TAU_D,
        tmp_tau_r=PARAM_TAU_R,
        approx_fps=30,
        spike_sampling_rate=500,
    )
    
    # Save the generated dataset
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
    t_upsampled = np.arange(len(cell['C_upsampled'])) / (cell['fps'] * len(cell['C_upsampled']) / len(cell['C']))
    
    # Plot C and C_upsampled
    fig.add_trace(go.Scatter(x=t, y=cell['C']/cell['upsample_factor'], name='C', line=dict(color='blue', width=2)), row=idx+1, col=1)
    fig.add_trace(go.Scatter(x=t_upsampled, y=cell['C_upsampled'], name='C_upsampled', line=dict(color='cyan', width=1)), row=idx+1, col=1)
    
    # Plot S and S_upsampled
    fig.add_trace(go.Scatter(x=t, y=cell['S'], name='S', line=dict(color='red', width=2, shape='hv')), row=idx+1, col=1)
    fig.add_trace(go.Scatter(x=t_upsampled, y=cell['S_upsampled'], name='S_upsampled', line=dict(color='orange', width=1, shape='hv')), row=idx+1, col=1)

# Update layout
fig.update_layout(height=300*num_cells_to_plot, width=1000, title_text='Simulated Calcium Traces and Spike Trains',
                  showlegend=False)
fig.update_xaxes(title_text='Time (s)', row=num_cells_to_plot, col=1)

# Show the plot
fig.show()

# %% 2. Upscale data to ~1KHz sampling
# %% 3. estimate initial guess at convolution kernel
# %% 4. Solve for non-binarized 's'
# %% 5. Binarize 's' using a single threshold across all cells and determine per cell scaling factor
# %% 6. Update free kernel based on binarized spikes
# %% 7. Optionally fit free kernel to bi-exponential and generate new kernel from this
# %% 8. Iterate back to step 4 and repeat until some metric is reached