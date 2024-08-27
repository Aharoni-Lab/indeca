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
import cvxpy as cp
import scipy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
from routine.simulation import simulate_traces



# %% 1. Generate or import dataset at normal FPS for calcium imaging
# TODO: Add noise to the data
generate_new_dataset = True
save_new_dataset = True

NUM_CELLS = 5
LENGTH_IN_SEC = 30 
SPIKE_SAMPLING_RATE = 200 # Hz
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
        spike_sampling_rate=SPIKE_SAMPLING_RATE, # This results in a 2ms resolution for the spikes (which is roughly the time for a spike and refactory period)
    )
    
    # Save the generated dataset
    if save_new_dataset:
        ds.to_pickle(OUT_PATH)
        print(f"Generated new dataset and saved to {OUT_PATH}")
# %% 1.1 Plot initial generated data

# Select a subset of cells to plot (e.g., first 5 cells)
num_cells_to_plot = 5
cells_to_plot = ds.iloc[:num_cells_to_plot]

# Create a figure with subplots for each cell
fig = make_subplots(rows=num_cells_to_plot, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=[f'Cell {i+1}' for i in range(num_cells_to_plot)])

for idx, (_, cell) in enumerate(cells_to_plot.iterrows()):
    # Calculate time arrays for both normal and upsampled data
    t = np.arange(len(cell['C'])) / cell['fps']
    t_upsampled = np.arange(len(cell['C_true'])) / (cell['fps'] * len(cell['C_true']) / len(cell['C']))
    
    # Plot C and C_upsampled
    fig.add_trace(go.Scatter(x=t, y=cell['C'], name='C', line=dict(color='blue', width=2)), row=idx+1, col=1)
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

# Define the kernel function
def kernel(t, A, tau_d, tau_r):
    return A * (np.exp(-t/tau_d) - np.exp(-t/tau_r))

# Calculate the kernel
t = np.arange(0, 5*TAU_D, 1/SPIKE_SAMPLING_RATE)  # 5*TAU_D should cover most of the kernel
k = kernel(t, 1, TAU_D, TAU_R)
# A = 1 / np.sum(k)  # Normalize the kernel to have area 1
# k *= A
# Plot the kernel
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=k, mode='lines', name='Kernel'))
fig.update_layout(
    title='Convolution Kernel',
    xaxis_title='Time (s)',
    yaxis_title='Amplitude',
    showlegend=True
)
fig.show()

print("Plotted the convolution kernel")

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
    fig.add_trace(go.Scatter(x=t_upsampled, y=cell['C_upsampled'], name='C_upsampled', mode='markers', marker=dict(color='red', size=2)), row=idx+1, col=1)

# Update layout
fig.update_layout(height=200*num_cells_to_plot, title_text='Comparison of C and C_upsampled',
                  showlegend=True)
fig.update_xaxes(title_text='Time (s)', row=num_cells_to_plot, col=1)
fig.update_yaxes(title_text='Fluorescence', col=1)

# Show the plot
fig.show()

print(f"Plotted comparison of C and C_upsampled for {num_cells_to_plot} cells")


# %% 3.3 Estimate spiking from kernel and upsampled C

# Function to solve for spike estimates given calcium trace and kernel
def solve_s(y, h, norm="l1", sparsity_penalty=0):
    y, h = y.squeeze(), h.squeeze()
    T = len(y)
    b = cp.Variable()  # Baseline fluorescence
    s = cp.Variable(T)  # Spike train
    conv_term = cp.conv(h, s)[:T]  # Convolution of kernel and spike train
    norm_ord = {"l1": 1, "l2": 2}[norm]
    # Objective function: minimize reconstruction error + sparsity penalty
    obj = cp.Minimize(
        cp.norm(y - conv_term - b, norm_ord)
        + sparsity_penalty * cp.norm(s, 1)
    )
    cons = [s >= 0, b >= 0]  # Constraints: non-negative spikes and baseline
    prob = cp.Problem(obj, cons)
    prob.solve()
    return s.value

# Estimate S for each cell
total_cells = len(ds)
print(f"Estimating S for {total_cells} cells")

# Vectorize operations for efficiency
y_values = np.stack(ds['C_upsampled'].values)
S_estimates = []

# Iterate through each cell's calcium trace
for i, y in enumerate(tqdm(y_values, desc="Estimating S")):
    # Solve for spike estimates using L2 norm and no sparsity penalty
    S_estimate = solve_s(y, k, norm="l2", sparsity_penalty=0.0)
    S_estimates.append(S_estimate)
    
    # Print progress every 10% or for the last cell
    if (i + 1) % max(1, total_cells // 10) == 0 or i == total_cells - 1:
        percent_complete = (i + 1) / total_cells * 100
        print(f"Progress: {percent_complete:.1f}% complete ({i + 1}/{total_cells} cells processed)")

# Add S_estimate column to the dataframe
ds['S_estimate'] = S_estimates

# Generate reconvolved C_estimates
print("Generating reconvolved C_estimates...")

# Function to vectorize reconvolution operation
def reconvolve_C_vectorized(S_estimates, kernel):
    return np.array([np.convolve(s, kernel, mode='full')[:len(s)] for s in S_estimates])

# Compute C_estimates by reconvolving S_estimates with the kernel
C_estimates = reconvolve_C_vectorized(S_estimates, k)

# Add C_estimate column to the dataframe
ds['C_estimate'] = C_estimates


# %% 3.4 Plot comparison of C_upsampled, S_estimate, and S_true for a few cells
num_cells_to_plot = 5
cells_to_plot = ds.iloc[:num_cells_to_plot]

fig = make_subplots(rows=num_cells_to_plot, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=[f'Cell {i+1}' for i in range(num_cells_to_plot)])

# Define trace properties
trace_properties = [
    {'name': 'C_upsampled', 'color': 'blue', 'width': 2, 'dash': None},
    {'name': 'C_estimate', 'color': 'purple', 'width': 2, 'dash': 'dash'},
    {'name': 'S_estimate', 'color': 'red', 'width': 1, 'dash': None},
    {'name': 'S_true', 'color': 'green', 'width': 1, 'dash': None}
]

for idx, (_, cell) in enumerate(cells_to_plot.iterrows()):
    t = np.arange(len(cell['C_upsampled'])) / spike_sampling_rate
    
    for prop in trace_properties:
        y_data = cell[prop['name']] if prop['name'] != 'S_estimate' else cell['S_estimate'] * 2
        fig.add_trace(
            go.Scatter(
                x=t, 
                y=y_data, 
                name=prop['name'],
                line=dict(color=prop['color'], width=prop['width'], dash=prop['dash']),
                legendgroup=prop['name'],
                showlegend=(idx == 0)  # Only show in legend for the first cell
            ),
            row=idx+1, 
            col=1
        )

# Update layout
fig.update_layout(
    height=200*num_cells_to_plot, 
    title_text='Comparison of C_upsampled, S_estimate, and S_true',
    showlegend=True, 
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_xaxes(title_text='Time (s)', row=num_cells_to_plot, col=1)
fig.update_yaxes(title_text='Fluorescence / Spike Amplitude', col=1)

# Show the plot
fig.show()

print(f"Plotted comparison of C_upsampled, S_estimate, and S_true for {num_cells_to_plot} cells")


# %% 4. Binarize 's' using a single threshold across all cells and determine per cell scaling factor
# %% 5. Update free kernel based on binarized spikes
# %% 6. Optionally fit free kernel to bi-exponential and generate new kernel from this
# %% 7. Iterate back to step 4 and repeat until some metric is reached