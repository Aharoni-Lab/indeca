# InDeCa
repository of Interpretable Deconvolution for Calcium imaging.

## Development Guide

### `pdm` workflow (recommended)

1. Obtain `pdm` globally on your system.
   Either follow the [official guide](https://pdm-project.org/en/latest/#installation), or if you prefer to use conda, `conda install -c conda-forge pdm` into your `base` environment.
1. Clone the repo and enter:
   ```bash
   git clone https://github.com/Aharoni-Lab/indeca.git
   cd indeca
   ```
1. If you want to use conda/mamba to handle dependencies, create a conda environment:
   ```
   conda create -n indeca -c conda-forge python=3.12
   conda activate indeca
   ```
   Otherwise skip to next step
1. `pdm install`

### setup cuosqp

1. Install cuda-toolkit 11.8 into the environment:
   ```bash
   mamba env create -n indeca-dev -f environment/cuda.yml
   conda activate indeca-dev
   ```
1. Obtain cuosqp source code:
   ```bash
   cd ..
   git clone https://github.com/osqp/cuosqp.git
   ```
1. Find the computing capability of your GPU and modify [line 120-125 of `osqp_sources/CMakeLists.txt`](https://github.com/osqp/osqp/blob/837c7a32baea1cc023566d2c82ab6dbb5f049af8/CMakeLists.txt#L120-L125) under the cuosqp repo accordingly:
   ```make
   set(CMAKE_CUDA_ARCHITECTURES 89) # modify the compute capability 89 to match your GPU
   # if (DFLOAT)
   #     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=compute_89 --gpu-code=sm_89")
   # else()
   #     # To use doubles we need compute capability 6.0 for atomic operations
   #     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=compute_89 --gpu-code=sm_89")
   # endif()
   ```
1. Obtain a copy of cuda examples 11.8 from [here](https://github.com/NVIDIA/cuda-samples/releases)
1. Add the extracted cuda examples folder to the include directories:
   - add the directory after [line 5 of `osqp_sources/lin_sys/cuda/cuda_pcg/CMakeLists.txt`](https://github.com/osqp/osqp/blob/837c7a32baea1cc023566d2c82ab6dbb5f049af8/lin_sys/cuda/cuda_pcg/CMakeLists.txt#L5)
   - modify the directory to point to the cuda examples folder at [line 27 of `osqp_sources/algebra/cuda/CMakeLists.txt`](https://github.com/osqp/osqp/blob/837c7a32baea1cc023566d2c82ab6dbb5f049af8/algebra/cuda/CMakeLists.txt#L27): `set(helper_cuda_header_dir /path/to/cuda-samples-11.8/Common)`
1. Run `CUDA_PATH=whatever python setup.py install`.
   If you followed the steps correctly, `CUDA_PATH` shouldn't matter (but it has to be set).
1. Verify that `cuosqp` is installed under your environment.

## Profiling

For comprehensive profiling documentation, see [`benchmarks/profile/README.md`](benchmarks/profile/README.md).

Quick reference:
- **Line-level profiling**: Use `kernprof -l -v your_script.py` for functions decorated with `@profile`
- **Pipeline-level profiling**: Use `yappi_profile` context manager for function-level attribution
- **Benchmark scripts**: Deterministic benchmarks in `benchmarks/profile/` for regression detection
   