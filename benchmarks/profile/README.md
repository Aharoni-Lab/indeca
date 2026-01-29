# Profiling

InDeCa provides two complementary profiling approaches for performance analysis and optimization.

## 1. Line-Level Profiling (line_profiler)

For deep numerical inspection of hot loops, solvers, and kernel construction.

Functions decorated with `@profile` can be profiled using:

```bash
kernprof -l -v your_script.py
```

This is particularly useful for:
- Hot inner loops
- Solver internals
- Kernel construction
- Deconvolution steps

## 2. Pipeline-Level Profiling (yappi + snakeviz)

For function-level attribution and call graph analysis.

### Quick Start

```python
from indeca.utils.profiling import yappi_profile
from indeca.pipeline import pipeline_bin_new, DeconvPipelineConfig

Y = load_your_data()
config = DeconvPipelineConfig(...)

with yappi_profile("pipeline.prof"):
    C, S, metrics = pipeline_bin_new(Y, config=config, spawn_dashboard=False)
```

View results:
```bash
snakeviz pipeline.prof
```

### Clock Types

- **wall** (default): Real elapsed time, includes I/O and waiting
- **cpu**: Actual computation time, excludes I/O

```python
with yappi_profile("cpu_profile.prof", clock="cpu"):
    ...
```

### Usage

The `yappi_profile` context manager wraps code execution and saves profiling data in pstat format:

```python
from indeca.utils.profiling import yappi_profile

with yappi_profile("output.prof", clock="wall"):
    # Your code here
    result = expensive_function()
```

## Benchmark Scripts

Deterministic benchmarks for performance regression detection. All scripts use fixed seeds and configurations for reproducible results.

### Small Benchmark (10 cells × 1K frames)

Quick iterations and fast profiling:

```bash
# Quick runtime check
python benchmarks/profile/profile_pipeline_small.py

# With yappi profiling (wall-clock time)
python benchmarks/profile/profile_pipeline_small.py --profile

# With yappi profiling (CPU time)
python benchmarks/profile/profile_pipeline_small.py --profile --clock cpu

# View results
snakeviz benchmarks/profile/output/profile_pipeline_small.prof
```

### Medium Benchmark (50 cells × 5K frames)

Realistic workload testing:

```bash
# Quick runtime check
python benchmarks/profile/profile_pipeline_medium.py

# With profiling
python benchmarks/profile/profile_pipeline_medium.py --profile

# View results
snakeviz benchmarks/profile/output/profile_pipeline_medium.prof
```

### Large Benchmark (100 cells × 10K frames)

Comprehensive profiling - **warning: may take several minutes**:

```bash
# Quick runtime check
python benchmarks/profile/profile_pipeline_large.py

# With profiling
python benchmarks/profile/profile_pipeline_large.py --profile

# View results
snakeviz benchmarks/profile/output/profile_pipeline_large.prof
```

## When to Use Each Tool

| Tool | Use Case |
|------|----------|
| line_profiler | Hot inner loops, solver internals, kernel construction |
| yappi | Pipeline flow, function call attribution, call graphs |
| Benchmark scripts | Performance regression detection, optimization validation |

## Performance Regression Workflow

1. **Baseline**: Run benchmark without profiling to establish baseline runtime
   ```bash
   python benchmarks/profile/profile_pipeline_small.py
   ```

2. **Profile**: Run with profiling to identify bottlenecks
   ```bash
   python benchmarks/profile/profile_pipeline_small.py --profile
   ```

3. **Analyze**: View call graph and function timings in snakeviz
   ```bash
   snakeviz benchmarks/profile/output/profile_pipeline_small.prof
   ```

4. **Optimize**: Focus on functions with highest cumulative time

5. **Validate**: Re-run benchmark to measure improvement

## Output Location

All profiling output files are saved in `benchmarks/profile/output/`:

- `profile_pipeline_small.prof`
- `profile_pipeline_medium.prof`
- `profile_pipeline_large.prof`

These files are in pstat format and can be viewed with:
- **snakeviz** (recommended): Interactive web-based visualization
- **gprof2dot**: Generate call graph diagrams
- **pyprof2calltree**: Convert for use with kcachegrind

