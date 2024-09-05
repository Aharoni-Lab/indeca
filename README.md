# minian-bin
repository of extended binary algorithms for minian

## Development Guide

### `pdm` workflow (recommended)

1. Obtain `pdm` globally on your system.
   Either follow the [official guide](https://pdm-project.org/en/latest/#installation), or if you prefer to use conda, `conda install -c conda-forge pdm` into your `base` environment.
1. Clone the repo and enter:
   ```bash
   git clone https://github.com/Aharoni-Lab/minian-bin.git
   cd minian-bin
   ```
1. If you want to use conda/mamba to handle dependencies, create the environment using the file under `environments`:
   ```
   mamba env create -n minian-bin-dev -f environments/generic.yml
   conda activate minian-bin-dev
   ```
   Otherwise skip to next step
1. `pdm install`