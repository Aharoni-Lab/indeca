# %% import and definitions
import fsspec

# %% download
fs = fsspec.filesystem("github", org="HelmchenLabSoftware", repo="Cascade")
fs.get("Ground_truth/", "./data/realds/", recursive=True)
