import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
tests_dir = Path(__file__).parent

print(f"Project root: {project_root}")
print(f"Tests dir: {tests_dir}")

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tests_dir))

# Try imports
from indeca.core.deconv import DeconvBin
from testing_utils.io import load_gt_ds

print("imports worked")