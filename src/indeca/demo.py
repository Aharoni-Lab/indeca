from pathlib import Path
import numpy as np
from pydantic import ConfigDict

from noob.asset import Asset


class NPYAsset(Asset):
    path: Path
    obj: np.ndarray = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def init(self):
        self.obj = np.load(self.path)
