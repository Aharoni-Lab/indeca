import numpy as np
from noob import Tube, SynchronousRunner
from noob.config import add_config_source

from indeca.config import tube_path


def _make_sin(frequency: float) -> np.ndarray:
    sampling_rate = 1000  # Hz
    duration = 1.0  # seconds
    amplitude = 1
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


def test_tube(tmp_path):
    """just show what tubes do!"""
    add_config_source(tube_path())

    wave = _make_sin(5)
    npy_path = tmp_path / "array.npy"
    np.save(npy_path, wave)

    tube = Tube.from_specification("indeca-demo", input={"array_path": npy_path})
    runner = SynchronousRunner(tube)

    incremented_by = 0
    for i in range(5):
        result = runner.process()
        incremented_by += result["a"]
        assert np.allclose(result["b"], wave + incremented_by)
