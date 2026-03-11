from pathlib import Path


def tube_path() -> Path:
    """
    Tell noob where our tubes are using the `noob.add_sources` entrypoint
    """
    return Path(__file__).parent / "tubes"
