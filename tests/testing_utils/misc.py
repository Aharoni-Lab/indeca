import os

import pandas as pd


def load_agg_result(res_path):
    try:
        res_files = list(filter(lambda fn: fn.endswith(".feat"), os.listdir(res_path)))
    except FileNotFoundError:
        return
    return pd.concat(
        [pd.read_feather(res_path / f) for f in res_files], ignore_index=True
    )
