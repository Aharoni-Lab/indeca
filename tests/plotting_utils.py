import plotly.graph_objects as go


def plot_traces(tr_dict, **kwargs):
    trs = []
    for tr_name, tr_dat in tr_dict.items():
        t = go.Scatter(y=tr_dat, name=tr_name, mode="lines", **kwargs)
        trs.append(t)
    return trs
