import numpy as np
import panel as pn
import plotly.graph_objects as go
from plotly.exceptions import PlotlyKeyError
from plotly.figure_factory import create_trisurf
from plotly.subplots import make_subplots
from scipy.spatial import Delaunay


class Dashboard:
    def __init__(
        self,
        Y: np.ndarray = None,
        ncell: int = None,
        T: int = None,
        max_iters: int = 20,
        kn_len: int = 60,
        port: int = 54321,
    ):
        super().__init__()
        self.title = "Dashboard"
        if Y is None:
            assert ncell is not None and T is not None
            Y = np.ones((ncell, T))
        else:
            ncell, T = Y.shape
        self.Y = Y
        self.ncell = ncell
        self.T = T
        self.kn_len = kn_len
        self.max_iters = max_iters
        self.it_update = 0
        self.it_view = 0
        self.it_vars = {
            "c": np.full((max_iters, ncell, T), np.nan),
            "s": np.full((max_iters, ncell, T), np.nan),
            "h": np.full((max_iters, ncell, kn_len), np.nan),
            "h_fit": np.full((max_iters, ncell, kn_len), np.nan),
            "scale": np.full((max_iters, ncell), np.nan),
            "tau_d": np.full((max_iters, ncell), np.nan),
            "tau_r": np.full((max_iters, ncell), np.nan),
            "err": np.full((max_iters, ncell), np.nan),
            "penal_err": np.array(
                [
                    [{"penal": [], "scale": [], "err": []} for _ in range(ncell)]
                    for _ in range(max_iters)
                ]
            ),
        }
        self._make_pane_cells()
        self._make_pane_iters()
        self.pn_main = pn.Column(self.pn_iters, self.pn_cells)
        self.dash = pn.template.MaterialTemplate(title="Minian-bin Dashboard")
        self.dash.main.append(self.pn_main)
        pn.serve(self.dash, port=port, threaded=True)

    def _make_pane_cells(self):
        self.fig_cells = [None] * self.ncell
        self.fig_penal = [go.Figure()] * self.ncell
        for icell, y in enumerate(self.Y):
            fig = make_subplots(
                cols=2,
                subplot_titles=("traces", "kernel"),
                horizontal_spacing=0.02,
                column_widths=[0.9, 0.1],
            )
            fig.add_trace(go.Scatter(y=y, mode="lines", name="y"), row=1, col=1)
            for v in ["c", "s"]:
                fig.add_trace(
                    go.Scatter(
                        y=self.it_vars[v][self.it_view, icell, :], mode="lines", name=v
                    ),
                    row=1,
                    col=1,
                )
            for v in ["h", "h_fit"]:
                fig.add_trace(
                    go.Scatter(
                        y=self.it_vars[v][self.it_view, icell, :], mode="lines", name=v
                    ),
                    row=1,
                    col=2,
                )
            fig.update_layout(
                autosize=True,
                margin={"l": 0, "r": 0, "t": 30, "b": 0},
                xaxis_title="frame",
            )
            self.fig_cells[icell] = fig
            fig = go.Figure(go.Heatmap(colorscale="viridis_r", zsmooth="best"))
            fig.update_layout(
                autosize=True,
                margin={"l": 0, "r": 0, "t": 30, "b": 0},
                title="error",
                xaxis_title="penalty",
                yaxis_title="scale",
            )
            self.fig_penal[icell] = fig
        self.pn_cells = pn.Feed(
            *[
                pn.Row(
                    *[
                        pn.pane.plotly.Plotly(
                            f, sizing_mode="stretch_width", height=300
                        ),
                        pn.pane.plotly.Plotly(p, width=450, height=300),
                    ]
                )
                for f, p in zip(self.fig_cells, self.fig_penal)
            ],
            # load_buffer=6,
            height=950,
            sizing_mode="stretch_width",
        )

    def _make_pane_iters(self):
        self.fig_iters = dict()
        for met in ["scale", "err"]:
            fig = go.Figure(
                data=[
                    go.Scatter(
                        y=self.it_vars[met][:, u],
                        name=met,
                        mode="lines+markers",
                        uid=u,
                        text="cell{}".format(u),
                        legendgroup=met,
                        showlegend=u == 0,
                    )
                    for u in range(self.ncell)
                ]
            )
            fig.update_layout(autosize=True, margin={"l": 0, "r": 0, "t": 0, "b": 0})
            self.fig_iters[met] = fig
        fig_tau = go.Figure()
        for met in ["tau_d", "tau_r"]:
            fig_tau.add_traces(
                [
                    go.Scatter(
                        y=self.it_vars[met][:, u],
                        name=met,
                        mode="lines+markers",
                        uid=u,
                        text="cell{}".format(u),
                        legendgroup=met,
                        showlegend=u == 0,
                    )
                    for u in range(self.ncell)
                ]
            )
        fig_tau.update_layout(autosize=True, margin={"l": 0, "r": 0, "t": 0, "b": 0})
        self.fig_iters["taus"] = fig_tau
        self.pn_iters = pn.Accordion(
            *[
                (k, pn.pane.plotly.Plotly(v, sizing_mode="stretch_both"))
                for k, v in self.fig_iters.items()
            ],
            sizing_mode="stretch_width",
            height=340,
        )

    def _update_cells_fig(self, data: np.ndarray, uid: int, vname: str):
        fig = self.fig_cells[uid]
        for d in fig.data:
            if d.name == vname:
                d.y = data
                break
        else:
            raise ValueError(f"no data with name {vname}")

    def _refresh_iters_fig(self, uid: int, vname: str):
        if vname in ["scale", "err"]:
            self.fig_iters[vname].data[uid].y = self.it_vars[vname][:, uid]
        elif vname in ["tau_d", "tau_r"]:
            dats = [
                d
                for d in self.fig_iters["taus"].data
                if d.name == vname and d.uid == str(uid)
            ]
            assert len(dats) == 1
            dats[0].y = self.it_vars[vname][:, uid]

    def _refresh_err_penal_fit(self, uid: int):
        ex = np.array(self.it_vars["penal_err"][self.it_view, uid]["penal"])
        ey = np.array(self.it_vars["penal_err"][self.it_view, uid]["scale"])
        err = np.log(np.array(self.it_vars["penal_err"][self.it_view, uid]["err"]))
        err = np.clip(err, 0, np.median(err))
        hm = go.Heatmap(x=ex, y=ey, z=err, type="heatmap")
        for a in ["x", "y", "z"]:
            self.fig_penal[uid].data[0][a] = hm[a]

    def set_iter(self, it: int):
        self.it_update = it

    def update(self, uid: int = None, **kwargs):
        if uid is None:
            uids = np.arange(self.ncell)
        else:
            uids = [uid]
        for u in uids:
            for vname, dat in kwargs.items():
                if vname in ["c", "s", "h", "h_fit"]:
                    self.it_vars[vname][self.it_update, u, :] = dat
                    if self.it_update == self.it_view:
                        self._update_cells_fig(dat, u, vname)
                elif vname in ["tau_d", "tau_r", "err", "scale"]:
                    try:
                        d = dat.item()
                    except ValueError:
                        d = dat[u]
                    except AttributeError:
                        d = dat
                    self.it_vars[vname][self.it_update, u] = d
                    self._refresh_iters_fig(u, vname)
                    if vname == "scale" and self.it_update == self.it_view:
                        self._update_cells_fig(self.Y[u] / d, u, "y")
                        self._refresh_err_penal_fit(u)
                elif vname in ["penal_err"]:
                    for v in ["penal", "scale", "err"]:
                        self.it_vars[vname][self.it_update, u][v].append(dat[v])
