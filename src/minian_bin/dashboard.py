import numpy as np
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
            "c": np.zeros((max_iters, ncell, T)),
            "s": np.zeros((max_iters, ncell, T)),
            "h": np.zeros((max_iters, ncell, kn_len)),
            "h_fit": np.zeros((max_iters, ncell, kn_len)),
            "scale": np.ones((max_iters, ncell)),
        }
        self._make_pane_cells()
        self.pn_main = pn.Column(self.pn_cells)
        self.dash = pn.template.MaterialTemplate(title="Minian-bin Dashboard")
        self.dash.main.append(self.pn_main)
        pn.serve(self.dash, port=port, threaded=True)

    def _make_pane_cells(self):
        self.fig_cells = [None] * self.ncell
        for icell, y in enumerate(self.Y):
            fig = make_subplots(
                cols=3,
                subplot_titles=("traces", "kernel", "error"),
                horizontal_spacing=0.02,
                column_widths=[0.7, 0.15, 0.15],
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
            fig.add_trace(
                go.Scatter(y=np.zeros(10), mode="lines", name="error"), row=1, col=3
            )
            fig.update_layout(autosize=True, margin={"l": 0, "r": 0, "t": 30, "b": 0})
            self.fig_cells[icell] = fig
        self.pn_cells = pn.Feed(
            *[
                pn.pane.plotly.Plotly(f, sizing_mode="stretch_width", height=200)
                for f in self.fig_cells
            ],
            # load_buffer=6,
            height=840,
            sizing_mode="stretch_width",
        )

    def _update_cells_fig(self, data: np.ndarray, uid: int, vname: str):
        fig = self.fig_cells[uid]
        for d in fig.data:
            if d.name == vname:
                d.y = data
                break
        else:
            raise ValueError(f"no data with name {vname}")

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
                elif vname in ["scale"]:
                    self.it_vars[vname][self.it_update, u] = dat
                    if self.it_update == self.it_view:
                        self._update_cells_fig(self.Y[u] / dat, u, "y")
