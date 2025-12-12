import numpy as np
import plotly.graph_objects as go
import pytest

from indeca.core.deconv import DeconvBin
from tests.conftest import fixt_deconv
from tests.testing_utils.plotting import plot_traces

pytestmark = pytest.mark.integration


class TestDeconvBin:
    @pytest.mark.parametrize("taus", [(6, 1), (10, 3)])
    @pytest.mark.parametrize("rand_seed", np.arange(3))
    @pytest.mark.parametrize("upsamp", [1])
    def test_masking(self, taus, rand_seed, upsamp, eq_atol, test_fig_path_html):
        # act
        deconv, y, c, c_org, s, s_org, scale = fixt_deconv(
            taus=taus, rand_seed=rand_seed, upsamp=upsamp, deconv_kws={"Hlim": None}
        )
        s_nomsk, b_nomsk = deconv._solve(amp_constraint=False)
        c_nomsk = deconv.H @ s_nomsk
        deconv._update_mask()
        s_msk, b_msk = deconv._solve(amp_constraint=False)
        c_msk = deconv.H @ s_msk
        s_msk = deconv._pad_s(s_msk)
        c_msk = deconv._pad_c(c_msk)
        # plotting
        fig = go.Figure()
        fig.add_traces(
            plot_traces(
                {
                    "c": c,
                    "s": s,
                    "c_org": c_org,
                    "s_org": s_org,
                    "s_nomsk": s_nomsk,
                    "c_nomsk": c_nomsk,
                    "s_msk": s_msk,
                    "c_msk": c_msk,
                }
            )
        )
        fig.write_html(test_fig_path_html)
        # assertion
        assert np.isclose(b_nomsk, 0, atol=eq_atol)
        assert np.isclose(b_msk, 0, atol=eq_atol)
        assert set(np.where(s)[0]).issubset(set(deconv.nzidx_s))
        assert np.isclose(s_org, s_nomsk, atol=eq_atol).all()
        assert np.isclose(s_org, s_msk, atol=eq_atol).all()
