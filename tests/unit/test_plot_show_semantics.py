"""Tests for the unified show/return semantics of the figure finalize helpers.

Contract (scanpy-style), identical across all three finalize helpers and the
plotting functions that delegate to them:

- ``show=True``  → the figure is displayed (``plt.show()``: inline in Jupyter,
  a window in GUI backends), then closed; the function returns ``None``.
- ``show=False`` → nothing is displayed; the figure is closed (dropped from
  pyplot's registry, so loops don't leak) and **returned** for saving,
  composition, or an explicit ``display(fig)``.
- ``save_path`` / ``save_dir`` writes the file in both modes.
"""

import warnings

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from spatioloji_s.visualization._spatial_helpers import finalize_plot  # noqa: E402
from spatioloji_s.visualization.basic_plots import _finalize_plot  # noqa: E402
from spatioloji_s.visualization.plots import _finalize  # noqa: E402


def _call(helper_name, fig, show, out_dir=None):
    """Invoke one of the three helpers, papering over signature differences."""
    if helper_name == "basic":
        path = str(out_dir / "fig.png") if out_dir else None
        return _finalize_plot(fig, path, dpi=72, show=show)
    if helper_name == "spatial":
        path = str(out_dir / "fig.png") if out_dir else None
        return finalize_plot(fig, path, dpi=72, show=show)
    if helper_name == "plots":
        return _finalize(fig, str(out_dir) if out_dir else None, "fig.png" if out_dir else None, 72, show)
    raise AssertionError(helper_name)


HELPERS = ["basic", "spatial", "plots"]


@pytest.mark.parametrize("helper", HELPERS)
class TestFinalizeHelpers:
    def test_show_false_returns_closed_figure(self, helper, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        # _finalize ("plots") saves unconditionally, so always hand it a dir
        result = _call(helper, fig, show=False,
                       out_dir=tmp_path if helper == "plots" else None)
        assert result is fig
        assert fig.number not in plt.get_fignums()

    def test_show_true_returns_none_and_closes(self, helper, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Agg's "non-interactive" warning
            result = _call(helper, fig, show=True,
                           out_dir=tmp_path if helper == "plots" else None)
        assert result is None
        assert plt.get_fignums() == []

    def test_save_writes_file(self, helper, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        _call(helper, fig, show=False, out_dir=tmp_path)
        assert (tmp_path / "fig.png").exists()


class TestPublicFunctions:
    """End-to-end: one delegate of each helper honours the contract."""

    def test_plot_violin_show_false_returns_fig(self, sp_basic):
        from spatioloji_s.visualization.basic_plots import plot_violin

        fig = plot_violin(sp_basic, genes="gene_0", group_by="cell_type",
                          layer=None, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_violin_show_true_returns_none(self, sp_basic):
        from spatioloji_s.visualization.basic_plots import plot_violin

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_violin(sp_basic, genes="gene_0", group_by="cell_type",
                                 layer=None, show=True)
        assert result is None
        assert plt.get_fignums() == []

    def test_xenium_plot_spatial_show_false_returns_fig(self, sp_basic):
        from spatioloji_s.visualization.plots import xenium_plot_spatial

        fig = xenium_plot_spatial(sp_basic, "cell_type", mode="dot", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_xenium_plot_spatial_show_true_returns_none(self, sp_basic):
        from spatioloji_s.visualization.plots import xenium_plot_spatial

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = xenium_plot_spatial(sp_basic, "cell_type", mode="dot", show=True)
        assert result is None
        assert plt.get_fignums() == []
