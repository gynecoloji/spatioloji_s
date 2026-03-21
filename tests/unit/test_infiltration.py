"""Tests for immune infiltration scoring."""

import pandas as pd
import pytest

from spatioloji_s.spatial._infiltration import score_infiltration
from spatioloji_s.spatial._infiltration_types import InfiltrationResult
from spatioloji_s.spatial._interface import identify_interface


class TestScoreInfiltrationBasic:
    """Basic infiltration scoring tests."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        return identify_interface(
            sp_gradient,
            group_col="cell_type",
            region_a="TypeA",
            region_b="TypeB",
            min_interface_cells=1,
        )

    def test_returns_infiltration_result(self, sp_gradient, interface_result):
        result = score_infiltration(
            sp_gradient,
            interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
            target_region="TypeA",
        )
        assert isinstance(result, InfiltrationResult)

    def test_cell_classifications(self, sp_gradient, interface_result):
        result = score_infiltration(
            sp_gradient,
            interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
            target_region="TypeA",
        )
        valid = {"infiltrating", "resident", "other"}
        assert set(result.cell_classifications.unique()).issubset(valid)
        assert len(result.cell_classifications) == len(sp_gradient.cell_index)

    def test_per_type_metrics_columns(self, sp_gradient, interface_result):
        result = score_infiltration(
            sp_gradient,
            interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T", "Macrophage"],
            target_region="TypeA",
        )
        expected_cols = {
            "median_depth",
            "max_depth",
            "density_slope",
            "density_pvalue",
            "infiltration_fraction",
            "n_infiltrating",
            "n_resident",
        }
        assert expected_cols.issubset(set(result.per_type_metrics.columns))

    def test_per_type_metrics_rows(self, sp_gradient, interface_result):
        result = score_infiltration(
            sp_gradient,
            interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T", "Macrophage"],
            target_region="TypeA",
        )
        assert set(result.per_type_metrics.index) == {"CD8_T", "Macrophage"}

    def test_target_region_auto_detect(self, sp_gradient, interface_result):
        result = score_infiltration(
            sp_gradient,
            interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
        )
        assert result.target_region in ("TypeA", "TypeB")

    def test_infiltration_fraction_range(self, sp_gradient, interface_result):
        result = score_infiltration(
            sp_gradient,
            interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
            target_region="TypeA",
        )
        fracs = result.per_type_metrics["infiltration_fraction"]
        assert (fracs >= 0).all() and (fracs <= 1).all()

    def test_distances_series(self, sp_gradient, interface_result):
        result = score_infiltration(
            sp_gradient,
            interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
            target_region="TypeA",
        )
        assert isinstance(result.distances, pd.Series)
        assert len(result.distances) == len(sp_gradient.cell_index)

    def test_region_labels_propagated(self, sp_gradient, interface_result):
        result = score_infiltration(
            sp_gradient,
            interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
            target_region="TypeA",
        )
        assert result.region_a == interface_result.region_a
        assert result.region_b == interface_result.region_b


class TestScoreInfiltrationValidation:
    """Input validation tests."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        return identify_interface(
            sp_gradient,
            group_col="cell_type",
            region_a="TypeA",
            region_b="TypeB",
            min_interface_cells=1,
        )

    def test_invalid_immune_col(self, sp_gradient, interface_result):
        with pytest.raises(ValueError, match="not found"):
            score_infiltration(
                sp_gradient,
                interface_result,
                immune_col="nonexistent",
                immune_types=["CD8_T"],
            )

    def test_invalid_target_region(self, sp_gradient, interface_result):
        with pytest.raises(ValueError, match="target_region"):
            score_infiltration(
                sp_gradient,
                interface_result,
                immune_col="immune_type",
                immune_types=["CD8_T"],
                target_region="InvalidRegion",
            )


class TestPointPolygonUnified:
    """Verify point and polygon export the same function."""

    def test_point_score_infiltration_is_same(self):
        from spatioloji_s.spatial.point.infiltration import score_infiltration as point_si
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration as poly_si

        assert point_si is poly_si


class TestPlotInfiltrationSummary:
    """Tests for plot_infiltration_summary."""

    def test_returns_figure(self, sp_gradient):
        import matplotlib
        import matplotlib.pyplot as plt

        from spatioloji_s.visualization.polygon_plots import plot_infiltration_summary

        iface = identify_interface(
            sp_gradient,
            group_col="cell_type",
            region_a="TypeA",
            region_b="TypeB",
            min_interface_cells=1,
        )
        result = score_infiltration(
            sp_gradient,
            iface,
            immune_col="immune_type",
            immune_types=["CD8_T", "Macrophage"],
            target_region="TypeA",
        )
        fig = plot_infiltration_summary(result, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_infiltration_workflow(self, sp_gradient):
        """Full workflow: interface -> infiltration -> plot."""
        import matplotlib.pyplot as plt

        from spatioloji_s.visualization.polygon_plots import plot_infiltration_summary

        iface = identify_interface(
            sp_gradient,
            group_col="cell_type",
            region_a="TypeA",
            region_b="TypeB",
            min_interface_cells=1,
        )
        result = score_infiltration(
            sp_gradient,
            iface,
            immune_col="immune_type",
            immune_types=["CD8_T", "Macrophage"],
            target_region="TypeA",
        )

        assert isinstance(result, InfiltrationResult)
        assert "CD8_T" in result.per_type_metrics.index

        fig = plot_infiltration_summary(result, show=False)
        plt.close(fig)
