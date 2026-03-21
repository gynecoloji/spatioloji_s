"""Tests for interface cell identification."""

import matplotlib
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spatioloji_s.spatial._interface_types import InterfaceResult


class TestInterfaceResult:
    """Tests for the InterfaceResult dataclass."""

    def test_dataclass_fields(self):
        """InterfaceResult has all required fields."""
        import geopandas as gpd

        labels = pd.Series(["interior_a", "interior_b"], index=["c0", "c1"])
        segs = gpd.GeoDataFrame(
            {
                "segment_id": pd.Series(dtype=int),
                "length": pd.Series(dtype=float),
                "tortuosity": pd.Series(dtype=float),
                "n_cells_a": pd.Series(dtype=int),
                "n_cells_b": pd.Series(dtype=int),
            },
            geometry=[],
        )
        result = InterfaceResult(
            cell_labels=labels,
            contour=None,
            segments=segs,
            summary={
                "total_length": 0.0,
                "n_segments": 0,
                "mean_tortuosity": 0.0,
                "n_interface_a": 0,
                "n_interface_b": 0,
            },
            region_a="Tumor",
            region_b="Stromal",
            method="grid",
        )
        assert result.contour is None
        assert result.method == "grid"
        assert result.summary["n_segments"] == 0

    def test_contour_accepts_multilinestring(self):
        """contour field accepts a MultiLineString."""
        import geopandas as gpd

        line = LineString([(0, 0), (1, 1)])
        contour = MultiLineString([line])
        labels = pd.Series(["region_a_interface", "region_b_interface"], index=["c0", "c1"])
        segs = gpd.GeoDataFrame(
            {"segment_id": [0], "length": [1.414], "tortuosity": [1.0], "n_cells_a": [1], "n_cells_b": [1]},
            geometry=[line],
        )
        result = InterfaceResult(
            cell_labels=labels,
            contour=contour,
            segments=segs,
            summary={
                "total_length": 1.414,
                "n_segments": 1,
                "mean_tortuosity": 1.0,
                "n_interface_a": 1,
                "n_interface_b": 1,
            },
            region_a="Tumor",
            region_b="Stromal",
            method="grid",
        )
        assert isinstance(result.contour, MultiLineString)
        assert len(result.segments) == 1


from spatioloji_s.spatial._interface import identify_interface  # noqa: E402


class TestValidation:
    """Tests for input validation in identify_interface."""

    def test_invalid_group_col_raises(self, sp_interface):
        with pytest.raises(ValueError, match="not found in cell_meta"):
            identify_interface(sp_interface, group_col="nonexistent", region_a="TypeA", region_b="TypeB")

    def test_invalid_region_label_raises(self, sp_interface):
        with pytest.raises(ValueError, match="not found"):
            identify_interface(sp_interface, group_col="cell_type", region_a="Tumor", region_b="TypeB")

    def test_overlapping_regions_raises(self, sp_interface):
        with pytest.raises(ValueError, match="overlap"):
            identify_interface(sp_interface, group_col="cell_type", region_a=["TypeA", "TypeB"], region_b="TypeA")


class TestGridMethod:
    """Tests for the grid-based interface identification."""

    def test_returns_interface_result(self, sp_interface):
        result = identify_interface(sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB")
        assert isinstance(result, InterfaceResult)
        assert result.method == "grid"

    def test_cell_labels_values(self, sp_interface):
        result = identify_interface(sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB")
        valid = {"region_a_interface", "region_b_interface", "interior_a", "interior_b", "other"}
        assert set(result.cell_labels.unique()).issubset(valid)

    def test_cell_labels_index_matches_cells(self, sp_interface):
        result = identify_interface(sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB")
        assert len(result.cell_labels) == len(sp_interface.cell_index)

    def test_interface_cells_detected(self, sp_interface):
        """With default grid, cells near the boundary should be interface."""
        result = identify_interface(
            sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB", min_interface_cells=1
        )
        n_a = (result.cell_labels == "region_a_interface").sum()
        n_b = (result.cell_labels == "region_b_interface").sum()
        assert n_a > 0, "Should detect TypeA interface cells"
        assert n_b > 0, "Should detect TypeB interface cells"

    def test_store_writes_to_cell_meta(self, sp_interface):
        identify_interface(sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB", store=True)
        assert "interface_label" in sp_interface.cell_meta.columns

    def test_store_false_no_modification(self, sp_interface):
        identify_interface(sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB", store=False)
        assert "interface_label" not in sp_interface.cell_meta.columns

    def test_list_region_labels(self, sp_interface):
        """region_a as a list should work."""
        result = identify_interface(sp_interface, group_col="cell_type", region_a=["TypeA"], region_b="TypeB")
        assert isinstance(result, InterfaceResult)

    def test_contour_geometry(self, sp_interface):
        result = identify_interface(sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB")
        if result.contour is not None:
            assert result.contour.geom_type in ("MultiLineString", "LineString")

    def test_grid_resolution_param(self, sp_interface):
        result = identify_interface(
            sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB", grid_resolution=20
        )
        assert isinstance(result, InterfaceResult)


from spatioloji_s.visualization.polygon_plots import (  # noqa: E402
    plot_interface_polygon_map,
    plot_interface_polygon_metrics,
)


class TestPlotInterfaceMap:
    """Tests for plot_interface_polygon_map."""

    def test_returns_figure(self, sp_interface):
        result = identify_interface(sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB")
        fig = plot_interface_polygon_map(sp_interface, result, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_empty_result_no_error(self, sp_interface):
        """Should handle empty InterfaceResult without crashing."""
        import geopandas as gpd

        empty = InterfaceResult(
            cell_labels=pd.Series("other", index=sp_interface.cell_index),
            contour=None,
            segments=gpd.GeoDataFrame(
                {
                    "segment_id": pd.Series(dtype=int),
                    "length": pd.Series(dtype=float),
                    "tortuosity": pd.Series(dtype=float),
                    "n_cells_a": pd.Series(dtype=int),
                    "n_cells_b": pd.Series(dtype=int),
                },
                geometry=[],
            ),
            summary={
                "total_length": 0.0,
                "n_segments": 0,
                "mean_tortuosity": 0.0,
                "n_interface_a": 0,
                "n_interface_b": 0,
            },
            region_a="TypeA",
            region_b="TypeB",
            method="grid",
        )
        fig = plot_interface_polygon_map(sp_interface, empty, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_ax(self, sp_interface):
        result = identify_interface(sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB")
        fig, ax = plt.subplots()
        plot_interface_polygon_map(sp_interface, result, ax=ax, show=False)
        plt.close("all")


class TestPlotInterfaceMetrics:
    """Tests for plot_interface_polygon_metrics."""

    def test_returns_figure(self, sp_interface):
        result = identify_interface(sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB")
        fig = plot_interface_polygon_metrics(result, metric="length", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_empty_segments(self):
        """Should handle empty segments without crashing."""
        import geopandas as gpd

        empty = InterfaceResult(
            cell_labels=pd.Series(dtype=str),
            contour=None,
            segments=gpd.GeoDataFrame(
                {
                    "segment_id": pd.Series(dtype=int),
                    "length": pd.Series(dtype=float),
                    "tortuosity": pd.Series(dtype=float),
                    "n_cells_a": pd.Series(dtype=int),
                    "n_cells_b": pd.Series(dtype=int),
                },
                geometry=[],
            ),
            summary={
                "total_length": 0.0,
                "n_segments": 0,
                "mean_tortuosity": 0.0,
                "n_interface_a": 0,
                "n_interface_b": 0,
            },
            region_a="A",
            region_b="B",
            method="grid",
        )
        fig = plot_interface_polygon_metrics(empty, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, sp_interface):
        """Full pipeline: identify interface -> plot."""
        result = identify_interface(
            sp_interface, group_col="cell_type", region_a="TypeA", region_b="TypeB", store=True
        )
        assert "interface_label" in sp_interface.cell_meta.columns
        assert result.summary["n_segments"] >= 0

        fig = plot_interface_polygon_map(sp_interface, result, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

        if result.summary["n_segments"] > 0:
            fig2 = plot_interface_polygon_metrics(result, metric="length", show=False)
            assert isinstance(fig2, plt.Figure)
            plt.close("all")

    def test_imports_from_top_level(self):
        """Verify imports work from package top-level paths."""
        from spatioloji_s.spatial.point import identify_interface as pi
        from spatioloji_s.spatial.polygon import InterfaceResult, identify_interface  # noqa: F401
        from spatioloji_s.visualization import plot_interface_polygon_map, plot_interface_polygon_metrics

        assert callable(identify_interface)
        assert callable(pi)
        assert callable(plot_interface_polygon_map)
        assert callable(plot_interface_polygon_metrics)

    def test_point_and_polygon_same_function(self):
        """Point and polygon should export the same identify_interface."""
        from spatioloji_s.spatial.point import identify_interface as pi
        from spatioloji_s.spatial.polygon import identify_interface as poly_ii

        assert pi is poly_ii
