"""Tests for interface cell identification."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString

from spatioloji_s.spatial._interface_types import InterfaceResult


class TestInterfaceResult:
    """Tests for the InterfaceResult dataclass."""

    def test_dataclass_fields(self):
        """InterfaceResult has all required fields."""
        import geopandas as gpd

        labels = pd.Series(["interior_a", "interior_b"], index=["c0", "c1"])
        segs = gpd.GeoDataFrame(
            {"segment_id": pd.Series(dtype=int), "length": pd.Series(dtype=float),
             "tortuosity": pd.Series(dtype=float),
             "n_cells_a": pd.Series(dtype=int), "n_cells_b": pd.Series(dtype=int)},
            geometry=[],
        )
        result = InterfaceResult(
            cell_labels=labels,
            contour=None,
            segments=segs,
            summary={"total_length": 0.0, "n_segments": 0, "mean_tortuosity": 0.0,
                     "n_interface_a": 0, "n_interface_b": 0},
            region_a="Tumor",
            region_b="Stromal",
            method="graph",
        )
        assert result.contour is None
        assert result.method == "graph"
        assert result.summary["n_segments"] == 0

    def test_contour_accepts_multilinestring(self):
        """contour field accepts a MultiLineString."""
        import geopandas as gpd

        line = LineString([(0, 0), (1, 1)])
        contour = MultiLineString([line])
        labels = pd.Series(["region_a_interface", "region_b_interface"], index=["c0", "c1"])
        segs = gpd.GeoDataFrame(
            {"segment_id": [0], "length": [1.414], "tortuosity": [1.0],
             "n_cells_a": [1], "n_cells_b": [1]},
            geometry=[line],
        )
        result = InterfaceResult(
            cell_labels=labels, contour=contour, segments=segs,
            summary={"total_length": 1.414, "n_segments": 1,
                     "mean_tortuosity": 1.0, "n_interface_a": 1, "n_interface_b": 1},
            region_a="Tumor", region_b="Stromal", method="graph",
        )
        assert isinstance(result.contour, MultiLineString)
        assert len(result.segments) == 1


from spatioloji_s.spatial.polygon.interface import identify_interface
from spatioloji_s.spatial.polygon.graph import build_buffer_graph, build_contact_graph


class TestPolygonValidation:
    """Tests for input validation in polygon identify_interface."""

    def test_invalid_group_col_raises(self, sp_interface):
        g = build_contact_graph(sp_interface)
        with pytest.raises(ValueError, match="not found in cell_meta"):
            identify_interface(sp_interface, g, group_col="nonexistent",
                               region_a="TypeA", region_b="TypeB")

    def test_invalid_region_label_raises(self, sp_interface):
        g = build_contact_graph(sp_interface)
        with pytest.raises(ValueError, match="not found"):
            identify_interface(sp_interface, g, group_col="cell_type",
                               region_a="Tumor", region_b="TypeB")

    def test_overlapping_regions_raises(self, sp_interface):
        g = build_contact_graph(sp_interface)
        with pytest.raises(ValueError, match="overlap"):
            identify_interface(sp_interface, g, group_col="cell_type",
                               region_a=["TypeA", "TypeB"], region_b="TypeA")

    def test_graph_required_for_graph_method(self, sp_interface):
        with pytest.raises(ValueError, match="graph.*required"):
            identify_interface(sp_interface, graph=None, group_col="cell_type",
                               region_a="TypeA", region_b="TypeB", method="graph")

    def test_density_without_graph_needs_threshold(self, sp_interface):
        with pytest.raises(ValueError, match="distance_threshold"):
            identify_interface(sp_interface, graph=None, group_col="cell_type",
                               region_a="TypeA", region_b="TypeB", method="density")


class TestPolygonGraphMethod:
    """Tests for the graph-based interface identification (polygon)."""

    def test_returns_interface_result(self, sp_interface):
        g = build_buffer_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        assert isinstance(result, InterfaceResult)
        assert result.method == "graph"

    def test_cell_labels_values(self, sp_interface):
        g = build_buffer_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        valid = {"region_a_interface", "region_b_interface",
                 "interior_a", "interior_b", "other"}
        assert set(result.cell_labels.unique()).issubset(valid)

    def test_cell_labels_index_matches_cells(self, sp_interface):
        g = build_buffer_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        assert len(result.cell_labels) == len(sp_interface.cell_index)

    def test_interface_cells_detected(self, sp_interface):
        """With buffer_distance=50, cells near x=500 should be interface."""
        g = build_buffer_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        n_a = (result.cell_labels == "region_a_interface").sum()
        n_b = (result.cell_labels == "region_b_interface").sum()
        assert n_a > 0, "Should detect TypeA interface cells"
        assert n_b > 0, "Should detect TypeB interface cells"

    def test_store_writes_to_cell_meta(self, sp_interface):
        g = build_buffer_graph(sp_interface, buffer_distance=50)
        identify_interface(sp_interface, g, group_col="cell_type",
                           region_a="TypeA", region_b="TypeB", store=True)
        assert "interface_label" in sp_interface.cell_meta.columns

    def test_store_false_no_modification(self, sp_interface):
        g = build_buffer_graph(sp_interface, buffer_distance=50)
        identify_interface(sp_interface, g, group_col="cell_type",
                           region_a="TypeA", region_b="TypeB", store=False)
        assert "interface_label" not in sp_interface.cell_meta.columns

    def test_list_region_labels(self, sp_interface):
        """region_a as a list should work."""
        g = build_buffer_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a=["TypeA"], region_b="TypeB")
        assert isinstance(result, InterfaceResult)

    def test_no_interface_returns_empty(self, sp_interface):
        """With no buffer (contact only), far-apart cells have no interface."""
        g = build_contact_graph(sp_interface)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    min_interface_cells=1)
        assert isinstance(result, InterfaceResult)
        assert isinstance(result.summary, dict)
        assert "n_segments" in result.summary


class TestPolygonDensityMethod:
    """Tests for the density-based interface identification."""

    def test_density_returns_interface_result(self, sp_interface):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        g = build_buffer_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    method="density")
        assert isinstance(result, InterfaceResult)
        assert result.method == "density"

    def test_density_without_graph_explicit_threshold(self, sp_interface):
        result = identify_interface(sp_interface, graph=None,
                                    group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    method="density",
                                    distance_threshold=30.0)
        assert isinstance(result, InterfaceResult)

    def test_density_cell_labels_valid(self, sp_interface):
        result = identify_interface(sp_interface, graph=None,
                                    group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    method="density",
                                    distance_threshold=30.0)
        valid = {"region_a_interface", "region_b_interface",
                 "interior_a", "interior_b", "other"}
        assert set(result.cell_labels.unique()).issubset(valid)

    def test_density_contour_is_geometry(self, sp_interface):
        result = identify_interface(sp_interface, graph=None,
                                    group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    method="density",
                                    distance_threshold=30.0)
        if result.contour is not None:
            assert result.contour.geom_type in ("MultiLineString", "LineString")

    def test_density_scikit_image_missing_raises(self, sp_interface, monkeypatch):
        """Should raise ImportError if scikit-image not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "skimage" or name.startswith("skimage."):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="scikit-image"):
            identify_interface(sp_interface, graph=None,
                               group_col="cell_type",
                               region_a="TypeA", region_b="TypeB",
                               method="density", distance_threshold=30.0)


from spatioloji_s.spatial.point.interface import (
    identify_interface as point_identify_interface,
)
from spatioloji_s.spatial.point.graph import build_knn_graph


class TestPointGraphMethod:
    """Tests for point-based interface identification."""

    def test_returns_interface_result(self, sp_interface):
        g = build_knn_graph(sp_interface, k=10)
        result = point_identify_interface(sp_interface, g, group_col="cell_type",
                                          region_a="TypeA", region_b="TypeB")
        assert isinstance(result, InterfaceResult)

    def test_cell_labels_valid(self, sp_interface):
        g = build_knn_graph(sp_interface, k=10)
        result = point_identify_interface(sp_interface, g, group_col="cell_type",
                                          region_a="TypeA", region_b="TypeB")
        valid = {"region_a_interface", "region_b_interface",
                 "interior_a", "interior_b", "other"}
        assert set(result.cell_labels.unique()).issubset(valid)

    def test_interface_cells_detected(self, sp_interface):
        g = build_knn_graph(sp_interface, k=10)
        result = point_identify_interface(sp_interface, g, group_col="cell_type",
                                          region_a="TypeA", region_b="TypeB")
        assert result.summary["n_interface_a"] > 0 or result.summary["n_interface_b"] > 0

    def test_contour_geometry(self, sp_interface):
        g = build_knn_graph(sp_interface, k=10)
        result = point_identify_interface(sp_interface, g, group_col="cell_type",
                                          region_a="TypeA", region_b="TypeB")
        if result.contour is not None:
            assert result.contour.geom_type in ("MultiLineString", "LineString")

    def test_validation_same_as_polygon(self, sp_interface):
        with pytest.raises(ValueError, match="not found in cell_meta"):
            point_identify_interface(sp_interface, None, group_col="bad",
                                     region_a="TypeA", region_b="TypeB",
                                     method="graph")
