"""Tests for signed distance to interface utility."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString

from spatioloji_s.spatial._distance_utils import signed_distance_to_interface
from spatioloji_s.spatial._interface_types import InterfaceResult


def _make_interface_result(cell_labels, contour, region_a="TypeA", region_b="TypeB"):
    """Helper to create a minimal InterfaceResult for testing."""
    import geopandas as gpd

    segments = gpd.GeoDataFrame(
        {
            "segment_id": [0],
            "geometry": [contour.geoms[0] if contour else None],
            "length": [1.0],
            "tortuosity": [1.0],
            "n_cells_a": [1],
            "n_cells_b": [1],
        },
    )
    return InterfaceResult(
        cell_labels=cell_labels,
        contour=contour,
        segments=segments,
        summary={"total_length": 1.0, "n_segments": 1, "mean_tortuosity": 1.0, "n_interface_a": 1, "n_interface_b": 1},
        region_a=region_a,
        region_b=region_b,
        method="graph",
    )


class TestSignedDistance:
    """Tests for signed_distance_to_interface."""

    def test_basic_signed_distance(self, sp_interface):
        """Cells on region A side get positive, region B side get negative."""
        from spatioloji_s.spatial.polygon.interface import identify_interface
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        graph = build_buffer_graph(sp_interface, buffer_distance=50)
        iface = identify_interface(
            sp_interface,
            graph,
            group_col="cell_type",
            region_a="TypeA",
            region_b="TypeB",
            method="graph",
            min_interface_cells=1,
        )
        distances = signed_distance_to_interface(sp_interface, iface, coord_type="global")

        assert isinstance(distances, pd.Series)
        assert len(distances) == len(sp_interface.cell_index)

        # Region A cells should have positive distances
        a_mask = iface.cell_labels.isin(["region_a_interface", "interior_a"])
        assert (distances[a_mask] >= 0).all()

        # Region B cells should have negative distances
        b_mask = iface.cell_labels.isin(["region_b_interface", "interior_b"])
        assert (distances[b_mask] <= 0).all()

    def test_unsigned_distance(self, sp_interface):
        """When unsigned=True, all distances should be non-negative."""
        from spatioloji_s.spatial.polygon.interface import identify_interface
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        graph = build_buffer_graph(sp_interface, buffer_distance=50)
        iface = identify_interface(
            sp_interface,
            graph,
            group_col="cell_type",
            region_a="TypeA",
            region_b="TypeB",
            method="graph",
            min_interface_cells=1,
        )
        distances = signed_distance_to_interface(
            sp_interface,
            iface,
            coord_type="global",
            unsigned=True,
        )
        assert (distances >= 0).all()

    def test_no_contour_raises(self, sp_interface):
        """Should raise ValueError when contour is None."""
        labels = pd.Series("other", index=sp_interface.cell_index)
        iface = _make_interface_result(labels, contour=None)
        with pytest.raises(ValueError, match="contour"):
            signed_distance_to_interface(sp_interface, iface)

    def test_returns_series_with_cell_index(self, sp_interface):
        """Result should be indexed by cell ID."""
        contour = MultiLineString([LineString([(500, 0), (500, 1000)])])
        labels = pd.Series(
            ["interior_a"] * 50 + ["interior_b"] * 50,
            index=sp_interface.cell_index,
        )
        iface = _make_interface_result(labels, contour)
        distances = signed_distance_to_interface(sp_interface, iface)
        assert distances.index.equals(sp_interface.cell_index)
