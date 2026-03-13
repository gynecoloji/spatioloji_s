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
