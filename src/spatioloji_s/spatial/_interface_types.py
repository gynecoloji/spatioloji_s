"""Shared data structures for interface analysis."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiLineString


@dataclass
class InterfaceResult:
    """Container for interface analysis results.

    Attributes:
        cell_labels: Series indexed by cell ID with values
            ``"region_a_interface"``, ``"region_b_interface"``,
            ``"interior_a"``, ``"interior_b"``, or ``"other"``.
        contour: Combined interface geometry (union of all segment lines).
            ``None`` when no interface is found.
        segments: GeoDataFrame with one row per disconnected interface
            segment. Columns: ``segment_id``, ``geometry`` (LineString),
            ``length``, ``tortuosity``, ``n_cells_a``, ``n_cells_b``.
            CRS is always ``None`` (pixel/micron coordinates).
        summary: Dict with keys ``total_length``, ``n_segments``,
            ``mean_tortuosity``, ``n_interface_a``, ``n_interface_b``.
        region_a: Region A label(s) used.
        region_b: Region B label(s) used.
        method: ``"graph"`` or ``"density"``.
    """

    cell_labels: pd.Series
    contour: MultiLineString | None
    segments: gpd.GeoDataFrame
    summary: dict
    region_a: str | list[str]
    region_b: str | list[str]
    method: str
