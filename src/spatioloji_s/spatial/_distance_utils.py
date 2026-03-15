"""Shared distance utilities for interface-based analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Point

from spatioloji_s.spatial._interface_types import InterfaceResult


def signed_distance_to_interface(
    sp,
    interface_result: InterfaceResult,
    coord_type: str = "global",
    unsigned: bool = False,
) -> pd.Series:
    """Compute signed distance from each cell to the interface contour.

    Positive distances indicate cells on the region A side, negative
    distances indicate cells on the region B side.

    Args:
        sp: spatioloji object.
        interface_result: Result from ``identify_interface``.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        unsigned: If True, return absolute distances.

    Returns:
        Series indexed by cell ID with signed (or unsigned) distances.

    Raises:
        ValueError: If ``interface_result.contour`` is None or
            ``coord_type`` is invalid.
    """
    if interface_result.contour is None:
        raise ValueError(
            "InterfaceResult.contour is None — cannot compute distances. "
            "Ensure identify_interface produced a valid contour."
        )

    if coord_type not in ("global", "local"):
        raise ValueError(f"coord_type must be 'global' or 'local', got '{coord_type}'")

    # Get cell coordinates
    if coord_type == "global":
        x = np.asarray(sp.spatial.x_global)
        y = np.asarray(sp.spatial.y_global)
    else:
        x = np.asarray(sp.spatial.x_local)
        y = np.asarray(sp.spatial.y_local)

    contour = interface_result.contour
    labels = interface_result.cell_labels
    cell_ids = sp.cell_index

    # Compute unsigned distances
    raw_distances = np.array([Point(xi, yi).distance(contour) for xi, yi in zip(x, y, strict=True)])

    if unsigned:
        return pd.Series(raw_distances, index=cell_ids, name="distance_to_interface")

    # Assign sign based on cell labels
    signs = np.ones(len(cell_ids))
    for i, cid in enumerate(cell_ids):
        label = labels.get(cid, "other")
        if label in ("region_b_interface", "interior_b"):
            signs[i] = -1.0
        # region_a_interface, interior_a, other → positive (default)

    signed = raw_distances * signs
    return pd.Series(signed, index=cell_ids, name="distance_to_interface")
