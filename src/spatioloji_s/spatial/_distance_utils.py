"""Shared distance utilities for interface-based analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Point

from spatioloji_s.spatial._interface_types import InterfaceResult


def _densify_contour(contour, max_spacing: float = 5.0) -> np.ndarray:
    """Sample points along contour lines at a maximum spacing.

    Interpolates additional points between contour vertices so that
    KDTree nearest-vertex distance closely approximates the true
    point-to-line distance.

    Args:
        contour: Shapely LineString or MultiLineString.
        max_spacing: Maximum distance between sampled points.

    Returns:
        np.ndarray of shape (n_points, 2).
    """
    lines = []
    if contour.geom_type == "MultiLineString":
        lines = list(contour.geoms)
    elif contour.geom_type == "LineString":
        lines = [contour]

    all_pts = []
    for line in lines:
        length = line.length
        if length == 0:
            continue
        n_pts = max(2, int(np.ceil(length / max_spacing)) + 1)
        fracs = np.linspace(0, 1, n_pts)
        pts = np.array([line.interpolate(f, normalized=True).coords[0] for f in fracs])
        all_pts.append(pts)

    if not all_pts:
        return np.empty((0, 2))
    return np.vstack(all_pts)


def classify_regions(
    sp,
    interface_result: InterfaceResult,
    group_col: str | None = None,
    min_segment_length: float = 0,
    coord_type: str = "global",
    store: bool = True,
    output_col: str = "region_side",
    grid_res: int = 500,
) -> pd.Series:
    """Classify ALL cells as region A or region B based on spatial zones.

    The interface contour segments partition the tissue into polygonal
    **zones**.  Each zone's identity (A or B) is determined by
    **majority vote** of the labeled cells inside it.  ALL cells
    within a zone (including immune, unclassified, etc.) inherit that
    zone's identity.

    This handles tissues with **multiple disconnected regions** (e.g.
    multiple Epi islands in stroma) correctly, and avoids misclassifying
    individual cells near the boundary.

    Args:
        sp: spatioloji object.
        interface_result: Result from ``identify_interface``.
        group_col: Column in ``cell_meta`` with cell type labels.
            Required to count region A vs B cells per zone.
            If ``None``, uses interface labels instead.
        min_segment_length: Minimum zone perimeter (in coordinate
            units).  Connected zones whose perimeter is shorter than
            this value are relabeled to match the surrounding region.
            Uses the same length scale as interface segment lengths
            in ``InterfaceResult.segments['length']``.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        store: If True, store result as ``output_col`` in ``cell_meta``.
        output_col: Column name for storing the classification.
        grid_res: Number of grid tiles along the longer tissue axis.
            Higher values preserve smaller regions but are slower.
            Default 500; increase if small regions are still lost.

    Returns:
        Series indexed by cell ID with values ``'region_a'`` or
        ``'region_b'`` for every cell.

    Examples:
        >>> sides = classify_regions(sp, interface_result,
        ...     group_col='cell_type', min_segment_length=500)
        >>> print(sides.value_counts())
        region_a    650000
        region_b    480000
    """
    region_a = interface_result.region_a
    region_b = interface_result.region_b
    a_list = [region_a] if isinstance(region_a, str) else list(region_a)
    b_list = [region_b] if isinstance(region_b, str) else list(region_b)

    # Get coordinates
    if coord_type == "global":
        x_all = np.asarray(sp.spatial.x_global)
        y_all = np.asarray(sp.spatial.y_global)
    else:
        x_all = np.asarray(sp.spatial.x_local)
        y_all = np.asarray(sp.spatial.y_local)

    cell_ids = sp.cell_index

    # Determine A/B masks
    if group_col is not None and group_col in sp.cell_meta.columns:
        cell_types = sp.cell_meta[group_col]
        a_mask = cell_types.isin(a_list).values
        b_mask = cell_types.isin(b_list).values
    else:
        labels_arr = interface_result.cell_labels.reindex(cell_ids).fillna("other").values
        a_mask = np.isin(labels_arr, ["region_a_interface", "interior_a"])
        b_mask = np.isin(labels_arr, ["region_b_interface", "interior_b"])

    # --- Grid-based region classification (fast, vectorized) ---
    # Partition tissue into a grid, majority-vote each tile, then assign
    # every cell to the region of its tile.  Same logic as the grid
    # interface method but used here for classification only.
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range >= y_range:
        nx = grid_res
        ny = max(1, int(grid_res * y_range / max(x_range, 1e-12)))
    else:
        ny = grid_res
        nx = max(1, int(grid_res * x_range / max(y_range, 1e-12)))

    x_bins = np.linspace(x_min, x_max, nx + 1)
    y_bins = np.linspace(y_min, y_max, ny + 1)
    x_idx = np.clip(np.digitize(x_all, x_bins) - 1, 0, nx - 1)
    y_idx = np.clip(np.digitize(y_all, y_bins) - 1, 0, ny - 1)

    # Count A and B cells per tile (vectorized)
    count_a = np.zeros((ny, nx), dtype=int)
    count_b = np.zeros((ny, nx), dtype=int)
    np.add.at(count_a, (y_idx[a_mask], x_idx[a_mask]), 1)
    np.add.at(count_b, (y_idx[b_mask], x_idx[b_mask]), 1)

    # Majority vote per tile; empty tiles (no A or B cells) are
    # filled from nearest non-empty neighbor so they don't default
    # to one side arbitrarily.
    has_cells = (count_a + count_b) > 0
    tile_is_a = np.where(has_cells, count_a > count_b, False)

    # Fill empty tiles from nearest labeled tile (flood-fill via
    # distance transform) so isolated gaps don't create artifacts.
    if not has_cells.all():
        from scipy.ndimage import distance_transform_edt

        # distance_transform_edt with return_indices gives indices
        # of the nearest non-zero (True) cell for every position.
        _, nearest_idx = distance_transform_edt(~has_cells, return_indices=True)
        tile_is_a[~has_cells] = tile_is_a[nearest_idx[0][~has_cells], nearest_idx[1][~has_cells]]

    # --- Filter small zones ---
    # Use connected-component labeling to find contiguous zones.
    # Zones whose perimeter (in coordinate units) is shorter than
    # min_segment_length are relabeled to match the surrounding zone.
    # Perimeter is measured the same way as interface segment lengths
    # (sum of boundary edge lengths in coordinate space).
    if min_segment_length > 0:
        from scipy.ndimage import label as nd_label

        tile_w = x_range / max(nx, 1)
        tile_h = y_range / max(ny, 1)

        for region_val in [True, False]:
            region_mask = tile_is_a == region_val
            labeled, n_components = nd_label(region_mask)
            for comp_id in range(1, n_components + 1):
                comp_mask = labeled == comp_id

                # Compute perimeter: count boundary edges between
                # this zone and the opposite region (or grid edge).
                # Horizontal edges (top/bottom of each tile)
                h_interior = np.sum(comp_mask[:-1, :] != comp_mask[1:, :])
                h_border = np.sum(comp_mask[0, :]) + np.sum(comp_mask[-1, :])
                # Vertical edges (left/right of each tile)
                v_interior = np.sum(comp_mask[:, :-1] != comp_mask[:, 1:])
                v_border = np.sum(comp_mask[:, 0]) + np.sum(comp_mask[:, -1])

                perimeter = (h_interior + h_border) * tile_h + (v_interior + v_border) * tile_w

                if perimeter < min_segment_length:
                    tile_is_a[comp_mask] = not region_val

    # Assign each cell to its tile's region
    cell_tile_is_a = tile_is_a[y_idx, x_idx]

    classification = pd.Series("region_b", index=cell_ids, name=output_col)
    classification[cell_tile_is_a] = "region_a"

    if store:
        sp._cell_meta[output_col] = classification.values
        n_a = (classification == "region_a").sum()
        n_b = (classification == "region_b").sum()
        print(f"  Region classification: {n_a:,} region_a, {n_b:,} region_b")
        print(f"  Stored in cell_meta['{output_col}']")

    return classification


def signed_distance_to_interface(
    sp,
    interface_result: InterfaceResult,
    coord_type: str = "global",
    unsigned: bool = False,
) -> pd.Series:
    """Compute signed distance from each cell to the interface contour.

    Positive distances indicate cells on the region A side, negative
    distances indicate cells on the region B side.

    The sign is determined by zone-based classification: contour segments
    partition the tissue into zones, each labeled A or B by majority vote.
    Labeled cells (region A/B) always keep their known sign.  "Other"
    cells get their sign from the zone they fall in.

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
    cell_ids = sp.cell_index

    # Compute unsigned distances using KDTree on contour sample points (fast)
    # instead of per-cell shapely Point.distance() calls (slow for >100k cells)
    from scipy.spatial import KDTree as ScipyKDTree

    # Densely sample points along all contour lines so that KDTree distance
    # closely approximates the true point-to-line distance.  We interpolate
    # at a spacing smaller than a typical cell diameter (~10 µm).
    contour_coords = _densify_contour(contour, max_spacing=5.0)

    if len(contour_coords) > 0:
        contour_tree = ScipyKDTree(contour_coords)
        all_coords = np.column_stack([x, y])
        raw_distances, _ = contour_tree.query(all_coords)
    else:
        raw_distances = np.array([Point(xi, yi).distance(contour) for xi, yi in zip(x, y, strict=True)])

    if unsigned:
        return pd.Series(raw_distances, index=cell_ids, name="distance_to_interface")

    # Assign signs:
    # - Labeled cells: from their known interface label (always correct)
    # - "Other" cells: from zone-based classification
    labels = interface_result.cell_labels
    labels_arr = labels.reindex(cell_ids).fillna("other").values
    a_mask = np.isin(labels_arr, ["region_a_interface", "interior_a"])
    b_mask = np.isin(labels_arr, ["region_b_interface", "interior_b"])
    other_mask = ~(a_mask | b_mask)

    signs = np.zeros(len(cell_ids))
    signs[a_mask] = 1.0
    signs[b_mask] = -1.0

    # For "other" cells, use zone-based classification
    if other_mask.any():
        region_sides = classify_regions(
            sp, interface_result, coord_type=coord_type, store=False,
        )
        zone_signs = np.where(region_sides.values == "region_a", 1.0, -1.0)
        signs[other_mask] = zone_signs[other_mask]

    signed = raw_distances * signs
    return pd.Series(signed, index=cell_ids, name="distance_to_interface")
