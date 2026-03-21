# src/spatioloji_s/spatial/_interface.py
"""Unified interface cell identification (grid-based).

Shared by both point and polygon modules. Uses only cell centroid
coordinates — no polygon geometry or spatial graph required.
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from spatioloji_s.spatial._interface_types import InterfaceResult


def _validate_inputs(
    sp,
    group_col: str,
    region_a: str | list[str],
    region_b: str | list[str],
) -> tuple[list[str], list[str]]:
    """Validate inputs and normalize region labels to lists.

    Args:
        sp: spatioloji object.
        group_col: Column in cell_meta.
        region_a: Region A label(s).
        region_b: Region B label(s).

    Returns:
        Tuple of (region_a_list, region_b_list).

    Raises:
        ValueError: On invalid inputs.
    """
    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"'{group_col}' not found in cell_meta. Available: {list(sp.cell_meta.columns)}")

    a_list = [region_a] if isinstance(region_a, str) else list(region_a)
    b_list = [region_b] if isinstance(region_b, str) else list(region_b)

    col_vals = set(sp.cell_meta[group_col].dropna().unique())
    for label in a_list + b_list:
        if label not in col_vals:
            raise ValueError(f"Label '{label}' not found in '{group_col}'. Available: {sorted(col_vals)}")

    overlap = set(a_list) & set(b_list)
    if overlap:
        raise ValueError(f"region_a and region_b overlap on: {overlap}")

    labels = sp.cell_meta[group_col]
    n_a = labels.isin(a_list).sum()
    n_b = labels.isin(b_list).sum()
    if n_a == 0:
        raise ValueError(f"region_a {a_list} has 0 cells in '{group_col}'")
    if n_b == 0:
        raise ValueError(f"region_b {b_list} has 0 cells in '{group_col}'")

    return a_list, b_list


def _empty_result(
    sp,
    a_list: list[str],
    b_list: list[str],
    group_col: str,
) -> InterfaceResult:
    """Build an empty InterfaceResult when no interface is found.

    Args:
        sp: spatioloji object.
        a_list: Region A labels.
        b_list: Region B labels.
        group_col: Column in cell_meta.

    Returns:
        InterfaceResult with no interface cells.
    """
    labels = sp.cell_meta[group_col]
    cell_labels = pd.Series("other", index=sp.cell_index)
    cell_labels[labels.isin(a_list)] = "interior_a"
    cell_labels[labels.isin(b_list)] = "interior_b"

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
    return InterfaceResult(
        cell_labels=cell_labels,
        contour=None,
        segments=segs,
        summary={"total_length": 0.0, "n_segments": 0, "mean_tortuosity": 0.0, "n_interface_a": 0, "n_interface_b": 0},
        region_a=a_list if len(a_list) > 1 else a_list[0],
        region_b=b_list if len(b_list) > 1 else b_list[0],
        method="grid",
    )


def _compute_tortuosity(geom) -> float:
    """Compute tortuosity of a LineString.

    For open contours: ``length / endpoint_distance`` (standard definition).
    For closed/looping contours (start ~ end): ``length / max_chord``,
    where max_chord is the maximum pairwise distance among sampled points
    along the contour.  This avoids ``inf`` for closed boundaries.

    Args:
        geom: A shapely LineString.

    Returns:
        Tortuosity value (>= 1.0). ``np.nan`` for degenerate cases.
    """
    if geom is None or geom.is_empty:
        return np.nan
    length = geom.length
    if length == 0:
        return np.nan
    start, end = geom.coords[0], geom.coords[-1]
    endpoint_dist = np.hypot(end[0] - start[0], end[1] - start[1])

    # Closed or near-closed contour: use max chord instead
    if endpoint_dist < length * 0.01:
        coords = np.array(geom.coords)
        if len(coords) < 2:
            return np.nan
        # Sample up to 200 points for efficiency
        if len(coords) > 200:
            idx = np.linspace(0, len(coords) - 1, 200, dtype=int)
            coords = coords[idx]
        # Max pairwise distance (diameter of the contour)
        from scipy.spatial import distance

        max_chord = distance.cdist(coords, coords).max()
        if max_chord == 0:
            return np.nan
        return length / max_chord

    return length / endpoint_dist


def _grid_method(
    sp,
    group_col,
    a_list,
    b_list,
    min_interface_cells,
    distance_threshold,
    grid_resolution,
    coord_type,
    close_contours,
) -> InterfaceResult:
    """Grid-based interface: partition tissue into tiles, majority-vote each tile.

    The interface is where A-tiles border B-tiles. Produces clean, rough
    boundaries suitable for visual inspection.

    Args:
        sp: spatioloji object.
        group_col: Column in cell_meta.
        a_list: Region A labels.
        b_list: Region B labels.
        min_interface_cells: Min cells per segment side.
        distance_threshold: Max distance from contour for interface cells.
        grid_resolution: Number of tiles along the longest axis.
        coord_type: 'global' or 'local'.
        close_contours: Whether to close contour loops.

    Returns:
        InterfaceResult.
    """
    from skimage.measure import find_contours

    labels = sp.cell_meta[group_col]

    if coord_type == "global":
        x_all = np.asarray(sp.spatial.x_global)
        y_all = np.asarray(sp.spatial.y_global)
    else:
        x_all = np.asarray(sp.spatial.x_local)
        y_all = np.asarray(sp.spatial.y_local)

    mask_a = labels.isin(a_list).values
    mask_b = labels.isin(b_list).values

    # Build grid
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Scale resolution to maintain aspect ratio
    if x_range >= y_range:
        nx = grid_resolution
        ny = max(1, int(grid_resolution * y_range / x_range))
    else:
        ny = grid_resolution
        nx = max(1, int(grid_resolution * x_range / y_range))

    # Assign cells to grid tiles
    x_bins = np.linspace(x_min, x_max, nx + 1)
    y_bins = np.linspace(y_min, y_max, ny + 1)
    x_idx = np.clip(np.digitize(x_all, x_bins) - 1, 0, nx - 1)
    y_idx = np.clip(np.digitize(y_all, y_bins) - 1, 0, ny - 1)

    # Count A and B cells per tile — vectorized with np.add.at
    count_a = np.zeros((ny, nx), dtype=int)
    count_b = np.zeros((ny, nx), dtype=int)
    np.add.at(count_a, (y_idx[mask_a], x_idx[mask_a]), 1)
    np.add.at(count_b, (y_idx[mask_b], x_idx[mask_b]), 1)

    # Majority vote per tile: +1 = A, -1 = B
    grid_label = np.zeros((ny, nx), dtype=float)
    has_cells = (count_a + count_b) > 0
    grid_label[has_cells] = np.where(
        count_a[has_cells] >= count_b[has_cells], 1.0, -1.0
    )

    # Fill empty tiles from nearest labeled tile so contours don't
    # fragment around gaps (empty tiles at level=0 would be treated
    # as the contour boundary itself).
    if not has_cells.all():
        from scipy.ndimage import distance_transform_edt

        _, nearest_idx = distance_transform_edt(~has_cells, return_indices=True)
        grid_label[~has_cells] = grid_label[nearest_idx[0][~has_cells], nearest_idx[1][~has_cells]]

    # Pad grid with a 1-cell border so enclosed regions produce
    # closed contours even when they touch the image edge.
    pad_val = grid_label[has_cells].mean()
    # Use the majority label as padding so the dominant region's contours
    # close properly at tissue edges.
    pad_label = 1.0 if pad_val >= 0 else -1.0
    grid_label = np.pad(grid_label, pad_width=1, constant_values=pad_label)

    # Extract contour at level=0 (boundary between A and B tiles)
    raw_contours = find_contours(grid_label, level=0.0)

    if not raw_contours:
        warnings.warn("No grid boundary found.", UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col)

    # Convert pixel coords to spatial coords.
    # Subtract 1 from contour indices to undo the padding offset.
    contour_lines = []
    for c in raw_contours:
        cx = c[:, 1] - 1  # undo pad
        cy = c[:, 0] - 1
        if nx > 1:
            spatial_x = x_bins[0] + cx * (x_bins[-1] - x_bins[0]) / (nx - 1)
        else:
            spatial_x = np.full(len(c), (x_min + x_max) / 2)
        if ny > 1:
            spatial_y = y_bins[0] + cy * (y_bins[-1] - y_bins[0]) / (ny - 1)
        else:
            spatial_y = np.full(len(c), (y_min + y_max) / 2)

        coords = np.column_stack([spatial_x, spatial_y])

        # Close contour if endpoints are near each other
        if close_contours and len(coords) >= 3:
            gap = np.linalg.norm(coords[0] - coords[-1])
            tile_diag = np.hypot(x_range / nx, y_range / ny)
            if gap > 0 and gap < tile_diag * 3:
                coords = np.vstack([coords, coords[:1]])

        if len(coords) >= 2:
            contour_lines.append(LineString(coords))

    if not contour_lines:
        return _empty_result(sp, a_list, b_list, group_col)

    # Clip contours to convex hull of all cells
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon as ShapelyPolygon

    all_pts = np.column_stack([x_all, y_all])
    hull = ConvexHull(all_pts)
    hull_poly = ShapelyPolygon(all_pts[hull.vertices])

    clipped_lines = []
    for line in contour_lines:
        clipped = line.intersection(hull_poly)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "LineString":
            clipped_lines.append(clipped)
        elif clipped.geom_type == "MultiLineString":
            clipped_lines.extend(clipped.geoms)

    if not clipped_lines:
        return _empty_result(sp, a_list, b_list, group_col)

    # Auto-estimate distance_threshold
    if distance_threshold is None:
        tile_size = max(x_range / nx, y_range / ny)
        distance_threshold = tile_size * 1.5
        print(f"  Auto distance_threshold={distance_threshold:.1f} (1.5x tile size)")

    # --- Vectorized cell labeling using KDTree on contour points ---
    from scipy.spatial import KDTree as ScipyKDTree

    # Sample points along all contour lines for fast distance queries
    contour_pts = []
    for line in clipped_lines:
        coords = np.array(line.coords)
        contour_pts.append(coords)
    contour_pts = np.vstack(contour_pts)

    # Build KDTree on contour sample points
    contour_tree = ScipyKDTree(contour_pts)

    # Query distance for all cells at once
    all_coords = np.column_stack([x_all, y_all])
    cell_dists_to_contour, _ = contour_tree.query(all_coords)

    # Label cells
    cell_labels = pd.Series("other", index=sp.cell_index)
    cell_labels[labels.isin(a_list)] = "interior_a"
    cell_labels[labels.isin(b_list)] = "interior_b"

    # Interface = labeled cells within distance_threshold of contour
    near_contour = cell_dists_to_contour <= distance_threshold
    cell_labels[(mask_a & near_contour)] = "region_a_interface"
    cell_labels[(mask_b & near_contour)] = "region_b_interface"

    # --- Vectorized per-segment cell counting ---
    # For each contour line, build a small KDTree and count nearby interface cells
    seg_rows = []
    interface_mask = cell_labels.isin(["region_a_interface", "region_b_interface"]).values
    interface_coords = all_coords[interface_mask]
    interface_is_a = mask_a[interface_mask]
    interface_is_b = mask_b[interface_mask]

    for line in clipped_lines:
        seg_pts = np.array(line.coords)
        seg_tree = ScipyKDTree(seg_pts)

        # Find interface cells near this segment
        dists, _ = seg_tree.query(interface_coords)
        near_seg = dists <= distance_threshold

        n_ca = int(interface_is_a[near_seg].sum())
        n_cb = int(interface_is_b[near_seg].sum())

        if n_ca < min_interface_cells or n_cb < min_interface_cells:
            continue

        seg_rows.append({
            "segment_id": len(seg_rows),
            "geometry": line,
            "length": line.length,
            "tortuosity": _compute_tortuosity(line),
            "n_cells_a": n_ca,
            "n_cells_b": n_cb,
        })

    if not seg_rows:
        warnings.warn("All grid segments dropped by min_interface_cells.", UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col)

    segments = gpd.GeoDataFrame(seg_rows, geometry="geometry")
    all_geoms = [row.geometry for _, row in segments.iterrows()]
    contour = MultiLineString(all_geoms) if all_geoms else None

    summary = {
        "total_length": float(segments["length"].sum()),
        "n_segments": len(segments),
        "mean_tortuosity": float(segments["tortuosity"].mean()),
        "n_interface_a": int((cell_labels == "region_a_interface").sum()),
        "n_interface_b": int((cell_labels == "region_b_interface").sum()),
        "grid_resolution": grid_resolution,
        "tile_size_x": x_range / nx,
        "tile_size_y": y_range / ny,
    }

    return InterfaceResult(
        cell_labels=cell_labels,
        contour=contour,
        segments=segments,
        summary=summary,
        region_a=a_list,
        region_b=b_list,
        method="grid",
    )


def identify_interface(
    sp,
    group_col: str = "cell_type",
    region_a: str | list[str] = "",
    region_b: str | list[str] = "",
    min_interface_cells: int = 3,
    distance_threshold: float | None = None,
    grid_resolution: int = 50,
    coord_type: str = "global",
    store: bool = True,
    close_contours: bool = True,
) -> InterfaceResult:
    """Identify interface cells between two spatial regions.

    Uses a grid-based approach: partition the tissue into tiles,
    majority-vote each tile as region A or B, then extract contour
    lines where A-tiles border B-tiles.  Produces clean, rough
    boundaries that match what you'd draw by eye.

    Works identically for point-based and polygon-based data — only
    cell centroid coordinates are used.

    Args:
        sp: spatioloji object.
        group_col: Column in ``cell_meta`` defining cell groups.
        region_a: Label(s) for region A.
        region_b: Label(s) for region B.
        min_interface_cells: Min cells per segment side.
        distance_threshold: Max distance from contour for interface
            cells.  If None, auto-estimated from tile size.
        grid_resolution: Number of tiles along the longest axis.
            Higher = more detailed, lower = rougher.  Default 50.
        coord_type: ``'global'`` or ``'local'``.
        store: If ``True``, add ``'interface_label'`` to ``cell_meta``.
        close_contours: If ``True`` (default), close contour loops whose
            endpoints are near each other. Ensures enclosed regions
            (e.g. stroma islands within epithelium) get proper closed
            boundaries.

    Returns:
        InterfaceResult.

    Examples:
        >>> result = identify_interface(sp, group_col="cell_type",
        ...     region_a="Epi", region_b="stroma", grid_resolution=50)
    """
    a_list, b_list = _validate_inputs(sp, group_col, region_a, region_b)

    print(f"\n[Interface] Identifying interface: {a_list} vs {b_list}")

    result = _grid_method(
        sp, group_col, a_list, b_list, min_interface_cells, distance_threshold, grid_resolution, coord_type,
        close_contours,
    )

    if store:
        sp._cell_meta["interface_label"] = result.cell_labels.values
        print("  Stored 'interface_label' in cell_meta")

    n_a = result.summary.get("n_interface_a", 0)
    n_b = result.summary.get("n_interface_b", 0)
    print(f"  {n_a + n_b} interface cells ({n_a} region_a, {n_b} region_b)")
    print(f"  {result.summary['n_segments']} segment(s)")

    return result


def filter_interface(
    sp,
    interface_result: InterfaceResult,
    min_segment_length: float = 0,
    group_col: str = "cell_type",
    distance_threshold: float | None = None,
    coord_type: str = "global",
    store: bool = True,
) -> InterfaceResult:
    """Filter interface segments by length and re-label cells.

    Drops segments shorter than ``min_segment_length``, rebuilds the
    contour geometry, and re-computes cell labels so that only cells
    near the surviving segments are marked as interface.

    Args:
        sp: spatioloji object.
        interface_result: Result from ``identify_interface``.
        min_segment_length: Minimum segment length (in coordinate
            units, same scale as ``segments['length']``).  Segments
            shorter than this are dropped.
        group_col: Column in ``cell_meta`` with cell type labels.
        distance_threshold: Max distance from a surviving segment
            for a cell to keep its interface label.  If ``None``,
            re-uses the value from ``interface_result.summary``
            (auto-estimated during ``identify_interface``).
        coord_type: ``'global'`` or ``'local'``.
        store: If ``True``, update ``'interface_label'`` in ``cell_meta``.

    Returns:
        New InterfaceResult with only the surviving segments.

    Examples:
        >>> result = identify_interface(sp, group_col="cell_type",
        ...     region_a="Epi", region_b="stroma")
        >>> filtered = filter_interface(sp, result, min_segment_length=3000)
    """
    segments = interface_result.segments

    if min_segment_length > 0:
        keep = segments["length"] >= min_segment_length
        segments = segments[keep].reset_index(drop=True)
        segments["segment_id"] = range(len(segments))

    region_a = interface_result.region_a
    region_b = interface_result.region_b
    a_list = [region_a] if isinstance(region_a, str) else list(region_a)
    b_list = [region_b] if isinstance(region_b, str) else list(region_b)

    if len(segments) == 0:
        warnings.warn(
            f"All segments dropped by min_segment_length={min_segment_length}.",
            UserWarning,
            stacklevel=2,
        )
        result = _empty_result(sp, a_list, b_list, group_col)
        if store:
            sp._cell_meta["interface_label"] = result.cell_labels.values
        return result

    # Rebuild contour from surviving segments
    all_geoms = list(segments.geometry)
    contour = MultiLineString(all_geoms)

    # Re-label cells: only cells near surviving segments keep interface labels
    labels = sp.cell_meta[group_col]
    mask_a = labels.isin(a_list).values
    mask_b = labels.isin(b_list).values

    if coord_type == "global":
        x_all = np.asarray(sp.spatial.x_global)
        y_all = np.asarray(sp.spatial.y_global)
    else:
        x_all = np.asarray(sp.spatial.x_local)
        y_all = np.asarray(sp.spatial.y_local)

    # Auto-estimate distance_threshold from tile size if available
    if distance_threshold is None:
        tile_x = interface_result.summary.get("tile_size_x")
        tile_y = interface_result.summary.get("tile_size_y")
        if tile_x is not None and tile_y is not None:
            distance_threshold = max(tile_x, tile_y) * 1.5
        else:
            # Fallback: estimate from segment point spacing
            from scipy.spatial import KDTree as ScipyKDTree

            seg_pts = np.vstack([np.array(g.coords) for g in all_geoms])
            tree = ScipyKDTree(seg_pts)
            dists, _ = tree.query(seg_pts, k=2)
            distance_threshold = float(np.median(dists[:, 1])) * 3

    from scipy.spatial import KDTree as ScipyKDTree

    contour_pts = np.vstack([np.array(g.coords) for g in all_geoms])
    contour_tree = ScipyKDTree(contour_pts)

    all_coords = np.column_stack([x_all, y_all])
    cell_dists, _ = contour_tree.query(all_coords)

    cell_labels = pd.Series("other", index=sp.cell_index)
    cell_labels[labels.isin(a_list)] = "interior_a"
    cell_labels[labels.isin(b_list)] = "interior_b"

    near_contour = cell_dists <= distance_threshold
    cell_labels[(mask_a & near_contour)] = "region_a_interface"
    cell_labels[(mask_b & near_contour)] = "region_b_interface"

    summary = {
        "total_length": float(segments["length"].sum()),
        "n_segments": len(segments),
        "mean_tortuosity": float(segments["tortuosity"].mean()),
        "n_interface_a": int((cell_labels == "region_a_interface").sum()),
        "n_interface_b": int((cell_labels == "region_b_interface").sum()),
    }
    # Carry over grid metadata if present
    for key in ("grid_resolution", "tile_size_x", "tile_size_y"):
        if key in interface_result.summary:
            summary[key] = interface_result.summary[key]

    n_dropped = len(interface_result.segments) - len(segments)
    print(f"  Filtered: kept {len(segments)} segments, dropped {n_dropped} "
          f"(min_length={min_segment_length})")

    result = InterfaceResult(
        cell_labels=cell_labels,
        contour=contour,
        segments=segments,
        summary=summary,
        region_a=region_a,
        region_b=region_b,
        method=interface_result.method,
    )

    if store:
        sp._cell_meta["interface_label"] = result.cell_labels.values
        print("  Updated 'interface_label' in cell_meta")

    return result
