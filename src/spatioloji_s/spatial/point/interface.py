# src/spatioloji_s/spatial/point/interface.py
"""Interface cell identification for point-based spatial analysis."""

from __future__ import annotations

import warnings
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

from spatioloji_s.spatial._interface_types import InterfaceResult

# Reuse validation and helpers from polygon module
from spatioloji_s.spatial.polygon.interface import (
    _compute_tortuosity,
    _empty_result,
    _validate_inputs,
)


def _point_graph_method(
    sp, graph, group_col, a_list, b_list, min_interface_cells, coord_type,
) -> InterfaceResult:
    """Graph-based interface for point data (uses midpoints for contour).

    Args:
        sp: spatioloji object.
        graph: PointSpatialGraph.
        group_col: Column in cell_meta.
        a_list: Region A labels.
        b_list: Region B labels.
        min_interface_cells: Min cells per segment side.
        coord_type: "global" or "local".

    Returns:
        InterfaceResult.
    """
    cell_ids_graph = graph.cell_ids
    adj = graph.adjacency
    labels = sp.cell_meta[group_col]

    graph_labels = labels.reindex(cell_ids_graph)
    mask_a = graph_labels.isin(a_list).values
    mask_b = graph_labels.isin(b_list).values

    adj_coo = adj.tocoo()
    row, col = adj_coo.row, adj_coo.col
    cross_mask = (mask_a[row] & mask_b[col]) | (mask_b[row] & mask_a[col])

    if not cross_mask.any():
        warnings.warn("No cross-region edges found.", UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col, "graph")

    cross_rows = row[cross_mask]
    cross_cols = col[cross_mask]

    interface_a_idx = set(cross_rows[mask_a[cross_rows]]) | set(cross_cols[mask_a[cross_cols]])
    interface_b_idx = set(cross_rows[mask_b[cross_rows]]) | set(cross_cols[mask_b[cross_cols]])

    # Cell labels
    cell_labels = pd.Series("other", index=sp.cell_index)
    cell_labels[labels.isin(a_list)] = "interior_a"
    cell_labels[labels.isin(b_list)] = "interior_b"

    for idx in interface_a_idx:
        cell_labels.loc[cell_ids_graph[idx]] = "region_a_interface"
    for idx in interface_b_idx:
        cell_labels.loc[cell_ids_graph[idx]] = "region_b_interface"

    # Get coordinates
    if coord_type == "global":
        x_all = np.asarray(sp.spatial.x_global)
        y_all = np.asarray(sp.spatial.y_global)
    else:
        x_all = np.asarray(sp.spatial.x_local)
        y_all = np.asarray(sp.spatial.y_local)

    # Build coordinate lookup by cell_id
    coord_dict = {}
    for i, cid in enumerate(sp.cell_index):
        coord_dict[cid] = (x_all[i], y_all[i])

    # Connected components
    upper = cross_rows < cross_cols
    cr_r, cr_c = cross_rows[upper], cross_cols[upper]
    all_cross_cells = sorted(set(cr_r) | set(cr_c))

    if len(all_cross_cells) == 0:
        return _empty_result(sp, a_list, b_list, group_col, "graph")

    idx_map = {old: new for new, old in enumerate(all_cross_cells)}
    n_sub = len(all_cross_cells)
    sub_r = np.array([idx_map[r] for r in cr_r])
    sub_c = np.array([idx_map[c] for c in cr_c])
    sub_adj = sparse.csr_matrix(
        (np.ones(len(sub_r)), (sub_r, sub_c)), shape=(n_sub, n_sub)
    )
    sub_adj = sub_adj + sub_adj.T
    n_components, comp_labels = connected_components(sub_adj, directed=False)

    # Build contour from midpoints between cross-region pairs
    midpoints_by_comp = {}
    seen_pairs = set()
    for r, c in zip(cr_r, cr_c, strict=True):
        pair = (min(r, c), max(r, c))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        cid_r = cell_ids_graph[r]
        cid_c = cell_ids_graph[c]
        if cid_r not in coord_dict or cid_c not in coord_dict:
            continue

        mx = (coord_dict[cid_r][0] + coord_dict[cid_c][0]) / 2
        my = (coord_dict[cid_r][1] + coord_dict[cid_c][1]) / 2
        comp_id = comp_labels[idx_map[r]]
        midpoints_by_comp.setdefault(comp_id, []).append((mx, my))

    # Build segments
    seg_rows = []
    for comp_id in range(n_components):
        comp_cell_indices = [all_cross_cells[i] for i in range(n_sub)
                            if comp_labels[i] == comp_id]
        n_ca = sum(1 for i in comp_cell_indices if i in interface_a_idx)
        n_cb = sum(1 for i in comp_cell_indices if i in interface_b_idx)

        if n_ca < min_interface_cells or n_cb < min_interface_cells:
            continue

        pts = midpoints_by_comp.get(comp_id, [])
        if len(pts) < 2:
            continue

        # Order midpoints by nearest-neighbor walk for a clean line
        pts_arr = np.array(pts)

        if len(pts_arr) == 2:
            line = LineString(pts_arr)
        else:
            # Greedy nearest-neighbor walk from the first midpoint
            visited = [0]
            remaining = set(range(1, len(pts_arr)))
            while remaining:
                curr = visited[-1]
                best_dist, best_idx = np.inf, -1
                for j in remaining:
                    d = np.linalg.norm(pts_arr[curr] - pts_arr[j])
                    if d < best_dist:
                        best_dist, best_idx = d, j
                if best_idx >= 0:
                    visited.append(best_idx)
                    remaining.discard(best_idx)
                else:
                    break
            ordered = pts_arr[visited]
            line = LineString(ordered)

        seg_rows.append({
            "segment_id": len(seg_rows),
            "geometry": line,
            "length": line.length,
            "tortuosity": _compute_tortuosity(line),
            "n_cells_a": n_ca,
            "n_cells_b": n_cb,
        })

    if not seg_rows:
        warnings.warn("All segments dropped by min_interface_cells.",
                      UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col, "graph")

    segments = gpd.GeoDataFrame(seg_rows, geometry="geometry")

    all_geoms = [row.geometry for _, row in segments.iterrows()]
    contour = MultiLineString(all_geoms) if all_geoms else None

    summary = {
        "total_length": float(segments["length"].sum()),
        "n_segments": len(segments),
        "mean_tortuosity": float(segments["tortuosity"].replace(np.inf, np.nan).mean()),
        "n_interface_a": int((cell_labels == "region_a_interface").sum()),
        "n_interface_b": int((cell_labels == "region_b_interface").sum()),
    }

    return InterfaceResult(
        cell_labels=cell_labels, contour=contour, segments=segments,
        summary=summary,
        region_a=a_list if len(a_list) > 1 else a_list[0],
        region_b=b_list if len(b_list) > 1 else b_list[0],
        method="graph",
    )


def _point_density_method(
    sp, graph, group_col, a_list, b_list, min_interface_cells,
    bandwidth, distance_threshold, coord_type,
) -> InterfaceResult:
    """Density method for point data — delegates to polygon implementation.

    Args:
        sp: spatioloji object.
        graph: Optional PointSpatialGraph.
        group_col: Column in cell_meta.
        a_list: Region A labels.
        b_list: Region B labels.
        min_interface_cells: Min cells per segment side.
        bandwidth: KDE bandwidth or None.
        distance_threshold: Max distance from contour or None.
        coord_type: "global" or "local".

    Returns:
        InterfaceResult.
    """
    # The density method is coordinate-based, not polygon-specific.
    # Reuse the polygon implementation directly.
    from spatioloji_s.spatial.polygon.interface import _density_method
    return _density_method(
        sp, graph, group_col, a_list, b_list, min_interface_cells,
        bandwidth, distance_threshold, coord_type,
    )


def identify_interface(
    sp,
    graph=None,
    group_col: str = "cell_type",
    region_a: str | list[str] = "",
    region_b: str | list[str] = "",
    method: Literal["graph", "density"] = "graph",
    min_interface_cells: int = 3,
    bandwidth: float | None = None,
    distance_threshold: float | None = None,
    coord_type: str = "global",
    store: bool = True,
) -> InterfaceResult:
    """Identify interface cells between two spatial regions (point-based).

    Uses cell centroid coordinates and KNN/radius/Delaunay graphs to find
    cross-region contacts. See the polygon version for full parameter docs.

    Args:
        sp: spatioloji object.
        graph: Pre-built ``PointSpatialGraph``. Required for
            ``method='graph'``.
        group_col: Column in ``cell_meta`` defining cell groups.
        region_a: Label(s) for region A.
        region_b: Label(s) for region B.
        method: ``'graph'`` or ``'density'``.
        min_interface_cells: Min cells per segment side.
        bandwidth: KDE bandwidth (density only).
        distance_threshold: Max contour distance (density only).
        coord_type: ``'global'`` or ``'local'``.
        store: If ``True``, add ``'interface_label'`` to ``cell_meta``.

    Returns:
        InterfaceResult.

    Example:
        >>> g = build_knn_graph(sp, k=10)
        >>> result = identify_interface(sp, g, "cell_type", "Tumor", "Stromal")
    """
    a_list, b_list = _validate_inputs(
        sp, graph, group_col, region_a, region_b, method, distance_threshold
    )

    print(f"\n[Interface/Point] Identifying interface: "
          f"{a_list} vs {b_list} (method={method})")

    if method == "graph":
        result = _point_graph_method(
            sp, graph, group_col, a_list, b_list, min_interface_cells, coord_type
        )
    else:
        result = _point_density_method(
            sp, graph, group_col, a_list, b_list, min_interface_cells,
            bandwidth, distance_threshold, coord_type
        )

    if store:
        sp._cell_meta["interface_label"] = result.cell_labels.values
        print("  Stored 'interface_label' in cell_meta")

    n_a = result.summary.get("n_interface_a", 0)
    n_b = result.summary.get("n_interface_b", 0)
    print(f"  {n_a + n_b} interface cells ({n_a} region_a, {n_b} region_b)")
    print(f"  {result.summary['n_segments']} segment(s)")

    return result
