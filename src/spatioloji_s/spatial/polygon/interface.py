# src/spatioloji_s/spatial/polygon/interface.py
"""Interface cell identification for polygon-based spatial analysis."""

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


def _validate_inputs(
    sp,
    graph,
    group_col: str,
    region_a: str | list[str],
    region_b: str | list[str],
    method: str,
    distance_threshold: float | None,
) -> tuple[list[str], list[str]]:
    """Validate inputs and normalize region labels to lists.

    Args:
        sp: spatioloji object.
        graph: Spatial graph or None.
        group_col: Column in cell_meta.
        region_a: Region A label(s).
        region_b: Region B label(s).
        method: "graph" or "density".
        distance_threshold: Distance threshold for density method.

    Returns:
        Tuple of (region_a_list, region_b_list).

    Raises:
        ValueError: On invalid inputs.
    """
    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"'{group_col}' not found in cell_meta. "
                         f"Available: {list(sp.cell_meta.columns)}")

    a_list = [region_a] if isinstance(region_a, str) else list(region_a)
    b_list = [region_b] if isinstance(region_b, str) else list(region_b)

    col_vals = set(sp.cell_meta[group_col].dropna().unique())
    for label in a_list + b_list:
        if label not in col_vals:
            raise ValueError(f"Label '{label}' not found in '{group_col}'. "
                             f"Available: {sorted(col_vals)}")

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

    if method == "graph" and graph is None:
        raise ValueError("graph is required for method='graph'")

    if method == "density" and graph is None and distance_threshold is None:
        raise ValueError(
            "distance_threshold must be set when method='density' and graph=None"
        )

    return a_list, b_list


def _empty_result(
    sp, a_list: list[str], b_list: list[str], group_col: str, method: str,
) -> InterfaceResult:
    """Build an empty InterfaceResult when no interface is found.

    Args:
        sp: spatioloji object.
        a_list: Region A labels.
        b_list: Region B labels.
        group_col: Column in cell_meta.
        method: "graph" or "density".

    Returns:
        InterfaceResult with no interface cells.
    """
    labels = sp.cell_meta[group_col]
    cell_labels = pd.Series("other", index=sp.cell_index)
    cell_labels[labels.isin(a_list)] = "interior_a"
    cell_labels[labels.isin(b_list)] = "interior_b"

    segs = gpd.GeoDataFrame(
        {"segment_id": pd.Series(dtype=int), "length": pd.Series(dtype=float),
         "tortuosity": pd.Series(dtype=float),
         "n_cells_a": pd.Series(dtype=int), "n_cells_b": pd.Series(dtype=int)},
        geometry=[],
    )
    return InterfaceResult(
        cell_labels=cell_labels, contour=None, segments=segs,
        summary={"total_length": 0.0, "n_segments": 0, "mean_tortuosity": 0.0,
                 "n_interface_a": 0, "n_interface_b": 0},
        region_a=a_list if len(a_list) > 1 else a_list[0],
        region_b=b_list if len(b_list) > 1 else b_list[0],
        method=method,
    )


def _compute_tortuosity(geom) -> float:
    """Compute tortuosity of a LineString.

    Args:
        geom: A shapely LineString.

    Returns:
        Tortuosity value (>= 1.0). np.inf for degenerate cases.
    """
    if geom is None or geom.is_empty:
        return np.inf
    length = geom.length
    if length == 0:
        return np.inf
    start, end = geom.coords[0], geom.coords[-1]
    endpoint_dist = np.hypot(end[0] - start[0], end[1] - start[1])
    if endpoint_dist == 0:
        return np.inf
    return length / endpoint_dist


def _graph_method(
    sp,
    graph,
    group_col: str,
    a_list: list[str],
    b_list: list[str],
    min_interface_cells: int,
    coord_type: str,
) -> InterfaceResult:
    """Graph-based interface identification.

    Args:
        sp: spatioloji object.
        graph: PolygonSpatialGraph.
        group_col: Column in cell_meta.
        a_list: Region A labels.
        b_list: Region B labels.
        min_interface_cells: Min cells per segment side.
        coord_type: "global" or "local".

    Returns:
        InterfaceResult.
    """
    cell_index = graph.cell_index
    adj = graph.adjacency
    labels = sp.cell_meta[group_col]

    # Build masks for cells in graph that belong to each region
    graph_labels = labels.reindex(cell_index)
    mask_a = graph_labels.isin(a_list).values
    mask_b = graph_labels.isin(b_list).values

    # Find cross-region edges from COO
    adj_coo = adj.tocoo()
    row, col = adj_coo.row, adj_coo.col
    cross_mask = (mask_a[row] & mask_b[col]) | (mask_b[row] & mask_a[col])

    if not cross_mask.any():
        warnings.warn("No cross-region edges found between the two regions.",
                      UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col, "graph")

    cross_rows = row[cross_mask]
    cross_cols = col[cross_mask]

    # Identify interface cell indices (in graph space)
    interface_a_idx = set(cross_rows[mask_a[cross_rows]]) | set(cross_cols[mask_a[cross_cols]])
    interface_b_idx = set(cross_rows[mask_b[cross_rows]]) | set(cross_cols[mask_b[cross_cols]])

    # Build cell labels
    cell_labels = pd.Series("other", index=sp.cell_index)
    cell_labels[labels.isin(a_list)] = "interior_a"
    cell_labels[labels.isin(b_list)] = "interior_b"

    for idx in interface_a_idx:
        cid = cell_index[idx]
        cell_labels.loc[cid] = "region_a_interface"
    for idx in interface_b_idx:
        cid = cell_index[idx]
        cell_labels.loc[cid] = "region_b_interface"

    # --- Connected components for segment detection ---
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

    # --- Build contour from shared polygon edges ---
    gdf = sp.to_geopandas(coord_type=coord_type, include_metadata=False)
    geom_dict = {cid: gdf.loc[cid, "geometry"] for cid in gdf.index
                 if cid in set(cell_index[list(interface_a_idx | interface_b_idx)])}

    shared_lines = []
    pair_component = []
    seen_pairs = set()
    for r, c in zip(cr_r, cr_c, strict=True):
        pair = (min(r, c), max(r, c))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        cid_r = cell_index[r]
        cid_c = cell_index[c]
        if cid_r not in geom_dict or cid_c not in geom_dict:
            continue

        try:
            shared = geom_dict[cid_r].intersection(geom_dict[cid_c])
        except Exception:
            continue

        if shared.is_empty:
            cr_geom = geom_dict[cid_r].centroid
            cc_geom = geom_dict[cid_c].centroid
            shared = LineString([cr_geom.coords[0], cc_geom.coords[0]])

        if shared.geom_type == "LineString":
            shared_lines.append(shared)
            pair_component.append(comp_labels[idx_map[r]])
        elif shared.geom_type == "MultiLineString":
            for line in shared.geoms:
                shared_lines.append(line)
                pair_component.append(comp_labels[idx_map[r]])
        elif shared.geom_type == "Point":
            cr_geom = geom_dict[cid_r].centroid
            cc_geom = geom_dict[cid_c].centroid
            shared_lines.append(LineString([cr_geom.coords[0], cc_geom.coords[0]]))
            pair_component.append(comp_labels[idx_map[r]])

    # --- Build segments GeoDataFrame ---
    seg_rows = []
    for comp_id in range(n_components):
        comp_cell_indices = [all_cross_cells[i] for i in range(n_sub)
                            if comp_labels[i] == comp_id]
        n_ca = sum(1 for i in comp_cell_indices if i in interface_a_idx)
        n_cb = sum(1 for i in comp_cell_indices if i in interface_b_idx)

        if n_ca < min_interface_cells or n_cb < min_interface_cells:
            continue

        comp_lines = [shared_lines[i] for i in range(len(shared_lines))
                      if pair_component[i] == comp_id]
        if not comp_lines:
            continue

        merged = unary_union(comp_lines)
        if merged.geom_type == "Point":
            continue
        if merged.geom_type not in ("LineString", "MultiLineString"):
            continue

        seg_rows.append({
            "segment_id": len(seg_rows),
            "geometry": merged,
            "length": merged.length,
            "tortuosity": _compute_tortuosity(merged) if merged.geom_type == "LineString"
                          else np.mean([_compute_tortuosity(g) for g in merged.geoms]),
            "n_cells_a": n_ca,
            "n_cells_b": n_cb,
        })

    if not seg_rows:
        warnings.warn("All interface segments dropped by min_interface_cells filter.",
                      UserWarning, stacklevel=3)
        segs = gpd.GeoDataFrame(
            {"segment_id": pd.Series(dtype=int), "length": pd.Series(dtype=float),
             "tortuosity": pd.Series(dtype=float),
             "n_cells_a": pd.Series(dtype=int), "n_cells_b": pd.Series(dtype=int)},
            geometry=[],
        )
        return InterfaceResult(
            cell_labels=cell_labels, contour=None, segments=segs,
            summary={"total_length": 0.0, "n_segments": 0, "mean_tortuosity": 0.0,
                     "n_interface_a": int((cell_labels == "region_a_interface").sum()),
                     "n_interface_b": int((cell_labels == "region_b_interface").sum())},
            region_a=a_list if len(a_list) > 1 else a_list[0],
            region_b=b_list if len(b_list) > 1 else b_list[0],
            method="graph",
        )

    segments = gpd.GeoDataFrame(seg_rows, geometry="geometry")
    segments.set_crs(epsg=None, inplace=True) if hasattr(segments, "set_crs") else None

    all_geoms = []
    for _, row in segments.iterrows():
        g = row.geometry
        if g.geom_type == "LineString":
            all_geoms.append(g)
        elif g.geom_type == "MultiLineString":
            all_geoms.extend(g.geoms)
    contour = MultiLineString(all_geoms) if all_geoms else None

    summary = {
        "total_length": float(segments["length"].sum()),
        "n_segments": len(segments),
        "mean_tortuosity": float(segments["tortuosity"].replace(np.inf, np.nan).mean()),
        "n_interface_a": int((cell_labels == "region_a_interface").sum()),
        "n_interface_b": int((cell_labels == "region_b_interface").sum()),
    }

    return InterfaceResult(
        cell_labels=cell_labels,
        contour=contour,
        segments=segments,
        summary=summary,
        region_a=a_list if len(a_list) > 1 else a_list[0],
        region_b=b_list if len(b_list) > 1 else b_list[0],
        method="graph",
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
    """Identify interface cells between two spatial regions.

    Finds cells at the boundary between two named cell groups (e.g.,
    tumor vs stroma) and computes the interface contour geometry and
    per-segment metrics.

    Args:
        sp: spatioloji object with polygon data.
        graph: Pre-built ``PolygonSpatialGraph``. Required for
            ``method='graph'``. Optional for ``method='density'``
            (used to auto-estimate ``distance_threshold``).
        group_col: Column in ``cell_meta`` defining cell groups.
        region_a: Label(s) for region A. If a list, all labels are
            treated as a single region.
        region_b: Label(s) for region B.
        method: ``'graph'`` (default) -- uses adjacency edges to find
            cross-region contacts. ``'density'`` -- uses KDE to find
            the density decision boundary.
        min_interface_cells: Minimum cells on each side of a segment
            for it to be retained.
        bandwidth: KDE bandwidth (density method only). Auto-estimated
            via Scott's rule if ``None``.
        distance_threshold: Max distance from KDE contour to label a
            cell as interface (density method only). Auto-estimated
            from graph if ``None``.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        store: If ``True``, add ``'interface_label'`` to ``cell_meta``.

    Returns:
        InterfaceResult with cell labels, contour geometry, segments,
        and summary metrics.

    Raises:
        ValueError: On invalid inputs.
        ImportError: If ``method='density'`` and scikit-image is not
            installed.

    Example:
        >>> g = build_contact_graph(sp)
        >>> result = identify_interface(sp, g, "cell_type", "Tumor", "Stromal")
        >>> print(result.summary)
    """
    a_list, b_list = _validate_inputs(
        sp, graph, group_col, region_a, region_b, method, distance_threshold
    )

    print(f"\n[Interface] Identifying interface: "
          f"{a_list} vs {b_list} (method={method})")

    if method == "graph":
        result = _graph_method(
            sp, graph, group_col, a_list, b_list, min_interface_cells, coord_type
        )
    else:
        result = _density_method(
            sp, graph, group_col, a_list, b_list, min_interface_cells,
            bandwidth, distance_threshold, coord_type
        )

    if store:
        sp._cell_meta["interface_label"] = result.cell_labels.values
        print("  Stored 'interface_label' in cell_meta")

    n_a = result.summary.get("n_interface_a", 0)
    n_b = result.summary.get("n_interface_b", 0)
    print(f"  {n_a + n_b} interface cells detected "
          f"({n_a} region_a, {n_b} region_b)")
    print(f"  {result.summary['n_segments']} segment(s), "
          f"total length={result.summary['total_length']:.1f}")

    return result


def _density_method(
    sp, graph, group_col, a_list, b_list, min_interface_cells,
    bandwidth, distance_threshold, coord_type,
) -> InterfaceResult:
    """KDE density-based interface identification. Placeholder for Task 5."""
    raise NotImplementedError("Density method not yet implemented")
