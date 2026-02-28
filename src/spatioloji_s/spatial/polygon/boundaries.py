"""
boundaries.py - Cell-cell contact quantification

Measures physical interactions at cell borders:
- Shared boundary length between touching/overlapping cells
- Fraction of perimeter in contact with each neighbor
- Fraction of perimeter exposed (free boundary)

Key design decisions:
- Contact length is measured as the portion of each cell's OWN boundary
  that intersects the neighboring cell (asymmetric for overlapping cells).
- Total contact fraction uses unary_union of all contact segments per cell
  to avoid double-counting when multiple neighbors share boundary regions.

Requires polygon data and a contact/buffer graph from graph.py.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from typing import Optional
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from .graph import PolygonSpatialGraph, _get_gdf


# ========== Helper ==========

def _boundary_intersection_length(geom_src, geom_nbr):
    """
    Length of geom_src's boundary that lies inside or on geom_nbr.

    This is the correct contact length from geom_src's perspective:
    - For touching cells (shared edge): returns shared edge length
    - For partially overlapping cells: returns the portion of geom_src's
      outer boundary that is enclosed by geom_nbr (not the overlap area perimeter)

    Parameters
    ----------
    geom_src : shapely geometry
        The cell whose boundary we are measuring
    geom_nbr : shapely geometry
        The neighboring cell

    Returns
    -------
    float
        Contact length from geom_src's perspective
    shapely geometry or None
        The actual contact segment geometry (for union later)
    """
    try:
        seg = geom_src.boundary.intersection(geom_nbr)
        if seg.is_empty:
            return 0.0, None
        return seg.length, seg
    except Exception:
        return 0.0, None


# ========== contact_length ==========

def contact_length(sp: 'spatioloji',
                   graph: PolygonSpatialGraph,
                   coord_type: str = 'global') -> pd.DataFrame:
    """
    Compute contact boundary length between neighboring cells.

    For touching cells: shared edge length (symmetric).
    For partially overlapping cells: asymmetric — reports the portion
    of each cell's OWN boundary that lies within the neighbor.

    Parameters
    ----------
    sp : spatioloji
    graph : PolygonSpatialGraph
    coord_type : str

    Returns
    -------
    pd.DataFrame
        Columns: cell_a, cell_b, contact_length_a, contact_length_b
        contact_length_a = length of cell_a's boundary inside cell_b
        contact_length_b = length of cell_b's boundary inside cell_a
    """
    print(f"\n[Boundaries] Computing contact lengths...")

    gdf = _get_gdf(sp, coord_type=coord_type)
    geom = gdf.geometry
    edges = graph.to_edge_list()

    if len(edges) == 0:
        print("  ⚠ No edges in graph")
        return pd.DataFrame(
            columns=['cell_a', 'cell_b', 'contact_length_a', 'contact_length_b']
        )

    len_a = np.zeros(len(edges), dtype=np.float64)
    len_b = np.zeros(len(edges), dtype=np.float64)

    for i, row in enumerate(edges.itertuples()):
        cell_a, cell_b = row.cell_a, row.cell_b

        if cell_a not in geom.index or cell_b not in geom.index:
            continue

        poly_a = geom.loc[cell_a]
        poly_b = geom.loc[cell_b]

        if poly_a is None or poly_b is None:
            continue
        if poly_a.is_empty or poly_b.is_empty:
            continue

        # Asymmetric: each cell's own boundary measured separately
        len_a[i], _ = _boundary_intersection_length(poly_a, poly_b)
        len_b[i], _ = _boundary_intersection_length(poly_b, poly_a)

    edges = edges.copy()
    edges['contact_length_a'] = len_a
    edges['contact_length_b'] = len_b

    n_contact = ((len_a > 0) | (len_b > 0)).sum()
    print(f"  ✓ {n_contact}/{len(edges)} pairs have contact")
    if n_contact > 0:
        all_lens = np.concatenate([len_a[len_a > 0], len_b[len_b > 0]])
        print(f"    Length range: {all_lens.min():.1f} – {all_lens.max():.1f}")
        print(f"    Mean: {all_lens.mean():.1f}")

    return edges


# ========== _collect_contact_segments (shared internal helper) ==========

def _collect_contact_segments(sp, graph, gdf):
    """
    For each cell, collect all boundary-intersection geometries with neighbors.

    Returns
    -------
    dict : cell_id → list of shapely geometries
        Each geometry is the portion of that cell's boundary touching a neighbor.
    """
    geom = gdf.geometry
    edges = graph.to_edge_list()

    shared_geoms = {cell_id: [] for cell_id in gdf.index}

    for _, row in edges.iterrows():
        cell_a, cell_b = row.cell_a, row.cell_b

        if cell_a not in geom.index or cell_b not in geom.index:
            continue

        poly_a = geom.loc[cell_a]
        poly_b = geom.loc[cell_b]

        if poly_a is None or poly_b is None:
            continue
        if poly_a.is_empty or poly_b.is_empty:
            continue

        _, seg_a = _boundary_intersection_length(poly_a, poly_b)
        _, seg_b = _boundary_intersection_length(poly_b, poly_a)

        if seg_a is not None:
            shared_geoms[cell_a].append(seg_a)
        if seg_b is not None:
            shared_geoms[cell_b].append(seg_b)

    return shared_geoms


# ========== contact_fraction ==========

def contact_fraction(sp: 'spatioloji',
                     graph: PolygonSpatialGraph,
                     coord_type: str = 'global',
                     store: bool = True) -> pd.DataFrame:
    """
    Compute fraction of each cell's perimeter in contact with each neighbor.

    Per-pair fractions (asymmetric):
        fraction_a = contact_length_a / perimeter(cell_a)
        fraction_b = contact_length_b / perimeter(cell_b)

    Total contact fraction per cell uses unary_union of all contact segments
    to avoid double-counting when neighbors share boundary regions.

    Parameters
    ----------
    sp : spatioloji
    graph : PolygonSpatialGraph
    coord_type : str
    store : bool
        If True, stores 'total_contact_fraction' in sp.cell_meta

    Returns
    -------
    pd.DataFrame
        Columns: cell_a, cell_b, contact_length_a, contact_length_b,
                 fraction_a, fraction_b
    """
    print(f"\n[Boundaries] Computing contact fractions...")

    gdf = _get_gdf(sp, coord_type=coord_type)
    geom = gdf.geometry
    perimeters = geom.length

    # Get per-pair contact lengths
    edges = contact_length(sp, graph, coord_type=coord_type)

    if len(edges) == 0:
        return edges

    # Per-pair fractions (asymmetric, no double-counting issue here)
    frac_a = np.zeros(len(edges), dtype=np.float64)
    frac_b = np.zeros(len(edges), dtype=np.float64)

    for i, row in enumerate(edges.itertuples()):
        peri_a = perimeters.get(row.cell_a, 0.0)
        peri_b = perimeters.get(row.cell_b, 0.0)

        if peri_a > 0:
            frac_a[i] = row.contact_length_a / peri_a
        if peri_b > 0:
            frac_b[i] = row.contact_length_b / peri_b

    edges = edges.copy()
    edges['fraction_a'] = frac_a
    edges['fraction_b'] = frac_b

    # Total contact fraction: union segments first to avoid double-counting
    if store:
        shared_geoms = _collect_contact_segments(sp, graph, gdf)

        total_frac = pd.Series(0.0, index=gdf.index, dtype=np.float64)
        for cell_id, geom_list in shared_geoms.items():
            if geom_list:
                peri = perimeters.get(cell_id, 0.0)
                if peri > 0:
                    total_frac[cell_id] = (
                        unary_union(geom_list).length / peri
                    ).clip(0.0, 1.0)

        total_frac = total_frac.reindex(sp.cell_index, fill_value=0.0)
        sp.cell_meta['total_contact_fraction'] = total_frac.values
        print(f"  ✓ Stored 'total_contact_fraction' in cell_meta")

    print(f"  ✓ Contact fractions computed for {len(edges)} pairs")

    return edges


# ========== free_boundary_fraction ==========

def free_boundary_fraction(sp: 'spatioloji',
                           graph: PolygonSpatialGraph,
                           coord_type: str = 'global',
                           store: bool = True) -> pd.Series:
    """
    Compute fraction of each cell's perimeter NOT touching any neighbor.

    free_fraction = 1 - union(all contact segments).length / perimeter

    Uses unary_union of all contact segments per cell before measuring,
    so overlapping segments at triple junctions are never double-counted.

    Biologically:
    - high free_fraction (~1.0) → isolated, exposed to ECM/stroma
    - low free_fraction (~0.0) → tightly packed among neighbors

    Parameters
    ----------
    sp : spatioloji
    graph : PolygonSpatialGraph
    coord_type : str
    store : bool
        If True, stores 'free_boundary_fraction' and 'total_contact_fraction'

    Returns
    -------
    pd.Series
        Free boundary fraction per cell (indexed by cell_id)
    """
    print(f"\n[Boundaries] Computing free boundary fractions...")

    gdf = _get_gdf(sp, coord_type=coord_type)
    geom = gdf.geometry
    perimeters = geom.length

    # Collect boundary segments per cell
    shared_geoms = _collect_contact_segments(sp, graph, gdf)

    # Union per cell → deduplicate overlapping segments at triple junctions
    total_contact_len = pd.Series(0.0, index=gdf.index, dtype=np.float64)
    for cell_id, geom_list in shared_geoms.items():
        if geom_list:
            total_contact_len[cell_id] = unary_union(geom_list).length

    # free_fraction = 1 - (unioned contact length / perimeter)
    free_frac = pd.Series(1.0, index=gdf.index, dtype=np.float64)
    nonzero_peri = perimeters > 0
    free_frac[nonzero_peri] = (
        1.0 - total_contact_len[nonzero_peri] / perimeters[nonzero_peri]
    ).clip(0.0, 1.0)  # safety only for float precision

    free_frac = free_frac.reindex(sp.cell_index, fill_value=1.0)

    if store:
        sp.cell_meta['free_boundary_fraction'] = free_frac.values
        # Exact complement — always consistent with free_boundary_fraction
        sp.cell_meta['total_contact_fraction'] = (1.0 - free_frac).values
        print(f"  ✓ Stored 'free_boundary_fraction' and 'total_contact_fraction'")

    print(f"  ✓ Free boundary: mean={free_frac.mean():.3f}, "
          f"median={free_frac.median():.3f}")
    print(f"    Fully enclosed (< 0.05): {(free_frac < 0.05).sum()} cells")
    print(f"    Mostly exposed (> 0.80): {(free_frac > 0.80).sum()} cells")

    return free_frac


# ========== contact_summary ==========

def contact_summary(sp: 'spatioloji',
                    graph: PolygonSpatialGraph,
                    group_col: str,
                    coord_type: str = 'global') -> pd.DataFrame:
    """
    Summarize contact lengths between cell type pairs.

    Uses contact_length_a + contact_length_b averaged as the pairwise
    contact measure (symmetric summary for reporting purposes).

    Parameters
    ----------
    sp : spatioloji
    graph : PolygonSpatialGraph
    group_col : str
        Column in cell_meta defining cell types
    coord_type : str

    Returns
    -------
    pd.DataFrame
        Columns: type_a, type_b, n_contacts, total_length,
                 mean_length, median_length
    """
    print(f"\n[Boundaries] Summarizing contacts by '{group_col}'...")

    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"'{group_col}' not found in cell_meta")

    edges = contact_length(sp, graph, coord_type=coord_type)

    if len(edges) == 0:
        print("  ⚠ No contacts found")
        return pd.DataFrame(
            columns=['type_a', 'type_b', 'n_contacts',
                     'total_length', 'mean_length', 'median_length']
        )

    cell_to_group = sp.cell_meta[group_col].to_dict()

    edges = edges.copy()
    edges['type_a'] = edges['cell_a'].map(cell_to_group)
    edges['type_b'] = edges['cell_b'].map(cell_to_group)
    edges = edges.dropna(subset=['type_a', 'type_b'])

    # Use mean of both directions as symmetric summary length
    edges['mean_contact_length'] = (
        edges['contact_length_a'] + edges['contact_length_b']
    ) / 2.0

    # Normalize pair order (alphabetical) so A-B and B-A are the same
    swapped = edges['type_a'] > edges['type_b']
    edges.loc[swapped, ['type_a', 'type_b']] = (
        edges.loc[swapped, ['type_b', 'type_a']].values
    )

    grouped = edges.groupby(['type_a', 'type_b'])['mean_contact_length']

    summary = pd.DataFrame({
        'n_contacts': grouped.count(),
        'total_length': grouped.sum(),
        'mean_length': grouped.mean(),
        'median_length': grouped.median()
    }).reset_index().sort_values('total_length', ascending=False)

    print(f"  ✓ {len(summary)} cell type pair combinations")
    print(f"\n  Top contacts:")
    for _, row in summary.head(5).iterrows():
        print(f"    {row.type_a} – {row.type_b}: "
              f"{row.n_contacts} contacts, "
              f"total={row.total_length:.1f}, "
              f"mean={row.mean_length:.1f}")

    return summary