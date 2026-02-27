"""
boundaries.py - Cell-cell contact quantification

Measures physical interactions at cell borders:
- Shared boundary length between touching cells
- Fraction of perimeter in contact with each neighbor
- Fraction of perimeter exposed (free boundary)

Requires polygon data and a contact/buffer graph from graph.py.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from scipy import sparse

from .graph import PolygonSpatialGraph, _get_gdf


def contact_length(sp: 'spatioloji',
                   graph: PolygonSpatialGraph,
                   coord_type: str = 'global') -> pd.DataFrame:
    """
    Compute shared boundary length between neighboring cells.
    
    For each pair of neighbors in the graph, calculates the length
    of the shared boundary (intersection of polygon boundaries).
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    graph : PolygonSpatialGraph
        Pre-built polygon graph (contact or buffer)
    coord_type : str
        'global' or 'local' coordinates
    
    Returns
    -------
    pd.DataFrame
        Edge list with columns: cell_a, cell_b, shared_length
        One row per neighbor pair (undirected).
    
    Examples
    --------
    >>> from spatioloji_s.spatial.polygon.graph import build_contact_graph
    >>> from spatioloji_s.spatial.polygon.boundaries import contact_length
    >>> graph = build_contact_graph(sp)
    >>> lengths = contact_length(sp, graph)
    >>> print(lengths.describe())
    """
    print(f"\n[Boundaries] Computing contact lengths...")
    
    gdf = _get_gdf(sp, coord_type=coord_type)
    geom = gdf.geometry
    
    # Get edge list from graph (upper triangle only)
    edges = graph.to_edge_list()
    
    if len(edges) == 0:
        print("  ⚠ No edges in graph")
        return pd.DataFrame(columns=['cell_a', 'cell_b', 'shared_length'])
    
    # Compute shared boundary length for each pair
    shared_lengths = np.zeros(len(edges), dtype=np.float64)
    
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
        
        # Intersection of two polygons → could be line, point, or polygon
        try:
            intersection = poly_a.intersection(poly_b)
            shared_lengths[i] = intersection.length
        except Exception:
            shared_lengths[i] = 0.0
    
    edges = edges.copy()
    edges['shared_length'] = shared_lengths
    
    n_contact = (shared_lengths > 0).sum()
    print(f"  ✓ {n_contact}/{len(edges)} pairs have shared boundary")
    if n_contact > 0:
        nonzero = shared_lengths[shared_lengths > 0]
        print(f"    Length range: {nonzero.min():.1f} – {nonzero.max():.1f}")
        print(f"    Mean: {nonzero.mean():.1f}")
    
    return edges


def contact_fraction(sp: 'spatioloji',
                     graph: PolygonSpatialGraph,
                     coord_type: str = 'global',
                     store: bool = True) -> pd.DataFrame:
    """
    Compute fraction of each cell's perimeter in contact with each neighbor.
    
    For cell A with perimeter P_A and shared boundary L_AB with cell B:
        fraction_AB = L_AB / P_A
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    graph : PolygonSpatialGraph
        Pre-built polygon graph
    coord_type : str
        'global' or 'local' coordinates
    store : bool
        If True, store total_contact_fraction in sp.cell_meta
    
    Returns
    -------
    pd.DataFrame
        Columns: cell_a, cell_b, shared_length, fraction_a, fraction_b
        fraction_a = shared_length / perimeter_of_cell_a
        fraction_b = shared_length / perimeter_of_cell_b
    
    Examples
    --------
    >>> graph = build_contact_graph(sp)
    >>> fractions = contact_fraction(sp, graph)
    >>> # Which cell pairs have the strongest contact?
    >>> fractions.sort_values('fraction_a', ascending=False).head(10)
    """
    print(f"\n[Boundaries] Computing contact fractions...")
    
    gdf = _get_gdf(sp, coord_type=coord_type)
    geom = gdf.geometry
    
    # Get shared lengths first
    edges = contact_length(sp, graph, coord_type=coord_type)
    
    if len(edges) == 0:
        return edges
    
    # Compute perimeters
    perimeters = geom.length
    
    # Calculate fractions for both directions
    frac_a = np.zeros(len(edges), dtype=np.float64)
    frac_b = np.zeros(len(edges), dtype=np.float64)
    
    for i, row in enumerate(edges.itertuples()):
        peri_a = perimeters.get(row.cell_a, 0.0)
        peri_b = perimeters.get(row.cell_b, 0.0)
        
        if peri_a > 0:
            frac_a[i] = row.shared_length / peri_a
        if peri_b > 0:
            frac_b[i] = row.shared_length / peri_b
    
    edges = edges.copy()
    edges['fraction_a'] = frac_a
    edges['fraction_b'] = frac_b
    
    # Store total contact fraction per cell
    if store:
        total_frac = pd.Series(0.0, index=sp.cell_index, dtype=np.float64)
        
        for _, row in edges.iterrows():
            if row.cell_a in total_frac.index:
                total_frac.loc[row.cell_a] += row.fraction_a
            if row.cell_b in total_frac.index:
                total_frac.loc[row.cell_b] += row.fraction_b
        
        # Clip to [0, 1]
        total_frac = total_frac.clip(0.0, 1.0)
        sp.cell_meta['total_contact_fraction'] = total_frac.values
        print(f"  ✓ Stored 'total_contact_fraction' in cell_meta")
    
    print(f"  ✓ Contact fractions computed for {len(edges)} pairs")
    
    return edges

def free_boundary_fraction(sp: 'spatioloji',
                           graph: PolygonSpatialGraph,
                           coord_type: str = 'global',
                           store: bool = True) -> pd.Series:
    """
    Compute fraction of each cell's perimeter not touching any neighbor.
    
    free_fraction = 1.0 - sum(shared_length_with_all_neighbors) / perimeter
    
    Biologically, high free boundary means a cell is exposed to
    extracellular matrix / stroma rather than packed among other cells.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    graph : PolygonSpatialGraph
        Pre-built polygon graph
    coord_type : str
        'global' or 'local' coordinates
    store : bool
        If True, store result in sp.cell_meta['free_boundary_fraction']
    
    Returns
    -------
    pd.Series
        Free boundary fraction per cell (0.0 = fully enclosed, 1.0 = isolated)
    
    Examples
    --------
    >>> graph = build_contact_graph(sp)
    >>> free_frac = free_boundary_fraction(sp, graph)
    >>> # Find cells at tissue edges (high free boundary)
    >>> edge_cells = sp.cell_index[free_frac > 0.8]
    """
    print(f"\n[Boundaries] Computing free boundary fractions...")
    
    gdf = _get_gdf(sp, coord_type=coord_type)
    geom = gdf.geometry
    
    # Get shared lengths
    edges = contact_length(sp, graph, coord_type=coord_type)
    
    # Sum shared length per cell
    total_shared = pd.Series(0.0, index=gdf.index, dtype=np.float64)
    
    for _, row in edges.iterrows():
        if row.cell_a in total_shared.index:
            total_shared.loc[row.cell_a] += row.shared_length
        if row.cell_b in total_shared.index:
            total_shared.loc[row.cell_b] += row.shared_length
    
    # Compute free fraction
    perimeters = geom.length
    free_frac = pd.Series(1.0, index=gdf.index, dtype=np.float64)
    
    nonzero_peri = perimeters > 0
    free_frac[nonzero_peri] = 1.0 - (
        total_shared[nonzero_peri] / perimeters[nonzero_peri]
    )
    
    # Clip to [0, 1] (numerical precision)
    free_frac = free_frac.clip(0.0, 1.0)
    
    # Reindex to full cell_index
    free_frac = free_frac.reindex(sp.cell_index, fill_value=1.0)
    
    if store:
        sp.cell_meta['free_boundary_fraction'] = free_frac.values
        print(f"  ✓ Stored 'free_boundary_fraction' in cell_meta")
    
    print(f"  ✓ Free boundary: mean={free_frac.mean():.3f}, "
          f"median={free_frac.median():.3f}")
    print(f"    Fully enclosed (< 0.05): {(free_frac < 0.05).sum()} cells")
    print(f"    Mostly exposed (> 0.80): {(free_frac > 0.80).sum()} cells")
    
    return free_frac


def contact_summary(sp: 'spatioloji',
                    graph: PolygonSpatialGraph,
                    group_col: str,
                    coord_type: str = 'global') -> pd.DataFrame:
    """
    Summarize contact lengths between cell type pairs.
    
    Aggregates shared boundary length across all pairs of a given
    cell type combination. Useful for understanding which cell types
    have the most physical interaction.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    graph : PolygonSpatialGraph
        Pre-built polygon graph
    group_col : str
        Column in cell_meta defining cell types
    coord_type : str
        'global' or 'local' coordinates
    
    Returns
    -------
    pd.DataFrame
        Columns: type_a, type_b, n_contacts, total_length,
        mean_length, median_length
    
    Examples
    --------
    >>> graph = build_contact_graph(sp)
    >>> summary = contact_summary(sp, graph, group_col='cell_type')
    >>> # Which cell type pairs have the most contact?
    >>> summary.sort_values('total_length', ascending=False).head(10)
    """
    print(f"\n[Boundaries] Summarizing contacts by '{group_col}'...")
    
    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"'{group_col}' not found in cell_meta")
    
    # Get contact lengths
    edges = contact_length(sp, graph, coord_type=coord_type)
    
    if len(edges) == 0:
        print("  ⚠ No contacts found")
        return pd.DataFrame(
            columns=['type_a', 'type_b', 'n_contacts',
                     'total_length', 'mean_length', 'median_length']
        )
    
    # Map cell IDs to group labels
    cell_to_group = sp.cell_meta[group_col].to_dict()
    
    edges = edges.copy()
    edges['type_a'] = edges['cell_a'].map(cell_to_group)
    edges['type_b'] = edges['cell_b'].map(cell_to_group)
    
    # Drop pairs where either cell has no group label
    edges = edges.dropna(subset=['type_a', 'type_b'])
    
    # Normalize pair order (alphabetical) so A-B and B-A are the same
    swapped = edges['type_a'] > edges['type_b']
    edges.loc[swapped, ['type_a', 'type_b']] = (
        edges.loc[swapped, ['type_b', 'type_a']].values
    )
    
    # Aggregate
    grouped = edges.groupby(['type_a', 'type_b'])['shared_length']
    
    summary = pd.DataFrame({
        'n_contacts': grouped.count(),
        'total_length': grouped.sum(),
        'mean_length': grouped.mean(),
        'median_length': grouped.median()
    }).reset_index()
    
    summary = summary.sort_values('total_length', ascending=False)
    
    print(f"  ✓ {len(summary)} cell type pair combinations")
    print(f"\n  Top contacts:")
    for _, row in summary.head(5).iterrows():
        print(f"    {row.type_a} – {row.type_b}: "
              f"{row.n_contacts} contacts, "
              f"total={row.total_length:.1f}, "
              f"mean={row.mean_length:.1f}")
    
    return summary

