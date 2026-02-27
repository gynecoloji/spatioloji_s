"""
statistics.py - Distance-based spatial statistics

Direct distance calculations and tests between cells and cell types.
Works from coordinates without requiring a pre-built graph (though
some functions can optionally use one for efficiency).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .graph import PointSpatialGraph


def nearest_neighbor_distances(
    sj: 'spatioloji',
    coord_type: str = 'global',
    cell_type_col: Optional[str] = None,
    k: int = 1,
    store: bool = True,
) -> pd.DataFrame:
    """
    Compute distance to k-th nearest neighbor for each cell.
    
    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    coord_type : str
        'global' or 'local'.
    cell_type_col : str, optional
        If provided, also compute summary statistics per cell type.
    k : int
        Which nearest neighbor (1 = closest, 2 = second closest, etc.).
    store : bool
        If True, add 'nnd' column to sj.cell_meta.
    
    Returns
    -------
    pd.DataFrame with columns:
        'nnd'       : nearest neighbor distance per cell
        'nn_cell_id': cell ID of the nearest neighbor
    Also prints per-type summary if cell_type_col is given.
    """
    coords = sj.get_spatial_coords(coord_type=coord_type)
    n_cells = len(coords)
    
    tree = KDTree(coords)
    # k+1 because query includes self
    dists, indices = tree.query(coords, k=k + 1)
    
    nnd = dists[:, k]           # k-th nearest neighbor distance
    nn_idx = indices[:, k]      # index of k-th nearest neighbor
    nn_ids = sj.cell_index[nn_idx]
    
    result = pd.DataFrame({
        'nnd': nnd,
        'nn_cell_id': nn_ids,
    }, index=sj.cell_index)
    
    if store:
        col_name = 'nnd' if k == 1 else f'nnd_k{k}'
        sj.cell_meta[col_name] = nnd
    
    # Report
    print(f"  ✓ Nearest neighbor distances (k={k}):")
    print(f"    mean={nnd.mean():.2f}, median={np.median(nnd):.2f}, "
          f"std={nnd.std():.2f}")
    
    # Per-type summary
    if cell_type_col is not None:
        if cell_type_col not in sj.cell_meta.columns:
            raise ValueError(f"'{cell_type_col}' not found in cell_meta")
        
        labels = sj.cell_meta[cell_type_col].values
        print(f"\n    Per-type summary:")
        
        for ct in np.unique(labels):
            mask = labels == ct
            ct_nnd = nnd[mask]
            print(f"      {ct:>20s}: mean={ct_nnd.mean():.2f}, "
                  f"median={np.median(ct_nnd):.2f}, n={mask.sum()}")
    
    return result

def cross_type_distances(
    sj: 'spatioloji',
    cell_type_col: str,
    source_type: str,
    target_type: str,
    coord_type: str = 'global',
    k: int = 1,
    store: bool = True,
) -> pd.DataFrame:
    """
    Compute distance from each source-type cell to its nearest
    target-type cell.
    
    Asymmetric: distance(A→B) != distance(B→A).
    
    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    cell_type_col : str
        Column with cell type labels.
    source_type : str
        "From" cell type (each cell of this type gets a distance).
    target_type : str
        "To" cell type (nearest cell of this type is found).
    coord_type : str
        'global' or 'local'.
    k : int
        k-th nearest target (1 = closest).
    store : bool
        If True, add result column to sj.cell_meta.
    
    Returns
    -------
    pd.DataFrame
        Indexed by source cell IDs with columns:
        'distance'       : distance to k-th nearest target
        'target_cell_id' : cell ID of that target
    """
    if cell_type_col not in sj.cell_meta.columns:
        raise ValueError(f"'{cell_type_col}' not found in cell_meta")
    
    labels = sj.cell_meta[cell_type_col].values
    coords = sj.get_spatial_coords(coord_type=coord_type)
    
    source_mask = labels == source_type
    target_mask = labels == target_type
    
    n_source = source_mask.sum()
    n_target = target_mask.sum()
    
    if n_source == 0:
        raise ValueError(f"No cells of source type '{source_type}'")
    if n_target == 0:
        raise ValueError(f"No cells of target type '{target_type}'")
    if k > n_target:
        raise ValueError(
            f"k={k} but only {n_target} target cells exist"
        )
    
    source_coords = coords[source_mask]
    target_coords = coords[target_mask]
    source_ids = sj.cell_index[source_mask]
    target_ids = sj.cell_index[target_mask]
    
    # Build tree on targets, query sources
    tree = KDTree(target_coords)
    dists, indices = tree.query(source_coords, k=k)
    
    # Handle k=1 vs k>1 (query returns different shapes)
    if k == 1:
        nn_dists = dists
        nn_target_ids = target_ids[indices]
    else:
        nn_dists = dists[:, k - 1]
        nn_target_ids = target_ids[indices[:, k - 1]]
    
    result = pd.DataFrame({
        'distance': nn_dists,
        'target_cell_id': nn_target_ids,
    }, index=source_ids)
    
    # Store in cell_meta (NaN for non-source cells)
    if store:
        col_name = f'dist_{source_type}_to_{target_type}'
        sj.cell_meta[col_name] = np.nan
        sj.cell_meta.loc[source_ids, col_name] = nn_dists
    
    print(f"  ✓ Cross-type distance ({source_type}→{target_type}):")
    print(f"    n_source={n_source}, n_target={n_target}")
    print(f"    mean={nn_dists.mean():.2f}, "
          f"median={np.median(nn_dists):.2f}, "
          f"std={nn_dists.std():.2f}")
    
    return result

def proximity_score(
    sj: 'spatioloji',
    cell_type_col: str,
    coord_type: str = 'global',
    normalize: bool = True,
    types: Optional[List[str]] = None,
) -> dict:
    """
    Compute pairwise proximity scores between all cell types.
    
    For each pair (A, B), computes the median distance from A cells
    to their nearest B cell. Optionally normalizes by self-distance.
    
    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    cell_type_col : str
        Column with cell type labels.
    coord_type : str
        'global' or 'local'.
    normalize : bool
        If True, divide each pair distance by the source type's
        self-distance (A→A). Scores <1 = closer than same-type.
    types : list of str, optional
        Cell types to include. If None, uses all types.
    
    Returns
    -------
    dict with keys:
        'median_distance' : pd.DataFrame (n_types x n_types)
        'normalized'      : pd.DataFrame or None (if normalize=True)
        'mean_distance'   : pd.DataFrame (for reference)
    """
    if cell_type_col not in sj.cell_meta.columns:
        raise ValueError(f"'{cell_type_col}' not found in cell_meta")
    
    labels = sj.cell_meta[cell_type_col].values
    coords = sj.get_spatial_coords(coord_type=coord_type)
    
    if types is None:
        types = list(np.unique(labels))
    
    n_types = len(types)
    
    # Build KDTree per type
    trees = {}
    type_ids = {}
    for ct in types:
        mask = labels == ct
        trees[ct] = KDTree(coords[mask])
        type_ids[ct] = np.where(mask)[0]
    
    # Compute pairwise median and mean distances
    median_dist = np.zeros((n_types, n_types))
    mean_dist = np.zeros((n_types, n_types))
    
    print(f"  Computing proximity for {n_types} types...")
    
    for i, src in enumerate(types):
        src_coords = coords[type_ids[src]]
        
        for j, tgt in enumerate(types):
            if src == tgt:
                # Self-distance: use k=2 (skip self)
                if len(type_ids[tgt]) < 2:
                    median_dist[i, j] = np.nan
                    mean_dist[i, j] = np.nan
                    continue
                dists, _ = trees[tgt].query(src_coords, k=2)
                nn_dists = dists[:, 1]  # second nearest = nearest other
            else:
                dists, _ = trees[tgt].query(src_coords, k=1)
                nn_dists = dists.flatten()
            
            median_dist[i, j] = np.median(nn_dists)
            mean_dist[i, j] = nn_dists.mean()
    
    median_df = pd.DataFrame(median_dist, index=types, columns=types)
    mean_df = pd.DataFrame(mean_dist, index=types, columns=types)
    
    # Normalize by self-distance
    normalized_df = None
    if normalize:
        self_dists = np.diag(median_dist)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_matrix = median_dist / self_dists[:, np.newaxis]
            norm_matrix = np.nan_to_num(norm_matrix, nan=1.0)
        normalized_df = pd.DataFrame(norm_matrix, index=types, columns=types)
    
    # Report
    print(f"  ✓ Proximity scores: {n_types}×{n_types} matrix")
    if normalize:
        # Find closest pair (excluding self)
        norm_vals = normalized_df.values.copy()
        np.fill_diagonal(norm_vals, np.inf)
        min_idx = np.unravel_index(norm_vals.argmin(), norm_vals.shape)
        closest = (types[min_idx[0]], types[min_idx[1]])
        print(f"    Closest pair: {closest[0]}→{closest[1]} "
              f"(norm={normalized_df.iloc[min_idx[0], min_idx[1]]:.2f})")
    
    return {
        'median_distance': median_df,
        'mean_distance': mean_df,
        'normalized': normalized_df,
    }


def permutation_test(
    sj: 'spatioloji',
    cell_type_col: str,
    metric_fn,
    n_permutations: int = 999,
    seed: Optional[int] = None,
    alternative: str = 'two-sided',
    **metric_kwargs,
) -> dict:
    """
    Generic permutation test for any spatial distance metric.
    
    Shuffles cell type labels while keeping coordinates fixed,
    recomputes the metric each time, and tests significance.
    
    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    cell_type_col : str
        Column with cell type labels (will be shuffled).
    metric_fn : callable
        Function that computes the metric. Signature:
        metric_fn(sj, cell_type_col, **kwargs) -> float
    n_permutations : int
        Number of label shuffles.
    seed : int, optional
        Random seed.
    alternative : str
        'two-sided', 'greater', or 'less'.
    **metric_kwargs
        Additional arguments passed to metric_fn.
    
    Returns
    -------
    dict with keys:
        'observed'     : float, observed metric value
        'null_dist'    : np.ndarray, permuted metric values
        'expected'     : float, mean of null distribution
        'zscore'       : float
        'pvalue'       : float
        'alternative'  : str
    
    Examples
    --------
    >>> # Test if median Tumor→Tcell distance is shorter than expected
    >>> def median_tumor_tcell(sj, col):
    ...     result = cross_type_distances(
    ...         sj, col, 'Tumor', 'T cell', store=False)
    ...     return result['distance'].median()
    >>> 
    >>> test = permutation_test(
    ...     my_sj, 'cell_type', median_tumor_tcell,
    ...     n_permutations=999, alternative='less')
    """
    rng = np.random.default_rng(seed)
    
    if cell_type_col not in sj.cell_meta.columns:
        raise ValueError(f"'{cell_type_col}' not found in cell_meta")
    
    # Store original labels
    original_labels = sj.cell_meta[cell_type_col].values.copy()
    
    # Observed metric
    observed = metric_fn(sj, cell_type_col, **metric_kwargs)
    
    # Permutations
    print(f"  Running {n_permutations} permutations...")
    null_dist = np.empty(n_permutations)
    
    for i in range(n_permutations):
        shuffled = rng.permutation(original_labels)
        sj.cell_meta[cell_type_col] = shuffled
        null_dist[i] = metric_fn(sj, cell_type_col, **metric_kwargs)
    
    # Restore original labels
    sj.cell_meta[cell_type_col] = original_labels
    
    # Statistics
    null_mean = null_dist.mean()
    null_std = null_dist.std()
    
    zscore = (observed - null_mean) / null_std if null_std > 0 else 0.0
    
    # P-value
    if alternative == 'two-sided':
        extreme = np.sum(
            np.abs(null_dist - null_mean) >= np.abs(observed - null_mean)
        )
    elif alternative == 'greater':
        extreme = np.sum(null_dist >= observed)
    elif alternative == 'less':
        extreme = np.sum(null_dist <= observed)
    else:
        raise ValueError(
            f"Unknown alternative: {alternative}. "
            "Use 'two-sided', 'greater', or 'less'."
        )
    
    pvalue = (extreme + 1) / (n_permutations + 1)
    
    # Report
    sig = "significant" if pvalue < 0.05 else "not significant"
    print(f"  ✓ Permutation test ({alternative}):")
    print(f"    observed={observed:.4f}, expected={null_mean:.4f}")
    print(f"    z={zscore:.2f}, p={pvalue:.4f} ({sig})")
    
    return {
        'observed': observed,
        'null_dist': null_dist,
        'expected': null_mean,
        'zscore': zscore,
        'pvalue': pvalue,
        'alternative': alternative,
    }

