"""
patterns.py - Spatial pattern detection with polygon topology

Detects large-scale spatial organization using polygon geometry
and polygon adjacency graphs:
- Local cell packing density from polygon areas
- Hotspot detection (Getis-Ord Gi*) on polygon graph
- Spatial autocorrelation (Moran's I, Geary's C) on polygon adjacency
- Colocalization of cell types via contact frequency

All spatial weights are derived from polygon adjacency, ensuring
analyses reflect true tissue topology.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd
from scipy import sparse

from .graph import PolygonSpatialGraph, _get_gdf


# ========== Density ==========

def cell_density_map(sp: 'spatioloji',
                     graph: PolygonSpatialGraph,
                     coord_type: str = 'global',
                     store: bool = True) -> pd.DataFrame:
    """
    Compute local cell packing density from polygon geometry.
    
    Two complementary density measures:
    - cell_density: 1 / polygon_area (smaller cell = more packed)
    - local_density: mean density of cell + its polygon neighbors
      (smoothed neighborhood density)
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    graph : PolygonSpatialGraph
        Pre-built polygon graph
    coord_type : str
        'global' or 'local' coordinates
    store : bool
        If True, store in sp.cell_meta
    
    Returns
    -------
    pd.DataFrame
        Columns: area, cell_density, local_density
    
    Examples
    --------
    >>> graph = build_contact_graph(sp)
    >>> density = cell_density_map(sp, graph)
    >>> # High local_density = tightly packed region
    """
    print(f"\n[Patterns] Computing cell density map...")
    
    gdf = _get_gdf(sp, coord_type=coord_type)
    
    # Per-cell density = 1 / area
    areas = gdf.geometry.area
    cell_density = pd.Series(0.0, index=gdf.index, dtype=np.float64)
    nonzero = areas > 0
    cell_density[nonzero] = 1.0 / areas[nonzero]
    
    # Reindex to graph cell_index
    cell_density = cell_density.reindex(graph.cell_index, fill_value=0.0)
    
    # Local density = mean density of self + neighbors
    adj = graph.adjacency
    n_cells = len(graph.cell_index)
    local_density = np.zeros(n_cells, dtype=np.float64)
    
    density_values = cell_density.values
    
    for i in range(n_cells):
        neighbor_indices = adj[i].nonzero()[1]
        # Include self
        all_indices = np.append(neighbor_indices, i)
        local_density[i] = density_values[all_indices].mean()
    
    # Reindex areas to graph cell_index
    areas = areas.reindex(graph.cell_index, fill_value=np.nan)
    
    # Build result
    result = pd.DataFrame({
        'area': areas.values,
        'cell_density': cell_density.values,
        'local_density': local_density
    }, index=graph.cell_index)
    
    # Reindex to full cell_index
    result = result.reindex(sp.cell_index)
    
    if store:
        sp.cell_meta['cell_density'] = result['cell_density'].values
        sp.cell_meta['local_density'] = result['local_density'].values
        print(f"  ✓ Stored 'cell_density' and 'local_density' in cell_meta")
    
    print(f"  ✓ Density: mean={result['local_density'].mean():.4f}, "
          f"max={result['local_density'].max():.4f}")
    
    return result


# ========== Hotspot Detection ==========

def hotspot_detection(sp: 'spatioloji',
                      graph: PolygonSpatialGraph,
                      metric: str,
                      store: bool = True) -> pd.DataFrame:
    """
    Detect spatial hotspots using Getis-Ord Gi* statistic.
    
    Identifies clusters of cells with significantly high or low
    values of a given metric, using the polygon adjacency graph
    as spatial weights.
    
    Gi* > 0: hot spot (high values cluster together)
    Gi* < 0: cold spot (low values cluster together)
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : PolygonSpatialGraph
        Pre-built polygon graph
    metric : str
        Column in cell_meta to analyze
    store : bool
        If True, store Gi* z-scores and p-values in sp.cell_meta
    
    Returns
    -------
    pd.DataFrame
        Columns: gi_star, z_score, p_value
        Indexed by cell_index
    
    Examples
    --------
    >>> graph = build_contact_graph(sp)
    >>> hotspots = hotspot_detection(sp, graph, metric='total_counts')
    >>> # Find significant hot spots
    >>> hot_mask = (hotspots['p_value'] < 0.05) & (hotspots['z_score'] > 0)
    >>> hot_cells = sp.cell_index[hot_mask]
    """
    from scipy import stats as scipy_stats
    
    print(f"\n[Patterns] Hotspot detection (Getis-Ord Gi*) for '{metric}'...")
    
    if metric not in sp.cell_meta.columns:
        raise ValueError(f"'{metric}' not found in cell_meta")
    
    # Get values aligned to graph
    values = sp.cell_meta[metric].reindex(graph.cell_index).values.astype(np.float64)
    
    # Handle NaN: replace with global mean
    nan_mask = np.isnan(values)
    if nan_mask.any():
        values[nan_mask] = np.nanmean(values)
        print(f"  ⚠ {nan_mask.sum()} NaN values replaced with mean")
    
    n = len(values)
    x_mean = values.mean()
    x_var = values.var()
    
    if x_var == 0:
        print("  ⚠ Zero variance in metric, cannot compute Gi*")
        result = pd.DataFrame({
            'gi_star': np.zeros(n),
            'z_score': np.zeros(n),
            'p_value': np.ones(n)
        }, index=graph.cell_index)
        return result.reindex(sp.cell_index)
    
    S = np.sqrt(x_var)
    
    # Compute Gi* for each cell (includes self in neighborhood)
    adj = graph.adjacency
    gi_star = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        neighbor_indices = adj[i].nonzero()[1]
        # Gi* includes self
        w_indices = np.append(neighbor_indices, i)
        w_count = len(w_indices)
        
        # Numerator: sum of values in neighborhood - expected
        local_sum = values[w_indices].sum()
        expected = w_count * x_mean
        
        # Denominator
        denom = S * np.sqrt(
            (n * w_count - w_count ** 2) / (n - 1)
        )
        
        if denom > 0:
            gi_star[i] = (local_sum - expected) / denom
    
    # P-values from standard normal
    p_values = 2.0 * (1.0 - scipy_stats.norm.cdf(np.abs(gi_star)))
    
    result = pd.DataFrame({
        'gi_star': gi_star,
        'z_score': gi_star,  # Gi* is already a z-score
        'p_value': p_values
    }, index=graph.cell_index)
    
    # Reindex to full cell_index
    result = result.reindex(sp.cell_index)
    
    if store:
        sp.cell_meta[f'gi_star_{metric}'] = result['gi_star'].values
        sp.cell_meta[f'gi_star_{metric}_pval'] = result['p_value'].values
        print(f"  ✓ Stored 'gi_star_{metric}' and p-values in cell_meta")
    
    n_hot = ((result['p_value'] < 0.05) & (result['z_score'] > 0)).sum()
    n_cold = ((result['p_value'] < 0.05) & (result['z_score'] < 0)).sum()
    print(f"  ✓ Hot spots (p<0.05): {n_hot}")
    print(f"    Cold spots (p<0.05): {n_cold}")
    
    return result


# ========== Spatial Autocorrelation ==========

def spatial_autocorrelation(sp: 'spatioloji',
                            graph: PolygonSpatialGraph,
                            metric: str,
                            method: str = 'moran',
                            store: bool = True) -> Dict[str, float]:
    """
    Compute global spatial autocorrelation on polygon adjacency graph.
    
    Tests whether a metric is spatially clustered (positive autocorrelation),
    dispersed (negative), or random.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : PolygonSpatialGraph
        Pre-built polygon graph
    metric : str
        Column in cell_meta to analyze
    method : str
        'moran': Moran's I (-1 to 1, positive = clustered)
        'geary': Geary's C (0 to ~2, <1 = clustered, >1 = dispersed)
    store : bool
        If True, store global statistic in sp.cell_meta attrs
    
    Returns
    -------
    dict
        Keys: statistic, expected, variance, z_score, p_value, method
    
    Examples
    --------
    >>> graph = build_contact_graph(sp)
    >>> result = spatial_autocorrelation(sp, graph, metric='total_counts')
    >>> print(f"Moran's I = {result['statistic']:.3f}, p = {result['p_value']:.2e}")
    """
    from scipy import stats as scipy_stats
    
    print(f"\n[Patterns] Spatial autocorrelation ({method}) for '{metric}'...")
    
    if metric not in sp.cell_meta.columns:
        raise ValueError(f"'{metric}' not found in cell_meta")
    
    if method not in ('moran', 'geary'):
        raise ValueError(f"method must be 'moran' or 'geary', got '{method}'")
    
    # Get values aligned to graph
    values = sp.cell_meta[metric].reindex(graph.cell_index).values.astype(np.float64)
    
    # Handle NaN
    nan_mask = np.isnan(values)
    if nan_mask.any():
        values[nan_mask] = np.nanmean(values)
        print(f"  ⚠ {nan_mask.sum()} NaN values replaced with mean")
    
    n = len(values)
    x_mean = values.mean()
    x_dev = values - x_mean
    
    # Build weight matrix from adjacency (binary weights)
    adj = graph.adjacency.astype(np.float64).toarray()
    
    # Total weight
    S0 = adj.sum()
    
    if S0 == 0:
        print("  ⚠ No edges in graph, cannot compute autocorrelation")
        return {
            'statistic': np.nan, 'expected': np.nan,
            'variance': np.nan, 'z_score': np.nan,
            'p_value': np.nan, 'method': method
        }
    
    if method == 'moran':
        # Moran's I = (n / S0) * (x_dev' W x_dev) / (x_dev' x_dev)
        numerator = x_dev @ adj @ x_dev
        denominator = x_dev @ x_dev
        
        if denominator == 0:
            print("  ⚠ Zero variance in metric")
            I = 0.0
        else:
            I = (n / S0) * (numerator / denominator)
        
        # Expected value under null
        E_I = -1.0 / (n - 1)
        
        # Variance under normality assumption
        S1 = 0.5 * ((adj + adj.T) ** 2).sum()
        S2 = ((adj.sum(axis=0) + adj.sum(axis=1)) ** 2).sum()
        
        k = (np.sum(x_dev ** 4) / n) / ((np.sum(x_dev ** 2) / n) ** 2)
        
        var_I_num = (
            n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
            - k * (n * (n - 1) * S1 - 2 * n * S2 + 6 * S0 ** 2)
        )
        var_I_denom = (n - 1) * (n - 2) * (n - 3) * S0 ** 2
        
        if var_I_denom > 0:
            V_I = (var_I_num / var_I_denom) - E_I ** 2
        else:
            V_I = 0.0
        
        statistic = I
        expected = E_I
        variance = max(V_I, 1e-20)
        
    else:  # geary
        # Geary's C = ((n-1)/(2*S0)) * sum w_ij(x_i - x_j)^2 / sum(x_i - x_mean)^2
        rows, cols = graph.adjacency.nonzero()
        sq_diffs = (values[rows] - values[cols]) ** 2
        weights = np.array([adj[r, c] for r, c in zip(rows, cols)])
        
        numerator = (weights * sq_diffs).sum()
        denominator = x_dev @ x_dev
        
        if denominator == 0:
            C = 1.0
        else:
            C = ((n - 1) / (2 * S0)) * (numerator / denominator)
        
        # Expected
        E_C = 1.0
        
        # Variance (simplified normal approximation)
        S1 = 0.5 * ((adj + adj.T) ** 2).sum()
        S2 = ((adj.sum(axis=0) + adj.sum(axis=1)) ** 2).sum()
        
        var_C_num = (
            (2 * S1 + S2) * (n - 1) - 4 * S0 ** 2
        )
        var_C_denom = 2 * (n + 1) * S0 ** 2
        
        if var_C_denom > 0:
            V_C = var_C_num / var_C_denom
        else:
            V_C = 0.0
        
        statistic = C
        expected = E_C
        variance = max(V_C, 1e-20)
    
    # Z-score and p-value
    z_score = (statistic - expected) / np.sqrt(variance)
    p_value = 2.0 * (1.0 - scipy_stats.norm.cdf(np.abs(z_score)))
    
    result = {
        'statistic': float(statistic),
        'expected': float(expected),
        'variance': float(variance),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'method': method
    }
    
    # Report
    stat_name = "Moran's I" if method == 'moran' else "Geary's C"
    print(f"  ✓ {stat_name} = {statistic:.4f}")
    print(f"    Expected = {expected:.4f}")
    print(f"    Z-score = {z_score:.2f}, p = {p_value:.2e}")
    
    if method == 'moran':
        if z_score > 1.96:
            print(f"    → Significant positive autocorrelation (clustered)")
        elif z_score < -1.96:
            print(f"    → Significant negative autocorrelation (dispersed)")
        else:
            print(f"    → Not significant (random)")
    else:
        if z_score < -1.96:
            print(f"    → Significant positive autocorrelation (clustered)")
        elif z_score > 1.96:
            print(f"    → Significant negative autocorrelation (dispersed)")
        else:
            print(f"    → Not significant (random)")
    
    return result


# ========== Colocalization ==========

def colocalization(sp: 'spatioloji',
                   graph: PolygonSpatialGraph,
                   type_a: str,
                   type_b: str,
                   group_col: str,
                   n_permutations: int = 1000,
                   seed: int = 42) -> Dict[str, float]:
    """
    Test if two cell types are spatially colocalized using contact frequency.
    
    Counts A-B contacts in the polygon graph and compares to
    null distribution from label permutations.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : PolygonSpatialGraph
        Pre-built polygon graph
    type_a : str
        First cell type
    type_b : str
        Second cell type
    group_col : str
        Column in cell_meta defining cell types
    n_permutations : int
        Number of permutations
    seed : int
        Random seed
    
    Returns
    -------
    dict
        Keys: observed, expected, z_score, p_value, fold_change, interpretation
        interpretation: 'colocalized', 'segregated', or 'random'
    
    Examples
    --------
    >>> graph = build_contact_graph(sp)
    >>> result = colocalization(
    ...     sp, graph, 'T_cell', 'Tumor', group_col='cell_type'
    ... )
    >>> print(f"Fold change: {result['fold_change']:.2f}")
    >>> print(f"p-value: {result['p_value']:.3f}")
    """
    print(f"\n[Patterns] Colocalization test: {type_a} – {type_b}...")
    
    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"'{group_col}' not found in cell_meta")
    
    labels = sp.cell_meta[group_col].reindex(graph.cell_index)
    
    # Validate types exist
    unique_types = labels.dropna().unique()
    if type_a not in unique_types:
        raise ValueError(f"'{type_a}' not found in '{group_col}'")
    if type_b not in unique_types:
        raise ValueError(f"'{type_b}' not found in '{group_col}'")
    
    # Get edges (upper triangle)
    rows, cols = graph.adjacency.nonzero()
    mask = rows < cols
    edge_rows = rows[mask]
    edge_cols = cols[mask]
    
    label_values = labels.values
    
    # Count A-B contacts
    def _count_ab(lab):
        count = 0
        for r, c in zip(edge_rows, edge_cols):
            lr, lc = lab[r], lab[c]
            if pd.isna(lr) or pd.isna(lc):
                continue
            if (lr == type_a and lc == type_b) or (lr == type_b and lc == type_a):
                count += 1
        return count
    
    observed = _count_ab(label_values)
    
    # Permutation null
    print(f"  → Running {n_permutations} permutations...")
    rng = np.random.default_rng(seed)
    perm_counts = np.zeros(n_permutations, dtype=np.float64)
    
    valid_indices = np.where(pd.notna(label_values))[0]
    
    for p in range(n_permutations):
        shuffled = label_values.copy()
        shuffled[valid_indices] = rng.permutation(shuffled[valid_indices])
        perm_counts[p] = _count_ab(shuffled)
    
    # Statistics
    expected = perm_counts.mean()
    perm_std = perm_counts.std()
    
    if perm_std > 0:
        z_score = (observed - expected) / perm_std
    else:
        z_score = 0.0
    
    # Two-sided p-value
    n_extreme = np.sum(
        np.abs(perm_counts - expected) >= np.abs(observed - expected)
    )
    p_value = (n_extreme + 1) / (n_permutations + 1)
    
    # Fold change
    fold_change = observed / expected if expected > 0 else np.inf
    
    # Interpretation
    if p_value < 0.05:
        if z_score > 0:
            interpretation = 'colocalized'
        else:
            interpretation = 'segregated'
    else:
        interpretation = 'random'
    
    result = {
        'observed': int(observed),
        'expected': float(expected),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'fold_change': float(fold_change),
        'interpretation': interpretation
    }
    
    print(f"  ✓ Observed: {observed}, Expected: {expected:.1f}")
    print(f"    Fold change: {fold_change:.2f}")
    print(f"    Z-score: {z_score:+.2f}, p = {p_value:.3f}")
    print(f"    → {interpretation}")
    
    return result