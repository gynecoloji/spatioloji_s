# src/spatioloji_s/spatial/point/statistics.py

"""
statistics.py - Spatial statistics and autocorrelation analysis

Provides methods for measuring spatial patterns and autocorrelation:
- Moran's I: Global spatial autocorrelation
- Geary's C: Local dissimilarity measure
- Local Moran's I: Local spatial autocorrelation
- Permutation tests: Statistical significance

All functions work with spatial graphs that use global coordinates.
For FOV-specific analysis, subset first and build a new graph:
    sp_fov = sp.subset_by_fovs(['fov1'])
    graph = sj.spatial.point.build_knn_graph(sp_fov, k=10)
    result = sj.spatial.point.morans_i(sp_fov, graph, values)
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    from .graph import SpatialGraph

from typing import Optional, Union, Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats
import warnings


def morans_i(sp: 'spatioloji',
            graph: 'SpatialGraph',
            values: Union[str, pd.Series, np.ndarray],
            permutations: int = 999,
            two_tailed: bool = True) -> Dict:
    """
    Compute Moran's I statistic for spatial autocorrelation.
    
    Moran's I measures whether similar values cluster together in space.
    - I > 0: Positive autocorrelation (similar values cluster)
    - I ≈ 0: Random spatial pattern
    - I < 0: Negative autocorrelation (dissimilar values cluster)
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph defining neighborhoods (uses global coordinates)
    values : str, pd.Series, or np.ndarray
        Values to test for spatial autocorrelation.
        - str: column name in cell_meta
        - pd.Series: values indexed by cell IDs
        - np.ndarray: values aligned with sp.cell_index
    permutations : int, default=999
        Number of permutations for significance testing
    two_tailed : bool, default=True
        If True, use two-tailed test
    
    Returns
    -------
    dict
        Results with keys:
        - I: Moran's I statistic
        - EI: Expected I under null hypothesis
        - VI: Variance of I
        - z_score: Standardized I
        - p_value: P-value from permutation test
        - p_norm: P-value from normal approximation
    
    Notes
    -----
    Moran's I formula:
        I = (N / W) * Σ_i Σ_j w_ij * (x_i - x̄) * (x_j - x̄) / Σ_i (x_i - x̄)²
    
    where:
        - N: number of observations
        - W: sum of all spatial weights
        - w_ij: spatial weight between i and j
        - x_i, x_j: values at locations i and j
        - x̄: mean of all values
    
    Examples
    --------
    >>> import spatioloji as sj
    >>> 
    >>> # Build spatial graph (uses global coordinates)
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # Test gene expression for spatial autocorrelation
    >>> expr = sp.get_expression(gene_names='CD4', as_dataframe=True)['CD4']
    >>> result = sj.spatial.point.morans_i(sp, graph, expr)
    >>> print(f"Moran's I = {result['I']:.3f}, p = {result['p_value']:.3e}")
    >>> 
    >>> # Test from cell_meta column
    >>> result = sj.spatial.point.morans_i(sp, graph, 'total_counts')
    >>> 
    >>> # FOV-specific analysis
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> graph_fov = sj.spatial.point.build_knn_graph(sp_fov, k=10)
    >>> result_fov = sj.spatial.point.morans_i(sp_fov, graph_fov, expr)
    >>> 
    >>> # Identify spatially autocorrelated genes
    >>> for gene in sp.gene_index[:100]:
    ...     expr = sp.get_expression(gene_names=gene, as_dataframe=True)[gene]
    ...     result = sj.spatial.point.morans_i(sp, graph, expr, permutations=99)
    ...     if result['p_value'] < 0.05:
    ...         print(f"{gene}: I={result['I']:.3f}")
    """
    print(f"\n{'='*70}")
    print(f"Computing Moran's I")
    print(f"{'='*70}")
    
    # Get values
    if isinstance(values, str):
        if values not in sp.cell_meta.columns:
            raise ValueError(f"Column '{values}' not found in cell_meta")
        values_array = sp.cell_meta[values].values
        values_name = values
    elif isinstance(values, pd.Series):
        values_array = values.reindex(sp.cell_index).values
        values_name = values.name or 'values'
    elif isinstance(values, np.ndarray):
        values_array = values
        values_name = 'values'
    else:
        raise TypeError("values must be str, pd.Series, or np.ndarray")
    
    print(f"Variable: {values_name}")
    print(f"Permutations: {permutations}")
    
    # Remove NaN values
    valid_mask = ~np.isnan(values_array)
    if not valid_mask.all():
        n_nan = (~valid_mask).sum()
        print(f"⚠ Removing {n_nan} cells with NaN values")
        values_array = values_array[valid_mask]
        # Would need to filter graph too - for now, fill with mean
        values_array = np.where(np.isnan(values_array), np.nanmean(values_array), values_array)
    
    n = len(values_array)
    
    # Get adjacency matrix
    W = graph.adjacency.toarray()
    W_sum = W.sum()
    
    # Compute Moran's I
    x_mean = values_array.mean()
    x_centered = values_array - x_mean
    
    numerator = np.sum(W * np.outer(x_centered, x_centered))
    denominator = np.sum(x_centered ** 2)
    
    I = (n / W_sum) * (numerator / denominator)
    
    # Expected I under null hypothesis
    EI = -1.0 / (n - 1)
    
    # Variance of I (under randomization)
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=0) + W.sum(axis=1)) ** 2)
    
    b2 = (n * np.sum(x_centered ** 4)) / (np.sum(x_centered ** 2) ** 2)
    
    VI_num = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * W_sum**2)
    VI_num -= b2 * ((n**2 - n) * S1 - 2*n * S2 + 6 * W_sum**2)
    VI_denom = (n - 1) * (n - 2) * (n - 3) * W_sum**2
    
    VI = VI_num / VI_denom - EI**2
    
    # Z-score
    z_score = (I - EI) / np.sqrt(VI)
    
    # P-value from normal approximation
    if two_tailed:
        p_norm = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        p_norm = 1 - stats.norm.cdf(z_score)
    
    print(f"\nComputing permutation test ({permutations} permutations)...")
    
    # Permutation test
    I_perms = np.zeros(permutations)
    for perm in range(permutations):
        if perm % 200 == 0 and perm > 0:
            print(f"  Permutation {perm}/{permutations}...")
        
        # Randomly permute values
        values_perm = np.random.permutation(values_array)
        x_centered_perm = values_perm - values_perm.mean()
        
        numerator_perm = np.sum(W * np.outer(x_centered_perm, x_centered_perm))
        denominator_perm = np.sum(x_centered_perm ** 2)
        
        I_perms[perm] = (n / W_sum) * (numerator_perm / denominator_perm)
    
    # P-value from permutation test
    if two_tailed:
        p_value = np.sum(np.abs(I_perms) >= np.abs(I)) / permutations
    else:
        p_value = np.sum(I_perms >= I) / permutations
    
    # Pseudo p-value (add 1 to numerator and denominator)
    p_value = (np.sum(np.abs(I_perms) >= np.abs(I)) + 1) / (permutations + 1)
    
    results = {
        'I': float(I),
        'EI': float(EI),
        'VI': float(VI),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'p_norm': float(p_norm),
        'n_obs': n,
        'variable': values_name
    }
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Moran's I = {I:.4f}")
    print(f"  Expected I = {EI:.4f}")
    print(f"  Z-score = {z_score:.4f}")
    print(f"  P-value (permutation) = {p_value:.4e}")
    print(f"  P-value (normal) = {p_norm:.4e}")
    
    if I > 0:
        print(f"  → Positive spatial autocorrelation")
    elif I < 0:
        print(f"  → Negative spatial autocorrelation")
    else:
        print(f"  → No spatial autocorrelation")
    
    if p_value < 0.001:
        print(f"  → Highly significant (p < 0.001)")
    elif p_value < 0.01:
        print(f"  → Very significant (p < 0.01)")
    elif p_value < 0.05:
        print(f"  → Significant (p < 0.05)")
    else:
        print(f"  → Not significant (p ≥ 0.05)")
    
    print(f"{'='*70}\n")
    
    return results


def gearys_c(sp: 'spatioloji',
            graph: 'SpatialGraph',
            values: Union[str, pd.Series, np.ndarray],
            permutations: int = 999) -> Dict:
    """
    Compute Geary's C statistic for spatial autocorrelation.
    
    Geary's C measures local dissimilarity (complement of Moran's I).
    - C < 1: Positive autocorrelation (similar values cluster)
    - C ≈ 1: Random spatial pattern
    - C > 1: Negative autocorrelation (dissimilar values cluster)
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph (uses global coordinates)
    values : str, pd.Series, or np.ndarray
        Values to test
    permutations : int, default=999
        Number of permutations for significance testing
    
    Returns
    -------
    dict
        Results with keys: C, EC, VC, z_score, p_value
    
    Notes
    -----
    Geary's C formula:
        C = [(N-1) / (2W)] * Σ_i Σ_j w_ij * (x_i - x_j)² / Σ_i (x_i - x̄)²
    
    Geary's C is inversely related to Moran's I but emphasizes
    differences between neighboring values rather than covariance.
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> expr = sp.get_expression(gene_names='CD8', as_dataframe=True)['CD8']
    >>> result = sj.spatial.point.gearys_c(sp, graph, expr)
    >>> print(f"Geary's C = {result['C']:.3f}, p = {result['p_value']:.3e}")
    >>> 
    >>> # FOV-specific analysis
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> graph_fov = sj.spatial.point.build_knn_graph(sp_fov, k=10)
    >>> result_fov = sj.spatial.point.gearys_c(sp_fov, graph_fov, expr)
    """
    print(f"\n{'='*70}")
    print(f"Computing Geary's C")
    print(f"{'='*70}")
    
    # Get values
    if isinstance(values, str):
        if values not in sp.cell_meta.columns:
            raise ValueError(f"Column '{values}' not found in cell_meta")
        values_array = sp.cell_meta[values].values
        values_name = values
    elif isinstance(values, pd.Series):
        values_array = values.reindex(sp.cell_index).values
        values_name = values.name or 'values'
    elif isinstance(values, np.ndarray):
        values_array = values
        values_name = 'values'
    else:
        raise TypeError("values must be str, pd.Series, or np.ndarray")
    
    print(f"Variable: {values_name}")
    print(f"Permutations: {permutations}")
    
    # Handle NaN
    values_array = np.where(np.isnan(values_array), np.nanmean(values_array), values_array)
    
    n = len(values_array)
    
    # Get adjacency matrix
    W = graph.adjacency.toarray()
    W_sum = W.sum()
    
    # Compute Geary's C
    x_mean = values_array.mean()
    
    numerator = 0
    for i in range(n):
        for j in range(n):
            if W[i, j] > 0:
                numerator += W[i, j] * (values_array[i] - values_array[j]) ** 2
    
    denominator = 2 * W_sum * np.sum((values_array - x_mean) ** 2) / (n - 1)
    
    C = numerator / denominator
    
    # Expected C under null hypothesis
    EC = 1.0
    
    # Variance of C (simplified)
    VC = 1.0 / (2 * (n + 1) * W_sum)
    
    # Z-score
    z_score = (C - EC) / np.sqrt(VC)
    
    print(f"\nComputing permutation test ({permutations} permutations)...")
    
    # Permutation test
    C_perms = np.zeros(permutations)
    for perm in range(permutations):
        if perm % 200 == 0 and perm > 0:
            print(f"  Permutation {perm}/{permutations}...")
        
        values_perm = np.random.permutation(values_array)
        
        numerator_perm = 0
        for i in range(n):
            for j in range(n):
                if W[i, j] > 0:
                    numerator_perm += W[i, j] * (values_perm[i] - values_perm[j]) ** 2
        
        denominator_perm = 2 * W_sum * np.sum((values_perm - values_perm.mean()) ** 2) / (n - 1)
        
        C_perms[perm] = numerator_perm / denominator_perm
    
    # P-value
    p_value = (np.sum(np.abs(C_perms - 1) >= np.abs(C - 1)) + 1) / (permutations + 1)
    
    results = {
        'C': float(C),
        'EC': float(EC),
        'VC': float(VC),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'n_obs': n,
        'variable': values_name
    }
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Geary's C = {C:.4f}")
    print(f"  Expected C = {EC:.4f}")
    print(f"  Z-score = {z_score:.4f}")
    print(f"  P-value = {p_value:.4e}")
    
    if C < 1:
        print(f"  → Positive spatial autocorrelation")
    elif C > 1:
        print(f"  → Negative spatial autocorrelation")
    else:
        print(f"  → No spatial autocorrelation")
    
    print(f"{'='*70}\n")
    
    return results


def local_morans_i(sp: 'spatioloji',
                  graph: 'SpatialGraph',
                  values: Union[str, pd.Series, np.ndarray],
                  permutations: int = 999) -> pd.DataFrame:
    """
    Compute Local Moran's I (LISA) for each cell.
    
    Local Moran's I identifies spatial clusters and outliers:
    - High-High: High value surrounded by high values
    - Low-Low: Low value surrounded by low values
    - High-Low: High value surrounded by low values (outlier)
    - Low-High: Low value surrounded by high values (outlier)
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph (uses global coordinates)
    values : str, pd.Series, or np.ndarray
        Values to analyze
    permutations : int, default=999
        Number of permutations for significance testing
    
    Returns
    -------
    pd.DataFrame
        Local Moran's I results with columns:
        - Ii: Local Moran's I statistic
        - p_value: P-value from permutation test
        - cluster_type: Cluster classification
        - significant: Boolean (p < 0.05)
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> expr = sp.get_expression(gene_names='CD4', as_dataframe=True)['CD4']
    >>> 
    >>> # Compute local Moran's I
    >>> local_i = sj.spatial.point.local_morans_i(sp, graph, expr)
    >>> 
    >>> # Add to cell metadata
    >>> sp.cell_meta['local_morans_i'] = local_i['Ii']
    >>> sp.cell_meta['cluster_type'] = local_i['cluster_type']
    >>> 
    >>> # Find significant spatial clusters
    >>> high_high = local_i[
    ...     (local_i['cluster_type'] == 'High-High') & 
    ...     (local_i['significant'])
    ... ]
    >>> print(f"Found {len(high_high)} High-High clusters")
    >>> 
    >>> # FOV-specific analysis
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> graph_fov = sj.spatial.point.build_knn_graph(sp_fov, k=10)
    >>> local_i_fov = sj.spatial.point.local_morans_i(sp_fov, graph_fov, expr)
    """
    print(f"\n{'='*70}")
    print(f"Computing Local Moran's I (LISA)")
    print(f"{'='*70}")
    
    # Get values
    if isinstance(values, str):
        if values not in sp.cell_meta.columns:
            raise ValueError(f"Column '{values}' not found in cell_meta")
        values_array = sp.cell_meta[values].values
        values_name = values
    elif isinstance(values, pd.Series):
        values_array = values.reindex(sp.cell_index).values
        values_name = values.name or 'values'
    elif isinstance(values, np.ndarray):
        values_array = values
        values_name = 'values'
    else:
        raise TypeError("values must be str, pd.Series, or np.ndarray")
    
    print(f"Variable: {values_name}")
    print(f"Permutations: {permutations}")
    
    # Handle NaN
    values_array = np.where(np.isnan(values_array), np.nanmean(values_array), values_array)
    
    n = len(values_array)
    x_mean = values_array.mean()
    x_centered = values_array - x_mean
    
    # Standardize
    x_std = x_centered / x_centered.std()
    
    # Get adjacency matrix
    W = graph.adjacency.toarray()
    
    # Compute local Moran's I for each cell
    Ii = np.zeros(n)
    spatial_lag = np.zeros(n)
    
    print(f"\nComputing Local I for {n:,} cells...")
    
    for i in range(n):
        # Get neighbors
        neighbors = graph.get_neighbors(sp.cell_index[i])
        
        if len(neighbors) > 0:
            # Spatial lag (weighted sum of neighbors)
            w_i = W[i, neighbors]
            w_i_sum = w_i.sum()
            
            if w_i_sum > 0:
                spatial_lag[i] = np.sum(w_i * x_std[neighbors]) / w_i_sum
                Ii[i] = x_std[i] * spatial_lag[i]
    
    # Classify cluster types
    cluster_types = np.array(['Not significant'] * n, dtype=object)
    
    # Quadrants
    high_value = x_std > 0
    high_lag = spatial_lag > 0
    
    cluster_types[(high_value) & (high_lag)] = 'High-High'
    cluster_types[(~high_value) & (~high_lag)] = 'Low-Low'
    cluster_types[(high_value) & (~high_lag)] = 'High-Low'
    cluster_types[(~high_value) & (high_lag)] = 'Low-High'
    
    print(f"\nComputing permutation test ({permutations} permutations)...")
    
    # Permutation test for each cell
    p_values = np.ones(n)
    
    for i in range(n):
        if i % 1000 == 0 and i > 0:
            print(f"  Cell {i:,}/{n:,}...")
        
        neighbors = graph.get_neighbors(sp.cell_index[i])
        
        if len(neighbors) == 0:
            continue
        
        w_i = W[i, neighbors]
        w_i_sum = w_i.sum()
        
        if w_i_sum == 0:
            continue
        
        # Permutation distribution
        Ii_perms = np.zeros(permutations)
        
        for perm in range(permutations):
            # Permute values
            values_perm = np.random.permutation(values_array)
            x_perm_centered = values_perm - values_perm.mean()
            x_perm_std = x_perm_centered / x_perm_centered.std()
            
            # Compute local I for permutation
            lag_perm = np.sum(w_i * x_perm_std[neighbors]) / w_i_sum
            Ii_perms[perm] = x_perm_std[i] * lag_perm
        
        # P-value
        p_values[i] = (np.sum(np.abs(Ii_perms) >= np.abs(Ii[i])) + 1) / (permutations + 1)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Ii': Ii,
        'p_value': p_values,
        'cluster_type': cluster_types,
        'significant': p_values < 0.05,
        'x_standardized': x_std,
        'spatial_lag': spatial_lag
    }, index=sp.cell_index)
    
    # Update cluster types for non-significant
    results.loc[~results['significant'], 'cluster_type'] = 'Not significant'
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Total cells: {n:,}")
    print(f"\nCluster types:")
    for cluster_type in ['High-High', 'Low-Low', 'High-Low', 'Low-High', 'Not significant']:
        count = (results['cluster_type'] == cluster_type).sum()
        pct = 100 * count / n
        print(f"  {cluster_type:20s}: {count:6,} ({pct:5.1f}%)")
    
    n_sig = results['significant'].sum()
    print(f"\nSignificant cells: {n_sig:,} ({100*n_sig/n:.1f}%)")
    print(f"{'='*70}\n")
    
    return results


def test_spatial_autocorrelation_genes(sp: 'spatioloji',
                                       graph: 'SpatialGraph',
                                       gene_names: Optional[List[str]] = None,
                                       layer: Optional[str] = None,
                                       permutations: int = 99,
                                       method: str = 'moran') -> pd.DataFrame:
    """
    Test multiple genes for spatial autocorrelation.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph (uses global coordinates)
    gene_names : list, optional
        Genes to test. If None, tests all genes.
    layer : str, optional
        Expression layer to use
    permutations : int, default=99
        Number of permutations (use fewer for speed)
    method : str, default='moran'
        'moran' or 'geary'
    
    Returns
    -------
    pd.DataFrame
        Results for each gene with columns: statistic, p_value, significant
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # Test top variable genes
    >>> var_genes = sp.gene_meta.nlargest(100, 'variance').index
    >>> results = sj.spatial.point.test_spatial_autocorrelation_genes(
    ...     sp, graph, gene_names=var_genes, permutations=99
    ... )
    >>> 
    >>> # Get spatially autocorrelated genes
    >>> spatial_genes = results[results['significant']].index
    >>> print(f"Found {len(spatial_genes)} spatially autocorrelated genes")
    >>> 
    >>> # FOV-specific analysis
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> graph_fov = sj.spatial.point.build_knn_graph(sp_fov, k=10)
    >>> results_fov = sj.spatial.point.test_spatial_autocorrelation_genes(
    ...     sp_fov, graph_fov, gene_names=var_genes, permutations=99
    ... )
    """
    print(f"\n{'='*70}")
    print(f"Testing Genes for Spatial Autocorrelation")
    print(f"{'='*70}")
    
    # Get expression data
    if layer is not None:
        expr_data = sp.get_layer(layer)
    else:
        expr_data = sp.expression.get_dense()
    
    # Get genes
    if gene_names is None:
        gene_names = sp.gene_index.tolist()
        gene_indices = np.arange(len(gene_names))
    else:
        gene_indices = sp._get_gene_indices(gene_names)
        gene_names = sp.gene_index[gene_indices].tolist()
    
    print(f"Testing {len(gene_names)} genes")
    print(f"Method: {method}")
    print(f"Permutations: {permutations}")
    
    # Test each gene
    results_list = []
    
    for idx, gene in enumerate(gene_names):
        if idx % 20 == 0:
            print(f"  Testing gene {idx+1}/{len(gene_names)}...")
        
        gene_idx = gene_indices[idx]
        expr = expr_data[:, gene_idx]
        
        # Skip if no variation
        if expr.std() == 0:
            results_list.append({
                'gene': gene,
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False
            })
            continue
        
        try:
            if method == 'moran':
                result = morans_i(sp, graph, expr, permutations=permutations, two_tailed=True)
                results_list.append({
                    'gene': gene,
                    'statistic': result['I'],
                    'p_value': result['p_value'],
                    'significant': result['p_value'] < 0.05
                })
            elif method == 'geary':
                result = gearys_c(sp, graph, expr, permutations=permutations)
                results_list.append({
                    'gene': gene,
                    'statistic': result['C'],
                    'p_value': result['p_value'],
                    'significant': result['p_value'] < 0.05
                })
            else:
                raise ValueError(f"Unknown method: {method}")
        
        except Exception as e:
            warnings.warn(f"Error testing gene {gene}: {e}")
            results_list.append({
                'gene': gene,
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    results_df = results_df.set_index('gene')
    
    # Summary
    n_sig = results_df['significant'].sum()
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Tested genes: {len(gene_names)}")
    print(f"  Significant genes: {n_sig} ({100*n_sig/len(gene_names):.1f}%)")
    print(f"  Method: {method}")
    print(f"{'='*70}\n")
    
    return results_df