"""
neighborhoods.py - Cell type neighborhood analysis

Analyzes the cellular composition of spatial neighborhoods,
identifying co-localization patterns and recurring microenvironments.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from typing import Optional, List, Union
import numpy as np
import pandas as pd
from scipy import sparse

from .graph import PointSpatialGraph


def neighborhood_composition(
    sj: 'spatioloji',
    graph: PointSpatialGraph,
    cell_type_col: str,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compute cell type composition of each cell's neighborhood.
    
    For each cell, counts (or proportions of) each cell type
    among its graph neighbors.
    
    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    graph : PointSpatialGraph
        Pre-built spatial graph.
    cell_type_col : str
        Column in sj.cell_meta containing cell type labels.
    normalize : bool
        If True, return proportions. If False, return raw counts.
    
    Returns
    -------
    pd.DataFrame
        (n_cells x n_types). Each row sums to 1 if normalized.
    """
    if cell_type_col not in sj.cell_meta.columns:
        raise ValueError(f"'{cell_type_col}' not found in cell_meta")
    
    labels = sj.cell_meta[cell_type_col].values
    categories = pd.Categorical(labels)
    types = categories.categories  # unique sorted types
    codes = categories.codes       # integer encoding
    
    # One-hot encode: (n_cells x n_types)
    n_cells = len(codes)
    n_types = len(types)
    onehot = sparse.csr_matrix(
        (np.ones(n_cells), (np.arange(n_cells), codes)),
        shape=(n_cells, n_types),
    )
    
    # Matrix multiply: adjacency @ onehot -> neighbor type counts
    counts = graph.adjacency.dot(onehot).toarray()  # (n_cells x n_types)
    
    result = pd.DataFrame(counts, index=graph.cell_ids, columns=types)
    
    if normalize:
        row_sums = result.sum(axis=1)
        # Avoid division by zero for isolated cells (degree=0)
        result = result.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    
    print(f"  ✓ Neighborhood composition: {n_cells} cells × {n_types} types"
          f" ({'proportions' if normalize else 'counts'})")
    
    return result

def neighborhood_enrichment(
    sj: 'spatioloji',
    graph: PointSpatialGraph,
    cell_type_col: str,
    n_permutations: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """
    Test whether cell type pairs co-locate more/less than expected.
    
    Uses permutation testing: shuffle cell type labels, recompute
    pairwise interaction counts, and compare observed vs. permuted.
    
    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    graph : PointSpatialGraph
        Pre-built spatial graph.
    cell_type_col : str
        Column in sj.cell_meta with cell type labels.
    n_permutations : int
        Number of label shuffles for null distribution.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    dict with keys:
        'zscore'   : pd.DataFrame (n_types x n_types), enrichment z-scores
        'observed' : pd.DataFrame, observed interaction counts
        'expected' : pd.DataFrame, mean expected counts from permutations
        'pvalue'   : pd.DataFrame, two-sided empirical p-values
    """
    if cell_type_col not in sj.cell_meta.columns:
        raise ValueError(f"'{cell_type_col}' not found in cell_meta")
    
    rng = np.random.default_rng(seed)
    
    labels = sj.cell_meta[cell_type_col].values
    categories = pd.Categorical(labels)
    types = categories.categories
    codes = categories.codes.copy()
    n_cells = len(codes)
    n_types = len(types)
    adj = graph.adjacency
    
    # --- Helper: compute type-type interaction counts from codes ---
    def _interaction_counts(codes_arr):
        onehot = sparse.csr_matrix(
            (np.ones(n_cells), (np.arange(n_cells), codes_arr)),
            shape=(n_cells, n_types),
        )
        # C.T @ A @ C -> (n_types x n_types) interaction matrix
        interactions = (onehot.T.dot(adj)).dot(onehot).toarray()
        return interactions
    
    # Observed
    observed = _interaction_counts(codes)
    
    # Permutations
    print(f"  Running {n_permutations} permutations...")
    perm_results = np.zeros((n_permutations, n_types, n_types))
    
    for i in range(n_permutations):
        shuffled = rng.permutation(codes)
        perm_results[i] = _interaction_counts(shuffled)
    
    # Statistics
    perm_mean = perm_results.mean(axis=0)
    perm_std = perm_results.std(axis=0)
    
    # Z-score: (observed - expected) / std
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (observed - perm_mean) / perm_std
        zscore = np.nan_to_num(zscore, nan=0.0)
    
    # Empirical p-value (two-sided): fraction of permutations
    # with value as extreme as observed
    pvalue = np.zeros((n_types, n_types))
    for i in range(n_types):
        for j in range(n_types):
            extreme = np.sum(
                np.abs(perm_results[:, i, j] - perm_mean[i, j])
                >= np.abs(observed[i, j] - perm_mean[i, j])
            )
            pvalue[i, j] = (extreme + 1) / (n_permutations + 1)
    
    # Wrap in DataFrames
    result = {
        'zscore': pd.DataFrame(zscore, index=types, columns=types),
        'observed': pd.DataFrame(observed, index=types, columns=types),
        'expected': pd.DataFrame(perm_mean, index=types, columns=types),
        'pvalue': pd.DataFrame(pvalue, index=types, columns=types),
    }
    
    # Report top enriched/depleted pairs
    z_flat = result['zscore'].values[np.triu_indices(n_types, k=1)]
    type_pairs = [
        (types[i], types[j])
        for i in range(n_types) for j in range(i + 1, n_types)
    ]
    top_enriched = sorted(zip(type_pairs, z_flat), key=lambda x: -x[1])[:3]
    top_depleted = sorted(zip(type_pairs, z_flat), key=lambda x: x[1])[:3]
    
    print(f"  ✓ Neighborhood enrichment: {n_types} types, "
          f"{n_permutations} permutations")
    print(f"    Top enriched:  {top_enriched[0][0]} (z={top_enriched[0][1]:.2f})")
    print(f"    Top depleted:  {top_depleted[0][0]} (z={top_depleted[0][1]:.2f})")
    
    return result

def identify_niches(
    sj: 'spatioloji',
    graph: PointSpatialGraph,
    cell_type_col: str,
    method: str = 'kmeans',
    n_niches: int = 5,
    resolution: float = 1.0,
    normalize: bool = True,
    label_col: str = 'spatial_niche',
    seed: Optional[int] = 42,
) -> dict:
    """
    Identify recurring spatial niches by clustering neighborhood profiles.
    
    Each cell's neighborhood composition (from the spatial graph) is 
    computed, then cells with similar compositions are grouped into niches.
    
    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    graph : PointSpatialGraph
        Pre-built spatial graph.
    cell_type_col : str
        Column in sj.cell_meta with cell type labels.
    method : str
        'kmeans' or 'leiden'.
    n_niches : int
        Number of niches (used by kmeans).
    resolution : float
        Resolution parameter (used by leiden). Higher = more niches.
    normalize : bool
        Whether to normalize composition to proportions.
    label_col : str
        Column name for storing niche labels in sj.cell_meta.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    dict with keys:
        'labels'     : pd.Series, niche label per cell
        'signatures' : pd.DataFrame, mean composition per niche
        'composition': pd.DataFrame, the full composition matrix used
    """
    # Step 1: Get neighborhood composition
    comp = neighborhood_composition(
        sj, graph, cell_type_col, normalize=normalize
    )
    
    # Step 2: Cluster
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        
        km = KMeans(n_clusters=n_niches, random_state=seed, n_init=10)
        labels = km.fit_predict(comp.values)
        labels = pd.Series(
            [f"niche_{i}" for i in labels],
            index=comp.index,
            name=label_col,
        )
        
    elif method == 'leiden':
        try:
            import leidenalg
            import igraph as ig
        except ImportError:
            raise ImportError(
                "Leiden requires leidenalg and igraph. "
                "Install with: pip install leidenalg igraph"
            )
        from sklearn.neighbors import NearestNeighbors
        
        # Build KNN graph in composition space (not spatial)
        k_comp = min(15, comp.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k_comp + 1, metric='cosine')
        nn.fit(comp.values)
        dist_matrix, idx_matrix = nn.kneighbors(comp.values)
        
        # Build igraph from KNN
        edges = []
        for i in range(comp.shape[0]):
            for j_idx in range(1, k_comp + 1):  # skip self
                neighbor = idx_matrix[i, j_idx]
                if i < neighbor:  # avoid duplicates
                    edges.append((i, neighbor))
        
        g = ig.Graph(n=comp.shape[0], edges=edges, directed=False)
        
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=seed if seed is not None else 0,
        )
        
        labels = pd.Series(
            [f"niche_{m}" for m in partition.membership],
            index=comp.index,
            name=label_col,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'leiden'.")
    
    # Step 3: Compute niche signatures (mean composition per niche)
    comp_with_niche = comp.copy()
    comp_with_niche[label_col] = labels.values
    signatures = comp_with_niche.groupby(label_col).mean()
    
    # Step 4: Store in cell_meta
    sj.cell_meta[label_col] = labels.values
    
    # Report
    niche_counts = labels.value_counts().sort_index()
    n_found = len(niche_counts)
    print(f"  ✓ Identified {n_found} niches (method={method}):")
    for niche, count in niche_counts.items():
        top_type = signatures.loc[niche].idxmax()
        top_pct = signatures.loc[niche].max() * 100
        print(f"    {niche}: {count:,} cells "
              f"(dominant: {top_type} {top_pct:.0f}%)")
    
    return {
        'labels': labels,
        'signatures': signatures,
        'composition': comp,
    }

def neighborhood_diversity(
    sj: 'spatioloji',
    graph: PointSpatialGraph,
    cell_type_col: str,
    metrics: Optional[List[str]] = None,
    store: bool = True,
) -> pd.DataFrame:
    """
    Compute diversity of each cell's neighborhood.
    
    Quantifies how mixed the cell types are around each cell.
    High diversity = many types intermingling. Low = homogeneous.
    
    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    graph : PointSpatialGraph
        Pre-built spatial graph.
    cell_type_col : str
        Column in sj.cell_meta with cell type labels.
    metrics : list of str, optional
        Which metrics to compute. Default: all three.
        Options: 'shannon', 'simpson', 'richness'.
    store : bool
        If True, add results to sj.cell_meta.
    
    Returns
    -------
    pd.DataFrame
        (n_cells x n_metrics). One diversity value per cell per metric.
    """
    if metrics is None:
        metrics = ['shannon', 'simpson', 'richness']
    
    valid = {'shannon', 'simpson', 'richness'}
    invalid = set(metrics) - valid
    if invalid:
        raise ValueError(f"Unknown metrics: {invalid}. Use {valid}")
    
    # Get proportions (normalized composition)
    comp = neighborhood_composition(
        sj, graph, cell_type_col, normalize=True
    )
    
    props = comp.values  # (n_cells, n_types)
    result = pd.DataFrame(index=graph.cell_ids)
    
    if 'shannon' in metrics:
        # H = -sum(p * log(p)), treating 0*log(0) = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            log_props = np.log(props)
            log_props[~np.isfinite(log_props)] = 0
        shannon = -np.sum(props * log_props, axis=1)
        result['shannon_entropy'] = shannon
    
    if 'simpson' in metrics:
        # D = 1 - sum(p^2)
        simpson = 1.0 - np.sum(props ** 2, axis=1)
        result['simpson_index'] = simpson
    
    if 'richness' in metrics:
        # Count of types with proportion > 0
        # Use raw counts to avoid floating point issues
        comp_counts = neighborhood_composition(
            sj, graph, cell_type_col, normalize=False
        )
        richness = (comp_counts.values > 0).sum(axis=1)
        result['richness'] = richness
    
    # Store in cell_meta
    if store:
        for col in result.columns:
            sj.cell_meta[col] = result[col].values
    
    # Report
    print(f"  ✓ Neighborhood diversity ({len(metrics)} metrics):")
    for col in result.columns:
        vals = result[col]
        print(f"    {col}: mean={vals.mean():.3f}, "
              f"range=[{vals.min():.3f}, {vals.max():.3f}]")
    
    return result