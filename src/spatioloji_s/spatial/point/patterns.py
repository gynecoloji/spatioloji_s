# src/spatioloji_s/spatial/point/patterns.py

"""
patterns.py - Spatial pattern detection and domain identification

Provides methods for:
- Identifying spatial domains/niches
- Finding spatially variable genes
- Co-localization analysis
- Spatial gradient detection
- Hotspot/coldspot identification

All functions use global coordinates. For FOV-specific analysis, subset first:
    sp_fov = sp.subset_by_fovs(['fov1'])
    graph = sj.spatial.point.build_knn_graph(sp_fov, k=10)
    domains = sj.spatial.point.identify_spatial_domains(sp_fov, graph)
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    from .graph import SpatialGraph

from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import warnings


def identify_spatial_domains(sp: 'spatioloji',
                            graph: 'SpatialGraph',
                            method: str = 'leiden',
                            n_domains: Optional[int] = None,
                            resolution: float = 1.0,
                            features: Optional[List[str]] = None,
                            use_expression: bool = True,
                            n_pcs: int = 20) -> pd.Series:
    """
    Identify spatial domains (spatially coherent regions).
    
    Combines spatial proximity with feature similarity to find
    biologically meaningful spatial domains.
    
    Uses global coordinates. For FOV-specific analysis, subset first:
        sp_fov = sp.subset_by_fovs(['fov1'])
        graph = sj.spatial.point.build_knn_graph(sp_fov, k=10)
        domains = sj.spatial.point.identify_spatial_domains(sp_fov, graph)
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph
    method : str, default='leiden'
        Clustering method:
        - 'leiden': Leiden clustering (requires leidenalg)
        - 'louvain': Louvain clustering (requires louvain)
        - 'kmeans': K-means on spatial coordinates + features
        - 'hierarchical': Hierarchical clustering
        - 'dbscan': DBSCAN clustering
    n_domains : int, optional
        Number of domains (for kmeans, hierarchical)
    resolution : float, default=1.0
        Resolution parameter (for leiden, louvain)
    features : list, optional
        Features from cell_meta to use. If None and use_expression=False,
        uses spatial coordinates only.
    use_expression : bool, default=True
        If True, include gene expression in clustering
    n_pcs : int, default=20
        Number of PCs to use if use_expression=True
    
    Returns
    -------
    pd.Series
        Domain assignments for each cell
    
    Examples
    --------
    >>> import spatioloji as sj
    >>> 
    >>> # Build spatial graph
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # Identify domains using expression + spatial info
    >>> domains = sj.spatial.point.identify_spatial_domains(
    ...     sp, graph, method='leiden', resolution=1.0
    ... )
    >>> sp.cell_meta['spatial_domain'] = domains
    >>> 
    >>> # FOV-specific analysis
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> graph = sj.spatial.point.build_knn_graph(sp_fov, k=10)
    >>> domains = sj.spatial.point.identify_spatial_domains(
    ...     sp_fov, graph, method='kmeans', n_domains=10
    ... )
    """
    print(f"\n{'='*70}")
    print(f"Identifying Spatial Domains")
    print(f"{'='*70}")
    print(f"Method: {method}")
    print(f"Use expression: {use_expression}")
    
    # Build feature matrix
    feature_list = []
    
    # Add spatial coordinates (normalized) - ALWAYS GLOBAL
    coords = sp.get_spatial_coords(coord_type='global')
    coords_norm = (coords - coords.mean(axis=0)) / coords.std(axis=0)
    feature_list.append(coords_norm)
    print(f"  Added spatial coordinates (2 features, global)")
    
    # Add expression features
    if use_expression:
        # Use PCA if available, otherwise compute
        if 'X_pca' in sp.embeddings:
            pca_coords = sp.embeddings['X_pca'][:, :n_pcs]
            print(f"  Using existing PCA ({n_pcs} PCs)")
        else:
            print(f"  Computing PCA...")
            from sklearn.decomposition import PCA
            
            expr = sp.expression.get_dense()
            pca = PCA(n_components=n_pcs)
            pca_coords = pca.fit_transform(expr)
            print(f"  Computed PCA ({n_pcs} PCs, {pca.explained_variance_ratio_[:5].sum():.1%} variance)")
        
        feature_list.append(pca_coords)
    
    # Add custom features
    if features is not None:
        for feat in features:
            if feat not in sp.cell_meta.columns:
                warnings.warn(f"Feature '{feat}' not found in cell_meta")
                continue
            
            feat_values = sp.cell_meta[feat].values
            
            # Handle categorical
            if sp.cell_meta[feat].dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                feat_values = le.fit_transform(feat_values)
            
            feat_values = feat_values.reshape(-1, 1)
            feat_values = (feat_values - feat_values.mean()) / (feat_values.std() + 1e-10)
            feature_list.append(feat_values)
        
        print(f"  Added {len(features)} custom features")
    
    # Combine features
    X = np.hstack(feature_list)
    print(f"\nFeature matrix: {X.shape}")
    
    # Clustering
    print(f"\nClustering with {method}...")
    
    if method == 'leiden':
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError(
                "Leiden requires python-igraph and leidenalg. "
                "Install with: pip install python-igraph leidenalg"
            )
        
        # Convert to igraph
        print(f"  Converting graph to igraph...")
        edges = graph.to_edge_list()
        g = ig.Graph()
        g.add_vertices(len(sp.cell_index))
        g.add_edges([(sp._cell_id_to_idx[s], sp._cell_id_to_idx[t]) 
                     for s, t in zip(edges['source'], edges['target'])])
        
        # Add weights based on feature similarity
        print(f"  Computing feature-based weights...")
        weights = []
        for s, t in zip(edges['source'], edges['target']):
            idx_s = sp._cell_id_to_idx[s]
            idx_t = sp._cell_id_to_idx[t]
            sim = 1.0 / (1.0 + np.linalg.norm(X[idx_s] - X[idx_t]))
            weights.append(sim)
        
        # Leiden clustering
        print(f"  Running Leiden (resolution={resolution})...")
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights=weights, resolution_parameter=resolution
        )
        
        domains = np.array(partition.membership)
        n_found = len(set(domains))
        print(f"  Found {n_found} domains")
    
    elif method == 'louvain':
        try:
            import igraph as ig
            import louvain
        except ImportError:
            raise ImportError(
                "Louvain requires python-igraph and louvain-igraph. "
                "Install with: pip install python-igraph louvain-igraph"
            )
        
        # Convert to igraph
        edges = graph.to_edge_list()
        g = ig.Graph()
        g.add_vertices(len(sp.cell_index))
        g.add_edges([(sp._cell_id_to_idx[s], sp._cell_id_to_idx[t]) 
                     for s, t in zip(edges['source'], edges['target'])])
        
        # Add weights
        weights = []
        for s, t in zip(edges['source'], edges['target']):
            idx_s = sp._cell_id_to_idx[s]
            idx_t = sp._cell_id_to_idx[t]
            sim = 1.0 / (1.0 + np.linalg.norm(X[idx_s] - X[idx_t]))
            weights.append(sim)
        
        # Louvain clustering
        partition = louvain.find_partition(
            g, louvain.RBConfigurationVertexPartition,
            weights=weights, resolution_parameter=resolution
        )
        
        domains = np.array(partition.membership)
        n_found = len(set(domains))
        print(f"  Found {n_found} domains")
    
    elif method == 'kmeans':
        if n_domains is None:
            raise ValueError("n_domains required for kmeans")
        
        kmeans = KMeans(n_clusters=n_domains, random_state=42)
        domains = kmeans.fit_predict(X)
        print(f"  Created {n_domains} domains")
    
    elif method == 'hierarchical':
        if n_domains is None:
            raise ValueError("n_domains required for hierarchical")
        
        clustering = AgglomerativeClustering(n_clusters=n_domains, linkage='ward')
        domains = clustering.fit_predict(X)
        print(f"  Created {n_domains} domains")
    
    elif method == 'dbscan':
        # Estimate eps from graph
        if graph.distances is not None:
            edges = graph.to_edge_list()
            eps = edges['distance'].quantile(0.5)
        else:
            eps = 1.0
        
        print(f"  Using eps={eps:.2f}")
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        domains = dbscan.fit_predict(X)
        
        n_domains = len(set(domains)) - (1 if -1 in domains else 0)
        n_noise = (domains == -1).sum()
        print(f"  Found {n_domains} domains, {n_noise} noise points")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to series
    domain_series = pd.Series(domains, index=sp.cell_index, name='spatial_domain')
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Domain Summary:")
    print(domain_series.value_counts().sort_index())
    print(f"{'='*70}\n")
    
    return domain_series


def find_spatially_variable_genes(sp: 'spatioloji',
                                  graph: 'SpatialGraph',
                                  n_genes: Optional[int] = None,
                                  method: str = 'moran',
                                  layer: Optional[str] = None,
                                  permutations: int = 99,
                                  fdr_threshold: float = 0.05) -> pd.DataFrame:
    """
    Identify genes with spatial expression patterns.
    
    Tests all (or top variable) genes for spatial autocorrelation to
    find genes with non-random spatial patterns.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph
    n_genes : int, optional
        Test top N most variable genes. If None, tests all genes.
    method : str, default='moran'
        'moran' or 'geary'
    layer : str, optional
        Expression layer to use
    permutations : int, default=99
        Permutations for significance testing
    fdr_threshold : float, default=0.05
        FDR threshold for significance
    
    Returns
    -------
    pd.DataFrame
        Spatially variable genes with columns:
        - statistic: Moran's I or Geary's C
        - p_value: P-value from permutation test
        - q_value: FDR-corrected p-value
        - significant: Boolean (q < fdr_threshold)
        - variance: Gene variance
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # Find top 500 most variable genes with spatial patterns
    >>> svg = sj.spatial.point.find_spatially_variable_genes(
    ...     sp, graph, n_genes=500, permutations=99
    ... )
    >>> 
    >>> # Get significant spatially variable genes
    >>> sig_svg = svg[svg['significant']].sort_values('statistic', ascending=False)
    >>> print(f"Found {len(sig_svg)} spatially variable genes")
    >>> 
    >>> # Top spatially variable genes
    >>> top_svg = sig_svg.head(20).index.tolist()
    """
    from .statistics import test_spatial_autocorrelation_genes
    from statsmodels.stats.multitest import multipletests
    
    print(f"\n{'='*70}")
    print(f"Finding Spatially Variable Genes")
    print(f"{'='*70}")
    
    # Get expression data
    if layer is not None:
        expr = sp.get_layer(layer)
    else:
        expr = sp.expression.get_dense()
    
    # Select genes
    if n_genes is not None:
        # Get top variable genes
        variances = expr.var(axis=0)
        top_indices = np.argsort(variances)[-n_genes:]
        gene_names = sp.gene_index[top_indices].tolist()
        print(f"Testing top {n_genes} most variable genes")
    else:
        gene_names = sp.gene_index.tolist()
        print(f"Testing all {len(gene_names)} genes")
    
    # Test for spatial autocorrelation
    results = test_spatial_autocorrelation_genes(
        sp, graph, gene_names=gene_names, layer=layer,
        permutations=permutations, method=method
    )
    
    # Add variance
    gene_indices = sp._get_gene_indices(results.index.tolist())
    results['variance'] = expr[:, gene_indices].var(axis=0)
    
    # FDR correction
    print(f"\nApplying FDR correction...")
    _, q_values, _, _ = multipletests(
        results['p_value'].values,
        method='fdr_bh'
    )
    results['q_value'] = q_values
    results['significant'] = q_values < fdr_threshold
    
    # Sort by statistic
    results = results.sort_values('statistic', ascending=(method == 'geary'))
    
    # Summary
    n_sig = results['significant'].sum()
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Tested genes: {len(results)}")
    print(f"  Significant (FDR < {fdr_threshold}): {n_sig} ({100*n_sig/len(results):.1f}%)")
    print(f"\nTop 10 spatially variable genes:")
    print(results.head(10)[['statistic', 'p_value', 'q_value']])
    print(f"{'='*70}\n")
    
    return results


def detect_colocalization(sp: 'spatioloji',
                         graph: 'SpatialGraph',
                         group1: str,
                         group2: Optional[str] = None,
                         groupby: str = 'cell_type',
                         method: str = 'neighborhood',
                         permutations: int = 999) -> Dict:
    """
    Test for spatial co-localization between cell groups.
    
    Tests whether two groups of cells are spatially associated
    (co-localized) or segregated. Uses global coordinates.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph
    group1 : str
        First group (value in groupby column)
    group2 : str, optional
        Second group. If None, tests group1 vs all others.
    groupby : str, default='cell_type'
        Column in cell_meta containing groups
    method : str, default='neighborhood'
        Method to test:
        - 'neighborhood': Neighborhood enrichment test
        - 'distance': Distance-based test
    permutations : int, default=999
        Permutations for significance testing
    
    Returns
    -------
    dict
        Results with keys:
        - observed: Observed co-localization metric
        - expected: Expected under null
        - enrichment: Observed / Expected
        - p_value: P-value from permutation test
        - interpretation: 'co-localized', 'segregated', or 'random'
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # Test if T cells and tumor cells co-localize
    >>> result = sj.spatial.point.detect_colocalization(
    ...     sp, graph, group1='T_cell', group2='Tumor', groupby='cell_type'
    ... )
    >>> print(f"Enrichment: {result['enrichment']:.2f}, p={result['p_value']:.3e}")
    >>> 
    >>> # Test all pairwise co-localizations
    >>> cell_types = sp.cell_meta['cell_type'].unique()
    >>> for ct1 in cell_types:
    ...     for ct2 in cell_types:
    ...         if ct1 < ct2:  # Avoid duplicates
    ...             result = sj.spatial.point.detect_colocalization(
    ...                 sp, graph, ct1, ct2
    ...             )
    ...             if result['p_value'] < 0.05:
    ...                 print(f"{ct1} - {ct2}: {result['interpretation']}")
    """
    print(f"\n{'='*70}")
    print(f"Testing Co-localization")
    print(f"{'='*70}")
    
    if groupby not in sp.cell_meta.columns:
        raise ValueError(f"Column '{groupby}' not found in cell_meta")
    
    print(f"Group 1: {group1}")
    print(f"Group 2: {group2 if group2 else 'all others'}")
    print(f"Method: {method}")
    
    # Get group assignments
    groups = sp.cell_meta[groupby].astype(str)
    
    # Define group masks
    mask1 = groups == group1
    if group2 is not None:
        mask2 = groups == group2
    else:
        mask2 = groups != group1
    
    n1 = mask1.sum()
    n2 = mask2.sum()
    
    print(f"\nGroup sizes:")
    print(f"  {group1}: {n1}")
    print(f"  {group2 if group2 else 'others'}: {n2}")
    
    if method == 'neighborhood':
        # Count group2 neighbors of group1 cells
        observed = 0
        total_neighbors = 0
        
        for cell_id in sp.cell_index[mask1]:
            neighbors = graph.get_neighbor_ids(cell_id)
            neighbor_groups = groups.loc[neighbors]
            
            if group2 is not None:
                observed += (neighbor_groups == group2).sum()
            else:
                observed += (neighbor_groups != group1).sum()
            
            total_neighbors += len(neighbors)
        
        # Expected under random distribution
        expected = total_neighbors * (n2 / len(groups))
        
        # Enrichment
        enrichment = observed / expected if expected > 0 else 0
        
        print(f"\nNeighborhood enrichment:")
        print(f"  Observed neighbors: {observed}")
        print(f"  Expected neighbors: {expected:.1f}")
        print(f"  Enrichment: {enrichment:.3f}")
        
        # Permutation test
        print(f"\nPermutation test ({permutations} permutations)...")
        
        enrichments_perm = np.zeros(permutations)
        
        for perm in range(permutations):
            if perm % 200 == 0 and perm > 0:
                print(f"  Permutation {perm}/{permutations}...")
            
            # Permute group labels
            groups_perm = np.random.permutation(groups.values)
            
            observed_perm = 0
            for i, cell_id in enumerate(sp.cell_index[mask1]):
                neighbors = graph.get_neighbor_ids(cell_id)
                neighbor_indices = sp._get_cell_indices(neighbors)
                neighbor_groups_perm = groups_perm[neighbor_indices]
                
                if group2 is not None:
                    observed_perm += (neighbor_groups_perm == group2).sum()
                else:
                    observed_perm += (neighbor_groups_perm != group1).sum()
            
            enrichments_perm[perm] = observed_perm / expected if expected > 0 else 0
        
        # P-value
        if enrichment > 1:
            p_value = (np.sum(enrichments_perm >= enrichment) + 1) / (permutations + 1)
            interpretation = 'co-localized' if p_value < 0.05 else 'random'
        else:
            p_value = (np.sum(enrichments_perm <= enrichment) + 1) / (permutations + 1)
            interpretation = 'segregated' if p_value < 0.05 else 'random'
    
    elif method == 'distance':
        # Average distance from group1 to nearest group2 cell
        # ALWAYS USES GLOBAL COORDINATES
        from .graph import compute_pairwise_distances
        
        cells1 = sp.cell_index[mask1].tolist()
        cells2 = sp.cell_index[mask2].tolist()
        
        distances = compute_pairwise_distances(sp, cells1, cells2)
        observed = distances.min(axis=1).mean()
        
        # Expected: average of all pairwise distances
        all_distances = compute_pairwise_distances(sp, cells1, sp.cell_index.tolist())
        expected = all_distances.min(axis=1).mean()
        
        enrichment = expected / observed if observed > 0 else 0
        
        print(f"\nDistance-based test:")
        print(f"  Mean min distance: {observed:.2f}")
        print(f"  Expected distance: {expected:.2f}")
        print(f"  Enrichment: {enrichment:.3f}")
        
        # Permutation test
        print(f"\nPermutation test ({permutations} permutations)...")
        
        distances_perm = np.zeros(permutations)
        
        for perm in range(permutations):
            if perm % 50 == 0 and perm > 0:
                print(f"  Permutation {perm}/{permutations}...")
            
            # Permute labels
            groups_perm = np.random.permutation(groups.values)
            mask2_perm = groups_perm != group1 if group2 is None else groups_perm == group2
            
            cells2_perm = sp.cell_index[mask2_perm].tolist()
            
            if len(cells2_perm) > 0:
                distances_perm_mat = compute_pairwise_distances(sp, cells1, cells2_perm)
                distances_perm[perm] = distances_perm_mat.min(axis=1).mean()
            else:
                distances_perm[perm] = np.inf
        
        # P-value
        if observed < expected:
            p_value = (np.sum(distances_perm <= observed) + 1) / (permutations + 1)
            interpretation = 'co-localized' if p_value < 0.05 else 'random'
        else:
            p_value = (np.sum(distances_perm >= observed) + 1) / (permutations + 1)
            interpretation = 'segregated' if p_value < 0.05 else 'random'
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    results = {
        'observed': float(observed),
        'expected': float(expected),
        'enrichment': float(enrichment),
        'p_value': float(p_value),
        'interpretation': interpretation,
        'method': method,
        'n_group1': int(n1),
        'n_group2': int(n2)
    }
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Enrichment: {enrichment:.3f}")
    print(f"  P-value: {p_value:.3e}")
    print(f"  Interpretation: {interpretation.upper()}")
    print(f"{'='*70}\n")
    
    return results


def detect_spatial_gradients(sp: 'spatioloji',
                            values: Union[str, pd.Series, np.ndarray],
                            method: str = 'linear') -> Dict:
    """
    Detect spatial gradients in expression or features.
    
    Tests whether values show a directional trend across space
    (e.g., expression increases from left to right).
    Uses global coordinates.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    values : str, pd.Series, or np.ndarray
        Values to test for gradients
    method : str, default='linear'
        'linear' or 'polynomial'
    
    Returns
    -------
    dict
        Results with keys:
        - coef_x: Coefficient for x-direction
        - coef_y: Coefficient for y-direction
        - r2_x: R² for x-direction
        - r2_y: R² for y-direction
        - p_value_x: P-value for x
        - p_value_y: P-value for y
        - gradient_direction: Angle in degrees (0° = right, 90° = up)
        - gradient_magnitude: Magnitude of gradient
    
    Examples
    --------
    >>> # Test gene for spatial gradient
    >>> expr = sp.get_expression(gene_names='CD4', as_dataframe=True)['CD4']
    >>> result = sj.spatial.point.detect_spatial_gradients(sp, expr)
    >>> 
    >>> if result['p_value_x'] < 0.05 or result['p_value_y'] < 0.05:
    ...     print(f"Gradient detected: {result['gradient_direction']:.1f}°")
    >>> 
    >>> # Test all genes
    >>> for gene in sp.gene_index[:100]:
    ...     expr = sp.get_expression(gene_names=gene, as_dataframe=True)[gene]
    ...     result = sj.spatial.point.detect_spatial_gradients(sp, expr)
    ...     if result['p_value_x'] < 0.01:
    ...         print(f"{gene}: gradient in x-direction")
    """
    print(f"\n{'='*70}")
    print(f"Detecting Spatial Gradients")
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
    print(f"Method: {method}")
    
    # Get coordinates - ALWAYS GLOBAL
    coords = sp.get_spatial_coords(coord_type='global')
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Remove NaN
    valid = ~np.isnan(values_array)
    x = x[valid]
    y = y[valid]
    values_array = values_array[valid]
    
    # Fit linear models
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # X-direction
    model_x = LinearRegression()
    model_x.fit(x.reshape(-1, 1), values_array)
    pred_x = model_x.predict(x.reshape(-1, 1))
    
    coef_x = model_x.coef_[0]
    r2_x = r2_score(values_array, pred_x)
    
    # Test significance
    n = len(values_array)
    se_x = np.sqrt(np.sum((values_array - pred_x)**2) / (n - 2)) / np.sqrt(np.sum((x - x.mean())**2))
    t_x = coef_x / se_x
    p_value_x = 2 * (1 - stats.t.cdf(abs(t_x), n - 2))
    
    # Y-direction
    model_y = LinearRegression()
    model_y.fit(y.reshape(-1, 1), values_array)
    pred_y = model_y.predict(y.reshape(-1, 1))
    
    coef_y = model_y.coef_[0]
    r2_y = r2_score(values_array, pred_y)
    
    se_y = np.sqrt(np.sum((values_array - pred_y)**2) / (n - 2)) / np.sqrt(np.sum((y - y.mean())**2))
    t_y = coef_y / se_y
    p_value_y = 2 * (1 - stats.t.cdf(abs(t_y), n - 2))
    
    # Gradient direction and magnitude
    gradient_direction = np.arctan2(coef_y, coef_x) * 180 / np.pi
    if gradient_direction < 0:
        gradient_direction += 360
    
    gradient_magnitude = np.sqrt(coef_x**2 + coef_y**2)
    
    results = {
        'coef_x': float(coef_x),
        'coef_y': float(coef_y),
        'r2_x': float(r2_x),
        'r2_y': float(r2_y),
        'p_value_x': float(p_value_x),
        'p_value_y': float(p_value_y),
        'gradient_direction': float(gradient_direction),
        'gradient_magnitude': float(gradient_magnitude),
        'variable': values_name
    }
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  X-direction: coef={coef_x:.4f}, R²={r2_x:.3f}, p={p_value_x:.3e}")
    print(f"  Y-direction: coef={coef_y:.4f}, R²={r2_y:.3f}, p={p_value_y:.3e}")
    print(f"  Gradient direction: {gradient_direction:.1f}°")
    print(f"  Gradient magnitude: {gradient_magnitude:.4f}")
    
    if p_value_x < 0.05 or p_value_y < 0.05:
        print(f"  → Significant spatial gradient detected")
    else:
        print(f"  → No significant gradient")
    
    print(f"{'='*70}\n")
    
    return results


def identify_hotspots(sp: 'spatioloji',
                     graph: 'SpatialGraph',
                     values: Union[str, pd.Series, np.ndarray],
                     method: str = 'getis_ord',
                     fdr_threshold: float = 0.05) -> pd.DataFrame:
    """
    Identify hotspots and coldspots (local clusters of high/low values).
    
    Uses Getis-Ord Gi* statistic to identify statistically significant
    spatial clusters of high or low values.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph
    values : str, pd.Series, or np.ndarray
        Values to analyze
    method : str, default='getis_ord'
        'getis_ord' (Gi* statistic)
    fdr_threshold : float, default=0.05
        FDR threshold for significance
    
    Returns
    -------
    pd.DataFrame
        Results with columns:
        - Gi_star: Getis-Ord Gi* statistic
        - p_value: P-value
        - q_value: FDR-corrected p-value
        - cluster_type: 'hotspot', 'coldspot', or 'not significant'
        - significant: Boolean
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> expr = sp.get_expression(gene_names='CD4', as_dataframe=True)['CD4']
    >>> 
    >>> # Identify hotspots
    >>> hotspots = sj.spatial.point.identify_hotspots(sp, graph, expr)
    >>> 
    >>> # Get significant hotspots
    >>> sig_hotspots = hotspots[
    ...     (hotspots['cluster_type'] == 'hotspot') & 
    ...     (hotspots['significant'])
    ... ].index
    >>> 
    >>> sp.cell_meta['is_hotspot'] = sp.cell_index.isin(sig_hotspots)
    """
    from statsmodels.stats.multitest import multipletests
    
    print(f"\n{'='*70}")
    print(f"Identifying Hotspots/Coldspots")
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
    print(f"Method: {method}")
    
    # Handle NaN
    values_array = np.where(np.isnan(values_array), np.nanmean(values_array), values_array)
    
    n = len(values_array)
    
    # Get adjacency matrix
    W = graph.adjacency.toarray()
    
    # Compute Getis-Ord Gi*
    print(f"\nComputing Gi* for {n:,} cells...")
    
    Gi_star = np.zeros(n)
    p_values = np.zeros(n)
    
    x_mean = values_array.mean()
    x_std = values_array.std()
    
    for i in range(n):
        if i % 5000 == 0 and i > 0:
            print(f"  Cell {i:,}/{n:,}...")
        
        # Get neighbors (including self for Gi*)
        w_i = W[i, :]
        w_i[i] = 1.0  # Include self
        
        w_sum = w_i.sum()
        
        if w_sum == 0:
            continue
        
        # Compute Gi*
        numerator = np.sum(w_i * values_array) - x_mean * w_sum
        denominator = x_std * np.sqrt((n * np.sum(w_i**2) - w_sum**2) / (n - 1))
        
        if denominator > 0:
            Gi_star[i] = numerator / denominator
            
            # P-value (two-tailed)
            p_values[i] = 2 * (1 - stats.norm.cdf(abs(Gi_star[i])))
        else:
            p_values[i] = 1.0
    
    # FDR correction
    print(f"\nApplying FDR correction...")
    _, q_values, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Classify clusters
    cluster_types = np.array(['not significant'] * n, dtype=object)
    significant = q_values < fdr_threshold
    
    cluster_types[(Gi_star > 0) & significant] = 'hotspot'
    cluster_types[(Gi_star < 0) & significant] = 'coldspot'
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Gi_star': Gi_star,
        'p_value': p_values,
        'q_value': q_values,
        'cluster_type': cluster_types,
        'significant': significant
    }, index=sp.cell_index)
    
    # Summary
    n_hotspots = (cluster_types == 'hotspot').sum()
    n_coldspots = (cluster_types == 'coldspot').sum()
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Hotspots: {n_hotspots} ({100*n_hotspots/n:.1f}%)")
    print(f"  Coldspots: {n_coldspots} ({100*n_coldspots/n:.1f}%)")
    print(f"  Not significant: {n - n_hotspots - n_coldspots}")
    print(f"{'='*70}\n")
    
    return results