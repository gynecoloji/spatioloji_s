# src/spatioloji_s/spatial/shared/utils.py

"""
utils.py - Shared utilities for spatial analysis

Common functions used by both point-based and polygon-based methods.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    import networkx as nx

from typing import Optional, Tuple, Dict, Union
import numpy as np
import pandas as pd
import warnings


def validate_spatial_data(sp: 'spatioloji',
                         require_polygons: bool = False,
                         coord_type: str = 'global') -> Dict[str, bool]:
    """
    Validate that spatioloji has required spatial data.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object to validate
    require_polygons : bool, default=False
        If True, check for polygon data
    coord_type : str, default='global'
        Coordinate type to validate ('local' or 'global')
    
    Returns
    -------
    dict
        Validation results with keys:
        - has_coords: bool
        - coords_valid: bool
        - has_polygons: bool (if require_polygons=True)
        - has_fov: bool
        - overall: bool (all checks passed)
    
    Examples
    --------
    >>> import spatioloji as sj
    >>> 
    >>> # Check if ready for point-based analysis
    >>> valid = sj.spatial.shared.validate_spatial_data(sp)
    >>> if not valid['overall']:
    ...     print("Missing required spatial data")
    >>> 
    >>> # Check if ready for polygon-based analysis
    >>> valid = sj.spatial.shared.validate_spatial_data(sp, require_polygons=True)
    """
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Validating Spatial Data")
    print(f"{'='*70}")
    
    # Check coordinates exist
    results['has_coords'] = sp.spatial is not None
    print(f"Has coordinates: {results['has_coords']}")
    
    if results['has_coords']:
        # Check coordinates are valid (not all NaN)
        coords = sp.get_spatial_coords(coord_type=coord_type)
        results['coords_valid'] = not np.all(np.isnan(coords))
        n_valid = np.sum(~np.isnan(coords).any(axis=1))
        print(f"Valid coordinates: {results['coords_valid']} ({n_valid:,} / {len(coords):,} cells)")
    else:
        results['coords_valid'] = False
    
    # Check FOV information
    fov_col = sp.config.fov_id_col
    results['has_fov'] = fov_col in sp.cell_meta.columns
    if results['has_fov']:
        n_fovs = sp.cell_meta[fov_col].nunique()
        print(f"Has FOV info: True ({n_fovs} FOVs)")
    else:
        print(f"Has FOV info: False")
    
    # Check polygons if required
    if require_polygons:
        results['has_polygons'] = sp.polygons is not None
        if results['has_polygons']:
            n_cells_with_poly = sp.polygons[sp.config.cell_id_col].nunique()
            print(f"Has polygons: True ({n_cells_with_poly:,} cells)")
        else:
            print(f"Has polygons: False")
    
    # Overall validation
    if require_polygons:
        results['overall'] = all([
            results['has_coords'],
            results['coords_valid'],
            results['has_polygons']
        ])
    else:
        results['overall'] = all([
            results['has_coords'],
            results['coords_valid']
        ])
    
    print(f"\n{'='*70}")
    if results['overall']:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")
    print(f"{'='*70}\n")
    
    return results


def get_centroid_coords(sp: 'spatioloji',
                       coord_type: str = 'global',
                       from_polygons: bool = False) -> np.ndarray:
    """
    Get cell centroid coordinates.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    coord_type : str, default='global'
        'local' or 'global'
    from_polygons : bool, default=False
        If True, calculate centroids from polygon vertices.
        If False, use stored spatial coordinates.
    
    Returns
    -------
    np.ndarray
        Centroid coordinates (n_cells × 2)
    
    Examples
    --------
    >>> # Get stored centroids
    >>> centroids = sj.spatial.shared.get_centroid_coords(sp)
    >>> 
    >>> # Calculate from polygons
    >>> centroids = sj.spatial.shared.get_centroid_coords(sp, from_polygons=True)
    """
    if from_polygons:
        if sp.polygons is None:
            raise ValueError("No polygon data available")
        
        # Calculate centroids from polygon vertices
        cell_id_col = sp.config.cell_id_col
        x_col, y_col = sp.config.get_coordinate_columns(coord_type)
        
        centroids = sp.polygons.groupby(cell_id_col).agg({
            x_col: 'mean',
            y_col: 'mean'
        })
        
        # Align to master cell index
        centroids = centroids.reindex(sp.cell_index)
        coords = centroids.values
    else:
        # Use stored coordinates
        coords = sp.get_spatial_coords(coord_type=coord_type)
    
    return coords


def convert_graph_to_networkx(graph,
                              add_distances: bool = True) -> 'nx.Graph':
    """
    Convert SpatialGraph to NetworkX graph.
    
    Useful for leveraging NetworkX algorithms and visualization.
    
    Parameters
    ----------
    graph : SpatialGraph
        Spatial graph to convert
    add_distances : bool, default=True
        If True, add distances as edge weights
    
    Returns
    -------
    networkx.Graph
        NetworkX graph
    
    Examples
    --------
    >>> import spatioloji as sj
    >>> import networkx as nx
    >>> 
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> G = sj.spatial.shared.convert_graph_to_networkx(graph)
    >>> 
    >>> # Use NetworkX algorithms
    >>> centrality = nx.betweenness_centrality(G)
    >>> communities = nx.community.louvain_communities(G)
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX required. Install with: pip install networkx")
    
    print(f"\nConverting to NetworkX graph...")
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(graph.cell_index)
    
    # Add edges
    rows, cols = graph.adjacency.nonzero()
    
    if add_distances and graph.distances is not None:
        # Add edges with weights
        for i, j in zip(rows, cols):
            if i < j:  # Only add once for undirected graph
                cell_i = graph.cell_index[i]
                cell_j = graph.cell_index[j]
                dist = graph.distances[i, j]
                G.add_edge(cell_i, cell_j, weight=dist)
    else:
        # Add edges without weights
        for i, j in zip(rows, cols):
            if i < j:
                cell_i = graph.cell_index[i]
                cell_j = graph.cell_index[j]
                G.add_edge(cell_i, cell_j)
    
    print(f"✓ Created NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def compute_graph_metrics(graph) -> pd.DataFrame:
    """
    Compute various graph metrics for each cell.
    
    Calculates:
    - Degree (number of neighbors)
    - Average neighbor degree
    - Clustering coefficient (requires NetworkX)
    
    Parameters
    ----------
    graph : SpatialGraph
        Spatial graph
    
    Returns
    -------
    pd.DataFrame
        Metrics for each cell
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> metrics = sj.spatial.shared.compute_graph_metrics(graph)
    >>> sp.cell_meta['degree'] = metrics['degree']
    >>> sp.cell_meta['clustering'] = metrics['clustering']
    """
    print(f"\nComputing graph metrics...")
    
    # Basic metrics
    degrees = graph.get_degree()
    
    metrics = pd.DataFrame({
        'degree': degrees,
    }, index=graph.cell_index)
    
    # Average neighbor degree
    avg_neighbor_degree = np.zeros(len(graph.cell_index))
    
    for i, cell_id in enumerate(graph.cell_index):
        neighbor_indices = graph.get_neighbors(cell_id)
        if len(neighbor_indices) > 0:
            neighbor_degrees = degrees[neighbor_indices]
            avg_neighbor_degree[i] = neighbor_degrees.mean()
    
    metrics['avg_neighbor_degree'] = avg_neighbor_degree
    
    # Try to compute clustering coefficient
    try:
        import networkx as nx
        G = convert_graph_to_networkx(graph, add_distances=False)
        clustering = nx.clustering(G)
        metrics['clustering'] = [clustering[cell_id] for cell_id in graph.cell_index]
        print(f"  ✓ Computed: degree, avg_neighbor_degree, clustering")
    except ImportError:
        print(f"  ✓ Computed: degree, avg_neighbor_degree")
        print(f"  ⚠ NetworkX not available for clustering coefficient")
    
    return metrics


def get_boundary_cells(sp: 'spatioloji',
                      coord_type: str = 'global',
                      method: str = 'convex_hull',
                      fov_id: Optional[str] = None) -> pd.Index:
    """
    Identify cells on the boundary of the tissue/FOV.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    coord_type : str, default='global'
        'local' or 'global'
    method : str, default='convex_hull'
        Method to identify boundary:
        - 'convex_hull': Cells on convex hull
        - 'percentile': Cells in outer percentile by distance from center
    fov_id : str, optional
        Specific FOV to analyze. If None, uses all cells.
    
    Returns
    -------
    pd.Index
        Cell IDs on boundary
    
    Examples
    --------
    >>> # Get boundary cells
    >>> boundary = sj.spatial.shared.get_boundary_cells(sp)
    >>> sp.cell_meta['is_boundary'] = sp.cell_index.isin(boundary)
    >>> 
    >>> # Boundary cells for specific FOV
    >>> boundary = sj.spatial.shared.get_boundary_cells(sp, fov_id='001')
    """
    from scipy.spatial import ConvexHull
    
    print(f"\nIdentifying boundary cells (method={method})...")
    
    # Get coordinates
    if fov_id is not None:
        cell_ids = sp.get_cells_in_fov(fov_id)
        if len(cell_ids) == 0:
            raise ValueError(f"No cells found for FOV '{fov_id}'")
        indices = sp._get_cell_indices(cell_ids)
        coords = sp.get_spatial_coords(coord_type=coord_type)[indices]
        cell_index = sp.cell_index[indices]
    else:
        coords = sp.get_spatial_coords(coord_type=coord_type)
        cell_index = sp.cell_index
    
    if method == 'convex_hull':
        # Compute convex hull
        hull = ConvexHull(coords)
        boundary_indices = hull.vertices
        boundary_cells = cell_index[boundary_indices]
        
        print(f"  ✓ Found {len(boundary_cells)} boundary cells (convex hull)")
    
    elif method == 'percentile':
        # Find cells in outer 10% by distance from center
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        threshold = np.percentile(distances, 90)
        boundary_mask = distances >= threshold
        boundary_cells = cell_index[boundary_mask]
        
        print(f"  ✓ Found {len(boundary_cells)} boundary cells (outer 10%)")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return boundary_cells


def compute_spatial_bins(sp: 'spatioloji',
                        n_bins: int = 10,
                        coord_type: str = 'global') -> pd.Series:
    """
    Divide spatial coordinates into grid bins.
    
    Useful for spatial aggregation and analysis.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    n_bins : int, default=10
        Number of bins per dimension
    coord_type : str, default='global'
        'local' or 'global'
    
    Returns
    -------
    pd.Series
        Bin ID for each cell (format: "x{i}_y{j}")
    
    Examples
    --------
    >>> # Create spatial bins
    >>> bins = sj.spatial.shared.compute_spatial_bins(sp, n_bins=10)
    >>> sp.cell_meta['spatial_bin'] = bins
    >>> 
    >>> # Count cells per bin
    >>> bin_counts = bins.value_counts()
    >>> 
    >>> # Average expression per bin
    >>> expr = sp.get_expression(gene_names='CD4', as_dataframe=True)
    >>> bin_avg = expr.groupby(bins).mean()
    """
    print(f"\nComputing spatial bins ({n_bins}×{n_bins})...")
    
    # Get coordinates
    coords = sp.get_spatial_coords(coord_type=coord_type)
    
    # Create bins
    x_bins = pd.cut(coords[:, 0], bins=n_bins, labels=False)
    y_bins = pd.cut(coords[:, 1], bins=n_bins, labels=False)
    
    # Create bin IDs
    bin_ids = [f"x{x}_y{y}" for x, y in zip(x_bins, y_bins)]
    bin_series = pd.Series(bin_ids, index=sp.cell_index, name='spatial_bin')
    
    n_occupied = bin_series.nunique()
    print(f"  ✓ Created {n_bins}×{n_bins} grid ({n_occupied} occupied bins)")
    
    return bin_series


def calculate_spatial_extent(sp: 'spatioloji',
                            coord_type: str = 'global',
                            fov_id: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate spatial extent (bounding box) of cells.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    coord_type : str, default='global'
        'local' or 'global'
    fov_id : str, optional
        Specific FOV. If None, uses all cells.
    
    Returns
    -------
    dict
        Bounding box with keys: xmin, xmax, ymin, ymax, width, height
    
    Examples
    --------
    >>> # Get spatial extent
    >>> extent = sj.spatial.shared.calculate_spatial_extent(sp)
    >>> print(f"Tissue size: {extent['width']} × {extent['height']}")
    >>> 
    >>> # Extent for specific FOV
    >>> extent = sj.spatial.shared.calculate_spatial_extent(sp, fov_id='001')
    """
    # Get cells
    if fov_id is not None:
        cell_ids = sp.get_cells_in_fov(fov_id)
        indices = sp._get_cell_indices(cell_ids)
        coords = sp.get_spatial_coords(coord_type=coord_type)[indices]
    else:
        coords = sp.get_spatial_coords(coord_type=coord_type)
    
    # Calculate extent
    extent = {
        'xmin': float(coords[:, 0].min()),
        'xmax': float(coords[:, 0].max()),
        'ymin': float(coords[:, 1].min()),
        'ymax': float(coords[:, 1].max()),
    }
    
    extent['width'] = extent['xmax'] - extent['xmin']
    extent['height'] = extent['ymax'] - extent['ymin']
    extent['area'] = extent['width'] * extent['height']
    
    return extent


def safe_divide(numerator: np.ndarray,
               denominator: np.ndarray,
               fill_value: float = 0.0) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator values
    denominator : np.ndarray
        Denominator values
    fill_value : float, default=0.0
        Value to use when denominator is zero
    
    Returns
    -------
    np.ndarray
        Result of division
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result[~np.isfinite(result)] = fill_value
    
    return result


def create_spatial_index(sp: 'spatioloji',
                        coord_type: str = 'global') -> 'KDTree':
    """
    Create KD-tree spatial index for fast queries.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    coord_type : str, default='global'
        'local' or 'global'
    
    Returns
    -------
    scipy.spatial.KDTree
        KD-tree for spatial queries
    
    Examples
    --------
    >>> from scipy.spatial import KDTree
    >>> 
    >>> # Create spatial index
    >>> tree = sj.spatial.shared.create_spatial_index(sp)
    >>> 
    >>> # Find 10 nearest neighbors to a point
    >>> point = [500, 500]
    >>> distances, indices = tree.query(point, k=10)
    >>> nearest_cells = sp.cell_index[indices]
    """
    from scipy.spatial import KDTree
    
    coords = sp.get_spatial_coords(coord_type=coord_type)
    tree = KDTree(coords)
    
    return tree


def normalize_coordinates(coords: np.ndarray,
                         method: str = 'minmax') -> np.ndarray:
    """
    Normalize spatial coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates to normalize (n_cells × 2)
    method : str, default='minmax'
        Normalization method:
        - 'minmax': Scale to [0, 1]
        - 'standard': Z-score normalization
        - 'center': Center at origin
    
    Returns
    -------
    np.ndarray
        Normalized coordinates
    
    Examples
    --------
    >>> coords = sp.get_spatial_coords()
    >>> coords_norm = sj.spatial.shared.normalize_coordinates(coords, method='minmax')
    """
    if method == 'minmax':
        # Scale to [0, 1]
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        normalized = (coords - min_vals) / (max_vals - min_vals)
    
    elif method == 'standard':
        # Z-score normalization
        mean = coords.mean(axis=0)
        std = coords.std(axis=0)
        normalized = (coords - mean) / std
    
    elif method == 'center':
        # Center at origin
        mean = coords.mean(axis=0)
        normalized = coords - mean
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return normalized

