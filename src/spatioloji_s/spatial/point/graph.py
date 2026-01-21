# src/spatioloji_s/spatial/point/graph.py

"""
graph.py - Point-based spatial graph construction

Builds spatial graphs using cell centroid coordinates (centerX, centerY).
Always uses global coordinates for consistency.

For FOV-specific analysis, subset your spatioloji object first:
    sp_fov = sp.subset_by_fovs(['fov1'])
    graph = sj.spatial.point.build_knn_graph(sp_fov, k=10)

All methods use cell centroids from sp.get_spatial_coords(coord_type='global').
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from typing import Optional, Tuple, List, Dict, Union
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, Delaunay
from scipy.sparse import csr_matrix, lil_matrix
import warnings


class SpatialGraph:
    """
    Spatial graph for point-based neighborhood analysis.
    
    Stores adjacency matrix and provides methods for neighborhood queries.
    Uses cell indices aligned with spatioloji master index.
    Always uses global coordinates.
    
    Attributes
    ----------
    adjacency : scipy.sparse.csr_matrix
        Sparse adjacency matrix (cells × cells)
        adjacency[i,j] = 1 if cells i and j are neighbors
    distances : scipy.sparse.csr_matrix, optional
        Sparse distance matrix (same shape as adjacency)
        distances[i,j] = Euclidean distance between cells i and j
    cell_index : pd.Index
        Cell IDs (aligned with spatioloji master index)
    method : str
        Graph construction method used (e.g., 'knn_k10', 'radius_r50')
    n_cells : int
        Number of cells in graph
    
    Examples
    --------
    >>> import spatioloji as sj
    >>> 
    >>> # Build graph for all cells
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # For single FOV analysis
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> graph = sj.spatial.point.build_knn_graph(sp_fov, k=10)
    >>> 
    >>> # Get neighbors for a cell
    >>> neighbors = graph.get_neighbor_ids('cell_1')
    >>> 
    >>> # Get graph degree
    >>> degree = graph.get_degree('cell_1')
    """
    
    def __init__(self,
                 adjacency: csr_matrix,
                 cell_index: pd.Index,
                 method: str,
                 distances: Optional[csr_matrix] = None):
        """
        Initialize spatial graph.
        
        Parameters
        ----------
        adjacency : csr_matrix
            Adjacency matrix (cells × cells)
        cell_index : pd.Index
            Cell identifiers
        method : str
            Construction method
        distances : csr_matrix, optional
            Distance matrix
        """
        self.adjacency = adjacency
        self.distances = distances
        self.cell_index = cell_index
        self.method = method
        self.n_cells = len(cell_index)
        
        # Create index map for fast lookup
        self._cell_id_to_idx = {cid: idx for idx, cid in enumerate(cell_index)}
    
    def get_neighbors(self, cell_id: str) -> np.ndarray:
        """
        Get neighbor indices for a cell.
        
        Parameters
        ----------
        cell_id : str
            Cell identifier
        
        Returns
        -------
        np.ndarray
            Integer indices of neighboring cells
        """
        if cell_id not in self._cell_id_to_idx:
            return np.array([], dtype=np.int64)
        
        idx = self._cell_id_to_idx[cell_id]
        neighbors = self.adjacency[idx].nonzero()[1]
        return neighbors
    
    def get_neighbor_ids(self, cell_id: str) -> List[str]:
        """
        Get neighbor cell IDs for a cell.
        
        Parameters
        ----------
        cell_id : str
            Cell identifier
        
        Returns
        -------
        list
            Neighbor cell IDs
        """
        neighbor_indices = self.get_neighbors(cell_id)
        return [self.cell_index[i] for i in neighbor_indices]
    
    def get_neighbor_distances(self, cell_id: str) -> Optional[np.ndarray]:
        """
        Get distances to neighbors.
        
        Parameters
        ----------
        cell_id : str
            Cell identifier
        
        Returns
        -------
        np.ndarray or None
            Distances to neighbors (None if distances not stored)
        """
        if self.distances is None:
            return None
        
        if cell_id not in self._cell_id_to_idx:
            return None
        
        idx = self._cell_id_to_idx[cell_id]
        neighbor_indices = self.get_neighbors(cell_id)
        
        if len(neighbor_indices) == 0:
            return np.array([])
        
        return np.array([self.distances[idx, j] for j in neighbor_indices])
    
    def get_degree(self, cell_id: Optional[str] = None) -> Union[int, np.ndarray]:
        """
        Get degree (number of neighbors) for cell(s).
        
        Parameters
        ----------
        cell_id : str, optional
            Specific cell ID. If None, returns degrees for all cells.
        
        Returns
        -------
        int or np.ndarray
            Degree for one cell or array of degrees for all cells
        """
        if cell_id is not None:
            return len(self.get_neighbors(cell_id))
        else:
            return np.array(self.adjacency.sum(axis=1)).flatten()
    
    def to_edge_list(self) -> pd.DataFrame:
        """
        Convert graph to edge list DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Edge list with columns: source, target, distance (if available)
            
        Examples
        --------
        >>> edges = graph.to_edge_list()
        >>> print(edges.head())
        """
        rows, cols = self.adjacency.nonzero()
        
        edge_data = {
            'source': [self.cell_index[i] for i in rows],
            'target': [self.cell_index[i] for i in cols]
        }
        
        if self.distances is not None:
            edge_data['distance'] = self.distances[rows, cols].A1
        
        return pd.DataFrame(edge_data)
    
    def summary(self) -> Dict:
        """
        Get graph summary statistics.
        
        Returns
        -------
        dict
            Summary with keys: n_cells, n_edges, method,
            avg_degree, min_degree, max_degree, has_distances
        """
        degrees = self.get_degree()
        
        return {
            'n_cells': self.n_cells,
            'n_edges': self.adjacency.nnz,
            'method': self.method,
            'avg_degree': float(degrees.mean()),
            'min_degree': int(degrees.min()),
            'max_degree': int(degrees.max()),
            'has_distances': self.distances is not None
        }
    
    def __repr__(self) -> str:
        """String representation."""
        s = self.summary()
        return (f"SpatialGraph ({s['method']})\n"
                f"  Cells: {s['n_cells']:,}\n"
                f"  Edges: {s['n_edges']:,}\n"
                f"  Avg degree: {s['avg_degree']:.1f}")


def build_knn_graph(sp: 'spatioloji',
                    k: int = 10,
                    include_self: bool = False) -> SpatialGraph:
    """
    Build k-nearest neighbors spatial graph using cell centroids.
    
    Uses global coordinates. For FOV-specific analysis, subset first:
        sp_fov = sp.subset_by_fovs(['fov1'])
        graph = build_knn_graph(sp_fov, k=10)
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    k : int, default=10
        Number of nearest neighbors
    include_self : bool, default=False
        Whether to include self-connections
    
    Returns
    -------
    SpatialGraph
        Spatial graph object with adjacency and distance matrices
    
    Examples
    --------
    >>> import spatioloji as sj
    >>> 
    >>> # Build k-NN graph for all cells
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # Single FOV analysis
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> graph = sj.spatial.point.build_knn_graph(sp_fov, k=10)
    >>> 
    >>> # Multiple FOVs separately
    >>> for fov_id in sp.fov_index:
    ...     sp_fov = sp.subset_by_fovs([fov_id])
    ...     graph = sj.spatial.point.build_knn_graph(sp_fov, k=10)
    ...     # ... analyze ...
    """
    print(f"\n{'='*70}")
    print(f"Building k-NN Graph (Global Coordinates)")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  k = {k}")
    print(f"  include_self = {include_self}")
    
    # Get coordinates (always global)
    coords = sp.get_spatial_coords(coord_type='global')
    n_cells = len(sp.cell_index)
    
    print(f"\nBuilding KD-tree for {n_cells:,} cells...")
    
    # Build KD-tree for efficient neighbor search
    tree = KDTree(coords)
    
    # Query k+1 neighbors (includes self as first neighbor)
    print(f"Querying {k} nearest neighbors...")
    distances, indices = tree.query(coords, k=k+1)
    
    # Build sparse adjacency matrix
    adjacency = lil_matrix((n_cells, n_cells), dtype=np.float32)
    distance_matrix = lil_matrix((n_cells, n_cells), dtype=np.float32)
    
    for i in range(n_cells):
        # Skip first neighbor if not including self (it's the point itself)
        start_idx = 0 if include_self else 1
        
        for j, dist in zip(indices[i, start_idx:], distances[i, start_idx:]):
            adjacency[i, j] = 1.0
            distance_matrix[i, j] = dist
    
    # Convert to CSR for efficiency
    print(f"Converting to CSR format...")
    adjacency = adjacency.tocsr()
    distance_matrix = distance_matrix.tocsr()
    
    # Create graph
    graph = SpatialGraph(
        adjacency=adjacency,
        cell_index=sp.cell_index,
        method=f'knn_k{k}',
        distances=distance_matrix
    )
    
    print(f"\n{'='*70}")
    print(f"Graph Summary:")
    print(f"  Total edges: {adjacency.nnz:,}")
    print(f"  Avg degree: {graph.get_degree().mean():.2f}")
    print(f"  Min degree: {graph.get_degree().min()}")
    print(f"  Max degree: {graph.get_degree().max()}")
    print(f"{'='*70}\n")
    
    return graph


def build_radius_graph(sp: 'spatioloji',
                       radius: float,
                       include_self: bool = False) -> SpatialGraph:
    """
    Build radius-based spatial graph using cell centroids.
    
    Uses global coordinates. For FOV-specific analysis, subset first.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    radius : float
        Radius for neighborhood (in coordinate units)
    include_self : bool, default=False
        Whether to include self-connections
    
    Returns
    -------
    SpatialGraph
        Spatial graph object
    
    Examples
    --------
    >>> # Build radius graph (50 pixel radius)
    >>> graph = sj.spatial.point.build_radius_graph(sp, radius=50)
    >>> 
    >>> # Single FOV
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> graph = sj.spatial.point.build_radius_graph(sp_fov, radius=50)
    >>> 
    >>> # Cells with many neighbors (high density regions)
    >>> degrees = graph.get_degree()
    >>> high_density = sp.cell_index[degrees > 20]
    """
    print(f"\n{'='*70}")
    print(f"Building Radius Graph (Global Coordinates)")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  radius = {radius}")
    print(f"  include_self = {include_self}")
    
    # Get coordinates (always global)
    coords = sp.get_spatial_coords(coord_type='global')
    n_cells = len(sp.cell_index)
    
    print(f"\nBuilding KD-tree for {n_cells:,} cells...")
    
    # Build KD-tree
    tree = KDTree(coords)
    
    # Query all neighbors within radius
    print(f"Querying neighbors within radius {radius}...")
    neighbors_list = tree.query_ball_tree(tree, radius)
    
    # Build sparse adjacency matrix
    adjacency = lil_matrix((n_cells, n_cells), dtype=np.float32)
    distance_matrix = lil_matrix((n_cells, n_cells), dtype=np.float32)
    
    for i, neighbors in enumerate(neighbors_list):
        for j in neighbors:
            if not include_self and i == j:
                continue
            
            dist = np.linalg.norm(coords[i] - coords[j])
            adjacency[i, j] = 1.0
            distance_matrix[i, j] = dist
    
    # Convert to CSR
    print(f"Converting to CSR format...")
    adjacency = adjacency.tocsr()
    distance_matrix = distance_matrix.tocsr()
    
    # Create graph
    graph = SpatialGraph(
        adjacency=adjacency,
        cell_index=sp.cell_index,
        method=f'radius_r{radius}',
        distances=distance_matrix
    )
    
    print(f"\n{'='*70}")
    print(f"Graph Summary:")
    print(f"  Total edges: {adjacency.nnz:,}")
    print(f"  Avg degree: {graph.get_degree().mean():.2f}")
    print(f"  Min degree: {graph.get_degree().min()}")
    print(f"  Max degree: {graph.get_degree().max()}")
    print(f"{'='*70}\n")
    
    return graph


def build_delaunay_graph(sp: 'spatioloji') -> SpatialGraph:
    """
    Build spatial graph using Delaunay triangulation.
    
    Uses global coordinates. For FOV-specific analysis, subset first.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    
    Returns
    -------
    SpatialGraph
        Spatial graph object
    
    Notes
    -----
    Delaunay triangulation creates a graph where:
    - Each cell is connected to its "natural" neighbors
    - No isolated cells (unless spatially separated)
    - Degree varies with local density
    
    Examples
    --------
    >>> # Build Delaunay graph for all cells
    >>> graph = sj.spatial.point.build_delaunay_graph(sp)
    >>> 
    >>> # Single FOV
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> graph = sj.spatial.point.build_delaunay_graph(sp_fov)
    """
    print(f"\n{'='*70}")
    print(f"Building Delaunay Graph (Global Coordinates)")
    print(f"{'='*70}")
    
    # Get coordinates (always global)
    coords = sp.get_spatial_coords(coord_type='global')
    n_cells = len(sp.cell_index)
    
    print(f"\nPerforming Delaunay triangulation on {n_cells:,} cells...")
    
    # Perform Delaunay triangulation
    try:
        tri = Delaunay(coords)
    except Exception as e:
        raise ValueError(f"Delaunay triangulation failed: {e}")
    
    print(f"  Created {tri.nsimplex:,} triangles")
    
    # Build adjacency from triangulation
    adjacency = lil_matrix((n_cells, n_cells), dtype=np.float32)
    distance_matrix = lil_matrix((n_cells, n_cells), dtype=np.float32)
    
    print(f"Building adjacency matrix...")
    
    # Each simplex (triangle) defines 3 edges
    for simplex in tri.simplices:
        # Add edges for all pairs in triangle
        for i in range(3):
            for j in range(i+1, 3):
                idx_i, idx_j = simplex[i], simplex[j]
                
                # Add bidirectional edges
                dist = np.linalg.norm(coords[idx_i] - coords[idx_j])
                
                adjacency[idx_i, idx_j] = 1.0
                adjacency[idx_j, idx_i] = 1.0
                distance_matrix[idx_i, idx_j] = dist
                distance_matrix[idx_j, idx_i] = dist
    
    # Convert to CSR
    print(f"Converting to CSR format...")
    adjacency = adjacency.tocsr()
    distance_matrix = distance_matrix.tocsr()
    
    # Create graph
    graph = SpatialGraph(
        adjacency=adjacency,
        cell_index=sp.cell_index,
        method='delaunay',
        distances=distance_matrix
    )
    
    print(f"\n{'='*70}")
    print(f"Graph Summary:")
    print(f"  Triangles: {tri.nsimplex:,}")
    print(f"  Total edges: {adjacency.nnz:,}")
    print(f"  Avg degree: {graph.get_degree().mean():.2f}")
    print(f"  Min degree: {graph.get_degree().min()}")
    print(f"  Max degree: {graph.get_degree().max()}")
    print(f"{'='*70}\n")
    
    return graph


def compute_distance_matrix(sp: 'spatioloji',
                            cell_ids: Optional[List[str]] = None,
                            sparse: bool = True,
                            max_distance: Optional[float] = None) -> Union[np.ndarray, csr_matrix]:
    """
    Compute pairwise distance matrix between cells.
    
    Uses global coordinates.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    cell_ids : list, optional
        Subset of cells to compute. If None, uses all cells.
    sparse : bool, default=True
        If True, return sparse matrix (only distances < max_distance)
    max_distance : float, optional
        Maximum distance for sparse matrix. Required if sparse=True.
    
    Returns
    -------
    np.ndarray or csr_matrix
        Distance matrix (cells × cells)
    
    Examples
    --------
    >>> # Dense distance matrix for subset
    >>> subset_cells = sp.cell_meta[sp.cell_meta['cell_type'] == 'Tumor'].index[:100]
    >>> distances = sj.spatial.point.compute_distance_matrix(
    ...     sp, cell_ids=subset_cells, sparse=False
    ... )
    >>> 
    >>> # Sparse distance matrix (only < 100 units)
    >>> distances_sparse = sj.spatial.point.compute_distance_matrix(
    ...     sp, sparse=True, max_distance=100
    ... )
    """
    if sparse and max_distance is None:
        raise ValueError("max_distance required when sparse=True")
    
    print(f"\n{'='*70}")
    print(f"Computing Distance Matrix (Global Coordinates)")
    print(f"{'='*70}")
    
    # Get cell subset
    if cell_ids is not None:
        indices = sp._get_cell_indices(cell_ids)
        coords = sp.get_spatial_coords(coord_type='global')[indices]
        cell_index = sp.cell_index[indices]
    else:
        coords = sp.get_spatial_coords(coord_type='global')
        cell_index = sp.cell_index
        indices = np.arange(len(cell_index))
    
    n_cells = len(coords)
    
    print(f"Parameters:")
    print(f"  n_cells = {n_cells:,}")
    print(f"  sparse = {sparse}")
    if sparse:
        print(f"  max_distance = {max_distance}")
    
    if sparse:
        # Use KDTree for efficient sparse computation
        print(f"\nBuilding KD-tree...")
        tree = KDTree(coords)
        
        print(f"Finding pairs within radius {max_distance}...")
        pairs = tree.query_pairs(max_distance, output_type='ndarray')
        
        distance_matrix = lil_matrix((n_cells, n_cells), dtype=np.float32)
        
        print(f"Computing distances for {len(pairs):,} pairs...")
        for i, j in pairs:
            dist = np.linalg.norm(coords[i] - coords[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
        
        distance_matrix = distance_matrix.tocsr()
        
        print(f"\n{'='*70}")
        print(f"Result:")
        print(f"  Matrix size: {n_cells} × {n_cells}")
        print(f"  Non-zero entries: {distance_matrix.nnz:,}")
        print(f"  Sparsity: {1 - distance_matrix.nnz / (n_cells**2):.2%}")
        print(f"{'='*70}\n")
        
        return distance_matrix
    
    else:
        # Compute dense distance matrix
        from scipy.spatial.distance import cdist
        
        print(f"\nComputing dense distance matrix...")
        distance_matrix = cdist(coords, coords, metric='euclidean')
        
        print(f"\n{'='*70}")
        print(f"Result:")
        print(f"  Matrix size: {n_cells} × {n_cells}")
        print(f"  Memory: {distance_matrix.nbytes / (1024**2):.2f} MB")
        print(f"{'='*70}\n")
        
        return distance_matrix


def compute_pairwise_distances(sp: 'spatioloji',
                               cell_ids_1: List[str],
                               cell_ids_2: Optional[List[str]] = None) -> np.ndarray:
    """
    Compute distances between two sets of cells.
    
    Uses global coordinates.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    cell_ids_1 : list
        First set of cell IDs
    cell_ids_2 : list, optional
        Second set of cell IDs. If None, uses same as cell_ids_1.
    
    Returns
    -------
    np.ndarray
        Distance matrix (len(cell_ids_1) × len(cell_ids_2))
    
    Examples
    --------
    >>> # Distance from tumor cells to immune cells
    >>> tumor = sp.cell_meta[sp.cell_meta['cell_type'] == 'Tumor'].index
    >>> immune = sp.cell_meta[sp.cell_meta['cell_type'] == 'Immune'].index
    >>> 
    >>> distances = sj.spatial.point.compute_pairwise_distances(sp, tumor, immune)
    >>> 
    >>> # Minimum distance from each tumor cell to any immune cell
    >>> min_distances = distances.min(axis=1)
    >>> sp.cell_meta.loc[tumor, 'dist_to_immune'] = min_distances
    """
    from scipy.spatial.distance import cdist
    
    print(f"\n{'='*70}")
    print(f"Computing Pairwise Distances (Global Coordinates)")
    print(f"{'='*70}")
    
    # Get coordinates for first set
    indices_1 = sp._get_cell_indices(cell_ids_1)
    coords_1 = sp.get_spatial_coords(coord_type='global')[indices_1]
    
    # Get coordinates for second set
    if cell_ids_2 is None:
        coords_2 = coords_1
        n_cells_2 = len(coords_1)
    else:
        indices_2 = sp._get_cell_indices(cell_ids_2)
        coords_2 = sp.get_spatial_coords(coord_type='global')[indices_2]
        n_cells_2 = len(coords_2)
    
    print(f"Parameters:")
    print(f"  Set 1: {len(coords_1):,} cells")
    print(f"  Set 2: {n_cells_2:,} cells")
    
    # Compute pairwise distances
    print(f"\nComputing distances...")
    distances = cdist(coords_1, coords_2, metric='euclidean')
    
    print(f"\n{'='*70}")
    print(f"Result:")
    print(f"  Matrix size: {len(coords_1)} × {n_cells_2}")
    print(f"  Min distance: {distances.min():.2f}")
    print(f"  Max distance: {distances.max():.2f}")
    print(f"  Mean distance: {distances.mean():.2f}")
    print(f"{'='*70}\n")
    
    return distances


def compute_distance_to_region(sp: 'spatioloji',
                               target_cells: List[str],
                               return_closest: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Compute distance from each cell to nearest cell in target region.
    
    Uses global coordinates.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    target_cells : list
        Cell IDs defining the target region
    return_closest : bool, default=False
        If True, also return ID of closest target cell
    
    Returns
    -------
    pd.Series or tuple of pd.Series
        Distance to nearest target cell for each cell.
        If return_closest=True, returns (distances, closest_ids)
    
    Examples
    --------
    >>> # Distance to tumor region
    >>> tumor_cells = sp.cell_meta[sp.cell_meta['cell_type'] == 'Tumor'].index
    >>> distances = sj.spatial.point.compute_distance_to_region(sp, tumor_cells)
    >>> sp.cell_meta['dist_to_tumor'] = distances
    >>> 
    >>> # Find cells within 50 units of tumor
    >>> near_tumor = sp.cell_meta[sp.cell_meta['dist_to_tumor'] < 50].index
    >>> 
    >>> # Get closest tumor cell for each cell
    >>> distances, closest = sj.spatial.point.compute_distance_to_region(
    ...     sp, tumor_cells, return_closest=True
    ... )
    """
    print(f"\n{'='*70}")
    print(f"Computing Distance to Region (Global Coordinates)")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  Target region: {len(target_cells):,} cells")
    
    # Get coordinates
    all_coords = sp.get_spatial_coords(coord_type='global')
    target_indices = sp._get_cell_indices(target_cells)
    target_coords = all_coords[target_indices]
    
    # Build KD-tree for target cells
    print(f"\nBuilding KD-tree for target cells...")
    tree = KDTree(target_coords)
    
    # Query nearest target for each cell
    print(f"Querying nearest target for all cells...")
    distances, indices = tree.query(all_coords)
    
    # Create result series
    distance_series = pd.Series(distances, index=sp.cell_index, name='distance_to_region')
    
    print(f"\n{'='*70}")
    print(f"Result:")
    print(f"  Min distance: {distances.min():.2f}")
    print(f"  Max distance: {distances.max():.2f}")
    print(f"  Mean distance: {distances.mean():.2f}")
    print(f"  Median distance: {np.median(distances):.2f}")
    print(f"{'='*70}\n")
    
    if return_closest:
        # Get IDs of closest target cells
        target_cells_list = list(target_cells)
        closest_ids = pd.Series(
            [target_cells_list[i] if i < len(target_cells_list) else None for i in indices],
            index=sp.cell_index,
            name='closest_target_cell'
        )
        return distance_series, closest_ids
    
    return distance_series


def filter_graph_by_fov(sp: 'spatioloji',
                       graph: SpatialGraph,
                       keep_inter_fov: bool = False) -> SpatialGraph:
    """
    Filter graph to remove/keep inter-FOV edges.
    
    Useful when analyzing neighborhoods within FOVs only.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph to filter
    keep_inter_fov : bool, default=False
        If False, remove edges between different FOVs.
        If True, keep only inter-FOV edges.
    
    Returns
    -------
    SpatialGraph
        Filtered graph
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # Remove cross-FOV edges
    >>> graph_intra = sj.spatial.point.filter_graph_by_fov(sp, graph)
    >>> 
    >>> # Keep only cross-FOV edges
    >>> graph_inter = sj.spatial.point.filter_graph_by_fov(
    ...     sp, graph, keep_inter_fov=True
    ... )
    """
    fov_col = sp.config.fov_id_col
    
    if fov_col not in sp.cell_meta.columns:
        warnings.warn(f"'{fov_col}' not in cell_meta, returning original graph")
        return graph
    
    print(f"\n{'='*70}")
    print(f"Filtering Graph by FOV")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  keep_inter_fov = {keep_inter_fov}")
    
    # Get FOV assignments
    fov_assignments = sp.cell_meta[fov_col].values
    
    # Filter adjacency matrix
    adjacency = graph.adjacency.tolil()
    if graph.distances is not None:
        distances = graph.distances.tolil()
    else:
        distances = None
    
    edges_removed = 0
    edges_kept = 0
    
    print(f"\nFiltering edges...")
    
    # Get all edges
    rows, cols = graph.adjacency.nonzero()
    
    for i, j in zip(rows, cols):
        # Check if same FOV
        same_fov = fov_assignments[i] == fov_assignments[j]
        
        if keep_inter_fov:
            # Keep only inter-FOV edges
            if same_fov:
                adjacency[i, j] = 0
                if distances is not None:
                    distances[i, j] = 0
                edges_removed += 1
            else:
                edges_kept += 1
        else:
            # Keep only intra-FOV edges
            if not same_fov:
                adjacency[i, j] = 0
                if distances is not None:
                    distances[i, j] = 0
                edges_removed += 1
            else:
                edges_kept += 1
    
    # Convert back to CSR
    adjacency = adjacency.tocsr()
    if distances is not None:
        distances = distances.tocsr()
    
    # Create new graph
    method_suffix = 'inter_fov' if keep_inter_fov else 'intra_fov'
    filtered_graph = SpatialGraph(
        adjacency=adjacency,
        cell_index=graph.cell_index,
        method=f"{graph.method}_{method_suffix}",
        distances=distances
    )
    
    print(f"\n{'='*70}")
    print(f"Result:")
    print(f"  Original edges: {graph.adjacency.nnz:,}")
    print(f"  Edges removed: {edges_removed:,}")
    print(f"  Edges kept: {edges_kept:,}")
    print(f"  New avg degree: {filtered_graph.get_degree().mean():.2f}")
    print(f"{'='*70}\n")
    
    return filtered_graph


def filter_graph_by_distance(graph: SpatialGraph,
                             max_distance: float) -> SpatialGraph:
    """
    Filter graph to keep only edges below distance threshold.
    
    Parameters
    ----------
    graph : SpatialGraph
        Spatial graph to filter (must have distances)
    max_distance : float
        Maximum distance threshold
    
    Returns
    -------
    SpatialGraph
        Filtered graph
    
    Examples
    --------
    >>> # Build k-NN graph
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=20)
    >>> 
    >>> # Keep only edges < 100 units
    >>> graph_filtered = sj.spatial.point.filter_graph_by_distance(graph, 100)
    """
    if graph.distances is None:
        raise ValueError("Graph must have distance information")
    
    print(f"\n{'='*70}")
    print(f"Filtering Graph by Distance")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  max_distance = {max_distance}")
    
    # Create mask for edges to keep
    adjacency = graph.adjacency.tolil()
    distances = graph.distances.tolil()
    
    edges_removed = 0
    edges_kept = 0
    
    # Get all edges
    rows, cols = graph.adjacency.nonzero()
    
    print(f"\nFiltering {len(rows):,} edges...")
    
    for i, j in zip(rows, cols):
        if distances[i, j] > max_distance:
            adjacency[i, j] = 0
            distances[i, j] = 0
            edges_removed += 1
        else:
            edges_kept += 1
    
    # Convert back to CSR
    adjacency = adjacency.tocsr()
    distances = distances.tocsr()
    
    # Create new graph
    filtered_graph = SpatialGraph(
        adjacency=adjacency,
        cell_index=graph.cell_index,
        method=f"{graph.method}_dist{max_distance}",
        distances=distances
    )
    
    print(f"\n{'='*70}")
    print(f"Result:")
    print(f"  Original edges: {graph.adjacency.nnz:,}")
    print(f"  Edges removed: {edges_removed:,}")
    print(f"  Edges kept: {edges_kept:,}")
    print(f"  New avg degree: {filtered_graph.get_degree().mean():.2f}")
    print(f"{'='*70}\n")
    
    return filtered_graph