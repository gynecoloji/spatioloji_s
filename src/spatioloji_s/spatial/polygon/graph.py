# src/spatioloji_s/spatial/polygon/graph.py

"""
graph.py - Polygon-based spatial graph construction

Provides graph construction and distance calculations using actual cell
boundary polygons instead of centroids.

Key differences from point-based:
- Distance: Boundary-to-boundary (minimum edge distance)
- Neighbors: Cells that physically touch (share boundary)
- Weights: Contact length (shared boundary length)

Requires polygon data to be available in sp.cell_boundaries.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import warnings


class PolygonSpatialGraph:
    """
    Spatial graph based on polygon boundaries.
    
    Similar to SpatialGraph but uses polygon contacts and boundary distances
    instead of centroid distances.
    
    Attributes
    ----------
    adjacency : scipy.sparse matrix
        Adjacency matrix (cells × cells)
    contact_lengths : scipy.sparse matrix or None
        Contact length matrix (shared boundary lengths)
    distances : scipy.sparse matrix or None
        Boundary-to-boundary distance matrix
    cell_index : pd.Index
        Cell IDs aligned with matrix rows/cols
    graph_type : str
        Type of graph ('contact', 'knn', 'radius')
    """
    
    def __init__(self, 
                 adjacency: sparse.spmatrix,
                 cell_index: pd.Index,
                 contact_lengths: Optional[sparse.spmatrix] = None,
                 distances: Optional[sparse.spmatrix] = None,
                 graph_type: str = 'contact'):
        """
        Initialize polygon spatial graph.
        
        Parameters
        ----------
        adjacency : scipy.sparse matrix
            Adjacency matrix
        cell_index : pd.Index
            Cell IDs
        contact_lengths : scipy.sparse matrix, optional
            Contact lengths (for contact graphs)
        distances : scipy.sparse matrix, optional
            Boundary distances
        graph_type : str
            Type of graph
        """
        self.adjacency = adjacency
        self.contact_lengths = contact_lengths
        self.distances = distances
        self.cell_index = cell_index
        self.graph_type = graph_type
        
        # Validate
        n = len(cell_index)
        if adjacency.shape != (n, n):
            raise ValueError(f"Adjacency shape {adjacency.shape} != ({n}, {n})")
        
        if contact_lengths is not None and contact_lengths.shape != (n, n):
            raise ValueError(f"Contact lengths shape mismatch")
        
        if distances is not None and distances.shape != (n, n):
            raise ValueError(f"Distances shape mismatch")
    
    def get_neighbors(self, cell_id: str) -> pd.DataFrame:
        """
        Get neighbors of a cell with contact info.
        
        Parameters
        ----------
        cell_id : str
            Cell ID
        
        Returns
        -------
        pd.DataFrame
            Neighbors with columns: neighbor_id, contact_length (if available),
            distance (if available)
        """
        if cell_id not in self.cell_index:
            raise ValueError(f"Cell {cell_id} not in graph")
        
        idx = self.cell_index.get_loc(cell_id)
        
        # Get neighbors from adjacency
        neighbor_indices = self.adjacency[idx].nonzero()[1]
        neighbor_ids = self.cell_index[neighbor_indices]
        
        # Build DataFrame
        result = pd.DataFrame({'neighbor_id': neighbor_ids})
        
        # Add contact lengths if available
        if self.contact_lengths is not None:
            lengths = self.contact_lengths[idx, neighbor_indices].toarray().flatten()
            result['contact_length'] = lengths
        
        # Add distances if available
        if self.distances is not None:
            dists = self.distances[idx, neighbor_indices].toarray().flatten()
            result['distance'] = dists
        
        return result
    
    def get_neighbor_ids(self, cell_id: str) -> List[str]:
        """Get list of neighbor cell IDs."""
        if cell_id not in self.cell_index:
            raise ValueError(f"Cell {cell_id} not in graph")
        
        idx = self.cell_index.get_loc(cell_id)
        neighbor_indices = self.adjacency[idx].nonzero()[1]
        return self.cell_index[neighbor_indices].tolist()
    
    def get_degree(self, cell_id: str) -> int:
        """Get degree (number of neighbors) for a cell."""
        return len(self.get_neighbor_ids(cell_id))
    
    def get_total_contact_length(self, cell_id: str) -> float:
        """
        Get total contact length for a cell.
        
        Returns sum of all shared boundary lengths.
        """
        if self.contact_lengths is None:
            raise ValueError("Graph does not have contact length information")
        
        if cell_id not in self.cell_index:
            raise ValueError(f"Cell {cell_id} not in graph")
        
        idx = self.cell_index.get_loc(cell_id)
        return self.contact_lengths[idx].sum()
    
    def to_edge_list(self) -> pd.DataFrame:
        """
        Convert graph to edge list format.
        
        Returns
        -------
        pd.DataFrame
            Edge list with columns: source, target, contact_length (if available),
            distance (if available)
        """
        rows, cols = self.adjacency.nonzero()
        
        edge_list = pd.DataFrame({
            'source': self.cell_index[rows],
            'target': self.cell_index[cols]
        })
        
        if self.contact_lengths is not None:
            edge_list['contact_length'] = self.contact_lengths[rows, cols].A1
        
        if self.distances is not None:
            edge_list['distance'] = self.distances[rows, cols].A1
        
        return edge_list
    
    def summary(self) -> None:
        """Print summary statistics of the graph."""
        n_cells = len(self.cell_index)
        n_edges = self.adjacency.nnz
        degrees = np.array([self.adjacency[i].nnz for i in range(n_cells)])
        
        print(f"\n{'='*70}")
        print(f"Polygon Spatial Graph Summary")
        print(f"{'='*70}")
        print(f"Graph type: {self.graph_type}")
        print(f"Cells: {n_cells:,}")
        print(f"Edges: {n_edges:,}")
        print(f"Average degree: {degrees.mean():.2f}")
        print(f"Degree range: {degrees.min()} - {degrees.max()}")
        
        if self.contact_lengths is not None:
            total_contact = self.contact_lengths.sum() / 2  # Divide by 2 (symmetric)
            print(f"Total contact length: {total_contact:.2f}")
            contact_per_edge = total_contact / (n_edges / 2)
            print(f"Average contact length: {contact_per_edge:.2f}")
        
        print(f"{'='*70}\n")


def build_contact_graph(sp: 'spatioloji',
                       min_contact_length: float = 0.0,
                       fov_id: Optional[str] = None,
                       verbose: bool = True) -> PolygonSpatialGraph:
    """
    Build spatial graph based on polygon contacts (physical touching).
    
    Two cells are neighbors if their boundary polygons touch or overlap.
    Edge weights are the length of shared boundary.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    min_contact_length : float, default=0.0
        Minimum contact length to consider as contact
    fov_id : str, optional
        Restrict to specific FOV
    verbose : bool, default=True
        Print progress messages
    
    Returns
    -------
    PolygonSpatialGraph
        Contact-based spatial graph
    
    Examples
    --------
    >>> import spatioloji as sj
    >>> 
    >>> # Build contact graph
    >>> graph = sj.spatial.polygon.build_contact_graph(sp)
    >>> 
    >>> # Get cells touching a specific cell
    >>> neighbors = graph.get_neighbors('cell_123')
    >>> print(neighbors[['neighbor_id', 'contact_length']])
    >>> 
    >>> # Filter by minimum contact length
    >>> graph = sj.spatial.polygon.build_contact_graph(sp, min_contact_length=5.0)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Building Contact Graph (Polygon-Based)")
        print(f"{'='*70}")
    
    # Validate polygon data
    from ..shared.utils import validate_spatial_data
    validate_spatial_data(sp, require_polygons=True)
    
    # Get cell subset
    if fov_id is not None:
        cell_ids = sp.get_cells_in_fov(fov_id)
        if verbose:
            print(f"FOV: {fov_id}")
            print(f"Cells: {len(cell_ids):,}")
    else:
        cell_ids = sp.cell_index.tolist()
        if verbose:
            print(f"Cells: {len(cell_ids):,}")
    
    if verbose:
        print(f"Min contact length: {min_contact_length}")
    
    n_cells = len(cell_ids)
    cell_id_to_idx = {cell_id: i for i, cell_id in enumerate(cell_ids)}
    
    # Get polygons
    polygons = {}
    if verbose:
        print(f"\nLoading polygons...")
    
    for cell_id in cell_ids:
        poly_data = sp.cell_boundaries.get_cell_polygon(cell_id)
        if poly_data is not None:
            polygons[cell_id] = poly_data
    
    if verbose:
        print(f"  Loaded {len(polygons):,} polygons")
    
    if len(polygons) == 0:
        raise ValueError("No polygons found for selected cells")
    
    # Build contact matrix
    if verbose:
        print(f"\nDetecting contacts...")
    
    rows, cols, contact_lengths = [], [], []
    
    # Use spatial index for efficiency
    from shapely.strtree import STRtree
    
    poly_list = list(polygons.values())
    poly_ids = list(polygons.keys())
    tree = STRtree(poly_list)
    
    n_contacts = 0
    for i, (cell_id, poly) in enumerate(polygons.items()):
        if verbose and i % 500 == 0 and i > 0:
            print(f"  Cell {i:,}/{len(polygons):,}... ({n_contacts:,} contacts)")
        
        # Query potential neighbors (geometries that might intersect)
        potential_neighbors = tree.query(poly)
        
        for neighbor_poly in potential_neighbors:
            # Get cell ID for neighbor
            neighbor_idx = poly_list.index(neighbor_poly)
            neighbor_id = poly_ids[neighbor_idx]
            
            # Skip self
            if neighbor_id == cell_id:
                continue
            
            # Check if already processed (graph is symmetric)
            idx_i = cell_id_to_idx[cell_id]
            idx_j = cell_id_to_idx[neighbor_id]
            if idx_j < idx_i:  # Already processed in reverse direction
                continue
            
            # Check if they actually touch
            if poly.touches(neighbor_poly) or poly.intersects(neighbor_poly):
                # Compute contact length (length of shared boundary)
                try:
                    intersection = poly.intersection(neighbor_poly)
                    
                    # Contact length is the length of the intersection
                    if intersection.is_empty:
                        contact_len = 0.0
                    elif intersection.geom_type == 'Point':
                        contact_len = 0.0  # Single point contact
                    elif intersection.geom_type in ['LineString', 'MultiLineString']:
                        contact_len = intersection.length
                    else:
                        # Overlapping polygons - use perimeter of overlap
                        contact_len = intersection.length
                    
                    if contact_len >= min_contact_length:
                        # Add both directions (symmetric graph)
                        rows.extend([idx_i, idx_j])
                        cols.extend([idx_j, idx_i])
                        contact_lengths.extend([contact_len, contact_len])
                        n_contacts += 1
                
                except Exception as e:
                    warnings.warn(f"Error computing contact between {cell_id} and {neighbor_id}: {e}")
                    continue
    
    if verbose:
        print(f"  Total contacts: {n_contacts:,}")
    
    # Create sparse matrices
    if len(rows) == 0:
        warnings.warn("No contacts found")
        adjacency = sparse.csr_matrix((n_cells, n_cells))
        contact_matrix = sparse.csr_matrix((n_cells, n_cells))
    else:
        adjacency = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_cells, n_cells)
        )
        contact_matrix = sparse.csr_matrix(
            (contact_lengths, (rows, cols)),
            shape=(n_cells, n_cells)
        )
    
    # Create graph
    graph = PolygonSpatialGraph(
        adjacency=adjacency,
        cell_index=pd.Index(cell_ids),
        contact_lengths=contact_matrix,
        distances=None,
        graph_type='contact'
    )
    
    if verbose:
        graph.summary()
    
    return graph


def compute_polygon_distance(poly1: Polygon, poly2: Polygon) -> float:
    """
    Compute minimum distance between two polygon boundaries.
    
    Parameters
    ----------
    poly1 : Polygon
        First polygon
    poly2 : Polygon
        Second polygon
    
    Returns
    -------
    float
        Minimum distance between polygon boundaries (0 if touching/overlapping)
    """
    # If polygons touch or overlap, distance is 0
    if poly1.touches(poly2) or poly1.intersects(poly2):
        return 0.0
    
    # Otherwise, compute minimum distance
    return poly1.distance(poly2)


def build_knn_graph(sp: 'spatioloji',
                   k: int = 10,
                   fov_id: Optional[str] = None,
                   include_contacts: bool = True,
                   verbose: bool = True) -> PolygonSpatialGraph:
    """
    Build k-nearest neighbors graph using polygon boundary distances.
    
    For each cell, finds k nearest cells based on minimum boundary-to-boundary
    distance (not centroid distance).
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    k : int, default=10
        Number of neighbors
    fov_id : str, optional
        Restrict to specific FOV
    include_contacts : bool, default=True
        If True, compute contact lengths for touching neighbors
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    PolygonSpatialGraph
        k-NN graph based on polygon distances
    
    Examples
    --------
    >>> # Build k-NN graph with polygon distances
    >>> graph = sj.spatial.polygon.build_knn_graph(sp, k=10)
    >>> 
    >>> # This is different from point-based k-NN:
    >>> graph_point = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> # Polygon version uses boundary distances, point version uses centroids
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Building k-NN Graph (Polygon Boundary Distance)")
        print(f"{'='*70}")
        print(f"k: {k}")
    
    # Validate polygon data
    from ..shared.utils import validate_spatial_data
    validate_spatial_data(sp, require_polygons=True)
    
    # Get cell subset
    if fov_id is not None:
        cell_ids = sp.get_cells_in_fov(fov_id)
        if verbose:
            print(f"FOV: {fov_id}")
            print(f"Cells: {len(cell_ids):,}")
    else:
        cell_ids = sp.cell_index.tolist()
        if verbose:
            print(f"Cells: {len(cell_ids):,}")
    
    n_cells = len(cell_ids)
    
    # Get polygons
    polygons = []
    valid_cells = []
    
    if verbose:
        print(f"\nLoading polygons...")
    
    for cell_id in cell_ids:
        poly = sp.cell_boundaries.get_cell_polygon(cell_id)
        if poly is not None:
            polygons.append(poly)
            valid_cells.append(cell_id)
    
    if verbose:
        print(f"  Loaded {len(polygons):,} polygons")
    
    if len(polygons) < k:
        warnings.warn(f"Number of cells ({len(polygons)}) < k ({k})")
        k = max(1, len(polygons) - 1)
    
    # Compute pairwise distances
    if verbose:
        print(f"\nComputing pairwise boundary distances...")
    
    n_valid = len(valid_cells)
    distances = np.full((n_valid, n_valid), np.inf)
    contact_lengths_array = np.zeros((n_valid, n_valid))
    
    for i in range(n_valid):
        if verbose and i % 100 == 0 and i > 0:
            print(f"  Cell {i:,}/{n_valid:,}...")
        
        for j in range(i + 1, n_valid):
            dist = compute_polygon_distance(polygons[i], polygons[j])
            distances[i, j] = dist
            distances[j, i] = dist
            
            # Compute contact length if touching
            if include_contacts and dist == 0:
                try:
                    intersection = polygons[i].intersection(polygons[j])
                    if intersection.geom_type in ['LineString', 'MultiLineString']:
                        contact_len = intersection.length
                    else:
                        contact_len = 0.0
                    
                    contact_lengths_array[i, j] = contact_len
                    contact_lengths_array[j, i] = contact_len
                except:
                    pass
    
    # Build k-NN graph
    if verbose:
        print(f"\nBuilding k-NN graph...")
    
    rows, cols, dists = [], [], []
    contact_rows, contact_cols, contact_lens = [], [], []
    
    for i in range(n_valid):
        # Get k nearest neighbors
        neighbor_dists = distances[i].copy()
        neighbor_dists[i] = np.inf  # Exclude self
        
        knn_indices = np.argpartition(neighbor_dists, min(k, len(neighbor_dists)-1))[:k]
        knn_indices = knn_indices[np.argsort(neighbor_dists[knn_indices])]
        
        for j in knn_indices:
            rows.append(i)
            cols.append(j)
            dists.append(distances[i, j])
            
            if include_contacts and contact_lengths_array[i, j] > 0:
                contact_rows.append(i)
                contact_cols.append(j)
                contact_lens.append(contact_lengths_array[i, j])
    
    # Create sparse matrices
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_valid, n_valid)
    )
    
    distance_matrix = sparse.csr_matrix(
        (dists, (rows, cols)),
        shape=(n_valid, n_valid)
    )
    
    if include_contacts and len(contact_rows) > 0:
        contact_matrix = sparse.csr_matrix(
            (contact_lens, (contact_rows, contact_cols)),
            shape=(n_valid, n_valid)
        )
    else:
        contact_matrix = None
    
    # Create graph
    graph = PolygonSpatialGraph(
        adjacency=adjacency,
        cell_index=pd.Index(valid_cells),
        contact_lengths=contact_matrix,
        distances=distance_matrix,
        graph_type='knn'
    )
    
    if verbose:
        graph.summary()
    
    return graph


def build_radius_graph(sp: 'spatioloji',
                      radius: float,
                      fov_id: Optional[str] = None,
                      include_contacts: bool = True,
                      verbose: bool = True) -> PolygonSpatialGraph:
    """
    Build radius graph using polygon boundary distances.
    
    Two cells are neighbors if their polygon boundaries are within
    the specified radius.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    radius : float
        Distance threshold for neighborhood
    fov_id : str, optional
        Restrict to specific FOV
    include_contacts : bool, default=True
        If True, compute contact lengths for touching neighbors
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    PolygonSpatialGraph
        Radius-based graph
    
    Examples
    --------
    >>> # Cells within 50 units (boundary distance)
    >>> graph = sj.spatial.polygon.build_radius_graph(sp, radius=50)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Building Radius Graph (Polygon Boundary Distance)")
        print(f"{'='*70}")
        print(f"Radius: {radius}")
    
    # Validate polygon data
    from ..shared.utils import validate_spatial_data
    validate_spatial_data(sp, require_polygons=True)
    
    # Get cell subset
    if fov_id is not None:
        cell_ids = sp.get_cells_in_fov(fov_id)
        if verbose:
            print(f"FOV: {fov_id}")
            print(f"Cells: {len(cell_ids):,}")
    else:
        cell_ids = sp.cell_index.tolist()
        if verbose:
            print(f"Cells: {len(cell_ids):,}")
    
    # Get polygons
    polygons = []
    valid_cells = []
    
    if verbose:
        print(f"\nLoading polygons...")
    
    for cell_id in cell_ids:
        poly = sp.cell_boundaries.get_cell_polygon(cell_id)
        if poly is not None:
            polygons.append(poly)
            valid_cells.append(cell_id)
    
    if verbose:
        print(f"  Loaded {len(polygons):,} polygons")
    
    n_valid = len(valid_cells)
    
    # Build radius graph using spatial index
    if verbose:
        print(f"\nFinding neighbors within radius...")
    
    from shapely.strtree import STRtree
    
    # Create buffered polygons for radius search
    buffered = [poly.buffer(radius) for poly in polygons]
    tree = STRtree(polygons)
    
    rows, cols, dists = [], [], []
    contact_rows, contact_cols, contact_lens = [], [], []
    
    for i, (poly, buff) in enumerate(zip(polygons, buffered)):
        if verbose and i % 100 == 0 and i > 0:
            print(f"  Cell {i:,}/{n_valid:,}...")
        
        # Query potential neighbors
        potential = tree.query(buff)
        
        for neighbor_poly in potential:
            j = polygons.index(neighbor_poly)
            
            # Skip self
            if i == j:
                continue
            
            # Skip if already processed
            if j < i:
                continue
            
            # Compute distance
            dist = compute_polygon_distance(poly, neighbor_poly)
            
            if dist <= radius:
                # Add both directions
                rows.extend([i, j])
                cols.extend([j, i])
                dists.extend([dist, dist])
                
                # Contact length if touching
                if include_contacts and dist == 0:
                    try:
                        intersection = poly.intersection(neighbor_poly)
                        if intersection.geom_type in ['LineString', 'MultiLineString']:
                            contact_len = intersection.length
                            contact_rows.extend([i, j])
                            contact_cols.extend([j, i])
                            contact_lens.extend([contact_len, contact_len])
                    except:
                        pass
    
    if verbose:
        print(f"  Found {len(rows)} edges")
    
    # Create sparse matrices
    if len(rows) == 0:
        adjacency = sparse.csr_matrix((n_valid, n_valid))
        distance_matrix = sparse.csr_matrix((n_valid, n_valid))
        contact_matrix = None
    else:
        adjacency = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_valid, n_valid)
        )
        distance_matrix = sparse.csr_matrix(
            (dists, (rows, cols)),
            shape=(n_valid, n_valid)
        )
        
        if include_contacts and len(contact_rows) > 0:
            contact_matrix = sparse.csr_matrix(
                (contact_lens, (contact_rows, contact_cols)),
                shape=(n_valid, n_valid)
            )
        else:
            contact_matrix = None
    
    # Create graph
    graph = PolygonSpatialGraph(
        adjacency=adjacency,
        cell_index=pd.Index(valid_cells),
        contact_lengths=contact_matrix,
        distances=distance_matrix,
        graph_type='radius'
    )
    
    if verbose:
        graph.summary()
    
    return graph


def compute_polygon_distance_matrix(sp: 'spatioloji',
                                    cell_ids: Optional[List[str]] = None,
                                    sparse_output: bool = True,
                                    max_distance: Optional[float] = None,
                                    verbose: bool = True) -> Union[np.ndarray, sparse.spmatrix]:
    """
    Compute pairwise polygon boundary distance matrix.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    cell_ids : list, optional
        Subset of cells. If None, uses all cells.
    sparse_output : bool, default=True
        Return sparse matrix (if max_distance specified)
    max_distance : float, optional
        Maximum distance to store (for sparse output)
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    np.ndarray or scipy.sparse matrix
        Distance matrix
    
    Examples
    --------
    >>> # Full distance matrix
    >>> dists = sj.spatial.polygon.compute_polygon_distance_matrix(sp)
    >>> 
    >>> # Sparse matrix for nearby cells only
    >>> dists = sj.spatial.polygon.compute_polygon_distance_matrix(
    ...     sp, max_distance=100, sparse_output=True
    ... )
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Computing Polygon Distance Matrix")
        print(f"{'='*70}")
    
    # Validate
    from ..shared.utils import validate_spatial_data
    validate_spatial_data(sp, require_polygons=True)
    
    # Get cells
    if cell_ids is None:
        cell_ids = sp.cell_index.tolist()
    
    if verbose:
        print(f"Cells: {len(cell_ids):,}")
        if max_distance:
            print(f"Max distance: {max_distance}")
    
    # Get polygons
    polygons = []
    valid_cells = []
    
    for cell_id in cell_ids:
        poly = sp.cell_boundaries.get_cell_polygon(cell_id)
        if poly is not None:
            polygons.append(poly)
            valid_cells.append(cell_id)
    
    n = len(polygons)
    
    if verbose:
        print(f"Valid polygons: {n:,}")
        print(f"\nComputing distances...")
    
    # Compute distances
    if sparse_output and max_distance is not None:
        rows, cols, dists = [], [], []
        
        for i in range(n):
            if verbose and i % 100 == 0 and i > 0:
                print(f"  Cell {i:,}/{n:,}...")
            
            for j in range(i + 1, n):
                dist = compute_polygon_distance(polygons[i], polygons[j])
                
                if dist <= max_distance:
                    rows.extend([i, j])
                    cols.extend([j, i])
                    dists.extend([dist, dist])
        
        distance_matrix = sparse.csr_matrix(
            (dists, (rows, cols)),
            shape=(n, n)
        )
    else:
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            if verbose and i % 100 == 0 and i > 0:
                print(f"  Cell {i:,}/{n:,}...")
            
            for j in range(i + 1, n):
                dist = compute_polygon_distance(polygons[i], polygons[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    if verbose:
        print(f"\n{'='*70}\n")
    
    return distance_matrix