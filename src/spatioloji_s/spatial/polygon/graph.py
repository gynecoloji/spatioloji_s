"""
graph.py - Polygon-based spatial graph construction

Builds cell adjacency graphs from actual polygon geometry.
All distances are polygon-to-polygon (boundary-to-boundary),
never centroid-to-centroid.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class PolygonSpatialGraph:
    """
    Stores polygon-based cell adjacency graph.
    
    Attributes
    ----------
    adjacency : sparse.csr_matrix
        Binary adjacency matrix (n_cells × n_cells)
    distances : sparse.csr_matrix
        Polygon-to-polygon distance matrix (0 for touching cells)
    cell_index : pd.Index
        Cell IDs aligned to matrix rows/columns
    method : str
        How the graph was built ('contact', 'buffer', 'knn')
    params : dict
        Parameters used to build the graph
    """
    adjacency: sparse.csr_matrix
    distances: sparse.csr_matrix
    cell_index: pd.Index
    method: str
    params: dict = field(default_factory=dict)
    
    @property
    def n_cells(self) -> int:
        return self.adjacency.shape[0]
    
    @property
    def n_edges(self) -> int:
        return self.adjacency.nnz // 2  # undirected
    
    @property
    def mean_degree(self) -> float:
        degrees = np.array(self.adjacency.sum(axis=1)).flatten()
        return degrees.mean()
    
    def get_neighbors(self, cell_id: str) -> List[str]:
        """Get neighbor cell IDs for a given cell."""
        if cell_id not in self.cell_index:
            raise ValueError(f"Cell '{cell_id}' not in graph")
        idx = self.cell_index.get_loc(cell_id)
        neighbor_indices = self.adjacency[idx].nonzero()[1]
        return self.cell_index[neighbor_indices].tolist()
    
    def get_degree(self, cell_id: Optional[str] = None) -> Union[int, np.ndarray]:
        """Get degree (number of neighbors) per cell or for one cell."""
        degrees = np.array(self.adjacency.sum(axis=1)).flatten()
        if cell_id is not None:
            idx = self.cell_index.get_loc(cell_id)
            return int(degrees[idx])
        return degrees
    
    def get_distance(self, cell_a: str, cell_b: str) -> float:
        """Get polygon-to-polygon distance between two cells."""
        idx_a = self.cell_index.get_loc(cell_a)
        idx_b = self.cell_index.get_loc(cell_b)
        return self.distances[idx_a, idx_b]
    
    def to_edge_list(self) -> pd.DataFrame:
        """Convert to edge list DataFrame."""
        rows, cols = self.adjacency.nonzero()
        # Keep only upper triangle (undirected)
        mask = rows < cols
        rows, cols = rows[mask], cols[mask]
        
        dist_values = np.array([
            self.distances[r, c] for r, c in zip(rows, cols)
        ])
        
        return pd.DataFrame({
            'cell_a': self.cell_index[rows],
            'cell_b': self.cell_index[cols],
            'distance': dist_values
        })
    
    def summary(self) -> Dict:
        """Get graph summary statistics."""
        degrees = self.get_degree()
        return {
            'n_cells': self.n_cells,
            'n_edges': self.n_edges,
            'method': self.method,
            'mean_degree': self.mean_degree,
            'median_degree': float(np.median(degrees)),
            'max_degree': int(degrees.max()),
            'min_degree': int(degrees.min()),
            'isolated_cells': int((degrees == 0).sum()),
            'params': self.params
        }


def _get_gdf(sp: 'spatioloji', coord_type: str = 'global') -> 'gpd.GeoDataFrame':
    """
    Get GeoDataFrame from spatioloji, with validation.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object (must have polygon data)
    coord_type : str
        'global' or 'local'
    
    Returns
    -------
    gpd.GeoDataFrame
        One row per cell with Polygon geometry
    """
    if sp.polygons is None:
        raise ValueError(
            "spatioloji has no polygon data. "
            "Polygon-based spatial analysis requires cell boundary polygons."
        )
    return sp.to_geopandas(coord_type=coord_type, include_metadata=False)

def build_contact_graph(sp: 'spatioloji',
                        coord_type: str = 'global',
                        predicate: str = 'intersects') -> PolygonSpatialGraph:
    """
    Build adjacency graph from polygon contacts.
    
    Two cells are neighbors if their polygons touch or overlap.
    Uses spatial index (STRtree) for efficient pairwise checks.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    coord_type : str
        'global' or 'local' coordinates
    predicate : str
        Spatial predicate: 'intersects' (touch or overlap) or 'touches' (shared edge only)
    
    Returns
    -------
    PolygonSpatialGraph
        Contact-based adjacency graph
    
    Examples
    --------
    >>> from spatioloji_s.spatial.polygon.graph import build_contact_graph
    >>> graph = build_contact_graph(sp)
    >>> print(graph.summary())
    >>> neighbors = graph.get_neighbors('cell_001')
    """
    import geopandas as gpd
    
    print(f"\n[Polygon Graph] Building contact graph (predicate='{predicate}')...")
    
    gdf = _get_gdf(sp, coord_type=coord_type)
    cell_index = gdf.index
    n_cells = len(gdf)
    
    # Drop cells with missing/invalid geometry
    valid_mask = gdf.geometry.notna() & gdf.geometry.is_valid
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  ⚠ Dropping {n_invalid} cells with invalid/missing geometry")
        gdf = gdf[valid_mask]
        cell_index = gdf.index
        n_cells = len(gdf)
    
    # Use spatial join for efficient pairwise detection
    # sjoin finds all pairs satisfying the predicate
    joined = gpd.sjoin(
        gdf[['geometry']],
        gdf[['geometry']],
        how='inner',
        predicate=predicate
    )
    
    # Remove self-joins
    joined = joined[joined.index != joined['index_right']]
    
    # Map cell IDs to integer positions
    id_to_pos = {cid: i for i, cid in enumerate(cell_index)}
    
    rows = np.array([id_to_pos[cid] for cid in joined.index])
    cols = np.array([id_to_pos[cid] for cid in joined['index_right']])
    
    # Build sparse adjacency (symmetric)
    data = np.ones(len(rows), dtype=np.float32)
    adjacency = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_cells, n_cells)
    )
    
    # Distance matrix: contacting cells have distance = 0
    distances = sparse.csr_matrix((n_cells, n_cells), dtype=np.float32)
    
    # Build graph
    graph = PolygonSpatialGraph(
        adjacency=adjacency,
        distances=distances,
        cell_index=cell_index,
        method='contact',
        params={'coord_type': coord_type, 'predicate': predicate}
    )
    
    print(f"  ✓ Contact graph: {graph.n_cells} cells, {graph.n_edges} edges")
    print(f"    Mean degree: {graph.mean_degree:.1f}")
    
    return graph

def build_buffer_graph(sp: 'spatioloji',
                       buffer_distance: float,
                       coord_type: str = 'global') -> PolygonSpatialGraph:
    """
    Build adjacency graph by buffering polygons.
    
    Expands each polygon by buffer_distance, then finds overlapping pairs.
    Stores actual polygon-to-polygon distances (not centroid distances).
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    buffer_distance : float
        Distance to expand each polygon (in coordinate units)
    coord_type : str
        'global' or 'local' coordinates
    
    Returns
    -------
    PolygonSpatialGraph
        Buffer-based adjacency graph with true polygon distances
    
    Examples
    --------
    >>> # Find cells within 50 units of each other
    >>> graph = build_buffer_graph(sp, buffer_distance=50.0)
    >>> print(graph.summary())
    >>> 
    >>> # Get actual polygon distance between two cells
    >>> dist = graph.get_distance('cell_001', 'cell_002')
    """
    import geopandas as gpd
    
    print(f"\n[Polygon Graph] Building buffer graph (buffer={buffer_distance})...")
    
    gdf = _get_gdf(sp, coord_type=coord_type)
    cell_index = gdf.index
    n_cells = len(gdf)
    
    # Drop invalid geometries
    valid_mask = gdf.geometry.notna() & gdf.geometry.is_valid
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  ⚠ Dropping {n_invalid} cells with invalid/missing geometry")
        gdf = gdf[valid_mask]
        cell_index = gdf.index
        n_cells = len(gdf)
    
    # Create buffered version for neighbor detection
    gdf_buffered = gdf.copy()
    gdf_buffered['geometry'] = gdf.geometry.buffer(buffer_distance)
    
    # Spatial join: find pairs where buffered polygons overlap
    joined = gpd.sjoin(
        gdf_buffered[['geometry']],
        gdf[['geometry']],
        how='inner',
        predicate='intersects'
    )
    
    # Remove self-joins
    joined = joined[joined.index != joined['index_right']]
    
    # Remove duplicate pairs (keep only one direction for distance calc)
    pairs = pd.DataFrame({
        'cell_a': joined.index,
        'cell_b': joined['index_right']
    })
    pairs = pairs[pairs['cell_a'] < pairs['cell_b']].drop_duplicates()
    
    # Compute actual polygon-to-polygon distances for each pair
    print(f"  → Computing polygon distances for {len(pairs)} pairs...")
    
    geom = gdf.geometry
    dist_values = np.array([
        geom.loc[row.cell_a].distance(geom.loc[row.cell_b])
        for row in pairs.itertuples()
    ], dtype=np.float32)
    
    # Map cell IDs to integer positions
    id_to_pos = {cid: i for i, cid in enumerate(cell_index)}
    
    rows_a = np.array([id_to_pos[cid] for cid in pairs['cell_a']])
    cols_b = np.array([id_to_pos[cid] for cid in pairs['cell_b']])
    
    # Build symmetric sparse matrices
    all_rows = np.concatenate([rows_a, cols_b])
    all_cols = np.concatenate([cols_b, rows_a])
    all_dists = np.concatenate([dist_values, dist_values])
    all_ones = np.ones(len(all_rows), dtype=np.float32)
    
    adjacency = sparse.csr_matrix(
        (all_ones, (all_rows, all_cols)), shape=(n_cells, n_cells)
    )
    
    distances = sparse.csr_matrix(
        (all_dists, (all_rows, all_cols)), shape=(n_cells, n_cells)
    )
    
    graph = PolygonSpatialGraph(
        adjacency=adjacency,
        distances=distances,
        cell_index=cell_index,
        method='buffer',
        params={
            'coord_type': coord_type,
            'buffer_distance': buffer_distance
        }
    )
    
    print(f"  ✓ Buffer graph: {graph.n_cells} cells, {graph.n_edges} edges")
    print(f"    Mean degree: {graph.mean_degree:.1f}")
    print(f"    Distance range: {dist_values.min():.1f} – {dist_values.max():.1f}")
    
    return graph

def build_knn_graph(sp: 'spatioloji',
                    k: int = 6,
                    coord_type: str = 'global',
                    candidate_multiplier: int = 3) -> PolygonSpatialGraph:
    """
    Build k-nearest neighbor graph using polygon-to-polygon distances.
    
    Strategy:
    1. Use centroid KNN to find candidate neighbors (k * multiplier)
    2. Compute exact polygon-to-polygon distance for each candidate pair
    3. Keep the k nearest by true polygon distance
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    k : int
        Number of nearest neighbors per cell
    coord_type : str
        'global' or 'local' coordinates
    candidate_multiplier : int
        How many candidates to consider per cell (k * multiplier).
        Higher = more accurate but slower.
    
    Returns
    -------
    PolygonSpatialGraph
        KNN graph based on polygon-to-polygon distances
    
    Examples
    --------
    >>> graph = build_knn_graph(sp, k=6)
    >>> print(graph.summary())
    >>> 
    >>> # Edges are NOT symmetric — cell A's 6 nearest may differ from cell B's
    >>> # Use symmetrize=True if you need undirected graph
    """
    from scipy.spatial import cKDTree
    
    print(f"\n[Polygon Graph] Building KNN graph (k={k})...")
    
    gdf = _get_gdf(sp, coord_type=coord_type)
    cell_index = gdf.index
    n_cells = len(gdf)
    
    # Drop invalid geometries
    valid_mask = gdf.geometry.notna() & gdf.geometry.is_valid
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  ⚠ Dropping {n_invalid} cells with invalid/missing geometry")
        gdf = gdf[valid_mask]
        cell_index = gdf.index
        n_cells = len(gdf)
    
    if k >= n_cells:
        raise ValueError(f"k={k} must be less than n_cells={n_cells}")
    
    # Step 1: Centroid-based candidate search (fast)
    n_candidates = min(k * candidate_multiplier, n_cells - 1)
    
    centroids = np.column_stack([
        gdf.geometry.centroid.x,
        gdf.geometry.centroid.y
    ])
    
    print(f"  → Finding {n_candidates} candidates per cell via centroid KDTree...")
    tree = cKDTree(centroids)
    # k+1 because query includes self
    _, candidate_indices = tree.query(centroids, k=n_candidates + 1)
    
    # Step 2: Compute exact polygon distances for candidates
    print(f"  → Computing exact polygon distances for {n_cells} × {n_candidates} pairs...")
    
    geom = gdf.geometry.values  # array of shapely geometries
    
    rows_list = []
    cols_list = []
    dist_list = []
    
    for i in range(n_cells):
        # Candidate indices (skip self at position 0)
        candidates = candidate_indices[i, 1:]
        
        # Compute polygon-to-polygon distances
        poly_i = geom[i]
        cand_dists = np.array([
            poly_i.distance(geom[j]) for j in candidates
        ])
        
        # Keep k nearest by true polygon distance
        nearest_order = np.argsort(cand_dists)[:k]
        nearest_indices = candidates[nearest_order]
        nearest_dists = cand_dists[nearest_order]
        
        rows_list.append(np.full(k, i, dtype=np.int32))
        cols_list.append(nearest_indices.astype(np.int32))
        dist_list.append(nearest_dists.astype(np.float32))
    
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    dists = np.concatenate(dist_list)
    
    # Step 3: Build sparse matrices (directed KNN → symmetrize)
    ones = np.ones(len(rows), dtype=np.float32)
    
    # Directed adjacency
    adj_directed = sparse.csr_matrix(
        (ones, (rows, cols)), shape=(n_cells, n_cells)
    )
    dist_directed = sparse.csr_matrix(
        (dists, (rows, cols)), shape=(n_cells, n_cells)
    )
    
    # Symmetrize: if A→B or B→A, both are neighbors
    # For distances, keep the minimum when both directions exist
    adjacency = (adj_directed + adj_directed.T).astype(bool).astype(np.float32)
    adjacency = sparse.csr_matrix(adjacency)
    
    # Symmetrize distances: take element-wise max of directed versions
    # (both entries should be identical, but handle missing direction)
    dist_sym_a = dist_directed.copy()
    dist_sym_b = dist_directed.T.copy()
    # Where only one direction exists, use that value
    distances = dist_sym_a.maximum(dist_sym_b)
    distances = sparse.csr_matrix(distances)
    
    graph = PolygonSpatialGraph(
        adjacency=adjacency,
        distances=distances,
        cell_index=cell_index,
        method='knn',
        params={
            'coord_type': coord_type,
            'k': k,
            'candidate_multiplier': candidate_multiplier
        }
    )
    
    print(f"  ✓ KNN graph: {graph.n_cells} cells, {graph.n_edges} edges")
    print(f"    Mean degree: {graph.mean_degree:.1f} (≥{k} due to symmetrization)")
    
    return graph

