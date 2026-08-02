"""
graph.py - Spatial graph construction from point coordinates

Builds cell-cell graphs using centroids/coordinates from the spatioloji object.
These graphs are the foundation for neighborhood analysis, spatial statistics,
and pattern detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


@dataclass
class PointSpatialGraph:
    """
    Container for a spatial graph built from point coordinates.

    Attributes
    ----------
    adjacency : sparse.csr_matrix
        Binary adjacency matrix (n_cells x n_cells).
    distances : sparse.csr_matrix
        Distance matrix (same sparsity as adjacency).
    cell_ids : pd.Index
        Cell IDs matching matrix rows/columns.
    method : str
        Construction method ('knn', 'radius', 'delaunay').
    params : dict
        Parameters used (e.g., {'k': 10}).
    coord_type : str
        Coordinate system used ('global' or 'local').
    """

    adjacency: sparse.csr_matrix
    distances: sparse.csr_matrix
    cell_ids: pd.Index
    method: str
    params: dict
    coord_type: str

    @property
    def n_cells(self) -> int:
        return self.adjacency.shape[0]

    @property
    def n_edges(self) -> int:
        """Edge count.  For symmetric graphs divides by 2 (undirected)."""
        is_sym = self.params.get("symmetrize", True) or self.method in ("radius", "delaunay")
        return self.adjacency.nnz // 2 if is_sym else self.adjacency.nnz

    @property
    def mean_degree(self) -> float:
        degrees = np.array(self.adjacency.sum(axis=1)).flatten()
        return degrees.mean()

    def get_neighbors(self, cell_id: str) -> pd.Index:
        """Get neighbor cell IDs for a given cell."""
        idx = self.cell_ids.get_loc(cell_id)
        neighbor_indices = self.adjacency[idx].nonzero()[1]
        return self.cell_ids[neighbor_indices]

    def get_neighbor_distances(self, cell_id: str) -> pd.Series:
        """Get distances to neighbors for a given cell."""
        idx = self.cell_ids.get_loc(cell_id)
        row = self.distances[idx]
        neighbor_indices = row.nonzero()[1]
        dists = np.array(row[0, neighbor_indices]).flatten()
        return pd.Series(dists, index=self.cell_ids[neighbor_indices])

    def degree_series(self) -> pd.Series:
        """Degree (neighbor count) for every cell."""
        degrees = np.array(self.adjacency.sum(axis=1)).flatten().astype(int)
        return pd.Series(degrees, index=self.cell_ids, name="degree")

    def summary(self) -> dict:
        degrees = np.array(self.adjacency.sum(axis=1)).flatten()
        dists = self.distances.data
        return {
            "method": self.method,
            "params": self.params,
            "coord_type": self.coord_type,
            "n_cells": self.n_cells,
            "n_edges": self.n_edges,
            "mean_degree": degrees.mean(),
            "min_degree": int(degrees.min()),
            "max_degree": int(degrees.max()),
            "mean_edge_distance": dists.mean() if len(dists) > 0 else 0,
            "max_edge_distance": dists.max() if len(dists) > 0 else 0,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"PointSpatialGraph (method={s['method']}, "
            f"{s['n_cells']} cells, {s['n_edges']} edges, "
            f"mean degree={s['mean_degree']:.1f})"
        )


def _symmetrize_sparse(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    """Symmetrize a sparse matrix: union of A→B and B→A, averaging shared entries."""
    sym = (mat + mat.T) / 2
    mask_one_way = (mat != 0).astype(float) - (mat.T != 0).astype(float)
    sym = sym + mat.multiply(mask_one_way > 0) / 2 + mat.T.multiply(mask_one_way < 0) / 2
    return sym.tocsr()


def build_knn_graph(
    sj: spatioloji,
    k: int = 10,
    coord_type: str = "global",
    symmetrize: bool = False,
) -> PointSpatialGraph:
    """
    Build K-nearest neighbors graph.

    Each cell connects to its k closest neighbors.  By default the
    graph is **directed** (asymmetric): A→B does not imply B→A.
    This preserves a fixed degree of k per cell and is standard for
    spatial statistics (Moran's I, Gi*).

    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    k : int
        Number of neighbors.
    coord_type : str
        'global' or 'local'.
    symmetrize : bool
        If True, symmetrize edges (union of A→B and B→A, averaging
        distances).  Increases degree above k for some cells.
        Default False (directed, fixed degree = k).

    Returns
    -------
    PointSpatialGraph
    """
    coords = sj.get_spatial_coords(coord_type=coord_type)  # (n_cells, 2)
    n_cells = coords.shape[0]

    # k+1 because sklearn includes the query point itself
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_cells), metric="euclidean")
    nn.fit(coords)
    dist_matrix, idx_matrix = nn.kneighbors(coords)

    # Build sparse distance matrix (directed)
    rows = np.repeat(np.arange(n_cells), k)
    cols = idx_matrix[:, 1:].flatten()  # skip self (column 0)
    dists = dist_matrix[:, 1:].flatten()

    dist_out = sparse.csr_matrix((dists, (rows, cols)), shape=(n_cells, n_cells))

    if symmetrize:
        dist_out = _symmetrize_sparse(dist_out)

    adjacency = (dist_out != 0).astype(np.float32)
    n_edges = adjacency.nnz if not symmetrize else adjacency.nnz // 2
    mean_deg = np.array(adjacency.sum(1)).mean()
    sym_label = "symmetric" if symmetrize else "directed"

    print(f"  ✓ KNN graph: k={k}, {n_edges} edges ({sym_label}), mean degree={mean_deg:.1f}")

    return PointSpatialGraph(
        adjacency=adjacency,
        distances=dist_out.tocsr(),
        cell_ids=sj.cell_index,
        method="knn",
        params={"k": k, "symmetrize": symmetrize},
        coord_type=coord_type,
    )


def build_weighted_knn_graph(
    sj: spatioloji,
    k: int = 10,
    coord_type: str = "global",
    kernel: str = "gaussian",
    bandwidth: float | None = None,
    symmetrize: bool = False,
) -> PointSpatialGraph:
    """
    Build K-nearest neighbors graph with distance-based edge weights.

    Like :func:`build_knn_graph`, but the adjacency matrix stores
    continuous weights instead of binary 1/0.  Closer cells receive
    higher weight, which improves spatial statistics (Moran's I,
    Getis-Ord Gi*, neighborhood enrichment) by letting proximity
    modulate influence.

    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    k : int
        Number of neighbors, by default 10.
    coord_type : str
        'global' or 'local', by default 'global'.
    kernel : {'gaussian', 'bisquare', 'inverse'}
        Weight function applied to distances.

        - ``'gaussian'``  — ``exp(-d² / (2 * bw²))``.  Smooth decay,
          most common in spatial statistics.
        - ``'bisquare'``  — ``(1 - (d/bw)²)²`` for ``d < bw``, else 0.
          Compact support (hard cutoff at bandwidth).
        - ``'inverse'``   — ``1 / (1 + d / bw)``.  Gentle decay.
    bandwidth : float or None
        Kernel bandwidth.  Controls how fast weight decays with distance.
        If None (default), set to the median of all neighbor distances
        (adaptive bandwidth).
    symmetrize : bool
        If True, symmetrize edges before applying kernel weights.
        Default False (directed, fixed degree = k).

    Returns
    -------
    PointSpatialGraph
        ``adjacency`` contains the kernel weights (not binary).
        ``distances`` contains raw Euclidean distances.

    Examples
    --------
    >>> # Gaussian-weighted graph (adaptive bandwidth)
    >>> g = sj.spatial.point.build_weighted_knn_graph(sp, k=15)
    >>>
    >>> # Fixed bandwidth, bisquare kernel
    >>> g = sj.spatial.point.build_weighted_knn_graph(
    ...     sp, k=10, kernel='bisquare', bandwidth=50.0)
    """
    coords = sj.get_spatial_coords(coord_type=coord_type)
    n_cells = coords.shape[0]

    nn = NearestNeighbors(n_neighbors=min(k + 1, n_cells), metric="euclidean")
    nn.fit(coords)
    dist_matrix, idx_matrix = nn.kneighbors(coords)

    # Build sparse distance matrix (directed)
    rows = np.repeat(np.arange(n_cells), k)
    cols = idx_matrix[:, 1:].flatten()
    dists = dist_matrix[:, 1:].flatten()

    dist_out = sparse.csr_matrix((dists, (rows, cols)), shape=(n_cells, n_cells))

    if symmetrize:
        dist_out = _symmetrize_sparse(dist_out)

    # Resolve bandwidth
    all_dists = dist_out.data
    if bandwidth is None:
        bandwidth = float(np.median(all_dists))
        print(f"  Adaptive bandwidth: {bandwidth:.2f} (median neighbor distance)")

    # Compute kernel weights
    if kernel == "gaussian":
        weight_data = np.exp(-(all_dists**2) / (2 * bandwidth**2))
    elif kernel == "bisquare":
        ratio = all_dists / bandwidth
        weight_data = np.where(ratio < 1.0, (1 - ratio**2) ** 2, 0.0)
    elif kernel == "inverse":
        weight_data = 1.0 / (1.0 + all_dists / bandwidth)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from: gaussian, bisquare, inverse")

    # Build weighted adjacency
    adjacency = dist_out.copy()
    adjacency.data = weight_data.astype(np.float32)
    adjacency.eliminate_zeros()

    n_edges = adjacency.nnz if not symmetrize else adjacency.nnz // 2
    mean_deg = np.array((adjacency != 0).astype(float).sum(1)).mean()
    mean_weight = float(adjacency.data.mean()) if adjacency.nnz > 0 else 0.0
    sym_label = "symmetric" if symmetrize else "directed"

    print(
        f"  ✓ Weighted KNN graph: k={k}, kernel={kernel}, bw={bandwidth:.2f}, "
        f"{n_edges} edges ({sym_label}), mean degree={mean_deg:.1f}, mean weight={mean_weight:.3f}"
    )

    return PointSpatialGraph(
        adjacency=adjacency.tocsr(),
        distances=dist_out.tocsr(),
        cell_ids=sj.cell_index,
        method="weighted_knn",
        params={"k": k, "kernel": kernel, "bandwidth": bandwidth, "symmetrize": symmetrize},
        coord_type=coord_type,
    )


def build_radius_graph(
    sj: spatioloji,
    radius: float,
    coord_type: str = "global",
) -> PointSpatialGraph:
    """
    Build distance-threshold graph.

    Connects all cell pairs within a given radius. This naturally
    reflects biological interaction distances (e.g., paracrine range).

    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    radius : float
        Maximum distance to connect two cells (in coordinate units).
    coord_type : str
        'global' or 'local'.

    Returns
    -------
    PointSpatialGraph
    """
    coords = sj.get_spatial_coords(coord_type=coord_type)

    nn = NearestNeighbors(radius=radius, metric="euclidean")
    nn.fit(coords)
    dist_sparse = nn.radius_neighbors_graph(coords, radius, mode="distance")

    # Symmetrize using element-wise max (keeps the distance for
    # entries that appear in only one direction, averages where both
    # exist).  Pure sparse operations — no dense intermediates.
    dist_t = dist_sparse.T.tocsr()
    # Element-wise max gives a symmetric matrix with no zeros dropped
    dist_sym = dist_sparse.maximum(dist_t)

    adjacency = (dist_sym != 0).astype(np.float32)
    mean_deg = np.array(adjacency.sum(1)).mean()

    # Warn if graph is very dense
    if mean_deg > 100:
        print(f"  ⚠ Very dense graph (mean degree={mean_deg:.0f}). Consider a smaller radius.")

    print(f"  ✓ Radius graph: r={radius}, {adjacency.nnz // 2} edges, mean degree={mean_deg:.1f}")

    return PointSpatialGraph(
        adjacency=adjacency,
        distances=dist_sym.tocsr(),
        cell_ids=sj.cell_index,
        method="radius",
        params={"radius": radius},
        coord_type=coord_type,
    )


def build_delaunay_graph(
    sj: spatioloji,
    coord_type: str = "global",
    max_edge_length: float | None = None,
) -> PointSpatialGraph:
    """
    Build Delaunay triangulation graph.

    Connects cells that are natural geometric neighbors. Adapts to
    local density without requiring k or radius parameters.

    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    coord_type : str
        'global' or 'local'.
    max_edge_length : float, optional
        Prune edges longer than this. Removes spurious long-range
        connections in sparse regions.

    Returns
    -------
    PointSpatialGraph
    """
    coords = sj.get_spatial_coords(coord_type=coord_type)
    n_cells = coords.shape[0]

    if n_cells < 3:
        raise ValueError("Need at least 3 cells for Delaunay triangulation")

    # Build triangulation
    tri = Delaunay(coords)

    # Extract edges from triangles
    # Each simplex (triangle) has 3 vertices -> 3 edges
    simplices = tri.simplices  # (n_triangles, 3)
    edges_set = set()
    for s in simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(s[i], s[j]), max(s[i], s[j]))
                edges_set.add(edge)

    edges = np.array(list(edges_set))  # (n_edges, 2)
    rows = edges[:, 0]
    cols = edges[:, 1]

    # Compute edge distances
    dists = np.sqrt((coords[rows, 0] - coords[cols, 0]) ** 2 + (coords[rows, 1] - coords[cols, 1]) ** 2)

    # Optional: prune long edges
    params = {}
    if max_edge_length is not None:
        keep = dists <= max_edge_length
        rows = rows[keep]
        cols = cols[keep]
        dists = dists[keep]
        n_pruned = (~keep).sum()
        params["max_edge_length"] = max_edge_length
        print(f"  → Pruned {n_pruned} edges longer than {max_edge_length}")

    # Build symmetric sparse matrices (add both directions)
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_dists = np.concatenate([dists, dists])

    dist_sparse = sparse.csr_matrix((all_dists, (all_rows, all_cols)), shape=(n_cells, n_cells))
    adjacency = (dist_sparse != 0).astype(np.float32)
    mean_deg = np.array(adjacency.sum(1)).mean()

    print(f"  ✓ Delaunay graph: {adjacency.nnz // 2} edges, mean degree={mean_deg:.1f}")

    return PointSpatialGraph(
        adjacency=adjacency,
        distances=dist_sparse,
        cell_ids=sj.cell_index,
        method="delaunay",
        params=params,
        coord_type=coord_type,
    )
