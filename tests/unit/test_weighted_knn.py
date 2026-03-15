"""Tests for build_weighted_knn_graph."""

import numpy as np
import pytest

from spatioloji_s.spatial.point.graph import PointSpatialGraph, build_weighted_knn_graph


class TestBuildWeightedKnnGraph:
    """Core functionality of distance-weighted KNN graph."""

    def test_returns_point_spatial_graph(self, sp_basic):
        g = build_weighted_knn_graph(sp_basic, k=5)
        assert isinstance(g, PointSpatialGraph)

    def test_method_and_params(self, sp_basic):
        g = build_weighted_knn_graph(sp_basic, k=5, kernel="gaussian")
        assert g.method == "weighted_knn"
        assert g.params["k"] == 5
        assert g.params["kernel"] == "gaussian"
        assert "bandwidth" in g.params

    def test_weights_between_zero_and_one(self, sp_basic):
        """Gaussian and inverse kernels should produce weights in (0, 1]."""
        for kernel in ("gaussian", "inverse"):
            g = build_weighted_knn_graph(sp_basic, k=5, kernel=kernel)
            w = g.adjacency.data
            assert np.all(w > 0), f"{kernel}: expected all weights > 0"
            assert np.all(w <= 1.0), f"{kernel}: expected all weights <= 1"

    def test_bisquare_compact_support(self, sp_basic):
        """Bisquare with small bandwidth should zero-out distant edges."""
        g = build_weighted_knn_graph(sp_basic, k=5, kernel="bisquare", bandwidth=1.0)
        # With bandwidth=1 and coords spread over 1000x1000,
        # almost all edges should be pruned
        assert g.adjacency.nnz < build_weighted_knn_graph(sp_basic, k=5, kernel="gaussian").adjacency.nnz

    def test_distances_preserved(self, sp_basic):
        """Raw distances should be stored unchanged regardless of kernel."""
        g = build_weighted_knn_graph(sp_basic, k=5, kernel="gaussian")
        assert g.distances.nnz > 0
        assert np.all(g.distances.data > 0)

    def test_symmetry(self, sp_basic):
        """Both adjacency and distances should be symmetric."""
        g = build_weighted_knn_graph(sp_basic, k=5)
        adj_diff = abs(g.adjacency - g.adjacency.T).sum()
        dist_diff = abs(g.distances - g.distances.T).sum()
        assert adj_diff < 1e-6, "adjacency not symmetric"
        assert dist_diff < 1e-6, "distances not symmetric"

    def test_cell_ids_match(self, sp_basic):
        g = build_weighted_knn_graph(sp_basic, k=5)
        assert list(g.cell_ids) == list(sp_basic.cell_index)

    def test_adaptive_bandwidth(self, sp_basic):
        """When bandwidth=None, should auto-set to median neighbor distance."""
        g = build_weighted_knn_graph(sp_basic, k=5, bandwidth=None)
        assert g.params["bandwidth"] > 0

    def test_fixed_bandwidth(self, sp_basic):
        g = build_weighted_knn_graph(sp_basic, k=5, bandwidth=100.0)
        assert g.params["bandwidth"] == 100.0

    def test_invalid_kernel_raises(self, sp_basic):
        with pytest.raises(ValueError, match="Unknown kernel"):
            build_weighted_knn_graph(sp_basic, k=5, kernel="invalid")

    def test_closer_neighbors_higher_weight(self, sp_basic):
        """Closer cells should get higher weight than farther ones."""
        g = build_weighted_knn_graph(sp_basic, k=10, kernel="gaussian")
        # Pick a cell and check weight-distance correlation
        row = g.adjacency[0].toarray().flatten()
        dist_row = g.distances[0].toarray().flatten()
        nonzero = row > 0
        if nonzero.sum() >= 2:
            weights = row[nonzero]
            dists = dist_row[nonzero]
            # Spearman correlation: weights should decrease with distance
            correlation = np.corrcoef(weights, dists)[0, 1]
            assert correlation < 0, "weights should be negatively correlated with distance"

    def test_coord_type_local(self, sp_basic):
        g = build_weighted_knn_graph(sp_basic, k=5, coord_type="local")
        assert g.coord_type == "local"
        assert g.n_cells == sp_basic.n_cells
