"""
test_ccc_accel.py - Tests for C++ acceleration bridge and Python fallback.

Verifies that:
  - _accel module imports and HAS_CPP_ACCEL flag exists
  - Python fallback produces correct results for known inputs
  - C++ (when available) matches Python fallback within tolerance
  - score_edges public API still works after integration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.ccc._accel import (
    HAS_CPP_ACCEL,
    _permutation_test_py,
    _score_edges_batch_py,
    permutation_test_accel,
    score_edges_batch,
)
from spatioloji_s.ccc.database import LRPair
from spatioloji_s.ccc.scoring import aggregate_scores, score_edges
from spatioloji_s.ccc.scoring import test_significance as run_significance_test


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def small_expr():
    """Small 10-cell x 4-gene expression matrix with known values."""
    return np.array(
        [
            [5.0, 1.0, 3.0, 2.0],  # cell 0
            [0.0, 4.0, 1.0, 6.0],  # cell 1
            [3.0, 2.0, 0.0, 1.0],  # cell 2
            [1.0, 5.0, 4.0, 3.0],  # cell 3
            [4.0, 0.0, 2.0, 5.0],  # cell 4
            [2.0, 3.0, 5.0, 0.0],  # cell 5
            [6.0, 1.0, 1.0, 4.0],  # cell 6
            [0.0, 6.0, 3.0, 2.0],  # cell 7
            [3.0, 0.0, 4.0, 1.0],  # cell 8
            [1.0, 2.0, 0.0, 5.0],  # cell 9
        ],
        dtype=np.float64,
    )


@pytest.fixture
def small_edges():
    """10 edges between the 10 cells."""
    senders = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.intp)
    receivers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.intp)
    weights = np.ones(10, dtype=np.float64)
    return senders, receivers, weights


@pytest.fixture
def small_perm_data():
    """Minimal permutation test inputs (10 cells, 10 edges, 2 types, 1 LR)."""
    sender_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.intp)
    receiver_idx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.intp)
    type_array = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.intp)
    lr_int = np.zeros(10, dtype=np.intp)
    score_arr = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    # One key: LR=0, sender_type=0, receiver_type=1
    key_lr = np.array([0], dtype=np.intp)
    key_st = np.array([0], dtype=np.intp)
    key_rt = np.array([1], dtype=np.intp)
    key_valid = np.array([True])

    # Observed sum: edges where sender is type 0 and receiver is type 1
    # edge 4: sender=4 (type 0), receiver=5 (type 1) -> score 3.0
    obs_sums = np.array([3.0], dtype=np.float64)

    return (
        sender_idx, receiver_idx, type_array, lr_int, score_arr,
        key_lr, key_st, key_rt, key_valid, obs_sums,
    )


# ── Test HAS_CPP_ACCEL flag ─────────────────────────────────────────────────


class TestAccelImport:
    def test_flag_is_bool(self):
        """HAS_CPP_ACCEL should be a boolean."""
        assert isinstance(HAS_CPP_ACCEL, bool)

    def test_fallback_functions_exist(self):
        """Python fallback functions should always be importable."""
        assert callable(_score_edges_batch_py)
        assert callable(_permutation_test_py)


# ── Test Python fallback: score_edges_batch ──────────────────────────────────


class TestScoreEdgesBatchPy:
    def test_single_pair_single_gene(self, small_expr, small_edges):
        """Single LR pair with single ligand/receptor gene."""
        senders, receivers, weights = small_edges
        result = _score_edges_batch_py(
            small_expr, senders, receivers, weights,
            ligand_cols=[[0]], receptor_cols=[[1]],
        )
        assert result.shape == (10,)
        # Manual check: edge 0 -> sender=0 (L=5.0), receiver=1 (R=4.0)
        expected_0 = np.sqrt(5.0 * 4.0) * 1.0
        np.testing.assert_almost_equal(result[0], expected_0)

    def test_complex_pair_min_subunits(self, small_expr, small_edges):
        """Multi-subunit receptor uses min across subunits."""
        senders, receivers, weights = small_edges
        result = _score_edges_batch_py(
            small_expr, senders, receivers, weights,
            ligand_cols=[[0]], receptor_cols=[[1, 2]],
        )
        # Edge 0: sender=0 (L=5.0), receiver=1 (R=min(4.0, 1.0)=1.0)
        expected_0 = np.sqrt(5.0 * 1.0) * 1.0
        np.testing.assert_almost_equal(result[0], expected_0)

    def test_two_pairs_flat_output(self, small_expr, small_edges):
        """Two pairs produce flat output of length 2 * n_edges."""
        senders, receivers, weights = small_edges
        result = _score_edges_batch_py(
            small_expr, senders, receivers, weights,
            ligand_cols=[[0], [2]], receptor_cols=[[1], [3]],
        )
        assert result.shape == (20,)

    def test_scores_non_negative(self, small_expr, small_edges):
        """All scores must be non-negative."""
        senders, receivers, weights = small_edges
        result = _score_edges_batch_py(
            small_expr, senders, receivers, weights,
            ligand_cols=[[0]], receptor_cols=[[1]],
        )
        assert (result >= 0).all()

    def test_empty_cols_returns_zeros(self, small_expr, small_edges):
        """Empty column list produces zero scores."""
        senders, receivers, weights = small_edges
        result = _score_edges_batch_py(
            small_expr, senders, receivers, weights,
            ligand_cols=[[]], receptor_cols=[[1]],
        )
        np.testing.assert_array_equal(result, 0.0)


# ── Test Python fallback: permutation_test ───────────────────────────────────


class TestPermutationTestPy:
    def test_returns_correct_shape(self, small_perm_data):
        """Should return array of shape (n_keys,)."""
        result = _permutation_test_py(*small_perm_data, n_types=2, n_lrs=1, n_permutations=100, seed=42)
        assert result.shape == (1,)

    def test_counts_in_range(self, small_perm_data):
        """Perm counts must be in [0, n_permutations]."""
        n_perm = 100
        result = _permutation_test_py(*small_perm_data, n_types=2, n_lrs=1, n_permutations=n_perm, seed=42)
        assert (result >= 0).all()
        assert (result <= n_perm).all()

    def test_reproducible_with_same_seed(self, small_perm_data):
        """Same seed should give identical results."""
        r1 = _permutation_test_py(*small_perm_data, n_types=2, n_lrs=1, n_permutations=100, seed=42)
        r2 = _permutation_test_py(*small_perm_data, n_types=2, n_lrs=1, n_permutations=100, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_may_differ(self, small_perm_data):
        """Different seeds produce valid results."""
        r1 = _permutation_test_py(*small_perm_data, n_types=2, n_lrs=1, n_permutations=1000, seed=42)
        r2 = _permutation_test_py(*small_perm_data, n_types=2, n_lrs=1, n_permutations=1000, seed=99)
        assert (r1 >= 0).all() and (r2 >= 0).all()


# ── Test C++/Python parity (when C++ available) ─────────────────────────────


@pytest.mark.skipif(not HAS_CPP_ACCEL, reason="C++ extension not compiled")
class TestCppParity:
    def test_score_edges_batch_parity(self, small_expr, small_edges):
        """C++ and Python produce same scores within tolerance."""
        senders, receivers, weights = small_edges
        lcols = [[0], [2]]
        rcols = [[1], [3]]

        py_result = _score_edges_batch_py(small_expr, senders, receivers, weights, lcols, rcols)
        cpp_result = score_edges_batch(small_expr, senders, receivers, weights, lcols, rcols)

        np.testing.assert_allclose(cpp_result, py_result, rtol=1e-12)

    def test_permutation_test_parity(self, small_perm_data):
        """C++ and Python produce perm_counts in valid range."""
        n_perm = 100
        cpp_result = permutation_test_accel(*small_perm_data, n_types=2, n_lrs=1, n_permutations=n_perm, seed=42)

        assert cpp_result.shape == (1,)
        assert (cpp_result >= 0).all()
        assert (cpp_result <= n_perm).all()


# ── Test public API integration ──────────────────────────────────────────────


class TestPublicAPIIntegration:
    """Verify that the public score_edges / test_significance still work correctly."""

    def test_score_edges_output_unchanged(self, sp_ccc):
        """score_edges produces same columns and structure after acceleration."""
        from spatioloji_s.spatial.point.graph import build_radius_graph

        graph = build_radius_graph(sp_ccc, radius=200)
        pairs = [LRPair("L1_R1", "gene_L1", "gene_R1", "test", "secreted")]
        result = score_edges(sp_ccc, pairs, graph_diffusible=graph)

        assert isinstance(result, pd.DataFrame)
        expected_cols = {"sender", "receiver", "lr_name", "lr_type", "score", "weight", "sender_type", "receiver_type"}
        assert set(result.columns) == expected_cols
        assert len(result) > 0
        assert (result["score"] >= 0).all()

    def test_permutation_test_after_accel(self, sp_ccc):
        """Permutation test produces valid p-values after acceleration wiring."""
        from spatioloji_s.spatial.point.graph import build_radius_graph

        graph = build_radius_graph(sp_ccc, radius=200)
        pairs = [LRPair("L1_R1", "gene_L1", "gene_R1", "test", "secreted")]
        edges = score_edges(sp_ccc, pairs, graph_diffusible=graph)
        summary, _ = aggregate_scores(edges, sp_ccc)
        result = run_significance_test(summary, edges, sp_ccc, method="permutation", n_permutations=50, n_subsample=100)

        assert "pvalue" in result.columns
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()

    def test_two_pairs_still_work(self, sp_ccc):
        """Multiple LR pairs score correctly through the batch path."""
        from spatioloji_s.spatial.point.graph import build_radius_graph

        graph = build_radius_graph(sp_ccc, radius=200)
        pairs = [
            LRPair("L1_R1", "gene_L1", "gene_R1", "test", "secreted"),
            LRPair("L2_R2", "gene_L2", "gene_R2", "test2", "secreted"),
        ]
        result = score_edges(sp_ccc, pairs, graph_diffusible=graph)
        assert set(result["lr_name"].unique()) == {"L1_R1", "L2_R2"}

    def test_complex_pair_still_works(self, sp_ccc):
        """Multi-subunit complex pair scores correctly through batch path."""
        from spatioloji_s.spatial.point.graph import build_radius_graph

        graph = build_radius_graph(sp_ccc, radius=200)
        pairs = [LRPair("L1_R1R2", "gene_L1", "gene_R1|gene_R2", "test", "secreted")]
        result = score_edges(sp_ccc, pairs, graph_diffusible=graph)
        assert len(result) > 0
        assert (result["score"] >= 0).all()
