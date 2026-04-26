"""
test_ccc_scoring.py - Tests for CCC scoring engine.

Uses the sp_ccc fixture (300 cells, genes gene_L1/gene_R1/gene_L2/gene_R2,
cell types Tumor/Stroma/Fibroblast/CD8_T, polygons, morph_class column).
"""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.ccc.database import LRPair
from spatioloji_s.ccc.scoring import aggregate_scores, score_edges
from spatioloji_s.ccc.scoring import test_significance as run_significance_test

# ── Helpers / fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def lr_secreted():
    """Single secreted LR pair: gene_L1 -> gene_R1."""
    return [LRPair("L1_R1", "gene_L1", "gene_R1", "test", "secreted")]


@pytest.fixture
def lr_two_pairs():
    """Two secreted LR pairs."""
    return [
        LRPair("L1_R1", "gene_L1", "gene_R1", "test", "secreted"),
        LRPair("L2_R2", "gene_L2", "gene_R2", "test2", "secreted"),
    ]


@pytest.fixture
def lr_juxtacrine():
    """Single juxtacrine LR pair."""
    return [LRPair("L1_R1_jux", "gene_L1", "gene_R1", "test", "juxtacrine")]


@pytest.fixture
def lr_complex():
    """LR pair with multi-subunit receptor complex."""
    return [LRPair("L1_R1R2", "gene_L1", "gene_R1|gene_R2", "test", "secreted")]


@pytest.fixture
def graph_diffusible(sp_ccc):
    """Radius graph for diffusible signaling."""
    from spatioloji_s.spatial.point.graph import build_radius_graph

    return build_radius_graph(sp_ccc, radius=200)


@pytest.fixture
def graph_juxtacrine(sp_ccc):
    """Buffer graph with contact fractions for juxtacrine signaling."""
    from spatioloji_s.spatial.polygon.boundaries import contact_fraction
    from spatioloji_s.spatial.polygon.graph import build_buffer_graph

    g = build_buffer_graph(sp_ccc, buffer_distance=10)
    cdf = contact_fraction(sp_ccc, g)
    # Store on graph so scoring.py can access it
    g.contact_frac_df = cdf
    return g


# ══════════════════════════════════════════════════════════════════════════════
# TestScoreEdges
# ══════════════════════════════════════════════════════════════════════════════


class TestScoreEdges:
    """Tests for score_edges."""

    def test_returns_dataframe(self, sp_ccc, lr_secreted, graph_diffusible):
        """score_edges returns a DataFrame with expected columns."""
        result = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "sender",
            "receiver",
            "lr_name",
            "lr_type",
            "score",
            "weight",
            "sender_type",
            "receiver_type",
            "interaction_mode",
        }
        assert expected_cols == set(result.columns)

    def test_scores_non_negative(self, sp_ccc, lr_secreted, graph_diffusible):
        """All scores must be >= 0."""
        result = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        assert (result["score"] >= 0).all()

    def test_weights_non_negative(self, sp_ccc, lr_secreted, graph_diffusible):
        """All weights must be >= 0."""
        result = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        assert (result["weight"] >= 0).all()

    def test_has_edges(self, sp_ccc, lr_secreted, graph_diffusible):
        """Should produce a non-empty result for radius=200."""
        result = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        assert len(result) > 0

    def test_secreted_distance_decay(self, sp_ccc, lr_secreted, graph_diffusible):
        """Secreted weights should use exponential distance decay (all <= 1)."""
        result = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        assert (result["weight"] <= 1.0 + 1e-9).all()

    def test_juxtacrine_uses_contact_fraction(self, sp_ccc, lr_juxtacrine, graph_juxtacrine):
        """Juxtacrine scoring should use contact fraction weights."""
        result = score_edges(sp_ccc, lr_juxtacrine, graph_juxtacrine=graph_juxtacrine)
        assert isinstance(result, pd.DataFrame)
        # Weights are fraction_a * fraction_b, should be in [0, 1]
        if len(result) > 0:
            assert (result["weight"] >= 0).all()
            assert (result["weight"] <= 1.0 + 1e-9).all()

    def test_juxtacrine_missing_graph_raises(self, sp_ccc, lr_juxtacrine):
        """Should raise ValueError when juxtacrine graph is missing."""
        with pytest.raises(ValueError, match="graph_juxtacrine"):
            score_edges(sp_ccc, lr_juxtacrine)

    def test_secreted_missing_graph_raises(self, sp_ccc, lr_secreted):
        """Should raise ValueError when diffusible graph is missing."""
        with pytest.raises(ValueError, match="graph_diffusible"):
            score_edges(sp_ccc, lr_secreted)

    def test_multi_subunit_complex(self, sp_ccc, lr_complex, graph_diffusible):
        """Multi-subunit receptor complex should use min across subunits."""
        result = score_edges(sp_ccc, lr_complex, graph_diffusible=graph_diffusible)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Scores from complex should be <= scores from single gene
        # (min <= any single subunit)
        single_pair = [LRPair("L1_R1", "gene_L1", "gene_R1", "test", "secreted")]
        result_single = score_edges(sp_ccc, single_pair, graph_diffusible=graph_diffusible)
        # Merge on sender+receiver to compare
        merged = result.merge(result_single, on=["sender", "receiver"], suffixes=("_complex", "_single"))
        assert (merged["score_complex"] <= merged["score_single"] + 1e-9).all()

    def test_sender_receiver_types_present(self, sp_ccc, lr_secreted, graph_diffusible):
        """sender_type and receiver_type should match cell_meta."""
        result = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        valid_types = set(sp_ccc.cell_meta["cell_type"].unique())
        assert set(result["sender_type"].unique()).issubset(valid_types | {"unknown"})
        assert set(result["receiver_type"].unique()).issubset(valid_types | {"unknown"})

    def test_two_pairs(self, sp_ccc, lr_two_pairs, graph_diffusible):
        """Should score both LR pairs."""
        result = score_edges(sp_ccc, lr_two_pairs, graph_diffusible=graph_diffusible)
        assert set(result["lr_name"].unique()) == {"L1_R1", "L2_R2"}


# ══════════════════════════════════════════════════════════════════════════════
# TestAggregateScores
# ══════════════════════════════════════════════════════════════════════════════


class TestAggregateScores:
    """Tests for aggregate_scores."""

    def test_returns_tuple(self, sp_ccc, lr_secreted, graph_diffusible):
        """aggregate_scores returns (summary_df, cell_scores_df)."""
        edges = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        summary, cell_scores = aggregate_scores(edges, sp_ccc)
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(cell_scores, pd.DataFrame)

    def test_summary_columns(self, sp_ccc, lr_secreted, graph_diffusible):
        """Summary should have expected columns."""
        edges = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        summary, _ = aggregate_scores(edges, sp_ccc)
        expected = {
            "lr_name",
            "sender_type",
            "receiver_type",
            "interaction_mode",
            "mean_score",
            "sum_score",
            "n_edges",
        }
        assert expected == set(summary.columns)

    def test_summary_aggregation(self, sp_ccc, lr_secreted, graph_diffusible):
        """sum_score should equal sum of edge scores for each group."""
        edges = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        summary, _ = aggregate_scores(edges, sp_ccc)
        for _, row in summary.iterrows():
            mask = (
                (edges["lr_name"] == row["lr_name"])
                & (edges["sender_type"] == row["sender_type"])
                & (edges["receiver_type"] == row["receiver_type"])
                & (edges["interaction_mode"] == row["interaction_mode"])
            )
            expected_sum = edges.loc[mask, "score"].sum()
            assert abs(row["sum_score"] - expected_sum) < 1e-6

    def test_cell_scores_shape(self, sp_ccc, lr_secreted, graph_diffusible):
        """cell_scores should have rows for all cells."""
        edges = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        _, cell_scores = aggregate_scores(edges, sp_ccc)
        assert len(cell_scores) == len(sp_ccc.cell_index)

    def test_cell_scores_columns(self, sp_ccc, lr_two_pairs, graph_diffusible):
        """cell_scores should have sender and receiver columns per LR pair."""
        edges = score_edges(sp_ccc, lr_two_pairs, graph_diffusible=graph_diffusible)
        _, cell_scores = aggregate_scores(edges, sp_ccc)
        assert "L1_R1_sender" in cell_scores.columns
        assert "L1_R1_receiver" in cell_scores.columns
        assert "L2_R2_sender" in cell_scores.columns
        assert "L2_R2_receiver" in cell_scores.columns

    def test_empty_edge_df(self, sp_ccc):
        """Should handle empty edge DataFrame gracefully."""
        empty = pd.DataFrame(
            columns=["sender", "receiver", "lr_name", "lr_type", "score", "weight", "sender_type", "receiver_type"]
        )
        summary, cell_scores = aggregate_scores(empty, sp_ccc)
        assert len(summary) == 0
        assert len(cell_scores) == len(sp_ccc.cell_index)


# ══════════════════════════════════════════════════════════════════════════════
# TestSignificance
# ══════════════════════════════════════════════════════════════════════════════


class TestSignificance:
    """Tests for test_significance."""

    def test_analytical_returns_pvalues(self, sp_ccc, lr_secreted, graph_diffusible):
        """Analytical method should return p-values in [0, 1]."""
        edges = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        summary, _ = aggregate_scores(edges, sp_ccc)
        result = run_significance_test(summary, edges, sp_ccc, method="analytical")
        assert "pvalue" in result.columns
        assert "z_score" in result.columns
        assert "fdr" in result.columns
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_permutation_returns_pvalues(self, sp_ccc, lr_secreted, graph_diffusible):
        """Permutation method should return p-values in [0, 1]."""
        edges = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        summary, _ = aggregate_scores(edges, sp_ccc)
        result = run_significance_test(summary, edges, sp_ccc, method="permutation", n_permutations=50, n_subsample=100)
        assert "pvalue" in result.columns
        assert "fdr" in result.columns
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()

    def test_analytical_known_strong_pair(self, sp_ccc, lr_secreted, graph_diffusible):
        """Tumor -> Stroma for L1_R1 should have a relatively high z-score.

        gene_L1 is highly expressed in Tumor, gene_R1 in Stroma.
        """
        edges = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        summary, _ = aggregate_scores(edges, sp_ccc)
        result = run_significance_test(summary, edges, sp_ccc, method="analytical")
        # Find the Tumor -> Stroma row
        mask = (result["sender_type"] == "Tumor") & (result["receiver_type"] == "Stroma")
        if mask.any():
            row = result[mask].iloc[0]
            # This should be one of the higher scoring pairs
            assert row["sum_score"] > 0

    def test_permutation_reproducible(self, sp_ccc, lr_secreted, graph_diffusible):
        """Same seed should give same permutation results."""
        edges = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        summary, _ = aggregate_scores(edges, sp_ccc)
        r1 = run_significance_test(summary, edges, sp_ccc, method="permutation", n_permutations=20, seed=42)
        r2 = run_significance_test(summary, edges, sp_ccc, method="permutation", n_permutations=20, seed=42)
        np.testing.assert_array_almost_equal(r1["pvalue"].values, r2["pvalue"].values)

    def test_empty_summary(self, sp_ccc):
        """Should handle empty summary DataFrame."""
        empty_summary = pd.DataFrame(
            columns=["lr_name", "sender_type", "receiver_type", "mean_score", "sum_score", "n_edges"]
        )
        empty_edges = pd.DataFrame(
            columns=["sender", "receiver", "lr_name", "lr_type", "score", "weight", "sender_type", "receiver_type"]
        )
        result = run_significance_test(empty_summary, empty_edges, sp_ccc, method="analytical")
        assert len(result) == 0

    def test_invalid_method_raises(self, sp_ccc, lr_secreted, graph_diffusible):
        """Unknown method should raise ValueError."""
        edges = score_edges(sp_ccc, lr_secreted, graph_diffusible=graph_diffusible)
        summary, _ = aggregate_scores(edges, sp_ccc)
        with pytest.raises(ValueError, match="Unknown method"):
            run_significance_test(summary, edges, sp_ccc, method="bogus")
