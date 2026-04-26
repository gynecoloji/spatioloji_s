"""
test_ccc_autocrine.py - Tests for autocrine self-edge support in CCC scoring.

Uses the sp_ccc fixture (300 cells; gene_L2/gene_R2 are uniformly expressed
so every cell is a candidate for autocrine signaling).
"""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.ccc.database import LRPair
from spatioloji_s.ccc.run import CCCConfig, run_ccc
from spatioloji_s.ccc.scoring import aggregate_scores, score_edges
from spatioloji_s.ccc.scoring import test_significance as run_significance_test


@pytest.fixture
def lr_uniform_secreted():
    """Secreted pair on the uniformly-expressed genes — every cell co-expresses."""
    return [LRPair("L2_R2", "gene_L2", "gene_R2", "uniform", "secreted")]


@pytest.fixture
def lr_uniform_juxtacrine():
    """Juxtacrine pair on the uniformly-expressed genes."""
    return [LRPair("L2_R2_jux", "gene_L2", "gene_R2", "uniform", "juxtacrine")]


@pytest.fixture
def graph_diffusible(sp_ccc):
    from spatioloji_s.spatial.point.graph import build_radius_graph

    return build_radius_graph(sp_ccc, radius=200)


@pytest.fixture
def graph_juxtacrine(sp_ccc):
    from spatioloji_s.spatial.polygon.boundaries import contact_fraction
    from spatioloji_s.spatial.polygon.graph import build_buffer_graph

    g = build_buffer_graph(sp_ccc, buffer_distance=10)
    g.contact_frac_df = contact_fraction(sp_ccc, g)
    return g


# ── score_edges flag-off baseline ────────────────────────────────────────────


class TestScoreEdgesAutocrineFlagOff:
    """include_autocrine=False must produce intercellular-only edges."""

    def test_no_self_edges_when_disabled(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        out = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=False)
        assert (out["sender"] != out["receiver"]).all()
        assert (out["interaction_mode"] == "paracrine").all()

    def test_only_paracrine_mode_when_disabled(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        out = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=False)
        assert set(out["interaction_mode"].unique()) == {"paracrine"}

    def test_default_now_includes_autocrine(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        """Default include_autocrine=True must add autocrine rows."""
        out = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible)
        assert (out["interaction_mode"] == "autocrine").any()


# ── score_edges flag-on autocrine ────────────────────────────────────────────


class TestScoreEdgesAutocrineFlagOn:
    """include_autocrine=True must add self-edges with weight=1 and correct typing."""

    def test_self_edges_appear(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        out = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=True)
        autocrine = out[out["interaction_mode"] == "autocrine"]
        assert len(autocrine) > 0

    def test_autocrine_sender_equals_receiver(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        out = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=True)
        autocrine = out[out["interaction_mode"] == "autocrine"]
        assert (autocrine["sender"] == autocrine["receiver"]).all()
        assert (autocrine["sender_type"] == autocrine["receiver_type"]).all()

    def test_autocrine_weight_is_one(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        out = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=True)
        autocrine = out[out["interaction_mode"] == "autocrine"]
        np.testing.assert_allclose(autocrine["weight"].values, 1.0)

    def test_autocrine_count_matches_coexpressing_cells(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        """One self-edge per co-expressing cell, per LR pair."""
        out = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=True)
        autocrine = out[out["interaction_mode"] == "autocrine"]
        expr = sp_ccc.expression.to_dataframe()
        n_coexpr = int(((expr["gene_L2"] > 0) & (expr["gene_R2"] > 0)).sum())
        assert len(autocrine) == n_coexpr

    def test_paracrine_unaffected(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        """Paracrine rows should be identical with and without autocrine."""
        baseline = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=False)
        with_auto = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=True)
        para = with_auto[with_auto["interaction_mode"] == "paracrine"].reset_index(drop=True)
        baseline = baseline.reset_index(drop=True)
        # Same rows in same order
        pd.testing.assert_frame_equal(
            baseline.drop(columns=["interaction_mode"]),
            para.drop(columns=["interaction_mode"]),
            check_like=False,
        )

    def test_juxtacrine_autocrine(self, sp_ccc, lr_uniform_juxtacrine, graph_juxtacrine):
        """Autocrine flag must apply to juxtacrine pairs too."""
        out = score_edges(
            sp_ccc,
            lr_uniform_juxtacrine,
            graph_juxtacrine=graph_juxtacrine,
            include_autocrine=True,
        )
        autocrine = out[out["interaction_mode"] == "autocrine"]
        assert len(autocrine) > 0


# ── aggregate_scores keeps modes separate ────────────────────────────────────


class TestAggregateAutocrine:
    def test_summary_has_interaction_mode_column(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        edges = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=True)
        summary, _ = aggregate_scores(edges, sp_ccc)
        assert "interaction_mode" in summary.columns
        modes = set(summary["interaction_mode"].unique())
        assert "paracrine" in modes
        assert "autocrine" in modes

    def test_paracrine_and_autocrine_rows_separate(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        """Same (lr, type, type) should appear twice — once per mode."""
        edges = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=True)
        summary, _ = aggregate_scores(edges, sp_ccc)
        # Same-type rows: every cell type pairs with itself in both modes
        for ctype in ["Tumor", "Stroma"]:
            rows = summary[
                (summary["lr_name"] == "L2_R2")
                & (summary["sender_type"] == ctype)
                & (summary["receiver_type"] == ctype)
            ]
            modes = set(rows["interaction_mode"].unique())
            assert "autocrine" in modes  # always present since gene_L2/R2 uniform
            # Paracrine same-type only present if same-type cells are within radius


# ── significance test runs cleanly with autocrine rows ───────────────────────


class TestSignificanceAutocrine:
    def test_analytical_runs(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        edges = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=True)
        summary, _ = aggregate_scores(edges, sp_ccc)
        result = run_significance_test(summary, edges, sp_ccc, method="analytical")
        assert "pvalue" in result.columns
        assert "fdr" in result.columns
        assert "z_score" in result.columns
        assert result["pvalue"].between(0.0, 1.0).all()

    def test_permutation_runs(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        edges = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible, include_autocrine=True)
        summary, _ = aggregate_scores(edges, sp_ccc)
        result = run_significance_test(summary, edges, sp_ccc, method="permutation", n_permutations=20, seed=0)
        assert result["pvalue"].between(0.0, 1.0).all()

    def test_legacy_edge_df_without_mode(self, sp_ccc, lr_uniform_secreted, graph_diffusible):
        """test_significance must still accept edge_df without interaction_mode."""
        edges = score_edges(sp_ccc, lr_uniform_secreted, graph_diffusible=graph_diffusible)
        summary, _ = aggregate_scores(edges, sp_ccc)
        legacy_edges = edges.drop(columns=["interaction_mode"])
        legacy_summary = summary.drop(columns=["interaction_mode"])
        result = run_significance_test(legacy_summary, legacy_edges, sp_ccc, method="analytical")
        assert result["pvalue"].between(0.0, 1.0).all()


# ── run_ccc end-to-end ───────────────────────────────────────────────────────


class TestRunCCCAutocrine:
    def test_default_includes_autocrine(self, sp_ccc):
        """Default CCCConfig now includes autocrine."""
        config = CCCConfig(
            db_source="builtin",
            test_method="analytical",
            verbose=False,
        )
        # Use only the uniform pair so we always get co-expression
        pairs = [LRPair("L2_R2", "gene_L2", "gene_R2", "uniform", "secreted")]
        result = run_ccc(sp_ccc, config, lr_pairs=pairs)
        assert (result.edge_df["interaction_mode"] == "autocrine").any()
        assert "interaction_mode" in result.scores.columns

    def test_explicit_disable_autocrine(self, sp_ccc):
        """Setting include_autocrine=False reverts to paracrine-only."""
        config = CCCConfig(
            db_source="builtin",
            test_method="analytical",
            include_autocrine=False,
            verbose=False,
        )
        pairs = [LRPair("L2_R2", "gene_L2", "gene_R2", "uniform", "secreted")]
        result = run_ccc(sp_ccc, config, lr_pairs=pairs)
        assert (result.edge_df["interaction_mode"] == "paracrine").all()
