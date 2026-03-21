"""
test_ccc_zones.py - Tests for CCC zone stratification functions.

Uses the sp_ccc fixture (300 cells, two regions, Tumor/Stroma/Fibroblast/CD8_T,
gene_L1/gene_R1/gene_L2/gene_R2, morph_class column with 'round'/'elongated').
"""

import pandas as pd
import pytest

from spatioloji_s.ccc.database import LRPair
from spatioloji_s.ccc.scoring import score_edges
from spatioloji_s.ccc.zones import (
    communication_gradient,
    compare_morphology,
    compare_zones,
)
from spatioloji_s.spatial._interface import identify_interface
from spatioloji_s.spatial.point.graph import build_radius_graph

# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def graph(sp_ccc):
    """Radius graph for diffusible signaling."""
    return build_radius_graph(sp_ccc, radius=200)


@pytest.fixture
def lr_pairs():
    """Single secreted LR pair with uniform expression."""
    return [LRPair("L2_R2", "gene_L2", "gene_R2", "test", "secreted")]


@pytest.fixture
def edges(sp_ccc, graph, lr_pairs):
    """Scored edges for the secreted LR pair."""
    return score_edges(sp_ccc, lr_pairs, graph_diffusible=graph, group_col="cell_type")


@pytest.fixture
def iface(sp_ccc):
    """Interface result between Tumor and Stroma."""
    return identify_interface(
        sp_ccc,
        group_col="cell_type",
        region_a="Tumor",
        region_b="Stroma",
        min_interface_cells=1,
        store=False,
    )


# ── TestCompareZones ──────────────────────────────────────────────────────────


class TestCompareZones:
    def test_returns_dataframe(self, sp_ccc, edges, iface):
        """compare_zones returns a DataFrame with the expected columns."""
        result = compare_zones(edges, sp_ccc, iface)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "lr_name",
            "sender_type",
            "receiver_type",
            "zone",
            "mean_score",
            "sum_score",
            "n_edges",
            "fold_change",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_zone_values(self, sp_ccc, edges, iface):
        """Zone values are a subset of the three allowed zone names."""
        result = compare_zones(edges, sp_ccc, iface)
        if result.empty:
            pytest.skip("No edges assigned to any zone")
        valid_zones = {"interface", "interior_a", "interior_b"}
        assert set(result["zone"].unique()).issubset(valid_zones)

    def test_fold_change_computed(self, sp_ccc, edges, iface):
        """fold_change column exists and contains numeric values."""
        result = compare_zones(edges, sp_ccc, iface)
        assert "fold_change" in result.columns
        if not result.empty:
            # fold_change should be numeric (float, possibly NaN)
            assert pd.api.types.is_float_dtype(result["fold_change"]) or pd.api.types.is_numeric_dtype(
                result["fold_change"]
            )

    def test_empty_edge_df_returns_empty(self, sp_ccc, iface):
        """compare_zones on an empty edge_df returns an empty DataFrame."""
        empty = pd.DataFrame(
            columns=["sender", "receiver", "lr_name", "lr_type", "score", "weight", "sender_type", "receiver_type"]
        )
        result = compare_zones(empty, sp_ccc, iface)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_column_raises(self, sp_ccc, iface):
        """compare_zones raises ValueError when edge_df lacks required columns."""
        bad_df = pd.DataFrame({"sender": ["cell_0"], "lr_name": ["L2_R2"]})
        with pytest.raises(ValueError, match="missing required columns"):
            compare_zones(bad_df, sp_ccc, iface)


# ── TestCommunicationGradient ─────────────────────────────────────────────────


class TestCommunicationGradient:
    def test_returns_dataframe(self, sp_ccc, edges, iface):
        """communication_gradient returns a DataFrame with the expected columns."""
        result = communication_gradient(edges, sp_ccc, iface)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"lr_name", "sender_type", "receiver_type", "slope", "pvalue", "r2", "trend"}
        assert expected_cols.issubset(set(result.columns))

    def test_trend_values(self, sp_ccc, edges, iface):
        """trend column only contains allowed values."""
        result = communication_gradient(edges, sp_ccc, iface)
        if result.empty:
            pytest.skip("No group produced a gradient result")
        valid_trends = {"increasing_toward_a", "increasing_toward_b", "flat"}
        assert set(result["trend"].unique()).issubset(valid_trends)

    def test_r2_bounded(self, sp_ccc, edges, iface):
        """R-squared values are in [0, 1]."""
        result = communication_gradient(edges, sp_ccc, iface)
        if result.empty:
            pytest.skip("No gradient results to check")
        assert (result["r2"] >= 0).all()
        assert (result["r2"] <= 1.0 + 1e-9).all()

    def test_empty_edge_df_returns_empty(self, sp_ccc, iface):
        """communication_gradient on an empty edge_df returns an empty DataFrame."""
        empty = pd.DataFrame(
            columns=["sender", "receiver", "lr_name", "lr_type", "score", "weight", "sender_type", "receiver_type"]
        )
        result = communication_gradient(empty, sp_ccc, iface)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_n_bins_parameter(self, sp_ccc, edges, iface):
        """n_bins parameter is accepted without error."""
        result = communication_gradient(edges, sp_ccc, iface, n_bins=5)
        assert isinstance(result, pd.DataFrame)


# ── TestCompareMorphology ─────────────────────────────────────────────────────


class TestCompareMorphology:
    def test_returns_dataframe(self, sp_ccc, edges):
        """compare_morphology returns a DataFrame with the expected columns."""
        result = compare_morphology(edges, sp_ccc, morphology_col="morph_class")
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "lr_name",
            "sender_type",
            "receiver_type",
            "morphology_group",
            "mean_score",
            "sum_score",
            "n_edges",
            "fold_change",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_morphology_groups_match_data(self, sp_ccc, edges):
        """morphology_group values match what is in the sp_ccc fixture."""
        result = compare_morphology(edges, sp_ccc, morphology_col="morph_class")
        if result.empty:
            pytest.skip("No morphology groups found")
        expected_groups = set(sp_ccc.cell_meta["morph_class"].unique())
        assert set(result["morphology_group"].unique()).issubset(expected_groups)

    def test_fold_change_computed(self, sp_ccc, edges):
        """fold_change column exists and contains numeric values."""
        result = compare_morphology(edges, sp_ccc, morphology_col="morph_class")
        assert "fold_change" in result.columns
        if not result.empty:
            assert pd.api.types.is_numeric_dtype(result["fold_change"])

    def test_invalid_morphology_col_raises(self, sp_ccc, edges):
        """compare_morphology raises ValueError for an unknown column."""
        with pytest.raises(ValueError, match="not found in cell_meta"):
            compare_morphology(edges, sp_ccc, morphology_col="nonexistent_col")

    def test_empty_edge_df_returns_empty(self, sp_ccc):
        """compare_morphology on an empty edge_df returns an empty DataFrame."""
        empty = pd.DataFrame(
            columns=["sender", "receiver", "lr_name", "lr_type", "score", "weight", "sender_type", "receiver_type"]
        )
        result = compare_morphology(empty, sp_ccc, morphology_col="morph_class")
        assert isinstance(result, pd.DataFrame)
        assert result.empty
