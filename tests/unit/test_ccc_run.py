"""
test_ccc_run.py - Tests for CCC pipeline orchestrator.

Uses the sp_ccc fixture (300 cells, genes gene_L1/gene_R1/gene_L2/gene_R2,
cell types Tumor/Stroma/Fibroblast/CD8_T, polygons, morph_class column).

The builtin LR database uses real gene names (EPCAM, TGFB1 etc.) which are
not in sp_ccc.  filter_to_expressed will therefore return 0 pairs.  Tests
that need actual edges supply lr_pairs directly via the run_ccc lr_pairs
parameter.
"""

from __future__ import annotations

import pandas as pd
import pytest

from spatioloji_s.ccc.database import LRPair
from spatioloji_s.ccc.run import CCCConfig, CCCResult, run_ccc

# ── Shared LR pair fixture ────────────────────────────────────────────────────


@pytest.fixture
def expressed_pairs():
    """Two secreted pairs using genes present in sp_ccc."""
    return [
        LRPair("L1_R1", "gene_L1", "gene_R1", "test_pathway", "secreted"),
        LRPair("L2_R2", "gene_L2", "gene_R2", "test_pathway2", "secreted"),
    ]


@pytest.fixture
def juxtacrine_pairs():
    """Single juxtacrine pair using genes present in sp_ccc."""
    return [LRPair("L1_R1_jux", "gene_L1", "gene_R1", "test_pathway", "juxtacrine")]


# ── TestCCCConfig ─────────────────────────────────────────────────────────────


class TestCCCConfig:
    def test_default_config(self):
        """Default CCCConfig has expected attribute values."""
        config = CCCConfig()
        assert config.test_method == "analytical"
        assert config.secreted_radius == 200.0
        assert config.ecm_radius == 200.0
        assert config.min_pct == 0.05
        assert config.group_col == "cell_type"
        assert config.db_source == "builtin"
        assert config.buffer_distance is None
        assert config.interface_result is None
        assert config.morphology_col is None
        assert config.verbose is True
        assert config.seed == 42

    def test_custom_config(self):
        """CCCConfig accepts and stores custom values."""
        config = CCCConfig(test_method="permutation", secreted_radius=300.0, group_col="my_col")
        assert config.test_method == "permutation"
        assert config.secreted_radius == 300.0
        assert config.group_col == "my_col"

    def test_lr_types_default_none(self):
        """lr_types defaults to None (all types included)."""
        config = CCCConfig()
        assert config.lr_types is None

    def test_n_distance_bins(self):
        """n_distance_bins defaults to 20."""
        assert CCCConfig().n_distance_bins == 20


# ── TestCCCResult ─────────────────────────────────────────────────────────────


class TestCCCResult:
    def test_fields(self):
        """CCCResult fields default to None / zero as expected."""
        result = CCCResult(scores=pd.DataFrame(), cell_scores=pd.DataFrame(), lr_pairs=[])
        assert result.zone_comparison is None
        assert result.zone_gradient is None
        assert result.morphology_comparison is None
        assert result.config is None
        assert result.n_cells == 0
        assert result.n_edges == 0
        assert result.runtime_seconds == 0.0

    def test_fields_set(self):
        """CCCResult stores provided values."""
        cfg = CCCConfig()
        result = CCCResult(
            scores=pd.DataFrame({"x": [1]}),
            cell_scores=pd.DataFrame(),
            lr_pairs=[],
            config=cfg,
            n_cells=10,
            n_edges=5,
            runtime_seconds=1.23,
        )
        assert result.config is cfg
        assert result.n_cells == 10
        assert result.n_edges == 5
        assert result.runtime_seconds == pytest.approx(1.23)


# ── TestRunCCC ────────────────────────────────────────────────────────────────


class TestRunCCC:
    def test_basic_run(self, sp_ccc):
        """run_ccc with builtin database returns CCCResult (0 expressed pairs expected)."""
        config = CCCConfig(group_col="cell_type", db_source="builtin", verbose=False)
        result = run_ccc(sp_ccc, config)
        assert isinstance(result, CCCResult)
        assert isinstance(result.scores, pd.DataFrame)
        assert isinstance(result.cell_scores, pd.DataFrame)
        assert isinstance(result.lr_pairs, list)

    def test_run_with_custom_pairs(self, sp_ccc, expressed_pairs):
        """Passing lr_pairs directly skips database loading and returns edges."""
        config = CCCConfig(group_col="cell_type", verbose=False)
        result = run_ccc(sp_ccc, config, lr_pairs=expressed_pairs)
        assert isinstance(result, CCCResult)
        assert result.n_edges > 0, "Expected edges for expressed gene pairs"
        assert "lr_name" in result.scores.columns
        assert "pvalue" in result.scores.columns
        assert "fdr" in result.scores.columns

    def test_run_with_juxtacrine_pairs(self, sp_ccc, juxtacrine_pairs):
        """Juxtacrine pairs trigger contact graph construction."""
        config = CCCConfig(group_col="cell_type", verbose=False)
        result = run_ccc(sp_ccc, config, lr_pairs=juxtacrine_pairs)
        assert isinstance(result, CCCResult)

    def test_run_with_interface(self, sp_ccc, expressed_pairs):
        """Interface result triggers zone_comparison and zone_gradient."""
        from spatioloji_s.spatial._interface import identify_interface

        iface = identify_interface(
            sp_ccc,
            group_col="cell_type",
            region_a="Tumor",
            region_b="Stroma",
            min_interface_cells=1,
            store=False,
        )
        config = CCCConfig(group_col="cell_type", interface_result=iface, verbose=False)
        result = run_ccc(sp_ccc, config, lr_pairs=expressed_pairs)
        assert result.zone_comparison is not None
        assert isinstance(result.zone_comparison, pd.DataFrame)
        assert result.zone_gradient is not None
        assert isinstance(result.zone_gradient, pd.DataFrame)

    def test_run_with_morphology(self, sp_ccc, expressed_pairs):
        """morphology_col triggers morphology_comparison."""
        config = CCCConfig(group_col="cell_type", morphology_col="morph_class", verbose=False)
        result = run_ccc(sp_ccc, config, lr_pairs=expressed_pairs)
        assert result.morphology_comparison is not None
        assert isinstance(result.morphology_comparison, pd.DataFrame)

    def test_run_permutation(self, sp_ccc, expressed_pairs):
        """Permutation test method runs without error."""
        config = CCCConfig(
            group_col="cell_type",
            test_method="permutation",
            n_subsample=100,
            n_permutations=10,
            verbose=False,
        )
        result = run_ccc(sp_ccc, config, lr_pairs=expressed_pairs)
        assert isinstance(result, CCCResult)

    def test_none_config_uses_defaults(self, sp_ccc):
        """Passing config=None creates a default CCCConfig internally."""
        result = run_ccc(sp_ccc)
        assert isinstance(result, CCCResult)

    def test_result_metadata(self, sp_ccc, expressed_pairs):
        """CCCResult metadata (n_cells, runtime, config) is populated."""
        config = CCCConfig(group_col="cell_type", verbose=False)
        result = run_ccc(sp_ccc, config, lr_pairs=expressed_pairs)
        assert result.n_cells == len(sp_ccc.cell_index)
        assert result.runtime_seconds > 0.0
        assert result.config is config

    def test_sender_receiver_filter(self, sp_ccc, expressed_pairs):
        """sender_types / receiver_types filtering reduces edges."""
        config_all = CCCConfig(group_col="cell_type", verbose=False)
        result_all = run_ccc(sp_ccc, config_all, lr_pairs=expressed_pairs)

        config_filtered = CCCConfig(
            group_col="cell_type",
            sender_types=["Tumor"],
            receiver_types=["Stroma"],
            verbose=False,
        )
        result_filtered = run_ccc(sp_ccc, config_filtered, lr_pairs=expressed_pairs)
        assert result_filtered.n_edges <= result_all.n_edges

    def test_zone_comparison_none_without_interface(self, sp_ccc, expressed_pairs):
        """zone_comparison is None when no interface_result is provided."""
        config = CCCConfig(group_col="cell_type", verbose=False)
        result = run_ccc(sp_ccc, config, lr_pairs=expressed_pairs)
        assert result.zone_comparison is None
        assert result.zone_gradient is None

    def test_morphology_none_without_col(self, sp_ccc, expressed_pairs):
        """morphology_comparison is None when morphology_col is not set."""
        config = CCCConfig(group_col="cell_type", verbose=False)
        result = run_ccc(sp_ccc, config, lr_pairs=expressed_pairs)
        assert result.morphology_comparison is None
