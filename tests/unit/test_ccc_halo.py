"""
test_ccc_halo.py - Tests for CCC halo/buffer strategy.

Uses the sp_ccc fixture (300 cells, genes gene_L1/gene_R1/gene_L2/gene_R2,
cell types Tumor/Stroma/Fibroblast/CD8_T, polygons, morph_class column).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.ccc.database import LRPair
from spatioloji_s.ccc.halo import create_halo_subset, filter_ccc_to_core
from spatioloji_s.ccc.run import CCCConfig, CCCResult, run_ccc


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def secreted_pairs():
    """Two secreted pairs using genes present in sp_ccc."""
    return [
        LRPair("L1_R1", "gene_L1", "gene_R1", "test_pathway", "secreted"),
        LRPair("L2_R2", "gene_L2", "gene_R2", "test_pathway2", "secreted"),
    ]


@pytest.fixture
def config():
    """Default CCC config for halo tests."""
    return CCCConfig(
        group_col="cell_type",
        secreted_radius=200.0,
        ecm_radius=150.0,
        test_method="analytical",
        verbose=False,
    )


@pytest.fixture
def core_ids(sp_ccc):
    """Core cell IDs: first 100 cells (a subset of sp_ccc)."""
    return sp_ccc.cell_index[:100]


# ── TestCreateHaloSubset ─────────────────────────────────────────────────────


class TestCreateHaloSubset:
    def test_returns_tuple(self, sp_ccc, core_ids, config):
        """create_halo_subset returns (spatioloji, pd.Index)."""
        sp_halo, returned_core = create_halo_subset(sp_ccc, core_ids, config=config)
        assert hasattr(sp_halo, "cell_index")
        assert isinstance(returned_core, pd.Index)

    def test_core_cells_included(self, sp_ccc, core_ids, config):
        """All core cells appear in the halo subset."""
        sp_halo, returned_core = create_halo_subset(sp_ccc, core_ids, config=config)
        assert returned_core.isin(sp_halo.cell_index).all()

    def test_halo_adds_cells(self, sp_ccc, core_ids, config):
        """Halo subset has more cells than just the core."""
        sp_halo, _ = create_halo_subset(sp_ccc, core_ids, config=config)
        assert sp_halo.n_cells >= len(core_ids)

    def test_halo_within_full(self, sp_ccc, core_ids, config):
        """All halo cells come from sp_full."""
        sp_halo, _ = create_halo_subset(sp_ccc, core_ids, config=config)
        assert sp_halo.cell_index.isin(sp_ccc.cell_index).all()

    def test_halo_distance_from_config(self, sp_ccc, core_ids, config):
        """When halo_distance is None, uses max(secreted_radius, ecm_radius)."""
        # config has secreted=200, ecm=150 -> max=200
        sp_200, _ = create_halo_subset(sp_ccc, core_ids, config=config)

        # Explicitly pass a smaller distance -> fewer cells
        sp_50, _ = create_halo_subset(sp_ccc, core_ids, halo_distance=50.0)

        assert sp_200.n_cells >= sp_50.n_cells

    def test_explicit_distance_overrides_config(self, sp_ccc, core_ids, config):
        """Explicit halo_distance takes precedence over config."""
        sp_small, _ = create_halo_subset(sp_ccc, core_ids, halo_distance=10.0, config=config)
        sp_large, _ = create_halo_subset(sp_ccc, core_ids, halo_distance=5000.0, config=config)
        assert sp_large.n_cells >= sp_small.n_cells

    def test_zero_distance_no_halo(self, sp_ccc, core_ids):
        """With halo_distance=0, only core cells are returned."""
        sp_halo, returned_core = create_halo_subset(sp_ccc, core_ids, halo_distance=0.0)
        assert sp_halo.n_cells == len(core_ids)

    def test_missing_distance_raises(self, sp_ccc, core_ids):
        """Raises ValueError when neither halo_distance nor config is given."""
        with pytest.raises(ValueError, match="halo_distance must be specified"):
            create_halo_subset(sp_ccc, core_ids)

    def test_invalid_core_ids_raises(self, sp_ccc, config):
        """Raises ValueError when no core IDs match sp_full."""
        bogus_ids = ["not_a_cell_1", "not_a_cell_2"]
        with pytest.raises(ValueError, match="No core_cell_ids found"):
            create_halo_subset(sp_ccc, bogus_ids, config=config)

    def test_partial_core_ids_warned(self, sp_ccc, config, capsys):
        """Warns when some core IDs are not in sp_full."""
        mixed_ids = list(sp_ccc.cell_index[:10]) + ["bogus_cell"]
        sp_halo, returned_core = create_halo_subset(sp_ccc, mixed_ids, config=config)
        captured = capsys.readouterr()
        assert "1 core cell IDs not found" in captured.out
        assert len(returned_core) == 10

    def test_large_distance_returns_all(self, sp_ccc, core_ids):
        """Very large halo distance returns all cells in sp_full."""
        sp_halo, _ = create_halo_subset(sp_ccc, core_ids, halo_distance=1e6)
        assert sp_halo.n_cells == sp_ccc.n_cells


# ── TestFilterCCCToCore ──────────────────────────────────────────────────────


class TestFilterCCCToCore:
    def test_cell_scores_only_core(self, sp_ccc, core_ids, config, secreted_pairs):
        """Filtered cell_scores only contain core cell IDs."""
        sp_halo, core = create_halo_subset(sp_ccc, core_ids, config=config)
        result = run_ccc(sp_halo, config, lr_pairs=secreted_pairs)

        sp_core = sp_ccc.subset_by_cells(core)
        filtered = filter_ccc_to_core(result, core, sp_core=sp_core)

        assert set(filtered.cell_scores.index).issubset(set(core))

    def test_n_cells_matches_core(self, sp_ccc, core_ids, config, secreted_pairs):
        """Filtered result has n_cells == len(core_ids)."""
        sp_halo, core = create_halo_subset(sp_ccc, core_ids, config=config)
        result = run_ccc(sp_halo, config, lr_pairs=secreted_pairs)

        sp_core = sp_ccc.subset_by_cells(core)
        filtered = filter_ccc_to_core(result, core, sp_core=sp_core)

        assert filtered.n_cells == len(core)

    def test_scores_recomputed(self, sp_ccc, core_ids, config, secreted_pairs):
        """Filtered scores summary has expected columns."""
        sp_halo, core = create_halo_subset(sp_ccc, core_ids, config=config)
        result = run_ccc(sp_halo, config, lr_pairs=secreted_pairs)

        sp_core = sp_ccc.subset_by_cells(core)
        filtered = filter_ccc_to_core(result, core, sp_core=sp_core)

        expected_cols = {"lr_name", "sender_type", "receiver_type", "mean_score", "sum_score", "n_edges"}
        assert expected_cols.issubset(set(filtered.scores.columns))

    def test_significance_with_sp_core(self, sp_ccc, core_ids, config, secreted_pairs):
        """When sp_core is provided, pvalue and fdr columns exist."""
        sp_halo, core = create_halo_subset(sp_ccc, core_ids, config=config)
        result = run_ccc(sp_halo, config, lr_pairs=secreted_pairs)

        sp_core = sp_ccc.subset_by_cells(core)
        filtered = filter_ccc_to_core(result, core, sp_core=sp_core)

        assert "pvalue" in filtered.scores.columns
        assert "fdr" in filtered.scores.columns

    def test_no_sp_core_omits_pvalue(self, sp_ccc, core_ids, config, secreted_pairs, capsys):
        """Without sp_core, pvalue/fdr are omitted and a warning is printed."""
        sp_halo, core = create_halo_subset(sp_ccc, core_ids, config=config)
        result = run_ccc(sp_halo, config, lr_pairs=secreted_pairs)

        filtered = filter_ccc_to_core(result, core)
        captured = capsys.readouterr()

        assert "sp_core not provided" in captured.out
        assert "pvalue" not in filtered.scores.columns

    def test_edge_df_preserved(self, sp_ccc, core_ids, config, secreted_pairs):
        """Filtered result still has edge_df (filtered version)."""
        sp_halo, core = create_halo_subset(sp_ccc, core_ids, config=config)
        result = run_ccc(sp_halo, config, lr_pairs=secreted_pairs)

        sp_core = sp_ccc.subset_by_cells(core)
        filtered = filter_ccc_to_core(result, core, sp_core=sp_core)

        assert filtered.edge_df is not None
        assert len(filtered.edge_df) <= len(result.edge_df)

    def test_edges_have_core_endpoint(self, sp_ccc, core_ids, config, secreted_pairs):
        """Every retained edge has at least one core cell endpoint."""
        sp_halo, core = create_halo_subset(sp_ccc, core_ids, config=config)
        result = run_ccc(sp_halo, config, lr_pairs=secreted_pairs)

        sp_core = sp_ccc.subset_by_cells(core)
        filtered = filter_ccc_to_core(result, core, sp_core=sp_core)

        core_set = set(core)
        for _, row in filtered.edge_df.iterrows():
            assert row["sender"] in core_set or row["receiver"] in core_set

    def test_missing_edge_df_raises(self, core_ids):
        """Raises ValueError when result.edge_df is None."""
        result = CCCResult(
            scores=pd.DataFrame(),
            cell_scores=pd.DataFrame(),
            lr_pairs=[],
            edge_df=None,
        )
        with pytest.raises(ValueError, match="edge_df is None"):
            filter_ccc_to_core(result, core_ids)

    def test_config_preserved(self, sp_ccc, core_ids, config, secreted_pairs):
        """Filtered result carries the original config."""
        sp_halo, core = create_halo_subset(sp_ccc, core_ids, config=config)
        result = run_ccc(sp_halo, config, lr_pairs=secreted_pairs)

        sp_core = sp_ccc.subset_by_cells(core)
        filtered = filter_ccc_to_core(result, core, sp_core=sp_core)

        assert filtered.config is config


# ── TestRoundTrip ────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_halo_improves_edge_coverage(self, sp_ccc, config, secreted_pairs):
        """Running with halo gives edge cells more scored edges than without.

        Cells at the boundary of the core set should have more neighbors
        when the halo is present, leading to >= edges after filtering.
        """
        core_ids = sp_ccc.cell_index[:100]

        # Without halo: run CCC on core only
        sp_core = sp_ccc.subset_by_cells(core_ids)
        result_no_halo = run_ccc(sp_core, config, lr_pairs=secreted_pairs)

        # With halo: enlarge, run, filter
        sp_halo, core = create_halo_subset(sp_ccc, core_ids, config=config)
        result_halo = run_ccc(sp_halo, config, lr_pairs=secreted_pairs)
        result_filtered = filter_ccc_to_core(result_halo, core, sp_core=sp_core)

        # Halo run should have >= edges touching core cells
        assert result_filtered.n_edges >= result_no_halo.n_edges
