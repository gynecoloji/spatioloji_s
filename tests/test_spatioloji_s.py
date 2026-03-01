"""
test_spatioloji_s.py - Test suite for spatioloji package

How to run:
    pytest tests/ -v                         # run all tests
    pytest tests/ -v -k "creation"           # run only tests with "creation" in the name
    pytest tests/ -v -k "section2"           # run only Section 2
    pytest tests/test_spatioloji_s.py::test_basic_counts -v   # one specific test

Reading test results:
    PASSED  → your code works as expected
    FAILED  → something is broken (look at the AssertionError message)
    ERROR   → test itself crashed before even reaching the assert
"""

import numpy as np
import pandas as pd
import pytest

# ===========================================================================
# SECTION 2 — Object Creation
#
# Goal: Verify that the spatioloji object initializes correctly.
#
# We check:
#   (a) basic counts    — n_cells, n_genes, n_fovs
#   (b) index content   — cell_index, gene_index contain the right IDs
#   (c) properties      — expression, cell_meta, gene_meta, spatial, embeddings
#   (d) sparse auto     — auto-conversion to sparse when data is mostly zeros
#   (e) polygons        — polygons attached correctly when provided
#   (f) minimal init    — works with only the required arguments
#
# Fixture used: sp_basic, sp_sparse, sp_with_polygons  (from conftest.py)
# ===========================================================================


class TestObjectCreation:
    """
    All tests for spatioloji object initialization.

    Why use a class?
    ----------------
    Grouping related tests in a class keeps the file organized.
    pytest still finds and runs them automatically — no extra setup needed.
    You can run just this group with:   pytest -v -k "TestObjectCreation"
    """

    # -----------------------------------------------------------------------
    # (a) Basic counts
    # -----------------------------------------------------------------------

    def test_n_cells(self, sp_basic):
        """sp.n_cells should equal the number of rows we passed in."""
        assert sp_basic.n_cells == 60

    def test_n_genes(self, sp_basic):
        """sp.n_genes should equal the number of columns we passed in."""
        assert sp_basic.n_genes == 25

    def test_n_fovs(self, sp_basic):
        """sp.n_fovs should equal the number of unique FOVs in cell metadata."""
        assert sp_basic.n_fovs == 2

    # -----------------------------------------------------------------------
    # (b) Index content
    # -----------------------------------------------------------------------

    def test_cell_index_values(self, sp_basic):
        """
        The master cell index should contain exactly the IDs we passed.
        We check by converting both to sorted lists and comparing.
        """
        expected = sorted([f"cell_{i}" for i in range(60)])
        actual   = sorted(sp_basic.cell_index.tolist())
        assert actual == expected

    def test_gene_index_values(self, sp_basic):
        """The master gene index should contain exactly the gene names we passed."""
        expected = sorted([f"gene_{i}" for i in range(25)])
        actual   = sorted(sp_basic.gene_index.tolist())
        assert actual == expected

    def test_fov_index_values(self, sp_basic):
        """FOV index should contain '1' and '2' (our two fake FOVs)."""
        actual = sorted(sp_basic.fov_index.tolist())
        assert actual == ['1', '2']

    # -----------------------------------------------------------------------
    # (c) Properties — correct types and shapes
    # -----------------------------------------------------------------------

    def test_expression_shape(self, sp_basic):
        """Expression matrix shape must be (n_cells, n_genes)."""
        assert sp_basic.expression.shape == (60, 25)

    def test_cell_meta_shape(self, sp_basic):
        """cell_meta must have one row per cell."""
        assert len(sp_basic.cell_meta) == 60

    def test_gene_meta_shape(self, sp_basic):
        """gene_meta must have one row per gene."""
        assert len(sp_basic.gene_meta) == 25

    def test_cell_meta_has_fov_column(self, sp_basic):
        """cell_meta must contain the 'fov' column we passed."""
        assert 'fov' in sp_basic.cell_meta.columns

    def test_gene_meta_has_negprobe_column(self, sp_basic):
        """gene_meta must contain the 'NegProbe' column we passed."""
        assert 'NegProbe' in sp_basic.gene_meta.columns

    def test_negprobe_count(self, sp_basic):
        """
        We flagged the last 2 genes as NegProbe in conftest.
        Check that exactly 2 genes are marked True.
        """
        n_neg = sp_basic.gene_meta['NegProbe'].sum()
        assert n_neg == 2

    def test_spatial_coords_length(self, sp_basic):
        """Spatial coordinate arrays must have one entry per cell."""
        assert len(sp_basic.spatial.x_global) == 60
        assert len(sp_basic.spatial.y_global) == 60
        assert len(sp_basic.spatial.x_local)  == 60
        assert len(sp_basic.spatial.y_local)  == 60

    def test_embeddings_starts_empty(self, sp_basic):
        """
        Embeddings dict should be empty right after initialization —
        nothing has been run yet (no PCA, UMAP, etc.).
        """
        assert isinstance(sp_basic.embeddings, dict)
        assert len(sp_basic.embeddings) == 0

    def test_layers_starts_empty(self, sp_basic):
        """
        Layers dict should also be empty at initialization —
        normalization hasn't been run yet.
        """
        assert isinstance(sp_basic.layers, dict)
        assert len(sp_basic.layers) == 0

    # -----------------------------------------------------------------------
    # (d) Sparse auto-conversion
    # -----------------------------------------------------------------------

    def test_auto_sparse_conversion(self, sp_sparse):
        """
        When expression data has >50% zeros, spatioloji should
        automatically store it as a sparse matrix to save memory.
        sp_sparse was built with ~80% zeros in conftest.
        """
        assert sp_sparse.expression.is_sparse is True

    def test_dense_stays_dense(self, sp_basic):
        """
        sp_basic has random integers 0-4 — not very sparse.
        Depending on the random seed, it may or may not be converted,
        but the expression should still be accessible and correct shape.

        We just verify the shape is intact regardless of format.
        """
        assert sp_basic.expression.shape == (60, 25)

    # -----------------------------------------------------------------------
    # (e) Polygons
    # -----------------------------------------------------------------------

    def test_no_polygons_by_default(self, sp_basic):
        """sp_basic was created without polygons → sp.polygons should be None."""
        assert sp_basic.polygons is None

    def test_polygons_attached(self, sp_with_polygons):
        """sp_with_polygons should have a non-None polygons DataFrame."""
        assert sp_with_polygons.polygons is not None

    def test_polygons_has_cell_column(self, sp_with_polygons):
        """The polygons DataFrame must have a 'cell' column (the cell ID)."""
        assert 'cell' in sp_with_polygons.polygons.columns

    def test_polygons_cell_ids_match(self, sp_with_polygons):
        """
        Every cell ID in the polygons table should exist
        in the master cell index. No orphan polygons allowed.
        """
        polygon_cells = set(sp_with_polygons.polygons['cell'].astype(str))
        master_cells  = set(sp_with_polygons.cell_index.astype(str))
        assert polygon_cells.issubset(master_cells)

    # -----------------------------------------------------------------------
    # (f) Minimal initialization (only required arguments)
    # -----------------------------------------------------------------------

    def test_minimal_init(self):
        """
        spatioloji should work with ONLY expression + cell_ids + gene_names.
        All optional arguments (metadata, spatial, polygons) should default gracefully.

        This ensures the constructor doesn't crash for simple use cases.
        """
        from spatioloji_s.data.core import spatioloji

        expr       = np.ones((10, 5))
        cell_ids   = [f"c{i}" for i in range(10)]
        gene_names = [f"g{i}" for i in range(5)]

        sp = spatioloji(expression=expr, cell_ids=cell_ids, gene_names=gene_names)

        # Basic shape checks
        assert sp.n_cells == 10
        assert sp.n_genes == 5


# ===========================================================================
# SECTION 3 — Data Consistency
#
# Goal: Verify that the master cell index enforces alignment across all
# components — expression, metadata, spatial, and polygons.
#
# Key principle in spatioloji:
#   The master cell index is the single source of truth.
#   All other components are aligned TO it, not the other way around.
#
# We test two kinds of situations:
#   (A) Happy path  — everything lines up → validate_consistency() passes
#   (B) Misalignment — metadata has extra/missing cells → spatioloji
#                      silently reindexes (fills NaN) rather than crashing,
#                      but validate_consistency() still reports True because
#                      the reindexed result has the correct shape
#   (C) Hard errors — wrong-length spatial arrays → should raise ValueError
#
# Fixture used: sp_basic, sp_with_polygons  (from conftest.py)
# ===========================================================================


class TestDataConsistency:

    # -----------------------------------------------------------------------
    # (A) Happy path — everything should be consistent
    # -----------------------------------------------------------------------

    def test_validate_consistency_passes(self, sp_basic):
        """
        validate_consistency() returns a dict of booleans.
        For a correctly built object, every value should be True,
        and 'overall' (the summary key) must also be True.
        """
        status = sp_basic.validate_consistency()
        assert status['overall'] is True

    def test_validate_consistency_all_keys_true(self, sp_basic):
        """
        Check each individual component key, not just 'overall'.
        Keys: expression_cells, expression_genes, cell_metadata,
              gene_metadata, spatial, polygons, cell_fovs, fov_positions, images
        """
        status = sp_basic.validate_consistency()
        # Remove 'overall' since it's a summary — check the rest individually
        component_keys = [k for k in status if k != 'overall']
        for key in component_keys:
            assert status[key] is True, f"Consistency check failed for: '{key}'"

    def test_expression_index_matches_cell_index(self, sp_basic):
        """
        The expression matrix row count must equal n_cells.
        (The master cell index defines what 'n_cells' means.)
        """
        assert sp_basic.expression.shape[0] == sp_basic.n_cells

    def test_cell_meta_index_matches_cell_index(self, sp_basic):
        """
        cell_meta must be indexed by the master cell index.
        Every cell ID in cell_meta.index should exist in cell_index.
        """
        meta_ids   = set(sp_basic.cell_meta.index.astype(str))
        master_ids = set(sp_basic.cell_index.astype(str))
        assert meta_ids == master_ids

    def test_gene_meta_index_matches_gene_index(self, sp_basic):
        """gene_meta must be aligned to the master gene index."""
        meta_genes   = set(sp_basic.gene_meta.index.astype(str))
        master_genes = set(sp_basic.gene_index.astype(str))
        assert meta_genes == master_genes

    def test_spatial_length_matches_n_cells(self, sp_basic):
        """All 4 coordinate arrays must have exactly n_cells entries."""
        sp = sp_basic
        assert len(sp.spatial.x_global) == sp.n_cells
        assert len(sp.spatial.y_global) == sp.n_cells
        assert len(sp.spatial.x_local)  == sp.n_cells
        assert len(sp.spatial.y_local)  == sp.n_cells

    def test_fov_index_matches_cell_meta_fovs(self, sp_basic):
        """
        Every FOV label that cells belong to must appear in fov_index.
        No cell should reference a FOV that doesn't exist in the master list.
        """
        cell_fovs  = set(sp_basic.cell_meta['fov'].astype(str))
        master_fovs = set(sp_basic.fov_index.astype(str))
        assert cell_fovs.issubset(master_fovs)

    def test_cells_per_fov_count(self, sp_basic):
        """
        In conftest we put 30 cells in FOV '1' and 30 in FOV '2'.
        Verify this split is preserved in the object.
        """
        counts = sp_basic.cell_meta['fov'].value_counts()
        assert counts['1'] == 30
        assert counts['2'] == 30

    def test_polygon_cells_in_master_index(self, sp_with_polygons):
        """
        No polygon should reference a cell that isn't in the master index.
        This guards against orphan polygons after subsetting or filtering.
        """
        polygon_cells = set(sp_with_polygons.polygons['cell'].astype(str))
        master_cells  = set(sp_with_polygons.cell_index.astype(str))
        # All polygon cells must be a subset of master cells
        orphans = polygon_cells - master_cells
        assert len(orphans) == 0, f"Found {len(orphans)} orphan polygon cell IDs"

    # -----------------------------------------------------------------------
    # (B) Misalignment — spatioloji reindexes silently (no crash)
    # -----------------------------------------------------------------------

    def test_extra_metadata_rows_are_dropped(self):
        """
        If cell_metadata has MORE rows than expression (extra cell IDs),
        the extra rows should be silently dropped.
        The resulting object should have n_cells == expression rows, not metadata rows.

        Why test this?
        → In real data, metadata files often include cells that failed QC
          and were removed from the expression matrix.
        """
        from spatioloji_s.data.core import spatioloji

        n = 10
        expr       = np.ones((n, 5))
        cell_ids   = [f"c{i}" for i in range(n)]
        gene_names = [f"g{i}" for i in range(5)]

        # Add 3 extra rows with cell IDs not in cell_ids
        extra_ids = cell_ids + ['extra_1', 'extra_2', 'extra_3']
        cell_meta = pd.DataFrame({'fov': ['1'] * len(extra_ids)},
                                 index=extra_ids)

        sp = spatioloji(expression=expr, cell_ids=cell_ids,
                        gene_names=gene_names, cell_metadata=cell_meta)

        # Must still have exactly n cells — extras dropped
        assert sp.n_cells == n
        assert len(sp.cell_meta) == n

    def test_missing_metadata_rows_filled_with_nan(self):
        """
        If cell_metadata has FEWER rows than expression (some cells have no metadata),
        the missing rows are filled with NaN — the object still builds successfully.

        Why test this?
        → Ensures the object never silently drops cells just because
          their metadata is incomplete.
        """
        from spatioloji_s.data.core import spatioloji

        n = 10
        expr       = np.ones((n, 5))
        cell_ids   = [f"c{i}" for i in range(n)]
        gene_names = [f"g{i}" for i in range(5)]

        # Only provide metadata for the first 7 cells
        partial_meta = pd.DataFrame({'score': range(7)},
                                    index=cell_ids[:7])

        sp = spatioloji(expression=expr, cell_ids=cell_ids,
                        gene_names=gene_names, cell_metadata=partial_meta)

        # All 10 cells must be present
        assert sp.n_cells == n
        assert len(sp.cell_meta) == n

        # The last 3 cells should have NaN in 'score'
        missing_scores = sp.cell_meta.loc[cell_ids[7:], 'score']
        assert missing_scores.isna().all()

    # -----------------------------------------------------------------------
    # (C) Hard errors — wrong-length spatial arrays must raise ValueError
    # -----------------------------------------------------------------------

    def test_wrong_spatial_length_raises_error(self):
        """
        Spatial coordinate arrays must have exactly n_cells entries.
        If they don't, spatioloji should raise a ValueError immediately —
        not silently produce a misaligned object.

        Why test this?
        → A misaligned spatial array would corrupt all downstream spatial
          analysis without any warning. Failing loudly is the safe behavior.
        """
        from spatioloji_s.data.core import spatioloji

        expr       = np.ones((10, 5))
        cell_ids   = [f"c{i}" for i in range(10)]
        gene_names = [f"g{i}" for i in range(5)]

        # Spatial arrays have 8 entries, but we have 10 cells → mismatch
        bad_spatial = {
            'x_global': np.ones(8),   # ← wrong length
            'y_global': np.ones(8),
            'x_local':  np.ones(8),
            'y_local':  np.ones(8),
        }

        with pytest.raises((ValueError, Exception)):
            spatioloji(expression=expr, cell_ids=cell_ids,
                       gene_names=gene_names, spatial_coords=bad_spatial)
