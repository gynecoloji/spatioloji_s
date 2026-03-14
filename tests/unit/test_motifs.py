"""Tests for hierarchical spatial motif discovery."""

import pandas as pd
import pytest

from spatioloji_s.spatial._motif_types import (
    AssemblyCatalog,
    MotifCatalog,
    MotifResult,
    StructureMatches,
)
from spatioloji_s.spatial.polygon.graph import build_buffer_graph
from spatioloji_s.spatial.polygon.motifs import discover_motifs


class TestMotifTypes:
    """Verify dataclass imports and basic construction."""

    def test_motif_catalog_fields(self):
        mc = MotifCatalog(
            labels=pd.Series([0, 1], index=["a", "b"]),
            signatures=pd.DataFrame({"T": [0.5, 0.5]}, index=[0, 1]),
            counts=pd.Series([1, 1], index=[0, 1]),
            group_col="cell_type",
            feature_matrix=None,
            params={},
        )
        assert mc.group_col == "cell_type"
        assert len(mc.labels) == 2

    def test_assembly_catalog_fields(self):
        ac = AssemblyCatalog(
            labels=pd.Series([0, -1], index=["a", "b"]),
            composition=pd.DataFrame(),
            instances=pd.DataFrame(),
            adjacency_pattern=pd.DataFrame(),
            params={},
        )
        assert (ac.labels == [0, -1]).all()

    def test_structure_matches_fields(self):
        sm = StructureMatches(
            matches=pd.DataFrame(),
            per_cell=pd.Series(dtype=str),
            signatures_used={},
        )
        assert sm.matches.empty

    def test_motif_result_fields(self):
        mc = MotifCatalog(
            labels=pd.Series(dtype=int),
            signatures=pd.DataFrame(),
            counts=pd.Series(dtype=int),
            group_col="ct",
            feature_matrix=None,
            params={},
        )
        mr = MotifResult(
            motif_catalog=mc,
            assembly_catalog=None,
            structure_matches=None,
            params={},
        )
        assert mr.assembly_catalog is None


# ---------------------------------------------------------------------------
# discover_motifs tests
# ---------------------------------------------------------------------------


class TestDiscoverMotifsKMeans:
    """KMeans-based motif discovery tests."""

    def test_returns_motif_catalog(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        catalog = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert isinstance(catalog, MotifCatalog)

    def test_labels_cover_all_cells(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        catalog = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert len(catalog.labels) == graph.n_cells
        assert catalog.labels.index.equals(graph.cell_index)

    def test_n_motifs_respected(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        catalog = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert catalog.labels.nunique() <= 5

    def test_signatures_have_values(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        catalog = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert catalog.signatures.shape[0] > 0
        assert catalog.signatures.shape[1] > 0
        # At least some non-zero values
        assert catalog.signatures.values.sum() > 0

    def test_group_col_stored(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        catalog = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert catalog.group_col == "cell_type"

    def test_feature_matrix_none_by_default(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        catalog = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert catalog.feature_matrix is None

    def test_keep_features_retains_matrix(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        catalog = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5, keep_features=True)
        assert catalog.feature_matrix is not None
        assert catalog.feature_matrix.shape[0] == graph.n_cells

    def test_store_writes_to_cell_meta(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5, store=True)
        assert "motif" in sp_motif.cell_meta.columns

    def test_auto_n_motifs(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        catalog = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=None)
        assert isinstance(catalog, MotifCatalog)
        assert catalog.labels.nunique() >= 2


class TestDiscoverMotifsLeiden:
    """Leiden-based motif discovery tests."""

    def test_returns_motif_catalog_correct_length(self, sp_motif):
        pytest.importorskip("leidenalg")
        pytest.importorskip("igraph")
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        catalog = discover_motifs(sp_motif, graph, group_col="cell_type", method="leiden")
        assert isinstance(catalog, MotifCatalog)
        assert len(catalog.labels) == graph.n_cells


class TestDiscoverMotifsValidation:
    """Validation and edge-case tests."""

    def test_invalid_method_raises(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        with pytest.raises(ValueError, match="Unknown clustering method"):
            discover_motifs(sp_motif, graph, group_col="cell_type", method="spectral")

    def test_invalid_group_col_raises(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        with pytest.raises(ValueError, match="not found"):
            discover_motifs(sp_motif, graph, group_col="nonexistent_col")

    def test_morphology_with_point_graph_raises(self, sp_motif):
        from spatioloji_s.spatial.point.graph import build_knn_graph

        graph = build_knn_graph(sp_motif, k=6)
        with pytest.raises(ValueError, match="Morphology features require a polygon graph"):
            discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5, use_morphology=True)

    def test_density_with_point_graph_raises(self, sp_motif):
        from spatioloji_s.spatial.point.graph import build_knn_graph

        graph = build_knn_graph(sp_motif, k=6)
        with pytest.raises(ValueError, match="Density features require a polygon graph"):
            discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5, use_density=True)

    def test_single_motif(self, sp_motif):
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        catalog = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=1)
        assert catalog.labels.nunique() == 1


class TestPointMotifReExport:
    """Verify point module re-exports the polygon function."""

    def test_same_function(self):
        from spatioloji_s.spatial.point.motifs import discover_motifs as point_discover
        from spatioloji_s.spatial.polygon.motifs import discover_motifs as polygon_discover

        assert point_discover is polygon_discover
