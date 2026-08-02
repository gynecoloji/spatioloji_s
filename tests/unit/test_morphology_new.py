"""Tests for new intrinsic + context morphology features."""

import pandas as pd
import pytest

from spatioloji_s.spatial.polygon.graph import build_contact_graph
from spatioloji_s.spatial.polygon.morphology import compute_context_morphology, compute_morphology

# ---------------------------------------------------------------------------
# Intrinsic shape metrics
# ---------------------------------------------------------------------------


class TestNewIntrinsicMetrics:
    """Test the 6 new intrinsic shape metrics in compute_morphology."""

    def test_all_new_metrics_returned(self, sp_with_polygons):
        df = compute_morphology(
            sp_with_polygons,
            metrics=[
                "rectangularity",
                "eccentricity",
                "feret_max",
                "feret_min",
                "feret_ratio",
                "n_concavities",
            ],
            store=False,
        )
        assert set(df.columns) == {
            "rectangularity",
            "eccentricity",
            "feret_max",
            "feret_min",
            "feret_ratio",
            "n_concavities",
        }

    def test_rectangularity_range(self, sp_with_polygons):
        """Rectangularity should be in (0, 1] — area / MBR area."""
        df = compute_morphology(sp_with_polygons, metrics=["rectangularity"], store=False)
        valid = df["rectangularity"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all()
        assert (valid <= 1.0 + 1e-6).all()

    def test_eccentricity_range(self, sp_with_polygons):
        """Eccentricity should be in [0, 1)."""
        df = compute_morphology(sp_with_polygons, metrics=["eccentricity"], store=False)
        valid = df["eccentricity"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()
        assert (valid < 1.0).all()

    def test_feret_positive(self, sp_with_polygons):
        """Feret diameters should be positive."""
        df = compute_morphology(sp_with_polygons, metrics=["feret_max", "feret_min"], store=False)
        for col in ["feret_max", "feret_min"]:
            valid = df[col].dropna()
            assert len(valid) > 0
            assert (valid > 0).all()

    def test_feret_ratio_range(self, sp_with_polygons):
        """feret_ratio = min/max, should be in (0, 1]."""
        df = compute_morphology(sp_with_polygons, metrics=["feret_max", "feret_min", "feret_ratio"], store=False)
        valid = df["feret_ratio"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all()
        assert (valid <= 1.0 + 1e-6).all()

    def test_feret_max_ge_feret_min(self, sp_with_polygons):
        df = compute_morphology(sp_with_polygons, metrics=["feret_max", "feret_min"], store=False)
        valid = df.dropna()
        assert (valid["feret_max"] >= valid["feret_min"] - 1e-6).all()

    def test_n_concavities_non_negative(self, sp_with_polygons):
        """Concavity count should be >= 0."""
        df = compute_morphology(sp_with_polygons, metrics=["n_concavities"], store=False)
        valid = df["n_concavities"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_squares_are_rectangular(self, sp_with_polygons):
        """Our test fixtures use square polygons — rectangularity should be ~1.0."""
        df = compute_morphology(sp_with_polygons, metrics=["rectangularity"], store=False)
        valid = df["rectangularity"].dropna()
        assert valid.mean() > 0.95

    def test_store_flag(self, sp_with_polygons):
        compute_morphology(sp_with_polygons, metrics=["rectangularity", "eccentricity"], store=True)
        assert "morph_rectangularity" in sp_with_polygons.cell_meta.columns
        assert "morph_eccentricity" in sp_with_polygons.cell_meta.columns

    def test_full_metrics_includes_new(self, sp_with_polygons):
        """Default metrics=None should include all 14 metrics."""
        df = compute_morphology(sp_with_polygons, store=False)
        assert "rectangularity" in df.columns
        assert "n_concavities" in df.columns
        assert len(df.columns) == 14


# ---------------------------------------------------------------------------
# Context morphology
# ---------------------------------------------------------------------------


class TestContextMorphology:
    """Test compute_context_morphology."""

    @pytest.fixture
    def sp_with_morph(self, sp_with_polygons):
        """sp_with_polygons with morphology already computed."""
        compute_morphology(sp_with_polygons, metrics=["area", "circularity", "elongation"], store=True)
        return sp_with_polygons

    def test_returns_dataframe(self, sp_with_morph):
        graph = build_contact_graph(sp_with_morph)
        df = compute_context_morphology(sp_with_morph, graph, store=False)
        assert isinstance(df, pd.DataFrame)

    def test_all_columns_present(self, sp_with_morph):
        graph = build_contact_graph(sp_with_morph)
        df = compute_context_morphology(sp_with_morph, graph, store=False)
        expected = {
            "area_ratio_neighbors",
            "circularity_deviation",
            "elongation_alignment",
            "local_shape_heterogeneity",
            "morphological_outlier_score",
            "area_ratio_voronoi",
        }
        assert set(df.columns) == expected

    def test_area_ratio_positive(self, sp_with_morph):
        graph = build_contact_graph(sp_with_morph)
        df = compute_context_morphology(sp_with_morph, graph, store=False)
        valid = df["area_ratio_neighbors"].dropna()
        if len(valid) > 0:
            assert (valid > 0).all()

    def test_circularity_deviation_non_negative(self, sp_with_morph):
        graph = build_contact_graph(sp_with_morph)
        df = compute_context_morphology(sp_with_morph, graph, store=False)
        valid = df["circularity_deviation"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all()

    def test_alignment_range(self, sp_with_morph):
        """cos(2*delta) should be in [-1, 1]."""
        graph = build_contact_graph(sp_with_morph)
        df = compute_context_morphology(sp_with_morph, graph, store=False)
        valid = df["elongation_alignment"].dropna()
        if len(valid) > 0:
            assert (valid >= -1 - 1e-6).all()
            assert (valid <= 1 + 1e-6).all()

    def test_voronoi_ratio_positive(self, sp_with_morph):
        graph = build_contact_graph(sp_with_morph)
        df = compute_context_morphology(sp_with_morph, graph, store=False)
        valid = df["area_ratio_voronoi"].dropna()
        if len(valid) > 0:
            assert (valid > 0).all()

    def test_store_flag(self, sp_with_morph):
        graph = build_contact_graph(sp_with_morph)
        compute_context_morphology(sp_with_morph, graph, store=True)
        assert "ctx_morph_area_ratio_neighbors" in sp_with_morph.cell_meta.columns
        assert "ctx_morph_area_ratio_voronoi" in sp_with_morph.cell_meta.columns

    def test_requires_morphology_first(self, sp_with_polygons):
        """Should raise if compute_morphology hasn't been run."""
        graph = build_contact_graph(sp_with_polygons)
        with pytest.raises(ValueError, match="Missing morphology columns"):
            compute_context_morphology(sp_with_polygons, graph, store=False)

    def test_index_matches_sp(self, sp_with_morph):
        graph = build_contact_graph(sp_with_morph)
        df = compute_context_morphology(sp_with_morph, graph, store=False)
        assert list(df.index) == list(sp_with_morph.cell_index)
