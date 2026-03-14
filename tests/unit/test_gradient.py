"""Tests for spatial gradient analysis."""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.spatial._gradient_types import GradientResult


class TestComputeGradientBasic:
    """Basic gradient computation tests."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        """Create InterfaceResult from sp_gradient fixture."""
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        return identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )

    def test_returns_gradient_result(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0", "gene_1"])
        assert isinstance(result, GradientResult)

    def test_gene_gradients_shape(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0", "gene_1", "gene_2"])
        assert result.gene_gradients.shape[0] == 3
        assert set(result.gene_gradients.columns) >= {"coef", "pvalue", "r2", "trend"}

    def test_gene_gradients_all_genes(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=None)
        assert result.gene_gradients.shape[0] == 10

    def test_bins_dataframe_columns(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"], n_bins=5)
        assert set(result.bins.columns) >= {"distance_bin", "gene", "mean_expr", "std_expr"}
        assert len(result.bins) <= 5

    def test_distances_series(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"])
        assert isinstance(result.distances, pd.Series)
        assert len(result.distances) == len(sp_gradient.cell_index)

    def test_trend_labels(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"])
        valid_trends = {"increasing_toward_a", "increasing_toward_b", "flat"}
        assert set(result.gene_gradients["trend"].unique()).issubset(valid_trends)

    def test_region_labels_propagated(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"])
        assert result.region_a == interface_result.region_a
        assert result.region_b == interface_result.region_b


class TestComputeGradientPrograms:
    """Tests for gene program gradient analysis."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface
        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        return identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )

    def test_user_programs(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        programs = {"gradient_set": ["gene_0", "gene_1"], "noise_set": ["gene_2", "gene_3"]}
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"], programs=programs)
        assert result.program_gradients.shape[0] == 2
        assert "gradient_set" in result.program_gradients.index
        assert result.program_scores.shape == (len(sp_gradient.cell_index), 2)

    def test_no_programs(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"])
        assert result.program_gradients.empty
        assert result.program_scores.empty


class TestComputeGradientAutoPrograms:
    """Tests for auto-discovered gene programs."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface
        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        return identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )

    def test_nmf_auto_programs(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"], auto_programs="nmf", n_auto_programs=3)
        assert result.program_gradients.shape[0] == 3

    def test_pca_auto_programs(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"], auto_programs="pca", n_auto_programs=3)
        assert result.program_gradients.shape[0] == 3

    def test_invalid_auto_programs(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        with pytest.raises(ValueError, match="auto_programs"):
            compute_gradient(sp_gradient, interface_result, auto_programs="invalid")


class TestComputeGradientValidation:
    """Validation and edge case tests."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface
        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        return identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )

    def test_invalid_method(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        with pytest.raises(ValueError, match="method"):
            compute_gradient(sp_gradient, interface_result, method="invalid")

    def test_missing_genes(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        with pytest.raises(ValueError, match="not found"):
            compute_gradient(sp_gradient, interface_result, genes=["nonexistent_gene"])

    def test_coord_type_local(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"], coord_type="local")
        assert isinstance(result, GradientResult)

    def test_unsigned_gradient(self, sp_gradient, interface_result):
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"], unsigned=True)
        assert (result.distances >= 0).all()


class TestPointGradientReExport:
    """Verify point module re-exports polygon gradient."""

    def test_point_compute_gradient_is_same(self):
        from spatioloji_s.spatial.point.gradient import compute_gradient as point_cg
        from spatioloji_s.spatial.polygon.gradient import compute_gradient as poly_cg
        assert point_cg is poly_cg
