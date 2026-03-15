"""Tests for proximity contact mode in boundaries.py."""

import numpy as np
import pandas as pd
import pytest


class TestProximityContactLength:
    """Tests for contact_length with buffer graphs."""

    @pytest.fixture
    def sp_pair(self):
        """Two cells separated by a gap — only connected via buffer graph."""
        from spatioloji_s.data.core import spatioloji

        cell_ids = ["cell_a", "cell_b"]
        # Two 10x10 squares, 20px apart (gap = 20)
        # cell_a at (0, 0), cell_b at (30, 0)
        polygons = pd.DataFrame(
            {
                "cell": ["cell_a"] * 5 + ["cell_b"] * 5,
                "x_global_px": [0, 10, 10, 0, 0, 30, 40, 40, 30, 30],
                "y_global_px": [0, 0, 10, 10, 0, 0, 0, 10, 10, 0],
            }
        )

        return spatioloji(
            expression=np.array([[1.0, 2.0], [3.0, 4.0]]),
            cell_ids=cell_ids,
            gene_names=["gene_0", "gene_1"],
            cell_metadata=pd.DataFrame({"cell_type": ["A", "B"]}, index=cell_ids),
            spatial_coords={
                "x_global": [5.0, 35.0],
                "y_global": [5.0, 5.0],
                "x_local": [5.0, 35.0],
                "y_local": [5.0, 5.0],
                "fov": [1, 1],
            },
            polygons=polygons,
        )

    def test_contact_graph_returns_zero_for_gap(self, sp_pair):
        """Contact graph: non-touching cells should have zero contact length."""
        from spatioloji_s.spatial.polygon.boundaries import contact_length
        from spatioloji_s.spatial.polygon.graph import build_contact_graph

        graph = build_contact_graph(sp_pair)
        edges = contact_length(sp_pair, graph)
        # Contact graph should have no edges (cells don't touch)
        assert len(edges) == 0 or (edges["contact_length_a"] == 0).all()

    def test_buffer_graph_returns_nonzero_for_gap(self, sp_pair):
        """Buffer graph: non-touching cells within buffer distance get facing length > 0."""
        from spatioloji_s.spatial.polygon.boundaries import contact_length
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        # Buffer distance 25 > gap of 20, so cells are connected
        graph = build_buffer_graph(sp_pair, buffer_distance=25)
        edges = contact_length(sp_pair, graph)

        assert len(edges) > 0
        # At least one pair should have positive contact length
        assert (edges["contact_length_a"] > 0).any()
        assert (edges["contact_length_b"] > 0).any()

    def test_buffer_fraction_in_range(self, sp_pair):
        """Buffer graph contact fractions should be in [0, 1]."""
        from spatioloji_s.spatial.polygon.boundaries import contact_fraction
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        graph = build_buffer_graph(sp_pair, buffer_distance=25)
        edges = contact_fraction(sp_pair, graph, store=False)

        if len(edges) > 0:
            assert (edges["fraction_a"] >= 0).all()
            assert (edges["fraction_a"] <= 1).all()
            assert (edges["fraction_b"] >= 0).all()
            assert (edges["fraction_b"] <= 1).all()

    def test_free_boundary_plus_contact_equals_one(self, sp_pair):
        """free_boundary_fraction + total_contact_fraction should equal 1."""
        from spatioloji_s.spatial.polygon.boundaries import free_boundary_fraction
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        graph = build_buffer_graph(sp_pair, buffer_distance=25)
        free_frac = free_boundary_fraction(sp_pair, graph, store=True)

        total_contact = sp_pair.cell_meta["total_contact_fraction"]
        result = free_frac.values + total_contact.values
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


class TestProximityWithMotifFixture:
    """Tests using the larger sp_motif fixture."""

    def test_buffer_graph_more_contact_than_contact_graph(self, sp_motif):
        """Buffer graph should detect more contact than direct contact graph."""
        from spatioloji_s.spatial.polygon.boundaries import contact_length
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph, build_contact_graph

        contact_g = build_contact_graph(sp_motif)
        buffer_g = build_buffer_graph(sp_motif, buffer_distance=20)

        edges_contact = contact_length(sp_motif, contact_g)
        edges_buffer = contact_length(sp_motif, buffer_g)

        total_contact = edges_contact["contact_length_a"].sum() if len(edges_contact) > 0 else 0
        total_buffer = edges_buffer["contact_length_a"].sum() if len(edges_buffer) > 0 else 0

        # Buffer mode should find at least as much total facing length
        assert total_buffer >= total_contact


class TestFallbackBehavior:
    """Test fallback when buffer_distance missing from params."""

    def test_missing_buffer_distance_warns(self, sp_motif):
        """Should warn and fall back to direct contact if buffer_distance missing."""
        from spatioloji_s.spatial.polygon.boundaries import contact_length
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        graph = build_buffer_graph(sp_motif, buffer_distance=20)
        # Remove buffer_distance from params to trigger fallback
        graph.params.pop("buffer_distance", None)

        with pytest.warns(UserWarning, match="buffer_distance"):
            contact_length(sp_motif, graph)
