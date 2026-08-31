"""Tests for per-cell-type contact-radius optimization (ccc/radius.py).

Geometry under test — two rows of square cells, sized so that the polygon
contact graph is known exactly and each type has a *different* correct radius:

    row A (y=0):    5 squares, side 19.8 um, centres 20 um apart  -> gaps 0.2 um
    row B (y=500):  5 squares, side  7.8 um, centres  8 um apart  -> gaps 0.2 um

Both rows touch within the 0.25 um truth buffer, so the ground-truth contact
graph is exactly the two 4-edge chains. Centroid distance along the chains is
20 um for A and 8 um for B, and the rows are far apart, so:

    optimal radius(A) = 20 um   (8 um misses every A-A contact)
    optimal radius(B) =  8 um   (20 um invents B-B edges two cells apart)

which is the whole point of the feature: one radius cannot serve both.
"""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.ccc.radius import (
    RadiusOptimizationResult,
    build_typed_radius_graph,
    optimize_contact_radius,
)
from spatioloji_s.data.core import spatioloji

N_A, N_B = 5, 5
SIDE_A, PITCH_A = 19.8, 20.0
SIDE_B, PITCH_B = 7.8, 8.0
ROW_B_Y = 500.0


@pytest.fixture
def sp_two_scales():
    """Two rows of square cells with deliberately different sizes/pitches."""
    centres, types = [], []
    for i in range(N_A):
        centres.append((i * PITCH_A, 0.0))
        types.append("Big")
    for i in range(N_B):
        centres.append((i * PITCH_B, ROW_B_Y))
        types.append("Small")

    cell_ids = [f"cell_{i}" for i in range(len(centres))]
    rows = []
    for cid, (cx, cy), t in zip(cell_ids, centres, types, strict=True):
        h = (SIDE_A if t == "Big" else SIDE_B) / 2
        for vx, vy in [
            (cx - h, cy - h),
            (cx + h, cy - h),
            (cx + h, cy + h),
            (cx - h, cy + h),
            (cx - h, cy - h),
        ]:
            rows.append({"cell": cid, "x_global_px": vx, "y_global_px": vy})

    n = len(cell_ids)
    rng = np.random.default_rng(0)
    xs = np.array([c[0] for c in centres], dtype=float)
    ys = np.array([c[1] for c in centres], dtype=float)
    return spatioloji(
        expression=rng.integers(1, 6, (n, 4)).astype(float),
        cell_ids=cell_ids,
        gene_names=["LIG", "REC", "G3", "G4"],
        cell_metadata=pd.DataFrame({"cell_type": types}, index=cell_ids),
        spatial_coords={"x_global": xs, "y_global": ys, "x_local": xs, "y_local": ys},
        polygons=pd.DataFrame(rows),
    )


RADII = (4.0, 8.0, 12.0, 20.0, 24.0)


class TestOptimizeContactRadius:
    def test_returns_result_with_expected_frames(self, sp_two_scales):
        res = optimize_contact_radius(
            sp_two_scales,
            group_col="cell_type",
            radii=RADII,
            min_type_edges=1,
            min_pair_edges=1,
            verbose=False,
        )
        assert isinstance(res, RadiusOptimizationResult)
        for col in ("cell_type", "best_radius", "jaccard", "n_truth_edges", "missed_rate", "false_rate"):
            assert col in res.per_type.columns
        assert {"pair", "best_radius", "jaccard"} <= set(res.per_pair.columns)
        assert set(res.sweep["radius"]) == set(RADII)

    def test_finds_the_distinct_per_type_optimum(self, sp_two_scales):
        res = optimize_contact_radius(
            sp_two_scales,
            group_col="cell_type",
            radii=RADII,
            min_type_edges=1,
            min_pair_edges=1,
            verbose=False,
        )
        assert res.radius_map["Big"] == 20.0
        assert res.radius_map["Small"] == 8.0
        # a perfect fit is reachable for this synthetic geometry
        assert res.per_type.set_index("cell_type").loc["Small", "jaccard"] == pytest.approx(1.0)

    def test_per_pair_table_matches_b3b_shape(self, sp_two_scales):
        res = optimize_contact_radius(
            sp_two_scales,
            group_col="cell_type",
            radii=RADII,
            min_type_edges=1,
            min_pair_edges=1,
            verbose=False,
        )
        pairs = set(res.per_pair["pair"])
        assert "Big|Big" in pairs and "Small|Small" in pairs
        opt = res.per_pair.set_index("pair")["best_radius"]
        assert opt["Big|Big"] == 20.0
        assert opt["Small|Small"] == 8.0

    def test_accepts_precomputed_truth_graph(self, sp_two_scales):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        truth = build_buffer_graph(sp_two_scales, buffer_distance=0.25)
        res = optimize_contact_radius(
            sp_two_scales,
            group_col="cell_type",
            radii=RADII,
            truth_graph=truth,
            min_type_edges=1,
            min_pair_edges=1,
            verbose=False,
        )
        assert res.radius_map["Big"] == 20.0

    def test_rejects_unknown_group_col(self, sp_two_scales):
        with pytest.raises(ValueError, match="not found"):
            optimize_contact_radius(sp_two_scales, group_col="nope", radii=RADII, verbose=False)


class TestBuildTypedRadiusGraph:
    def test_union_rule_edges(self, sp_two_scales):
        g = build_typed_radius_graph(
            sp_two_scales,
            radius_map={"Big": 20.0, "Small": 8.0},
            group_col="cell_type",
        )
        ids = list(g.cell_ids)
        adj = g.adjacency.toarray()

        def connected(a, b):
            return adj[ids.index(a), ids.index(b)] > 0

        assert connected("cell_0", "cell_1")  # Big-Big, d=20 <= max(20,20)
        assert not connected("cell_0", "cell_2")  # Big-Big, d=40 > 20
        assert connected("cell_5", "cell_6")  # Small-Small, d=8 <= 8
        assert not connected("cell_5", "cell_7")  # Small-Small, d=16 > 8

    def test_union_uses_the_larger_radius_for_mixed_pairs(self, sp_two_scales):
        # Place the two rows 12 um apart: reachable under Big's radius only.
        sp = sp_two_scales
        sp._spatial.y_global[N_A:] = 12.0
        sp._spatial.y_local[N_A:] = 12.0
        g = build_typed_radius_graph(
            sp,
            radius_map={"Big": 20.0, "Small": 8.0},
            group_col="cell_type",
        )
        ids = list(g.cell_ids)
        adj = g.adjacency.toarray()
        # cell_0 (Big, 0,0) and cell_5 (Small, 0,12): d=12 <= max(20,8)=20 -> edge
        assert adj[ids.index("cell_0"), ids.index("cell_5")] > 0

    def test_symmetric_and_carries_uniform_contact_fractions(self, sp_two_scales):
        g = build_typed_radius_graph(
            sp_two_scales,
            radius_map={"Big": 20.0, "Small": 8.0},
            group_col="cell_type",
        )
        adj = g.adjacency.toarray()
        np.testing.assert_array_equal(adj, adj.T)
        cdf = g.contact_frac_df
        assert {"cell_a", "cell_b", "fraction_a", "fraction_b"} <= set(cdf.columns)
        # uniform weighting: every juxtacrine edge weighs 1.0
        assert (cdf["fraction_a"] == 1.0).all()
        assert (cdf["fraction_b"] == 1.0).all()
        assert len(cdf) == g.n_edges

    def test_missing_type_falls_back_to_default(self, sp_two_scales):
        g = build_typed_radius_graph(
            sp_two_scales,
            radius_map={"Big": 20.0},
            group_col="cell_type",
            default_radius=8.0,
        )
        ids = list(g.cell_ids)
        assert g.adjacency.toarray()[ids.index("cell_5"), ids.index("cell_6")] > 0

    def test_missing_type_without_default_raises(self, sp_two_scales):
        with pytest.raises(ValueError, match="Small"):
            build_typed_radius_graph(
                sp_two_scales,
                radius_map={"Big": 20.0},
                group_col="cell_type",
            )

    def test_graph_scores_juxtacrine_edges(self, sp_two_scales):
        """The typed graph must drop straight into score_edges as the juxtacrine graph."""
        from spatioloji_s.ccc.database import LRPair
        from spatioloji_s.ccc.scoring import score_edges

        g = build_typed_radius_graph(
            sp_two_scales,
            radius_map={"Big": 20.0, "Small": 8.0},
            group_col="cell_type",
        )
        pairs = [LRPair(lr_name="LIG-REC", ligand="LIG", receptor="REC", pathway="test", lr_type="juxtacrine")]
        edges = score_edges(sp_two_scales, pairs, graph_juxtacrine=g, group_col="cell_type", include_autocrine=False)
        assert len(edges) > 0
        assert (edges["weight"] == 1.0).all()


class TestRunCCCIntegration:
    """CCCConfig.juxtacrine_radius_map swaps the polygon graph for the typed one."""

    def test_run_ccc_uses_typed_radius_graph(self, sp_two_scales):
        from spatioloji_s.ccc import CCCConfig, run_ccc
        from spatioloji_s.ccc.database import LRPair

        pairs = [LRPair(lr_name="LIG-REC", ligand="LIG", receptor="REC", pathway="test", lr_type="juxtacrine")]
        config = CCCConfig(
            group_col="cell_type",
            juxtacrine_radius_map={"Big": 20.0, "Small": 8.0},
            include_autocrine=False,
            verbose=False,
        )
        result = run_ccc(sp_two_scales, config, lr_pairs=pairs)
        assert result.edge_df is not None and len(result.edge_df) > 0
        # uniform juxtacrine weighting under the radius graph
        assert (result.edge_df["weight"] == 1.0).all()
        # Big-Big contacts (d=20) only exist under the fitted per-type radius
        big = result.edge_df[(result.edge_df.sender_type == "Big") & (result.edge_df.receiver_type == "Big")]
        assert len(big) > 0

    def test_exported_from_package_namespace(self):
        import spatioloji_s as sj

        assert hasattr(sj.ccc, "optimize_contact_radius")
        assert hasattr(sj.ccc, "build_typed_radius_graph")
        assert hasattr(sj.ccc, "RadiusOptimizationResult")
