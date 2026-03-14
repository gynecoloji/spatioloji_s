"""Tests for hierarchical spatial motif discovery."""

import pandas as pd

from spatioloji_s.spatial._motif_types import (
    AssemblyCatalog,
    MotifCatalog,
    MotifResult,
    StructureMatches,
)


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
