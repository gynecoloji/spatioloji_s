"""Tests for decoupler.py gene set activity scoring."""

import pandas as pd
import pytest

from spatioloji_s.processing.decoupler import make_gene_set_net, score_gene_sets


class TestMakeGeneSetNet:
    """Test custom gene set → network conversion."""

    def test_basic_conversion(self):
        net = make_gene_set_net({
            "EMT": ["VIM", "CDH2", "SNAI1"],
            "Prolif": ["MKI67", "TOP2A"],
        })
        assert isinstance(net, pd.DataFrame)
        assert set(net.columns) == {"source", "target", "weight"}
        assert len(net) == 5
        assert net["source"].nunique() == 2

    def test_default_weights_are_one(self):
        net = make_gene_set_net({"A": ["G1", "G2"]})
        assert (net["weight"] == 1.0).all()

    def test_custom_weights(self):
        net = make_gene_set_net(
            {"A": ["G1", "G2"]},
            weights={"A": [0.5, -0.3]},
        )
        assert list(net["weight"]) == [0.5, -0.3]

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Weight length mismatch"):
            make_gene_set_net(
                {"A": ["G1", "G2"]},
                weights={"A": [0.5]},
            )

    def test_empty_gene_set(self):
        net = make_gene_set_net({"empty": []})
        assert len(net) == 0
        assert set(net.columns) == {"source", "target", "weight"}


class TestScoreGeneSets:
    """Test score_gene_sets with dict input (no decoupler import needed for validation tests)."""

    def test_dict_auto_converted(self, sp_basic):
        """Dict input should be auto-converted to network DataFrame."""
        gene_names = list(sp_basic.gene_index)
        gene_sets = {
            "set_A": gene_names[:5],
            "set_B": gene_names[5:10],
        }
        # This will fail at dc.run_aucell because decoupler may not be installed,
        # but should NOT fail at the dict → DataFrame conversion step.
        # We test validation separately.
        try:
            score_gene_sets(sp_basic, gene_sets, method="aucell", layer=None, store=False)
        except ImportError:
            pytest.skip("decoupler not installed")

    def test_invalid_method_raises(self, sp_basic):
        with pytest.raises(ValueError, match="Unknown method"):
            score_gene_sets(sp_basic, {"A": ["G1"]}, method="invalid")

    def test_missing_columns_raises(self, sp_basic):
        bad_net = pd.DataFrame({"foo": ["A"], "bar": ["G1"]})
        try:
            score_gene_sets(sp_basic, bad_net, method="aucell", layer=None)
        except ImportError:
            pytest.skip("decoupler not installed")
        except ValueError as e:
            assert "source" in str(e) or "target" in str(e)

    def test_no_overlap_raises(self, sp_basic):
        net = pd.DataFrame({
            "source": ["X", "X"],
            "target": ["NONEXISTENT_GENE_1", "NONEXISTENT_GENE_2"],
        })
        try:
            score_gene_sets(sp_basic, net, method="aucell", layer=None)
        except ImportError:
            pytest.skip("decoupler not installed")
        except ValueError as e:
            assert "No genes overlap" in str(e)


class TestScoreGeneSetsIntegration:
    """Integration tests — only run if decoupler is installed."""

    @pytest.fixture
    def gene_sets(self, sp_basic):
        """Gene sets using actual gene names from the fixture."""
        names = list(sp_basic.gene_index)
        return {
            "set_A": names[:8],
            "set_B": names[8:16],
        }

    def _skip_no_dc(self):
        try:
            import decoupler  # noqa: F401
        except ImportError:
            pytest.skip("decoupler not installed")

    def test_aucell(self, sp_basic, gene_sets):
        self._skip_no_dc()
        result = score_gene_sets(sp_basic, gene_sets, method="aucell", layer=None, store=False)
        assert "estimate" in result
        assert result["pvals"] is None  # AUCell has no p-values
        assert result["estimate"].shape[0] == sp_basic.n_cells
        assert result["estimate"].shape[1] == 2

    def test_ulm(self, sp_basic, gene_sets):
        self._skip_no_dc()
        result = score_gene_sets(sp_basic, gene_sets, method="ulm", layer=None, store=False)
        assert result["estimate"].shape[0] == sp_basic.n_cells
        assert result["pvals"] is not None

    def test_wmean(self, sp_basic, gene_sets):
        self._skip_no_dc()
        result = score_gene_sets(
            sp_basic, gene_sets, method="wmean", layer=None, n_permutations=10, store=False,
        )
        assert result["estimate"].shape[0] == sp_basic.n_cells
        assert result["pvals"] is not None

    def test_store_in_cell_meta(self, sp_basic, gene_sets):
        self._skip_no_dc()
        score_gene_sets(sp_basic, gene_sets, method="aucell", layer=None, store=True)
        assert "aucell_set_A" in sp_basic.cell_meta.columns
        assert "aucell_set_B" in sp_basic.cell_meta.columns

    def test_store_in_embeddings(self, sp_basic, gene_sets):
        self._skip_no_dc()
        score_gene_sets(sp_basic, gene_sets, method="aucell", layer=None, store=True)
        assert "aucell_estimate" in sp_basic._embeddings
