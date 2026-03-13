"""Unit tests for processing/DEG.py."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon

from spatioloji_s.processing.DEG import (
    _apply_correction,
    _build_cell_mask,
    _build_result_df,
)


# ---------------------------------------------------------------------------
# _build_cell_mask tests
# ---------------------------------------------------------------------------


class TestBuildCellMask:
    def test_one_vs_rest(self, sp_deg):
        fg, bg = _build_cell_mask(sp_deg, "cell_type", "TypeA", "rest", None, 10)
        assert len(fg) == 250
        assert len(bg) == 250
        # fg indices are all < 250 (TypeA cells are first 250)
        assert (fg < 250).all()
        assert (bg >= 250).all()

    def test_pairwise(self, sp_deg):
        fg, bg = _build_cell_mask(sp_deg, "cell_type", "TypeA", "TypeB", None, 10)
        assert len(fg) == 250
        assert len(bg) == 250

    def test_custom_multi_group(self, sp_deg):
        fg, bg = _build_cell_mask(sp_deg, "cell_type", ["TypeA"], ["TypeB"], None, 10)
        assert len(fg) == 250
        assert len(bg) == 250

    def test_invalid_groupby_raises(self, sp_deg):
        with pytest.raises(ValueError, match="not found in cell_meta columns"):
            _build_cell_mask(sp_deg, "nonexistent", "TypeA", "rest", None, 10)

    def test_invalid_group_fg_raises(self, sp_deg):
        with pytest.raises(ValueError, match="group_fg values"):
            _build_cell_mask(sp_deg, "cell_type", "TypeC", "rest", None, 10)

    def test_min_cells_fg_raises(self, sp_deg):
        # min_cells=999 > 250 TypeA cells -> foreground check triggers first
        with pytest.raises(ValueError, match="Foreground has"):
            _build_cell_mask(sp_deg, "cell_type", "TypeA", "rest", None, min_cells=999)

    def test_min_cells_bg_raises(self):
        """bg check enforced independently from fg check."""
        # Build a minimal sp where fg=10 cells, bg=5 cells, min_cells=8
        # fg (10) passes the guard; bg (5) does not -> raises "Background has"
        from spatioloji_s.data.core import spatioloji as _spatioloji

        rng = np.random.default_rng(0)
        n = 15
        expr = rng.random((n, 5))
        cell_ids = [f"c{i}" for i in range(n)]
        cm = pd.DataFrame({"grp": ["A"] * 10 + ["B"] * 5}, index=cell_ids)
        sp_small = _spatioloji(
            expression=expr,
            cell_ids=cell_ids,
            gene_names=[f"g{j}" for j in range(5)],
            cell_metadata=cm,
            spatial_coords={
                "x_global": np.zeros(n),
                "y_global": np.zeros(n),
                "x_local": np.zeros(n),
                "y_local": np.zeros(n),
            },
        )
        with pytest.raises(ValueError, match="Background has"):
            _build_cell_mask(sp_small, "grp", "A", "B", None, min_cells=8)

    def test_spatial_filter_bbox(self, sp_deg):
        # Use pairwise contrast (TypeA vs TypeB) with bbox that includes both types
        # TypeA in x [0, 500], TypeB in x [500, 1000]; bbox [0, 800] reduces TypeB
        sf = {"x_range": (0, 800), "y_range": (0, 1000)}
        fg, bg = _build_cell_mask(sp_deg, "cell_type", "TypeA", "TypeB", sf, min_cells=5)
        x = sp_deg.spatial.x_global
        assert (x[fg] <= 800).all()
        assert (x[bg] <= 800).all()
        # TypeB bg should be reduced (some cells have x > 800)
        assert len(bg) < 250

    def test_spatial_filter_polygon(self, sp_deg):
        # Polygon covering x [0, 800] x y [0, 1000] — includes both types
        poly = Polygon([(0, 0), (800, 0), (800, 1000), (0, 1000)])
        sf = {"polygon": poly}
        fg, bg = _build_cell_mask(sp_deg, "cell_type", "TypeA", "TypeB", sf, min_cells=5)
        x = sp_deg.spatial.x_global
        assert (x[fg] <= 800).all()
        assert (x[bg] <= 800).all()

    def test_indices_are_positional(self, sp_deg):
        """fg_idx/bg_idx must be usable as positional row indices into X."""
        fg, bg = _build_cell_mask(sp_deg, "cell_type", "TypeA", "rest", None, 10)
        assert np.issubdtype(fg.dtype, np.signedinteger), f"Expected signed int, got {fg.dtype}"
        assert fg.max() < len(sp_deg.cell_meta)


# ---------------------------------------------------------------------------
# _apply_correction tests
# ---------------------------------------------------------------------------


class TestApplyCorrection:
    def test_bh_monotone(self):
        """BH-adjusted p-values must be non-decreasing when sorted by raw p."""
        # Use unsorted input to exercise the full rank-mapping logic
        pvals = np.array([0.05, 0.001, 0.1, 0.02, 0.5, 0.01])
        padj = _apply_correction(pvals, method="fdr_bh")
        # Monotonicity must hold in SORTED order, not input order
        sorted_padj = padj[np.argsort(pvals)]
        assert (sorted_padj >= pvals[np.argsort(pvals)]).all(), "padj must be >= raw pval"
        assert (np.diff(sorted_padj) >= -1e-10).all(), "padj must be non-decreasing in sorted order"

    def test_bh_nan_passthrough(self):
        """NaN p-values must remain NaN in padj."""
        pvals = np.array([0.01, np.nan, 0.05])
        padj = _apply_correction(pvals, method="fdr_bh")
        assert np.isnan(padj[1])
        assert not np.isnan(padj[0])

    def test_bonferroni(self):
        pvals = np.array([0.01, 0.05])
        padj = _apply_correction(pvals, method="bonferroni")
        np.testing.assert_allclose(padj, np.minimum(pvals * 2, 1.0))

    def test_all_nan_returns_nan(self):
        pvals = np.array([np.nan, np.nan])
        padj = _apply_correction(pvals)
        assert np.all(np.isnan(padj))


# ---------------------------------------------------------------------------
# _build_result_df tests
# ---------------------------------------------------------------------------


class TestBuildResultDf:
    def test_schema(self):
        """Output DataFrame must have all required columns, sorted by padj."""
        n_genes = 10
        gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
        stats = {
            "pval": np.random.uniform(0, 1, n_genes),
            "mean_fg": np.random.rand(n_genes),
            "mean_bg": np.random.rand(n_genes),
            "pct_fg": np.random.rand(n_genes),
            "pct_bg": np.random.rand(n_genes),
        }
        padj = stats["pval"] * 2  # dummy
        df = _build_result_df(gene_names, stats, padj, n_fg=50, n_bg=50)

        required = {"gene", "log2fc", "mean_fg", "mean_bg", "pct_fg", "pct_bg", "pval", "padj", "n_fg", "n_bg"}
        assert required.issubset(set(df.columns))
        assert df.shape[0] == n_genes

    def test_sorted_by_padj_nan_last(self):
        gene_names = np.array(["g0", "g1", "g2"])
        stats = {
            "pval": np.array([0.5, 0.1, np.nan]),
            "mean_fg": np.ones(3),
            "mean_bg": np.ones(3) * 0.5,
            "pct_fg": np.ones(3),
            "pct_bg": np.ones(3),
        }
        padj = np.array([0.5, 0.1, np.nan])
        df = _build_result_df(gene_names, stats, padj, n_fg=10, n_bg=10)
        # Sorted: 0.1, 0.5, NaN
        assert df.iloc[0]["gene"] == "g1"
        assert np.isnan(df.iloc[-1]["padj"])

    def test_log2fc_formula(self):
        """log2fc = log2((mean_fg + 1e-9) / (mean_bg + 1e-9))."""
        gene_names = np.array(["g0"])
        stats = {
            "pval": np.array([0.01]),
            "mean_fg": np.array([4.0]),
            "mean_bg": np.array([1.0]),
            "pct_fg": np.array([1.0]),
            "pct_bg": np.array([0.5]),
        }
        padj = np.array([0.05])
        df = _build_result_df(gene_names, stats, padj, n_fg=10, n_bg=10)
        expected_log2fc = np.log2((4.0 + 1e-9) / (1.0 + 1e-9))
        np.testing.assert_allclose(df["log2fc"].values[0], expected_log2fc, rtol=1e-5)


# ---------------------------------------------------------------------------
# Backend tests — synthetic data with known signal
# ---------------------------------------------------------------------------


def _make_fg_bg(n_fg=250, n_bg=250, n_genes=50, seed=0):
    """Synthetic fg/bg dense float32 arrays with known upregulation in fg genes 0-24."""
    rng = np.random.default_rng(seed)
    X_fg = rng.poisson(1.0, (n_fg, n_genes)).astype(np.float32)
    n_up = min(25, n_genes)
    X_fg[:, :n_up] += rng.poisson(8.0, (n_fg, n_up)).astype(np.float32)
    X_bg = rng.poisson(1.0, (n_bg, n_genes)).astype(np.float32)
    return X_fg, X_bg


class TestWilcoxonBackend:
    def test_returns_required_keys(self):
        from spatioloji_s.processing.DEG import _wilcoxon_backend

        X_fg, X_bg = _make_fg_bg(n_genes=10)
        result = _wilcoxon_backend(X_fg, X_bg)
        assert set(result.keys()) == {"pval", "mean_fg", "mean_bg", "pct_fg", "pct_bg"}

    def test_pval_shape(self):
        from spatioloji_s.processing.DEG import _wilcoxon_backend

        X_fg, X_bg = _make_fg_bg(n_genes=15)
        result = _wilcoxon_backend(X_fg, X_bg)
        assert result["pval"].shape == (15,)

    def test_pval_in_range(self):
        from spatioloji_s.processing.DEG import _wilcoxon_backend

        X_fg, X_bg = _make_fg_bg(n_genes=10)
        result = _wilcoxon_backend(X_fg, X_bg)
        assert (result["pval"] >= 0).all() and (result["pval"] <= 1).all()

    def test_detects_upregulated_genes(self):
        """Genes 0-24 (upregulated in fg) should have very small p-values."""
        from spatioloji_s.processing.DEG import _wilcoxon_backend

        X_fg, X_bg = _make_fg_bg(n_fg=250, n_bg=250, n_genes=50)
        result = _wilcoxon_backend(X_fg, X_bg)
        assert result["pval"][:25].max() < 0.05
        assert result["mean_fg"][:25].mean() > result["mean_fg"][25:].mean()

    def test_n_jobs_identical_results(self):
        """n_jobs=1 and n_jobs=4 must produce identical results."""
        from spatioloji_s.processing.DEG import _wilcoxon_backend

        X_fg, X_bg = _make_fg_bg(n_genes=20)
        r1 = _wilcoxon_backend(X_fg, X_bg, n_jobs=1)
        r4 = _wilcoxon_backend(X_fg, X_bg, n_jobs=4)
        np.testing.assert_array_equal(r1["pval"], r4["pval"])


class TestTtestBackend:
    def test_returns_required_keys(self):
        from spatioloji_s.processing.DEG import _ttest_backend

        X_fg, X_bg = _make_fg_bg(n_genes=10)
        result = _ttest_backend(X_fg, X_bg)
        assert set(result.keys()) == {"pval", "mean_fg", "mean_bg", "pct_fg", "pct_bg"}

    def test_pval_shape(self):
        from spatioloji_s.processing.DEG import _ttest_backend

        X_fg, X_bg = _make_fg_bg(n_genes=15)
        assert _ttest_backend(X_fg, X_bg)["pval"].shape == (15,)

    def test_detects_upregulated_genes(self):
        from spatioloji_s.processing.DEG import _ttest_backend

        X_fg, X_bg = _make_fg_bg(n_fg=250, n_bg=250, n_genes=50)
        result = _ttest_backend(X_fg, X_bg)
        assert result["pval"][:25].max() < 0.05

    def test_n_jobs_ignored_but_identical(self):
        """t-test is fully vectorized; n_jobs is accepted but results are identical."""
        from spatioloji_s.processing.DEG import _ttest_backend

        X_fg, X_bg = _make_fg_bg(n_genes=20)
        r1 = _ttest_backend(X_fg, X_bg, n_jobs=1)
        r4 = _ttest_backend(X_fg, X_bg, n_jobs=4)
        np.testing.assert_array_equal(r1["pval"], r4["pval"])


# ---------------------------------------------------------------------------
# run_deg integration tests (Wilcoxon + t-test paths)
# ---------------------------------------------------------------------------


class TestRunDegCore:
    def test_returns_dict(self, sp_deg):
        from spatioloji_s.processing.DEG import run_deg

        results = run_deg(sp_deg, "cell_type", "TypeA", methods=["wilcoxon"])
        assert isinstance(results, dict)
        assert "wilcoxon" in results

    def test_output_schema(self, sp_deg):
        from spatioloji_s.processing.DEG import run_deg

        results = run_deg(sp_deg, "cell_type", "TypeA", methods=["ttest"])
        df = results["ttest"]
        required = {"gene", "log2fc", "mean_fg", "mean_bg", "pct_fg", "pct_bg", "pval", "padj", "n_fg", "n_bg"}
        assert required.issubset(set(df.columns))
        assert df.shape[0] == 50  # 50 genes in sp_deg

    def test_sorted_by_padj(self, sp_deg):
        from spatioloji_s.processing.DEG import run_deg

        df = run_deg(sp_deg, "cell_type", "TypeA", methods=["wilcoxon"])["wilcoxon"]
        non_nan = df["padj"].dropna()
        assert (non_nan.diff().dropna() >= 0).all()

    def test_both_methods_returned(self, sp_deg):
        from spatioloji_s.processing.DEG import run_deg

        results = run_deg(sp_deg, "cell_type", "TypeA", methods=["wilcoxon", "ttest"])
        assert "wilcoxon" in results and "ttest" in results

    def test_unknown_method_raises(self, sp_deg):
        from spatioloji_s.processing.DEG import run_deg

        with pytest.raises(ValueError, match="Unknown method"):
            run_deg(sp_deg, "cell_type", "TypeA", methods=["bad_method"])

    def test_deseq2_without_replicate_key_raises(self, sp_deg):
        from spatioloji_s.processing.DEG import run_deg

        with pytest.raises(ValueError, match="replicate_key"):
            run_deg(sp_deg, "cell_type", "TypeA", methods=["deseq2"])

    def test_invalid_replicate_key_raises(self, sp_deg):
        from spatioloji_s.processing.DEG import run_deg

        with pytest.raises(ValueError, match="replicate_key"):
            run_deg(sp_deg, "cell_type", "TypeA", methods=["deseq2"], replicate_key="nonexistent")

    def test_detects_true_deg(self, sp_deg):
        """Genes 0-24 upregulated in TypeA should be top hits."""
        from spatioloji_s.processing.DEG import run_deg

        df = run_deg(sp_deg, "cell_type", "TypeA", methods=["wilcoxon"])["wilcoxon"]
        top_genes = set(df.head(25)["gene"].values)
        expected = {f"gene_{i}" for i in range(25)}
        # At least 10 of the top 25 hits should be true positives
        # (both groups have signal, so some bg genes are also significant)
        assert len(top_genes & expected) >= 10

    def test_sparse_input(self, sp_deg):
        """run_deg must work when the expression matrix is sparse."""
        from spatioloji_s.processing.DEG import run_deg

        # Add a sparse layer
        X = sp_deg.expression.get_dense()
        sp_deg.add_layer("sparse_layer", csr_matrix(X), overwrite=True)
        results = run_deg(sp_deg, "cell_type", "TypeA", methods=["ttest"], layer="sparse_layer")
        assert "ttest" in results

    def test_one_vs_rest(self, sp_deg):
        from spatioloji_s.processing.DEG import run_deg

        df = run_deg(sp_deg, "cell_type", "TypeA", group_bg="rest", methods=["ttest"])["ttest"]
        assert df["n_fg"].iloc[0] == 250
        assert df["n_bg"].iloc[0] == 250

    def test_spatial_filter_reduces_cells(self, sp_deg):
        from spatioloji_s.processing.DEG import run_deg

        # Only cells in x [0, 800] — reduces bg (TypeB is in x [500, 1000])
        sf = {"x_range": (0, 800), "y_range": (0, 1000)}
        df = run_deg(sp_deg, "cell_type", "TypeA", group_bg="TypeB", methods=["ttest"], spatial_filter=sf)["ttest"]
        assert df["n_bg"].iloc[0] < 250
