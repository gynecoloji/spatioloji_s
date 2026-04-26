"""Unit tests for processing/annotation.py (CellTypist wrapper)."""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.processing.annotation import (
    _build_celltypist_anndata,
    _check_layer_for_celltypist,
)

CELLTYPIST_AVAILABLE = (
    importlib.util.find_spec("celltypist") is not None
    and importlib.util.find_spec("anndata") is not None
)
celltypist_only = pytest.mark.skipif(
    not CELLTYPIST_AVAILABLE, reason="celltypist + anndata not installed"
)


# =========================================================================
# Backend-agnostic tests (always run)
# =========================================================================


class TestImportErrorPath:
    def test_celltypist_annotate_raises_when_missing(self, sp_basic, monkeypatch):
        if CELLTYPIST_AVAILABLE:
            pytest.skip("celltypist is installed; cannot test missing-dep path")
        from spatioloji_s.processing.annotation import celltypist_annotate

        with pytest.raises(ImportError, match="celltypist"):
            celltypist_annotate(sp_basic, model="Immune_All_Low.pkl")


class TestLayerWarning:
    def test_layer_none_warns(self, sp_basic):
        with pytest.warns(UserWarning, match="raw counts"):
            _check_layer_for_celltypist(sp_basic, None)

    def test_non_log_layer_warns(self, sp_basic):
        with pytest.warns(UserWarning, match="log-normalized"):
            _check_layer_for_celltypist(sp_basic, "scaled")

    def test_log_layer_silent(self, sp_basic):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_layer_for_celltypist(sp_basic, "log_normalized")


@celltypist_only
class TestAnnDataBridge:
    def test_build_anndata_from_main_matrix(self, sp_basic):
        adata = _build_celltypist_anndata(sp_basic, layer=None)
        assert adata.n_obs == sp_basic.n_cells
        assert adata.n_vars == sp_basic.n_genes
        # obs / var names round-trip from spatioloji indices
        assert list(adata.obs_names) == list(sp_basic.cell_index.astype(str))
        assert list(adata.var_names) == list(sp_basic.gene_index.astype(str))


# =========================================================================
# Live CellTypist annotation — only if package + a model are available
# =========================================================================


@celltypist_only
class TestCellTypistAnnotate:
    def test_annotate_with_minimal_custom_model(self, sp_basic, tmp_path, monkeypatch):
        """Train a tiny in-memory CellTypist model on synthetic data and annotate
        sp_basic with it. Avoids any network download from celltypist's CDN."""
        import anndata
        import celltypist

        # Reuse sp_basic's gene names so the model's features align.
        rng = np.random.default_rng(0)
        n_train = 60
        gene_names = sp_basic.gene_index.astype(str).tolist()
        X_train = rng.poisson(2.0, size=(n_train, len(gene_names))).astype(np.float32)
        # log1p-normalize so it matches CellTypist's expected input
        X_train = np.log1p(X_train / X_train.sum(axis=1, keepdims=True) * 1e4)
        labels = np.array(["TypeA"] * 30 + ["TypeB"] * 30)
        train_adata = anndata.AnnData(
            X=X_train,
            obs=pd.DataFrame({"cell_type": labels}, index=[f"t{i}" for i in range(n_train)]),
            var=pd.DataFrame(index=gene_names),
        )

        # Train a quick (very low-iter) logistic-regression model and save it.
        model_path = tmp_path / "tiny.pkl"
        model = celltypist.train(
            train_adata,
            labels="cell_type",
            n_jobs=1,
            max_iter=50,
            check_expression=False,
        )
        model.write(str(model_path))

        # Add a log_normalized layer to sp_basic so the annotate call uses it.
        from spatioloji_s.processing import log_transform, normalize_total

        normalize_total(sp_basic, target_sum=1e4, inplace=True, device="cpu")
        log_transform(sp_basic, layer="normalized_counts", inplace=True, device="cpu")

        from spatioloji_s.processing import celltypist_annotate

        celltypist_annotate(
            sp_basic,
            model=str(model_path),
            layer="log_normalized",
            majority_voting=False,  # avoid the extra clustering step in the test
        )

        assert "celltypist_label" in sp_basic.cell_meta.columns
        assert "celltypist_score" in sp_basic.cell_meta.columns
        assert sp_basic.cell_meta["celltypist_label"].isin({"TypeA", "TypeB", "Unassigned"}).all()
        # Scores are valid probabilities or NaN (when below p_thres)
        scores = sp_basic.cell_meta["celltypist_score"].dropna()
        assert ((scores >= 0) & (scores <= 1)).all()


class TestPublicAPI:
    def test_exports_present(self):
        from spatioloji_s import processing

        for name in ("celltypist_annotate", "list_celltypist_models", "download_celltypist_models"):
            assert hasattr(processing, name), f"{name} missing from spatioloji_s.processing"
