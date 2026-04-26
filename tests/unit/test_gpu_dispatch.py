"""
test_gpu_dispatch.py - GPU acceleration dispatch tests for processing.

Two layers:

1. Backend-agnostic tests (always run): verify that ``device='auto'`` and
   ``device='cpu'`` produce sensible results, that ``device='gpu'`` raises
   when RAPIDS is missing, that the deprecated ``gpu=`` flag still works
   and emits a DeprecationWarning, and that all returns are numpy / scipy
   / pandas (no leaked cupy arrays).

2. Parity / smoke tests on GPU (skipped when RAPIDS not importable):
   - Deterministic ops (PCA, KMeans): CPU and GPU agree within tolerance.
   - Stochastic ops (UMAP, t-SNE, Leiden): outputs have the right shape /
     label-count properties.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse

from spatioloji_s.processing._gpu import _gpu_available, _resolve_device

GPU_AVAILABLE = _gpu_available()
gpu_only = pytest.mark.skipif(not GPU_AVAILABLE, reason="RAPIDS (cuML/cuPy) not importable")


# =========================================================================
# Layer 1 — backend-agnostic dispatcher behaviour (always runs)
# =========================================================================


class TestResolveDevice:
    def test_explicit_cpu_returns_cpu(self):
        assert _resolve_device("cpu", "test op") == "cpu"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="device must be"):
            _resolve_device("tpu", "test op")  # type: ignore[arg-type]

    def test_auto_picks_cpu_when_gpu_missing(self):
        if GPU_AVAILABLE:
            pytest.skip("GPU is available; cannot test missing-GPU branch")
        assert _resolve_device("auto", "test op") == "cpu"

    def test_explicit_gpu_raises_when_missing(self):
        if GPU_AVAILABLE:
            pytest.skip("GPU is available; cannot test missing-GPU branch")
        with pytest.raises(ImportError, match="RAPIDS"):
            _resolve_device("gpu", "test op")

    def test_auto_picks_gpu_when_available(self):
        if not GPU_AVAILABLE:
            pytest.skip("GPU not available")
        assert _resolve_device("auto", "test op") == "gpu"


class TestPublicAPIDeviceParam:
    """Verify each touched function exposes ``device`` and accepts 'cpu'."""

    def test_pca_accepts_device_cpu(self, sp_basic):
        from spatioloji_s.processing.dimension_reduction import pca

        sp_basic._gene_meta["highly_variable"] = True  # use all genes
        # provide a 'scaled' layer so default layer arg works
        sp_basic._layers = getattr(sp_basic, "_layers", {})
        result = pca(
            sp_basic, layer=None, n_comps=5,
            inplace=False, device="cpu",
        )
        assert isinstance(result["X_pca"], np.ndarray)
        assert result["X_pca"].shape == (sp_basic.n_cells, 5)

    def test_kmeans_accepts_device_cpu(self, sp_basic):
        from spatioloji_s.processing.clustering import kmeans_clustering

        labels = kmeans_clustering(
            sp_basic, layer=None, n_clusters=3, use_pca=False,
            inplace=False, device="cpu",
        )
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (sp_basic.n_cells,)
        assert len(np.unique(labels)) <= 3

    def test_leiden_accepts_device_cpu(self, sp_basic):
        from spatioloji_s.processing.clustering import leiden_clustering

        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not installed")

        labels = leiden_clustering(
            sp_basic, layer=None, use_highly_variable=False,
            use_pca=False, n_neighbors=5,
            inplace=False, device="cpu",
        )
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (sp_basic.n_cells,)


class TestDeprecatedGpuFlag:
    def test_tsne_gpu_kwarg_emits_deprecation(self, sp_basic, monkeypatch):
        """Calling tsne(gpu=False) should warn and route to device='cpu'."""
        from spatioloji_s.processing import dimension_reduction as dr

        # Stub out the heavy CPU path so the test stays fast.
        called = {}

        def _stub_get_pca(*a, **k):
            return np.random.RandomState(0).randn(20, 4)

        monkeypatch.setattr(dr, "_get_pca_or_expr", _stub_get_pca)

        # Replace sklearn TSNE with a dummy that just returns input cols.
        class _FakeTSNE:
            kl_divergence_ = 0.0

            def __init__(self, **kw):
                pass

            def fit_transform(self, X):
                called["ran"] = True
                return X[:, :2]

        import sklearn.manifold
        monkeypatch.setattr(sklearn.manifold, "TSNE", _FakeTSNE)
        # Force openTSNE to be unavailable so we hit the sklearn branch.
        import builtins

        real_import = builtins.__import__

        def _no_opentsne(name, *a, **k):
            if name == "openTSNE":
                raise ImportError("disabled for test")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _no_opentsne)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dr.tsne(sp_basic, gpu=False, inplace=False)
            messages = [str(x.message) for x in w]

        assert any("deprecated" in m.lower() for m in messages), messages
        assert called.get("ran") is True


class TestExplicitGpuRaisesWithoutRapids:
    def test_pca_device_gpu_raises(self, sp_basic):
        if GPU_AVAILABLE:
            pytest.skip("GPU available; this test exercises the missing-GPU path")
        from spatioloji_s.processing.dimension_reduction import pca

        with pytest.raises(ImportError, match="RAPIDS"):
            pca(sp_basic, layer=None, n_comps=3, inplace=False, device="gpu")

    def test_kmeans_device_gpu_raises(self, sp_basic):
        if GPU_AVAILABLE:
            pytest.skip("GPU available; this test exercises the missing-GPU path")
        from spatioloji_s.processing.clustering import kmeans_clustering

        with pytest.raises(ImportError, match="RAPIDS"):
            kmeans_clustering(
                sp_basic, layer=None, n_clusters=3, use_pca=False,
                inplace=False, device="gpu",
            )


# =========================================================================
# Layer 2 — GPU parity / smoke tests (skipped when RAPIDS missing)
# =========================================================================


@gpu_only
class TestPCAParity:
    """PCA is deterministic up to sign flips of the eigenvectors."""

    def test_pca_cpu_gpu_agree(self, sp_basic):
        from spatioloji_s.processing.dimension_reduction import pca

        sp_basic._gene_meta["highly_variable"] = True

        cpu = pca(sp_basic, layer=None, n_comps=5, inplace=False, device="cpu")
        gpu = pca(sp_basic, layer=None, n_comps=5, inplace=False, device="gpu")

        # Compare absolute values (sign of each PC is arbitrary).
        # Compare absolute variance ratios (these ARE sign-invariant).
        np.testing.assert_allclose(
            cpu["variance_ratio"], gpu["variance_ratio"], rtol=1e-3, atol=1e-4,
        )
        # Each PC should match up to a sign flip.
        for i in range(5):
            corr = np.corrcoef(cpu["X_pca"][:, i], gpu["X_pca"][:, i])[0, 1]
            assert abs(corr) > 0.99, f"PC{i} differs (corr={corr:.3f})"


@gpu_only
class TestKMeansParity:
    def test_kmeans_cpu_gpu_label_counts_match(self, sp_basic):
        """KMeans with the same seed should produce the same number of clusters
        and similar cluster size distribution. Labels themselves can permute."""
        from spatioloji_s.processing.clustering import kmeans_clustering

        cpu = kmeans_clustering(
            sp_basic, layer=None, n_clusters=4, use_pca=False,
            inplace=False, random_state=0, device="cpu",
        )
        gpu = kmeans_clustering(
            sp_basic, layer=None, n_clusters=4, use_pca=False,
            inplace=False, random_state=0, device="gpu",
        )
        assert len(np.unique(cpu)) == len(np.unique(gpu)) == 4
        # Cluster size distributions should be roughly comparable.
        cpu_sizes = sorted(np.bincount(cpu).tolist())
        gpu_sizes = sorted(np.bincount(gpu).tolist())
        # Within ±25% of total cells, generously
        n = sp_basic.n_cells
        assert sum(abs(a - b) for a, b in zip(cpu_sizes, gpu_sizes, strict=True)) <= 0.5 * n


@gpu_only
class TestUMAPSmoke:
    def test_umap_gpu_returns_numpy_with_expected_shape(self, sp_basic):
        from spatioloji_s.processing.dimension_reduction import umap

        sp_basic._gene_meta["highly_variable"] = True
        out = umap(
            sp_basic, layer=None, use_pca=False,
            n_components=2, n_neighbors=5, min_dist=0.1,
            inplace=False, device="gpu",
        )
        assert isinstance(out, np.ndarray)
        assert out.shape == (sp_basic.n_cells, 2)
        assert np.isfinite(out).all()


@gpu_only
class TestTSNESmoke:
    def test_tsne_gpu_returns_numpy_with_expected_shape(self, sp_basic):
        from spatioloji_s.processing.dimension_reduction import tsne

        sp_basic._gene_meta["highly_variable"] = True
        out = tsne(
            sp_basic, layer=None, use_pca=False,
            n_components=2, perplexity=5, n_iter=250,
            inplace=False, device="gpu",
        )
        assert isinstance(out, np.ndarray)
        assert out.shape == (sp_basic.n_cells, 2)
        assert np.isfinite(out).all()


@gpu_only
class TestLeidenSmoke:
    def test_leiden_gpu_runs_and_returns_numpy(self, sp_basic):
        from spatioloji_s.processing.clustering import leiden_clustering

        labels = leiden_clustering(
            sp_basic, layer=None, use_highly_variable=False,
            use_pca=False, n_neighbors=5, resolution=1.0,
            inplace=False, device="gpu",
        )
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (sp_basic.n_cells,)
        assert len(np.unique(labels)) >= 1


@gpu_only
class TestKNNGPU:
    def test_build_knn_gpu_matches_shape(self):
        from spatioloji_s.processing.clustering import _build_knn

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 8))
        d_cpu, i_cpu = _build_knn(X, n_neighbors=5, device="cpu")
        d_gpu, i_gpu = _build_knn(X, n_neighbors=5, device="gpu")
        assert d_cpu.shape == d_gpu.shape == (50, 5)
        assert i_cpu.shape == i_gpu.shape == (50, 5)
        # Same first neighbour for each point (closest is unambiguous in dense gauss data)
        # Allow up to 5% mismatch for floating-point ordering ties.
        agreement = (i_cpu[:, 0] == i_gpu[:, 0]).mean()
        assert agreement > 0.95, f"first-NN agreement {agreement:.2f}"


@gpu_only
class TestUMAPConnectivitiesParity:
    def test_umap_connectivities_cpu_gpu_close(self):
        from spatioloji_s.processing.clustering import _compute_umap_connectivities

        rng = np.random.default_rng(0)
        n, k = 40, 5
        d = np.sort(rng.uniform(0, 1, (n, k)), axis=1)
        i = np.tile(np.arange(k), (n, 1)) + np.arange(n)[:, None]
        i = i % n
        W_cpu = _compute_umap_connectivities(d, i, device="cpu")
        W_gpu = _compute_umap_connectivities(d, i, device="gpu")
        assert sparse.issparse(W_cpu) and sparse.issparse(W_gpu)
        # Sums and nnz should match closely
        assert W_cpu.nnz == W_gpu.nnz
        np.testing.assert_allclose(W_cpu.sum(), W_gpu.sum(), rtol=1e-3, atol=1e-4)


# =========================================================================
# Tier 1 — normalization & feature_selection dispatch
# =========================================================================


class TestNormalizationDevice:
    """device='cpu' should always work and return numpy/scipy."""

    def test_normalize_total_cpu(self, sp_basic):
        from spatioloji_s.processing.normalization import normalize_total

        out = normalize_total(sp_basic, target_sum=1e4, device="cpu")
        assert out.shape == (sp_basic.n_cells, sp_basic.n_genes)

    def test_log_transform_cpu(self, sp_basic):
        from spatioloji_s.processing.normalization import log_transform

        out = log_transform(sp_basic, device="cpu")
        assert out.shape == (sp_basic.n_cells, sp_basic.n_genes)

    def test_scale_cpu(self, sp_basic):
        from spatioloji_s.processing.normalization import scale

        out = scale(sp_basic, device="cpu")
        assert out.shape == (sp_basic.n_cells, sp_basic.n_genes)
        assert isinstance(out, np.ndarray)

    def test_scale_by_batch_cpu(self, sp_basic):
        from spatioloji_s.processing.normalization import scale_by_batch_normalization

        out = scale_by_batch_normalization(sp_basic, batch_key="fov", layer=None, device="cpu")
        assert out.shape == (sp_basic.n_cells, sp_basic.n_genes)

    def test_pearson_residuals_cpu(self, sp_basic):
        from spatioloji_s.processing.normalization import normalize_pearson_residuals

        out = normalize_pearson_residuals(sp_basic, device="cpu")
        assert out.shape == (sp_basic.n_cells, sp_basic.n_genes)

    def test_hvg_cpu(self, sp_basic):
        from spatioloji_s.processing.feature_selection import highly_variable_genes

        out = highly_variable_genes(sp_basic, n_top_genes=5, method="seurat", device="cpu", inplace=False)
        assert out["highly_variable"].sum() > 0

    def test_normalize_total_gpu_raises_when_missing(self, sp_basic):
        if GPU_AVAILABLE:
            pytest.skip("GPU available; this exercises the missing-GPU path")
        from spatioloji_s.processing.normalization import normalize_total

        with pytest.raises(ImportError, match="RAPIDS"):
            normalize_total(sp_basic, device="gpu")


@gpu_only
class TestNormalizationParity:
    """Numerical parity for deterministic reductions / elementwise ops."""

    def test_normalize_total_cpu_gpu_match(self, sp_basic):
        from spatioloji_s.processing.normalization import normalize_total

        cpu = normalize_total(sp_basic, target_sum=1e4, device="cpu")
        gpu = normalize_total(sp_basic, target_sum=1e4, device="gpu")
        cpu_arr = cpu.toarray() if sparse.issparse(cpu) else np.asarray(cpu)
        gpu_arr = gpu.toarray() if sparse.issparse(gpu) else np.asarray(gpu)
        np.testing.assert_allclose(cpu_arr, gpu_arr, rtol=1e-4, atol=1e-4)

    def test_log_transform_cpu_gpu_match(self, sp_basic):
        from spatioloji_s.processing.normalization import log_transform

        cpu = log_transform(sp_basic, device="cpu")
        gpu = log_transform(sp_basic, device="gpu")
        cpu_arr = cpu.toarray() if sparse.issparse(cpu) else np.asarray(cpu)
        gpu_arr = gpu.toarray() if sparse.issparse(gpu) else np.asarray(gpu)
        np.testing.assert_allclose(cpu_arr, gpu_arr, rtol=1e-5, atol=1e-5)

    def test_scale_standard_cpu_gpu_match(self, sp_basic):
        from spatioloji_s.processing.normalization import scale

        cpu = np.asarray(scale(sp_basic, method="standard", device="cpu"))
        gpu = np.asarray(scale(sp_basic, method="standard", device="gpu"))
        np.testing.assert_allclose(cpu, gpu, rtol=1e-4, atol=1e-4)

    def test_pearson_residuals_cpu_gpu_match(self, sp_basic):
        from spatioloji_s.processing.normalization import normalize_pearson_residuals

        cpu = normalize_pearson_residuals(sp_basic, device="cpu")
        gpu = normalize_pearson_residuals(sp_basic, device="gpu")
        np.testing.assert_allclose(cpu, gpu, rtol=1e-4, atol=1e-4)

    def test_hvg_pearson_cpu_gpu_match(self, sp_basic):
        from spatioloji_s.processing.feature_selection import highly_variable_genes

        cpu = highly_variable_genes(
            sp_basic, n_top_genes=5, method="pearson_residuals",
            device="cpu", inplace=False,
        )
        gpu = highly_variable_genes(
            sp_basic, n_top_genes=5, method="pearson_residuals",
            device="gpu", inplace=False,
        )
        # Same top-gene set
        cpu_set = set(cpu.index[cpu["highly_variable"]])
        gpu_set = set(gpu.index[gpu["highly_variable"]])
        # Allow small overlap mismatch from floating-point ranking ties
        jaccard = len(cpu_set & gpu_set) / max(len(cpu_set | gpu_set), 1)
        assert jaccard > 0.8
