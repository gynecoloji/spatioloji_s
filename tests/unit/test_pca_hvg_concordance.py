"""Numerical concordance of pca() and highly_variable_genes(seurat_v3) with scanpy.

These guard the two properties that make spatioloji_s a drop-in replacement for the
scanpy preprocessing path:

* ``pca()`` must resolve the *exact* leading eigenspace, not an approximation.
  sklearn's ``svd_solver='auto'`` picks the randomized solver for typical
  transcriptomics shapes, whose trailing components drift by >10 degrees.
* ``highly_variable_genes(method='seurat_v3')`` must implement the Seurat v3 VST
  the way Seurat and scanpy do, since PCA runs on whatever gene subset it returns.

The scanpy-comparison tests skip when scanpy/scikit-misc are absent; the solver
accuracy tests run everywhere by comparing against an exact full SVD.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA as _SKPCA

from spatioloji_s.data.core import spatioloji
from spatioloji_s.processing.dimension_reduction import pca
from spatioloji_s.processing.feature_selection import highly_variable_genes


def _max_angle(A: np.ndarray, B: np.ndarray) -> float:
    """Largest principal angle (degrees) between two column subspaces."""
    return float(np.degrees(np.max(subspace_angles(A, B))))


def _make_sp(counts: np.ndarray) -> spatioloji:
    """Wrap a counts matrix in a minimal spatioloji object."""
    n_cells, n_genes = counts.shape
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    rng = np.random.default_rng(0)
    return spatioloji(
        expression=counts,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=pd.DataFrame({"fov": ["1"] * n_cells}, index=cell_ids),
        gene_metadata=pd.DataFrame(index=gene_names),
        spatial_coords={
            "x_global": rng.uniform(0, 100, n_cells),
            "y_global": rng.uniform(0, 100, n_cells),
            "x_local": rng.uniform(0, 100, n_cells),
            "y_local": rng.uniform(0, 100, n_cells),
        },
    )


@pytest.fixture(scope="module")
def slow_spectrum() -> np.ndarray:
    """Dense matrix with a slowly-decaying spectrum, like log-normalized counts.

    A fast-decaying spectrum hides solver error: the trailing PCs are pure noise
    and any solver "agrees" on them. Real gene-expression covariance decays
    slowly, which is exactly where the randomized solver loses accuracy.
    """
    rng = np.random.default_rng(1)
    n_cells, n_genes, n_latent = 2000, 600, 200
    loads = rng.normal(size=(n_latent, n_genes)) * (1.0 / np.arange(1, n_latent + 1) ** 0.6)[:, None]
    scores = rng.normal(size=(n_cells, n_latent))
    return np.asarray(scores @ loads + rng.normal(scale=0.35, size=(n_cells, n_genes)), dtype=np.float64)


@pytest.fixture(scope="module")
def counts() -> np.ndarray:
    """Poisson counts with cell-type structure and a wide per-gene dynamic range."""
    rng = np.random.default_rng(7)
    n_cells, n_genes, n_types = 1500, 800, 6
    labels = rng.integers(0, n_types, n_cells)
    base = rng.lognormal(0.0, 1.5, n_genes)
    prog = rng.lognormal(0.0, 1.0, (n_types, n_genes))
    depth = rng.lognormal(0.0, 0.4, n_cells)[:, None]
    return rng.poisson(base[None, :] * prog[labels, :] * depth).astype(np.float64)


# ---------------------------------------------------------------------------
# pca(): solver accuracy
# ---------------------------------------------------------------------------


def test_pca_resolves_exact_eigenspace(slow_spectrum):
    """pca() must match an exact full SVD across all 50 returned components."""
    sp = _make_sp(slow_spectrum)
    sp.add_layer("log_normalized", slow_spectrum)

    got = pca(sp, layer="log_normalized", use_highly_variable=False, n_comps=50,
              random_state=42, inplace=False, device="cpu")["X_pca"]
    exact = _SKPCA(n_components=50, svd_solver="full").fit_transform(slow_spectrum)

    assert _max_angle(got, exact) < 0.01


def test_pca_variance_ratio_matches_exact(slow_spectrum):
    """explained_variance_ratio_ must be exact, including the trailing components."""
    sp = _make_sp(slow_spectrum)
    sp.add_layer("log_normalized", slow_spectrum)

    got = pca(sp, layer="log_normalized", use_highly_variable=False, n_comps=50,
              random_state=42, inplace=False, device="cpu")["variance_ratio"]
    exact = _SKPCA(n_components=50, svd_solver="full").fit(slow_spectrum).explained_variance_ratio_

    np.testing.assert_allclose(got, exact, rtol=1e-6, atol=1e-12)


def test_pca_svd_solver_is_overridable(slow_spectrum):
    """The approximate solver stays reachable for users who want the speed."""
    sp = _make_sp(slow_spectrum)
    sp.add_layer("log_normalized", slow_spectrum)

    res = pca(sp, layer="log_normalized", use_highly_variable=False, n_comps=10,
              random_state=42, inplace=False, device="cpu", svd_solver="randomized")
    assert res["X_pca"].shape == (slow_spectrum.shape[0], 10)


def test_pca_handles_full_rank_request(counts):
    """n_comps == min(n_cells, n_genes) must not crash (arpack cannot do k == min)."""
    small = counts[:40, :30]
    sp = _make_sp(small)
    sp.add_layer("log_normalized", small)

    res = pca(sp, layer="log_normalized", use_highly_variable=False, n_comps=30,
              random_state=42, inplace=False, device="cpu")
    assert res["X_pca"].shape[1] == 30


def test_pca_accepts_sparse_input(counts):
    """Sparse layers must give the same answer as their dense equivalent."""
    from scipy import sparse

    dense = counts[:, :200]
    sp_dense = _make_sp(dense)
    sp_dense.add_layer("log_normalized", dense)
    sp_sparse_obj = _make_sp(dense)
    sp_sparse_obj.add_layer("log_normalized", sparse.csr_matrix(dense))

    a = pca(sp_dense, layer="log_normalized", use_highly_variable=False, n_comps=20,
            random_state=42, inplace=False, device="cpu")["X_pca"]
    b = pca(sp_sparse_obj, layer="log_normalized", use_highly_variable=False, n_comps=20,
            random_state=42, inplace=False, device="cpu")["X_pca"]

    assert _max_angle(a, b) < 0.01


# ---------------------------------------------------------------------------
# highly_variable_genes(seurat_v3): agreement with Seurat/scanpy
# ---------------------------------------------------------------------------


def test_seurat_v3_reports_scanpy_columns(counts):
    """seurat_v3 must expose the same per-gene statistics scanpy writes to .var."""
    sp = _make_sp(counts)
    df = highly_variable_genes(sp, layer=None, method="seurat_v3", n_top_genes=200,
                               inplace=False, device="cpu")

    assert {"means", "variances", "variances_norm", "highly_variable_rank"} <= set(df.columns)
    assert int(df["highly_variable"].sum()) == 200
    # rank 0..n-1 for selected genes, NaN for the rest
    sel = df.loc[df["highly_variable"], "highly_variable_rank"]
    assert sorted(sel.tolist()) == list(range(200))
    assert df.loc[~df["highly_variable"], "highly_variable_rank"].isna().all()


def test_seurat_v3_uses_unbiased_variance(counts):
    """Seurat/scanpy report the ddof=1 variance, not the population variance."""
    sp = _make_sp(counts)
    df = highly_variable_genes(sp, layer=None, method="seurat_v3", n_top_genes=200,
                               inplace=False, device="cpu")

    np.testing.assert_allclose(df["variances"].to_numpy(), counts.var(axis=0, ddof=1), rtol=1e-8)


# ---------------------------------------------------------------------------
# End-to-end comparison against scanpy itself
# ---------------------------------------------------------------------------


def _skip_without_scanpy():
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    pytest.importorskip("anndata", reason="anndata not installed")
    pytest.importorskip("skmisc", reason="scikit-misc not installed")
    return sc


def test_seurat_v3_matches_scanpy(counts):
    """Same gene set as sc.pp.highly_variable_genes(flavor='seurat_v3')."""
    sc = _skip_without_scanpy()
    import anndata as ad

    n_top = 200
    adata = ad.AnnData(counts.copy())
    adata.var_names = [f"gene_{i}" for i in range(counts.shape[1])]
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top)
    want = set(adata.var_names[adata.var["highly_variable"]])

    sp = _make_sp(counts)
    highly_variable_genes(sp, layer=None, method="seurat_v3", n_top_genes=n_top,
                          inplace=True, device="cpu")
    got = set(sp.gene_meta.index[sp.gene_meta["highly_variable"]].astype(str))

    jaccard = len(want & got) / len(want | got)
    assert jaccard > 0.99, f"HVG Jaccard vs scanpy = {jaccard:.4f}"


def test_pca_matches_scanpy_end_to_end(counts):
    """normalize -> log1p -> HVG -> PCA must land in scanpy's top-50 subspace."""
    sc = _skip_without_scanpy()
    import anndata as ad

    import spatioloji_s as sj

    n_top, n_comps = 200, 50

    adata = ad.AnnData(counts.copy())
    adata.var_names = [f"gene_{i}" for i in range(counts.shape[1])]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top, layer="counts")
    sc.pp.pca(adata, n_comps=n_comps, random_state=42, use_highly_variable=True)

    sp = _make_sp(counts)
    sj.processing.normalize_total(sp, target_sum=1e4, inplace=True, device="cpu")
    sj.processing.log_transform(sp, layer="normalized_counts", inplace=True, device="cpu")
    highly_variable_genes(sp, layer="log_normalized", method="seurat_v3",
                          n_top_genes=n_top, inplace=True, device="cpu")
    pca(sp, layer="log_normalized", use_highly_variable=True, n_comps=n_comps,
        random_state=42, inplace=True, device="cpu")

    angle = _max_angle(np.asarray(sp.embeddings["X_pca"])[:, :n_comps],
                       np.asarray(adata.obsm["X_pca"])[:, :n_comps])
    assert angle < 1.0, f"PCA subspace angle vs scanpy = {angle:.4f} deg"


# ---------------------------------------------------------------------------
# accumulation precision
# ---------------------------------------------------------------------------


def test_hvg_moments_use_float64_accumulator():
    """means/variances must be exact at scale, not float32-accumulated.

    numpy does not use pairwise summation for a strided axis-0 reduction, so a
    float32 accumulator over hundreds of thousands of cells drifts ~1e-3 in the
    variance. That is orders of magnitude larger than the gap between adjacent
    genes at the HVG cut, so it silently reshuffles the selection. Caught only
    with a genuinely 2-D matrix -- a 1-D array is contiguous and stays accurate.
    """
    rng = np.random.default_rng(3)
    n_cells, n_genes = 250_000, 24
    scale = rng.lognormal(1.0, 0.8, n_genes)  # spread the mean-variance curve
    counts = np.where(rng.random((n_cells, n_genes)) < 0.9, 0.0,
                      rng.lognormal(0.0, 0.7, (n_cells, n_genes)) * scale).astype(np.float32)

    n = counts.shape[0]
    cells = [f"c{i}" for i in range(n)]
    genes = [f"g{i}" for i in range(n_genes)]
    sp = spatioloji(
        expression=counts,
        cell_ids=cells,
        gene_names=genes,
        cell_metadata=pd.DataFrame({"fov": ["1"] * n, "grp": ["a"] * n}, index=cells),
        gene_metadata=pd.DataFrame(index=genes),
        spatial_coords={k: rng.uniform(0, 10, n)
                        for k in ("x_global", "y_global", "x_local", "y_local")},
    )

    # 'seurat' reports the moments straight from the shared accumulator, with no
    # loess in the way, so this isolates the precision question.
    df = highly_variable_genes(sp, layer=None, method="seurat", n_top_genes=5,
                               inplace=False, device="cpu")

    exact = counts.astype(np.float64)
    np.testing.assert_allclose(df["means"].to_numpy(), exact.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(df["variances"].to_numpy(), exact.var(axis=0), rtol=1e-10)
