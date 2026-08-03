"""Concordance of the Leiden cell graph with scanpy's sc.pp.neighbors.

Leiden itself is not the problem — given scanpy's own connectivities, leidenalg
reproduces scanpy's labels exactly. What decides whether clustering agrees is the
graph, so that is what these tests pin down:

* ``n_neighbors`` must mean what it means in scanpy, UMAP and Seurat — the
  neighbourhood size *including the cell itself*, so ``n_neighbors=15`` uses 14
  real neighbours.
* the fuzzy-simplicial-set bandwidth must target ``log2(n_neighbors)``, which is
  what fixes the overall scale of every edge weight in the graph.

The scanpy/umap comparisons skip when those packages are missing; the convention
tests run everywhere.
"""

import numpy as np
import pytest
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from spatioloji_s.processing.clustering import _build_leiden_graph, _compute_umap_connectivities

def _skip_if_umap_has_fastmath_bug():
    """Skip when the installed umap-learn escapes its bandwidth to infinity.

    Up to 0.5.9 the bisection was compiled with fastmath=True, which let LLVM
    drop the `hi == inf` guard, so rows still below target at mid=1 escaped to
    sigma=inf and got all-1.0 memberships. 0.5.12 removed fastmath and seeds hi
    with NPY_FLOATMAX. We implement the corrected behaviour, so against an old
    umap -- and against a scanpy built on one -- the graphs legitimately differ.
    """
    umap_mod = pytest.importorskip("umap.umap_", reason="umap-learn not installed")
    rng = np.random.default_rng(0)
    probe = np.sort(rng.lognormal(1.0, 0.5, (200, K)).astype(np.float32), axis=1)
    probe[:, 0] = 0.0
    sigma, _ = umap_mod.smooth_knn_dist(probe, float(K), local_connectivity=1.0)
    if np.isinf(sigma).any():
        pytest.skip("installed umap-learn (<0.5.11) has the fastmath escape-to-infinity "
                    "bug; upgrade umap-learn to compare against a correct reference")


K = 15


@pytest.fixture(scope="module")
def blobs() -> np.ndarray:
    """Overlapping, unequally-sized blobs in PCA-like space."""
    rng = np.random.default_rng(17)
    n_dim = 30
    centers = np.zeros((6, n_dim))
    centers[:, :5] = rng.normal(scale=3.2, size=(6, 5))
    sizes = [800, 600, 400, 300, 200, 150]
    return np.vstack([
        c + rng.normal(scale=1.6, size=(m, n_dim)) for c, m in zip(centers, sizes, strict=True)
    ]).astype(np.float32)


@pytest.fixture(scope="module")
def knn(blobs):
    """Exact k-NN split into UMAP's (self-inclusive) and real-neighbour views.

    The self column is forced to exactly 0.0. sklearn returns a tiny positive
    residual there for float32 input, and UMAP would take that residual as rho,
    which is not what scanpy feeds it.
    """
    n = blobs.shape[0]
    nn = NearestNeighbors(n_neighbors=K + 1, metric="euclidean").fit(blobs)
    d, i = nn.kneighbors(blobs)
    d_real, i_real = np.ascontiguousarray(d[:, 1:K]), np.ascontiguousarray(i[:, 1:K])
    d_incl = np.hstack([np.zeros((n, 1)), d_real])
    i_incl = np.hstack([np.arange(n)[:, None], i_real])
    return {"incl": (d_incl, i_incl), "real": (d_real, i_real)}


def test_connectivities_match_umap(knn):
    """Edge weights must equal umap.fuzzy_simplicial_set on the same neighbours."""
    umap_mod = pytest.importorskip("umap.umap_", reason="umap-learn not installed")
    _skip_if_umap_has_fastmath_bug()

    d_incl, i_incl = knn["incl"]
    ref = umap_mod.fuzzy_simplicial_set(
        sparse.coo_matrix(([], ([], [])), shape=(d_incl.shape[0], 1)),
        K, None, None, knn_indices=i_incl, knn_dists=d_incl,
        set_op_mix_ratio=1.0, local_connectivity=1.0,
    )
    ref = (ref[0] if isinstance(ref, tuple) else ref).tocsr()
    ref.eliminate_zeros()

    d_real, i_real = knn["real"]
    got = _compute_umap_connectivities(d_real, i_real, device="cpu", n_neighbors=K).tocsr()
    got.eliminate_zeros()

    rel = sparse.linalg.norm(ref - got) / sparse.linalg.norm(ref)
    assert rel < 0.02, f"relative Frobenius difference vs UMAP = {rel:.4f}"


def test_bandwidth_targets_log2_n_neighbors(knn):
    """sum(exp(-(d-rho)/sigma)) over the real neighbours must equal log2(n_neighbors).

    The target is log2(n_neighbors) even though only n_neighbors-1 terms are
    summed — UMAP counts the cell itself in n_neighbors but skips it in the sum.
    """
    from spatioloji_s.processing.clustering import _smooth_knn_bandwidth

    d_real, _ = knn["real"]
    sigma, rho = _smooth_knn_bandwidth(d_real.astype(np.float64), n_neighbors=K)

    solved = np.isfinite(sigma)
    assert solved.any(), "no rows solved"
    psum = np.exp(-np.maximum(d_real - rho[:, None], 0.0) / sigma[:, None]).sum(axis=1)

    assert np.allclose(psum[solved], np.log2(K), atol=1e-3), (
        f"median row sum {np.median(psum[solved]):.5f}, expected log2({K})={np.log2(K):.5f}"
    )


def test_bandwidth_always_converges(knn):
    """Our search must always produce a finite bandwidth.

    We deliberately do not reproduce umap's pre-0.5.11 escape-to-infinity bug;
    matching it would be the thing causing a discrepancy against a current
    umap/scanpy.
    """
    from spatioloji_s.processing.clustering import _smooth_knn_bandwidth

    d_real, _ = knn["real"]
    sigma, rho = _smooth_knn_bandwidth(d_real.astype(np.float64), n_neighbors=K)

    assert np.isfinite(sigma).all(), f"{int(np.isinf(sigma).sum())} bandwidths escaped to inf"
    assert (sigma > 0).all()
    assert np.isfinite(rho).all()


def test_n_neighbors_counts_self(blobs):
    """n_neighbors=K must build K-1 real edges per cell, as scanpy does."""
    _, knn_d, knn_i, _ = _build_leiden_graph(blobs, K, "umap", device="cpu")
    assert knn_d.shape[1] == K - 1, f"got {knn_d.shape[1]} real neighbours, expected {K - 1}"
    assert knn_i.shape[1] == K - 1


def _scanpy_reference(blobs, resolution=None):
    """sc.pp.neighbors (+ optionally sc.tl.leiden) on the same matrix."""
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    ad = pytest.importorskip("anndata", reason="anndata not installed")
    pytest.importorskip("umap", reason="umap-learn not installed")
    _skip_if_umap_has_fastmath_bug()

    adata = ad.AnnData(blobs.copy())
    adata.obsm["X_pca"] = blobs.copy()
    # explicit use_rep: with n_vars <= 50 scanpy silently falls back to .X
    sc.pp.neighbors(adata, n_neighbors=K, use_rep="X_pca", random_state=42)
    if resolution is not None:
        sc.tl.leiden(adata, resolution=resolution, random_state=42, key_added="ref",
                     flavor="igraph", n_iterations=2)
    return adata


def test_connectivities_match_scanpy_given_its_own_knn(blobs):
    """Fed scanpy's own neighbours, our weights must reproduce its connectivities.

    This isolates the weight math from k-NN approximation: identical neighbour
    sets in, so any difference is ours.
    """
    adata = _scanpy_reference(blobs)
    ref = adata.obsp["connectivities"].tocsr()
    ref.eliminate_zeros()

    dist = adata.obsp["distances"].tocsr()
    n = blobs.shape[0]
    kk = int(np.diff(dist.indptr).min())
    d = np.zeros((n, kk))
    idx = np.zeros((n, kk), dtype=np.int64)
    for r in range(n):
        s, e = dist.indptr[r], dist.indptr[r + 1]
        order = np.argsort(dist.data[s:e])[:kk]
        d[r] = dist.data[s:e][order]
        idx[r] = dist.indices[s:e][order]

    got = _compute_umap_connectivities(d, idx, device="cpu", n_neighbors=K).tocsr()
    got.eliminate_zeros()

    assert got.nnz == ref.nnz, f"edge count {got.nnz} vs scanpy {ref.nnz}"
    rel = sparse.linalg.norm(ref - got) / sparse.linalg.norm(ref)
    assert rel < 0.02, f"relative Frobenius difference vs scanpy = {rel:.4f}"


def test_graph_matches_scanpy_neighbors(blobs):
    """The end-to-end graph must match scanpy up to approximate-k-NN noise.

    pynndescent is approximate and threaded, so a fraction of a percent of edges
    legitimately differ between two runs; the exact weight math is pinned by
    test_connectivities_match_scanpy_given_its_own_knn.
    """
    adata = _scanpy_reference(blobs)
    ref = adata.obsp["connectivities"].tocsr()
    ref.eliminate_zeros()

    _, _, _, got = _build_leiden_graph(blobs, K, "umap", device="cpu", random_state=42)
    got = got.tocsr()
    got.eliminate_zeros()

    assert abs(got.nnz - ref.nnz) / ref.nnz < 0.01, f"edge count {got.nnz} vs scanpy {ref.nnz}"
    assert abs(got.sum() - ref.sum()) / ref.sum() < 0.02, (
        f"total edge weight {got.sum():.1f} vs scanpy {ref.sum():.1f}"
    )


def test_leiden_matches_scanpy(blobs):
    """End-to-end: same cluster assignment as sc.tl.leiden at the same resolution."""
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    ad = pytest.importorskip("anndata", reason="anndata not installed")
    pytest.importorskip("leidenalg", reason="leidenalg not installed")
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score

    from spatioloji_s.data.core import spatioloji
    from spatioloji_s.processing.clustering import leiden_clustering

    res = 0.5
    adata = ad.AnnData(blobs.copy())
    adata.obsm["X_pca"] = blobs.copy()
    sc.pp.neighbors(adata, n_neighbors=K, use_rep="X_pca", random_state=42)
    sc.tl.leiden(adata, resolution=res, random_state=42, key_added="ref",
                 flavor="igraph", n_iterations=2)
    ref = adata.obs["ref"].astype(str).to_numpy()

    n_cells = blobs.shape[0]
    cids = [f"c{i}" for i in range(n_cells)]
    rng = np.random.default_rng(0)
    sp = spatioloji(
        expression=np.zeros((n_cells, 3), dtype=np.float32),
        cell_ids=cids,
        gene_names=["a", "b", "c"],
        cell_metadata=pd.DataFrame({"fov": ["1"] * n_cells}, index=cids),
        gene_metadata=pd.DataFrame(index=["a", "b", "c"]),
        spatial_coords={k: rng.uniform(0, 10, n_cells)
                        for k in ("x_global", "y_global", "x_local", "y_local")},
    )
    sp._embeddings["X_pca"] = blobs.copy()

    got = leiden_clustering(sp, use_pca=True, n_pcs=blobs.shape[1], n_neighbors=K,
                            resolution=res, random_state=42, inplace=False, device="cpu")

    score = adjusted_rand_score(got, ref)
    assert score > 0.90, f"Leiden ARI vs scanpy = {score:.4f}"


def test_internal_pca_fallback_is_exact(blobs):
    """leiden_clustering's own PCA fallback must match processing.pca().

    Callers who cluster without running pca() first still go through this path,
    so it must not silently drop back to the approximate randomized solver.
    """
    from scipy.linalg import subspace_angles
    from sklearn.decomposition import PCA as _SKPCA

    from spatioloji_s.processing.clustering import _fallback_pca

    got = _fallback_pca(blobs, 20, 42).fit_transform(blobs)
    exact = _SKPCA(n_components=20, svd_solver="full").fit_transform(blobs)

    angle = float(np.degrees(np.max(subspace_angles(got, exact))))
    assert angle < 0.01, f"fallback PCA is {angle:.4f} deg off the exact solution"
