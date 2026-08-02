"""
clustering.py - Clustering methods for spatial transcriptomics data

Provides various clustering algorithms optimized for spatioloji objects,
including spatial-aware clustering methods.

GPU acceleration
----------------
``leiden_clustering``, ``kmeans_clustering``, ``find_optimal_clusters``
and ``leiden_resolution_sweep`` accept a ``device`` parameter
(``'auto' | 'cpu' | 'gpu'``).  Internal helpers ``_build_knn`` and
``_compute_umap_connectivities`` also accept ``device``.

When ``device='auto'`` (default) and RAPIDS (``cuml``, ``cugraph``,
``cupy``) is importable, k-NN runs on cuML, the UMAP fuzzy connectivity
search runs on cupy, and Leiden runs on cuGraph.  Hierarchical
clustering has no good GPU equivalent and remains CPU-only.
"""

import warnings
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    # igraph is an optional dependency ([clustering]); it is imported lazily
    # inside _build_igraph(). This binding exists only so the quoted
    # "ig.Graph" annotations below resolve for type checkers.
    import igraph as ig

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ._gpu import Device, _resolve_device, _to_cupy, _to_numpy, _warn_fallback


def _get_expression_matrix(spatioloji_obj, layer: str | None) -> np.ndarray:
    """Extract dense expression matrix from the specified layer."""
    if layer is None:
        return spatioloji_obj.expression.get_dense()
    X = spatioloji_obj.get_layer(layer)
    if sparse.issparse(X):
        X = X.toarray()
    return X


def _subset_hvg(spatioloji_obj, X: np.ndarray) -> np.ndarray:
    """Subset to highly variable genes if available."""
    if "highly_variable" not in spatioloji_obj.gene_meta.columns:
        warnings.warn(
            "No 'highly_variable' column in gene_meta. "
            "Run highly_variable_genes() first or set use_highly_variable=False. "
            "Using all genes.",
            stacklevel=2,
        )
        return X
    hvg_mask = spatioloji_obj.gene_meta["highly_variable"].values
    n_hvg = int(hvg_mask.sum())
    if n_hvg == 0:
        warnings.warn("No highly variable genes found. Using all genes.", stacklevel=2)
        return X
    print(f"  Using {n_hvg} highly variable genes (out of {hvg_mask.shape[0]} total)")
    return X[:, hvg_mask]


def _compute_umap_connectivities(
    knn_distances: np.ndarray,
    knn_indices: np.ndarray,
    device: Device = "auto",
) -> "sparse.csr_matrix":
    """Build UMAP fuzzy-simplicial-set edge weights.

    For each cell *i* we estimate a local bandwidth σ_i via binary search so
    that the sum of fuzzy membership strengths over its k neighbours equals
    log₂(k).  The directed weights are then symmetrised with the fuzzy union
    (W = A + Aᵀ − A ⊙ Aᵀ), matching the UMAP paper (McInnes et al. 2018).

    Parameters
    ----------
    knn_distances : np.ndarray, shape (n_cells, k)
        Distances to the k nearest neighbours (self excluded, nearest first).
    knn_indices : np.ndarray, shape (n_cells, k)
        Global cell indices of the k nearest neighbours.
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  GPU runs the same vectorised σ search on cupy and
        builds a cupyx CSR adjacency, then converts to scipy CSR for return.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric (n_cells × n_cells) weight matrix with values in [0, 1].
    """
    backend = _resolve_device(device, "UMAP connectivities")

    if backend == "gpu":
        try:
            return _compute_umap_connectivities_gpu(knn_distances, knn_indices)
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("UMAP connectivities", exc)

    n_cells, k = knn_distances.shape
    target = float(np.log2(k))

    rho = knn_distances[:, 0].astype(np.float64)
    shifted_all = np.maximum(knn_distances.astype(np.float64) - rho[:, np.newaxis], 0.0)

    lo = np.zeros(n_cells, dtype=np.float64)
    hi = np.full(n_cells, np.inf, dtype=np.float64)
    mid = np.ones(n_cells, dtype=np.float64)

    for _ in range(64):
        psum = np.exp(-shifted_all / mid[:, np.newaxis]).sum(axis=1)
        converged = np.abs(psum - target) < 1e-5
        if converged.all():
            break
        too_high = psum > target
        too_low = ~too_high & ~converged
        hi = np.where(too_high, mid, hi)
        mid_new = np.where(too_high, (lo + mid) / 2.0, mid)
        lo = np.where(too_low, mid, lo)
        hi_inf = np.isinf(hi)
        mid_new = np.where(too_low & hi_inf, mid * 2.0, mid_new)
        mid_new = np.where(too_low & ~hi_inf, (lo + hi) / 2.0, mid_new)
        mid = mid_new

    sigma = np.maximum(mid, 1e-8)
    weights = np.exp(-shifted_all / sigma[:, np.newaxis]).astype(np.float32)

    row_idx = np.repeat(np.arange(n_cells, dtype=np.int32), k)
    col_idx = knn_indices.ravel().astype(np.int32)
    A = sparse.csr_matrix((weights.ravel(), (row_idx, col_idx)), shape=(n_cells, n_cells))

    At = A.T.tocsr()
    W = A + At - A.multiply(At)
    return W


def _compute_umap_connectivities_gpu(knn_distances: np.ndarray, knn_indices: np.ndarray):
    """cupy port of the σ binary search and fuzzy-union symmetrisation."""
    import cupy as cp
    from cupyx.scipy import sparse as cusparse

    n_cells, k = knn_distances.shape
    target = float(np.log2(k))

    d = cp.asarray(knn_distances, dtype=cp.float64)
    rho = d[:, 0]
    shifted_all = cp.maximum(d - rho[:, cp.newaxis], 0.0)

    lo = cp.zeros(n_cells, dtype=cp.float64)
    hi = cp.full(n_cells, cp.inf, dtype=cp.float64)
    mid = cp.ones(n_cells, dtype=cp.float64)

    for _ in range(64):
        psum = cp.exp(-shifted_all / mid[:, cp.newaxis]).sum(axis=1)
        converged = cp.abs(psum - target) < 1e-5
        if bool(converged.all()):
            break
        too_high = psum > target
        too_low = ~too_high & ~converged
        hi = cp.where(too_high, mid, hi)
        mid_new = cp.where(too_high, (lo + mid) / 2.0, mid)
        lo = cp.where(too_low, mid, lo)
        hi_inf = cp.isinf(hi)
        mid_new = cp.where(too_low & hi_inf, mid * 2.0, mid_new)
        mid_new = cp.where(too_low & ~hi_inf, (lo + hi) / 2.0, mid_new)
        mid = mid_new

    sigma = cp.maximum(mid, 1e-8)
    weights = cp.exp(-shifted_all / sigma[:, cp.newaxis]).astype(cp.float32).ravel()

    row_idx = cp.repeat(cp.arange(n_cells, dtype=cp.int32), k)
    col_idx = cp.asarray(knn_indices, dtype=cp.int32).ravel()

    A = cusparse.csr_matrix((weights, (row_idx, col_idx)), shape=(n_cells, n_cells))
    At = A.T.tocsr()
    W = A + At - A.multiply(At)
    return _to_numpy(W)


def _build_knn(
    X_reduced: np.ndarray,
    n_neighbors: int,
    device: Device = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute k-nearest neighbours.

    Backends (in order of preference within each device):

    - GPU: cuML ``NearestNeighbors`` (brute-force, very fast on PCA-space).
    - CPU: pynndescent (approximate) → sklearn ``NearestNeighbors`` fallback.

    Parameters
    ----------
    X_reduced : np.ndarray
        Cell × feature matrix.
    n_neighbors : int
        Number of nearest neighbours (self excluded).
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  Default ``'auto'``.

    Returns
    -------
    knn_distances : np.ndarray, shape (n_cells, n_neighbors)
    knn_indices : np.ndarray, shape (n_cells, n_neighbors)
    """
    backend = _resolve_device(device, "k-NN")

    if backend == "gpu":
        try:
            from cuml.neighbors import NearestNeighbors as cuNN

            print("    Using cuML NearestNeighbors (GPU)...")
            X_gpu = _to_cupy(X_reduced)
            nn = cuNN(n_neighbors=n_neighbors + 1, metric="euclidean")
            nn.fit(X_gpu)
            d_gpu, i_gpu = nn.kneighbors(X_gpu)
            knn_distances_full = _to_numpy(d_gpu)
            knn_indices_full = _to_numpy(i_gpu)
            return knn_distances_full[:, 1:], knn_indices_full[:, 1:]
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("k-NN", exc)

    # CPU path
    try:
        from pynndescent import NNDescent

        print("    Using pynndescent (approximate KNN)...")
        index = NNDescent(X_reduced, n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1)
        knn_indices_full, knn_distances_full = index.neighbor_graph
        return knn_distances_full[:, 1:], knn_indices_full[:, 1:]
    except ImportError:
        print("    Using sklearn NearestNeighbors (install pynndescent for ~10x speedup)...")
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1).fit(X_reduced)
        knn_distances_full, knn_indices_full = nbrs.kneighbors(X_reduced)
        return knn_distances_full[:, 1:], knn_indices_full[:, 1:]


def _build_leiden_graph(
    X_reduced: np.ndarray,
    n_neighbors: int,
    graph_method: str,
    device: Device = "auto",
) -> tuple["ig.Graph", np.ndarray, np.ndarray, "sparse.csr_matrix"]:
    """Build the weighted igraph used by Leiden, using the chosen connectivity method.

    Parameters
    ----------
    X_reduced : np.ndarray
        Cell × feature matrix (PCA space or raw features).
    n_neighbors : int
        Number of nearest neighbours (self excluded).
    graph_method : {'umap', 'simple'}
        ``'umap'``   — UMAP fuzzy simplicial set weights (McInnes et al. 2018).
        ``'simple'`` — inverse-distance weights: 1 / (1 + d).
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend for the underlying k-NN and σ search.

    Returns
    -------
    g : ig.Graph
        Undirected igraph with ``weight`` edge attribute.
    knn_distances : np.ndarray
        Raw k-NN distances (self excluded).
    knn_indices : np.ndarray
        k-NN indices (self excluded).
    W_sym : scipy.sparse.csr_matrix
        Symmetrised adjacency (returned so cuGraph callers can skip the
        igraph round-trip).
    """
    try:
        import igraph as ig
    except ImportError as err:
        raise ImportError("igraph is required. Install with: pip install igraph leidenalg") from err

    n_cells = X_reduced.shape[0]

    knn_distances, knn_indices = _build_knn(X_reduced, n_neighbors, device=device)

    if graph_method == "umap":
        W = _compute_umap_connectivities(knn_distances, knn_indices, device=device)
        W_sym = (W + W.T) / 2
    else:  # simple
        row_ids = np.repeat(np.arange(n_cells, dtype=np.int32), n_neighbors)
        col_ids = knn_indices.ravel().astype(np.int32)
        weights_arr = (1.0 / (1.0 + knn_distances.ravel())).astype(np.float32)
        W = sparse.csr_matrix((weights_arr, (row_ids, col_ids)), shape=(n_cells, n_cells))
        W_sym = (W + W.T) / 2

    g = ig.Graph.Weighted_Adjacency(W_sym, mode="undirected", loops=False)
    return g, knn_distances, knn_indices, W_sym


def _leiden_cugraph(W_sym, resolution: float, random_state: int) -> tuple[np.ndarray, float]:
    """Run Leiden via cuGraph on a symmetric scipy CSR adjacency.

    Returns ``(membership, modularity)`` where:

    - ``membership`` is a numpy int64 array of cluster labels, one per row
      of ``W_sym``.
    - ``modularity`` is the partition modularity scalar reported by cuGraph
      (Newman-style modularity computed against the same weighted graph
      that igraph reports — the two backends are directly comparable, up
      to numerical precision).
    """
    import cudf
    import cugraph
    import cupy as cp

    print("  Running Leiden algorithm (cuGraph, GPU)...")

    coo = W_sym.tocoo()
    src = cp.asarray(coo.row, dtype=cp.int32)
    dst = cp.asarray(coo.col, dtype=cp.int32)
    w = cp.asarray(coo.data, dtype=cp.float32)
    edge_df = cudf.DataFrame({"src": src, "dst": dst, "weight": w})

    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(edge_df, source="src", destination="dst", edge_attr="weight", renumber=False)

    parts, modularity = cugraph.leiden(
        G,
        resolution=resolution,
        random_state=random_state,
        max_iter=100,
    )
    parts = parts.sort_values("vertex").reset_index(drop=True)
    membership = np.asarray(parts["partition"].to_numpy(), dtype=np.int64)
    try:
        mod = float(modularity)
    except (TypeError, ValueError):
        mod = float("nan")
    return membership, mod


def leiden_clustering(
    spatioloji_obj,
    layer: str | None = "normalized",
    use_highly_variable: bool = True,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    use_pca: bool = True,
    graph_method: Literal["umap", "simple"] = "umap",
    random_state: int = 42,
    output_column: str = "leiden",
    inplace: bool = True,
    device: Device = "auto",
):
    """
    Leiden clustering for single-cell data.

    Graph-based clustering using the Leiden algorithm (improved Louvain).
    The cell graph can be built with UMAP-style fuzzy simplicial set weights
    (default) or simple inverse-distance weights.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data.
    layer : str, optional
        Which expression layer to use, by default ``'normalized'``.
    use_highly_variable : bool, optional
        Use only highly variable genes if available, by default True.
        Requires running ``highly_variable_genes()`` first.
    resolution : float, optional
        Resolution parameter (higher = more clusters), by default 1.0.
    n_neighbors : int, optional
        Number of nearest neighbours for the cell graph, by default 15.
    n_pcs : int, optional
        Number of PCs to use when ``use_pca=True``, by default 50.
    use_pca : bool, optional
        Reduce to PCA space before building the graph, by default True.
        Stored ``X_pca`` embeddings are reused if available.
    graph_method : {'umap', 'simple'}, optional
        Edge-weight method, by default ``'umap'``.

        - ``'umap'`` — UMAP fuzzy simplicial set weights.  For each cell *i* a
          local bandwidth σ_i is estimated via binary search so that the fuzzy
          membership strengths sum to log₂(k).  The directed graph is then
          symmetrised with the fuzzy union (W = A + Aᵀ − A ⊙ Aᵀ), giving edge
          weights in [0, 1] that reflect local connectivity strength.
        - ``'simple'`` — inverse-distance weights: ``1 / (1 + d)``.

    random_state : int, optional
        Random seed, by default 42.
    output_column : str, optional
        Column name for cluster labels, by default ``'leiden'``.
    inplace : bool, optional
        Add cluster labels to cell_meta, by default True.
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  ``'auto'`` (default) accelerates k-NN, the UMAP
        σ search, and Leiden itself with cuML/cupy/cuGraph when RAPIDS is
        importable.  Falls back to leidenalg on CPU otherwise.  Note that
        cuGraph Leiden uses a different optimisation than leidenalg, so
        cluster *labels* will not match between backends — but cluster
        *structure* is comparable.

    Returns
    -------
    np.ndarray or None
        Cluster labels if ``inplace=False``, otherwise None.

    Raises
    ------
    ImportError
        If ``igraph`` or ``leidenalg`` are not installed (CPU backend), or
        if RAPIDS is missing and ``device='gpu'`` was requested.

    Examples
    --------
    >>> # Standard workflow — UMAP connectivities (default)
    >>> sj.processing.highly_variable_genes(sp, layer='log_normalized')
    >>> sj.processing.leiden_clustering(sp, layer='scaled', resolution=0.5)
    >>>
    >>> # Force GPU
    >>> sj.processing.leiden_clustering(sp, resolution=0.5, device='gpu')
    """
    print(f"\nLeiden clustering (resolution={resolution}, n_neighbors={n_neighbors}, graph={graph_method})")

    backend = _resolve_device(device, "Leiden")

    # ── PCA / feature matrix ──────────────────────────────────────────────────
    if use_pca:
        if hasattr(spatioloji_obj, "_embeddings") and "X_pca" in spatioloji_obj._embeddings:
            X_reduced = np.asarray(spatioloji_obj._embeddings["X_pca"])[:, :n_pcs]
            print(f"  Using stored PCA (first {n_pcs} PCs)")
        else:
            print(f"  No stored PCA found, computing PCA (n_pcs={n_pcs})...")
            X = _get_expression_matrix(spatioloji_obj, layer)
            if use_highly_variable:
                X = _subset_hvg(spatioloji_obj, X)
            pca_model = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]), random_state=random_state)
            X_reduced = pca_model.fit_transform(X)
            print(f"    Variance explained: {pca_model.explained_variance_ratio_.sum() * 100:.1f}%")
    else:
        X = _get_expression_matrix(spatioloji_obj, layer)
        if use_highly_variable:
            X = _subset_hvg(spatioloji_obj, X)
        X_reduced = X

    # ── Build cell graph ──────────────────────────────────────────────────────
    print(f"  Building {graph_method} connectivity graph...")
    g, _, _, W_sym = _build_leiden_graph(X_reduced, n_neighbors, graph_method, device=backend)

    # ── Leiden ────────────────────────────────────────────────────────────────
    if backend == "gpu":
        try:
            clusters, _ = _leiden_cugraph(W_sym, resolution=resolution, random_state=random_state)
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("Leiden", exc)
            backend = "cpu"

    if backend == "cpu":
        try:
            import leidenalg
        except ImportError as err:
            raise ImportError(
                "Leiden clustering requires igraph and leidenalg. Install with: pip install igraph leidenalg"
            ) from err
        print("  Running Leiden algorithm (leidenalg, CPU)...")
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=random_state,
        )
        clusters = np.array(partition.membership)

    n_clusters = len(np.unique(clusters))
    _, counts = np.unique(clusters, return_counts=True)

    print(f"  ✓ Found {n_clusters} clusters")
    print(f"    Cluster sizes - min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")

    if inplace:
        spatioloji_obj._cell_meta[output_column] = pd.Categorical(clusters)
        return None
    else:
        return clusters


def kmeans_clustering(
    spatioloji_obj,
    layer: str | None = "normalized",
    n_clusters: int = 10,
    n_pcs: int = 50,
    use_pca: bool = True,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    output_column: str = "kmeans",
    inplace: bool = True,
    device: Device = "auto",
):
    """
    K-means clustering.

    Classic k-means clustering on expression data or PCA space.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which expression layer to use, by default 'normalized'
    n_clusters : int, optional
        Number of clusters, by default 10
    n_pcs : int, optional
        Number of PCs to use if use_pca=True, by default 50
    use_pca : bool, optional
        Whether to use PCA before clustering, by default True
    n_init : int, optional
        Number of k-means initializations, by default 10
    max_iter : int, optional
        Maximum iterations, by default 300
    random_state : int, optional
        Random seed, by default 42
    output_column : str, optional
        Column name for cluster labels, by default 'kmeans'
    inplace : bool, optional
        Add cluster labels to spatioloji object, by default True
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  ``'auto'`` (default) uses cuML KMeans on GPU when
        RAPIDS is importable, otherwise scikit-learn on CPU.

    Returns
    -------
    np.ndarray or None
        Cluster labels if inplace=False, otherwise None

    Examples
    --------
    >>> # K-means with 10 clusters
    >>> sp.processing.kmeans_clustering(sp, n_clusters=10)
    >>>
    >>> # Force GPU
    >>> sp.processing.kmeans_clustering(sp, n_clusters=10, device='gpu')
    """
    print(f"\nK-means clustering (n_clusters={n_clusters})")

    backend = _resolve_device(device, "K-means")

    # PCA / feature matrix
    if use_pca:
        if hasattr(spatioloji_obj, "_embeddings") and "X_pca" in spatioloji_obj._embeddings:
            X_reduced = np.asarray(spatioloji_obj._embeddings["X_pca"])[:, :n_pcs]
            print(f"  Using stored PCA (first {n_pcs} PCs)")
        else:
            print(f"  No stored PCA found, computing PCA (n_pcs={n_pcs})...")
            X = _get_expression_matrix(spatioloji_obj, layer)
            pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]), random_state=random_state)
            X_reduced = pca.fit_transform(X)
            print(f"    Variance explained: {pca.explained_variance_ratio_.sum() * 100:.1f}%")
    else:
        X_reduced = _get_expression_matrix(spatioloji_obj, layer)

    inertia = None
    if backend == "gpu":
        try:
            from cuml.cluster import KMeans as cuKMeans

            print("  Running k-means (cuML GPU)...")
            km = cuKMeans(
                n_clusters=n_clusters, n_init=n_init,
                max_iter=max_iter, random_state=random_state,
            )
            X_gpu = _to_cupy(X_reduced)
            km.fit(X_gpu)
            clusters = _to_numpy(km.labels_).astype(np.int64)
            inertia = float(_to_numpy(km.inertia_)) if km.inertia_ is not None else None
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("K-means", exc)
            backend = "cpu"

    if backend == "cpu":
        print("  Running k-means (scikit-learn)...")
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=random_state)
        clusters = kmeans.fit_predict(X_reduced)
        inertia = float(kmeans.inertia_)

    # Calculate cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"  ✓ Clustered into {n_clusters} groups")
    print(f"    Cluster sizes - min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")
    if inertia is not None:
        print(f"    Inertia: {inertia:.2f}")

    if inplace:
        spatioloji_obj._cell_meta[output_column] = pd.Categorical(clusters)
        return None
    else:
        return clusters


def hierarchical_clustering(
    spatioloji_obj,
    layer: str | None = "normalized",
    n_clusters: int = 10,
    n_pcs: int = 50,
    use_pca: bool = True,
    linkage: Literal["ward", "complete", "average", "single"] = "ward",
    distance_threshold: float | None = None,
    output_column: str = "hierarchical",
    inplace: bool = True,
):
    """
    Hierarchical/Agglomerative clustering.

    Bottom-up hierarchical clustering with various linkage methods.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which expression layer to use, by default 'normalized'
    n_clusters : int, optional
        Number of clusters, by default 10
        Ignored if distance_threshold is set
    n_pcs : int, optional
        Number of PCs to use if use_pca=True, by default 50
    use_pca : bool, optional
        Whether to use PCA before clustering, by default True
    linkage : {'ward', 'complete', 'average', 'single'}, optional
        Linkage criterion, by default 'ward'
    distance_threshold : float, optional
        Distance threshold for cutting dendrogram, by default None
        If set, n_clusters is ignored
    output_column : str, optional
        Column name for cluster labels, by default 'hierarchical'
    inplace : bool, optional
        Add cluster labels to spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        Cluster labels if inplace=False, otherwise None

    Examples
    --------
    >>> # Hierarchical clustering with Ward linkage
    >>> sp.processing.hierarchical_clustering(sp, n_clusters=8, linkage='ward')
    >>>
    >>> # Cut dendrogram at specific distance
    >>> sp.processing.hierarchical_clustering(sp, distance_threshold=10.0)
    """
    print(f"\nHierarchical clustering (linkage={linkage})")

    # PCA / feature matrix
    if use_pca:
        if hasattr(spatioloji_obj, "_embeddings") and "X_pca" in spatioloji_obj._embeddings:
            X_reduced = np.asarray(spatioloji_obj._embeddings["X_pca"])[:, :n_pcs]
            print(f"  Using stored PCA (first {n_pcs} PCs)")
        else:
            print(f"  No stored PCA found, computing PCA (n_pcs={n_pcs})...")
            X = _get_expression_matrix(spatioloji_obj, layer)
            pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]))
            X_reduced = pca.fit_transform(X)
            print(f"    Variance explained: {pca.explained_variance_ratio_.sum() * 100:.1f}%")
    else:
        X_reduced = _get_expression_matrix(spatioloji_obj, layer)

    # Run hierarchical clustering
    print("  Running hierarchical clustering...")

    if distance_threshold is not None:
        clustering = AgglomerativeClustering(n_clusters=None, linkage=linkage, distance_threshold=distance_threshold)
    else:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    clusters = clustering.fit_predict(X_reduced)

    n_clusters_found = len(np.unique(clusters))

    # Calculate cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"  ✓ Found {n_clusters_found} clusters")
    print(f"    Cluster sizes - min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")

    if inplace:
        spatioloji_obj._cell_meta[output_column] = pd.Categorical(clusters)
        return None
    else:
        return clusters


def spatial_clustering(
    spatioloji_obj,
    coord_type: Literal["local", "global"] = "global",
    method: Literal["dbscan", "kmeans"] = "dbscan",
    eps: float = 100.0,
    min_samples: int = 5,
    n_clusters: int | None = None,
    output_column: str = "spatial_cluster",
    inplace: bool = True,
):
    """
    Spatial clustering based on cell coordinates.

    Cluster cells based on their spatial positions rather than expression.
    Useful for identifying spatial domains.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with spatial coordinates
    coord_type : {'local', 'global'}, optional
        Which coordinates to use, by default 'global'
    method : {'dbscan', 'kmeans'}, optional
        Clustering method, by default 'dbscan'
    eps : float, optional
        DBSCAN epsilon (max distance for neighbors), by default 100.0
    min_samples : int, optional
        DBSCAN minimum samples per cluster, by default 5
    n_clusters : int, optional
        Number of clusters for k-means, by default None
    output_column : str, optional
        Column name for cluster labels, by default 'spatial_cluster'
    inplace : bool, optional
        Add cluster labels to spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        Cluster labels if inplace=False, otherwise None

    Examples
    --------
    >>> # DBSCAN spatial clustering
    >>> sp.processing.spatial_clustering(sp, method='dbscan', eps=50)
    >>>
    >>> # K-means spatial clustering
    >>> sp.processing.spatial_clustering(sp, method='kmeans', n_clusters=5)
    """
    print(f"\nSpatial clustering (method={method}, coord_type={coord_type})")

    # Get spatial coordinates
    coords = spatioloji_obj.get_spatial_coords(coord_type=coord_type)

    # Run clustering
    if method == "dbscan":
        print(f"  Running DBSCAN (eps={eps}, min_samples={min_samples})...")
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = clustering.fit_predict(coords)

        n_clusters = len(np.unique(clusters[clusters != -1]))
        n_noise = (clusters == -1).sum()

        print(f"  ✓ Found {n_clusters} spatial clusters")
        print(f"    Noise points: {n_noise} ({n_noise / len(clusters) * 100:.1f}%)")

    elif method == "kmeans":
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for k-means")

        print(f"  Running k-means (n_clusters={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(coords)

        print(f"  ✓ Clustered into {n_clusters} spatial groups")

    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate cluster sizes
    unique, counts = np.unique(clusters[clusters != -1], return_counts=True)
    if len(unique) > 0:
        print(f"    Cluster sizes - min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")

    if inplace:
        spatioloji_obj._cell_meta[output_column] = pd.Categorical(clusters)
        return None
    else:
        return clusters


def spatially_constrained_clustering(
    spatioloji_obj,
    layer: str | None = "normalized",
    coord_type: Literal["local", "global"] = "global",
    n_clusters: int = 10,
    spatial_weight: float = 0.5,
    n_pcs: int = 50,
    use_pca: bool = True,
    random_state: int = 42,
    output_column: str = "spatial_constrained",
    inplace: bool = True,
):
    """
    Clustering that considers both expression and spatial information.

    Combines expression similarity with spatial proximity for clustering.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression and spatial data
    layer : str, optional
        Which expression layer to use, by default 'normalized'
    coord_type : {'local', 'global'}, optional
        Which coordinates to use, by default 'global'
    n_clusters : int, optional
        Number of clusters, by default 10
    spatial_weight : float, optional
        Weight for spatial vs expression (0=expression only, 1=spatial only),
        by default 0.5
    n_pcs : int, optional
        Number of PCs for expression, by default 50
    use_pca : bool, optional
        Whether to use PCA on expression, by default True
    random_state : int, optional
        Random seed, by default 42
    output_column : str, optional
        Column name for cluster labels, by default 'spatial_constrained'
    inplace : bool, optional
        Add cluster labels to spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        Cluster labels if inplace=False, otherwise None

    Examples
    --------
    >>> # Equal weighting of expression and space
    >>> sp.processing.spatially_constrained_clustering(sp, spatial_weight=0.5)
    >>>
    >>> # More emphasis on spatial proximity
    >>> sp.processing.spatially_constrained_clustering(sp, spatial_weight=0.8)
    """
    print(f"\nSpatially constrained clustering (spatial_weight={spatial_weight})")

    # PCA / feature matrix
    if use_pca:
        if hasattr(spatioloji_obj, "_embeddings") and "X_pca" in spatioloji_obj._embeddings:
            X_expr_reduced = np.asarray(spatioloji_obj._embeddings["X_pca"])[:, :n_pcs]
            print(f"  Using stored PCA (first {n_pcs} PCs)")
        else:
            print(f"  No stored PCA found, computing PCA (n_pcs={n_pcs})...")
            X_expr = _get_expression_matrix(spatioloji_obj, layer)
            pca = PCA(n_components=min(n_pcs, X_expr.shape[0], X_expr.shape[1]), random_state=random_state)
            X_expr_reduced = pca.fit_transform(X_expr)
            print(f"    Variance explained: {pca.explained_variance_ratio_.sum() * 100:.1f}%")
    else:
        X_expr_reduced = _get_expression_matrix(spatioloji_obj, layer)

    # Get spatial coordinates
    X_spatial = spatioloji_obj.get_spatial_coords(coord_type=coord_type)

    # Normalize both to [0, 1] range
    from sklearn.preprocessing import MinMaxScaler

    scaler_expr = MinMaxScaler()
    X_expr_norm = scaler_expr.fit_transform(X_expr_reduced)

    scaler_spatial = MinMaxScaler()
    X_spatial_norm = scaler_spatial.fit_transform(X_spatial)

    # Combine expression and spatial data
    X_combined = np.hstack([X_expr_norm * (1 - spatial_weight), X_spatial_norm * spatial_weight])

    print(f"  Combined feature space: {X_combined.shape[1]} dimensions")
    print(f"    Expression: {X_expr_norm.shape[1]} dims × {1 - spatial_weight:.2f} weight")
    print(f"    Spatial: {X_spatial_norm.shape[1]} dims × {spatial_weight:.2f} weight")

    # Run k-means on combined space
    print("  Running k-means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_combined)

    # Calculate cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"  ✓ Found {n_clusters} clusters")
    print(f"    Cluster sizes - min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")

    if inplace:
        spatioloji_obj._cell_meta[output_column] = pd.Categorical(clusters)
        return None
    else:
        return clusters


def _kmeans_fit_predict(X, k: int, n_init: int, random_state: int, backend: str):
    """Run KMeans on the chosen backend, return (labels, inertia) as numpy/float."""
    if backend == "gpu":
        from cuml.cluster import KMeans as cuKMeans

        km = cuKMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        X_gpu = _to_cupy(X)
        km.fit(X_gpu)
        labels = _to_numpy(km.labels_).astype(np.int64)
        inertia = float(_to_numpy(km.inertia_)) if km.inertia_ is not None else float("nan")
        return labels, inertia

    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, float(km.inertia_)


def find_optimal_clusters(
    spatioloji_obj,
    layer: str | None = "normalized",
    method: Literal["elbow", "silhouette", "gap", "davies_bouldin", "calinski_harabasz", "all"] = "silhouette",
    k_range: tuple[int, int] = (2, 20),
    n_pcs: int = 50,
    use_pca: bool = True,
    sample_size: int | None = 5000,
    random_state: int = 42,
    device: Device = "auto",
) -> dict[str, int | list[float]]:
    """
    Find optimal number of clusters using one or all quality metrics.

    Sweeps k-means over a range of k values and evaluates clustering quality
    using the specified metric(s). The ``'all'`` method combines silhouette,
    Davies-Bouldin, and Calinski-Harabasz scores into a consensus ranking.

    Args:
        spatioloji_obj: Spatioloji object with expression data.
        layer: Expression layer to use, by default 'normalized'.
        method: Scoring method. One of:
            - ``'elbow'`` — within-cluster inertia (K-Means only).
            - ``'silhouette'`` — mean silhouette score (higher = better).
            - ``'gap'`` — gap statistic vs. random reference (higher = better).
            - ``'davies_bouldin'`` — ratio of scatter to separation (lower = better).
            - ``'calinski_harabasz'`` — variance ratio criterion (higher = better).
            - ``'all'`` — runs silhouette + davies_bouldin + calinski_harabasz and
              returns a normalized consensus score plus individual scores.
        k_range: (min_k, max_k) range to evaluate, by default (2, 20).
        n_pcs: Number of PCs to use when use_pca=True, by default 50.
        use_pca: Whether to reduce to PCA space before clustering, by default True.
        sample_size: Max cells to use for silhouette computation (subsamples if
            n_cells > sample_size). Set to None to use all cells. By default 5000.
        random_state: Random seed, by default 42.

    Returns:
        dict with keys:
            - ``optimal_k`` (int): Recommended number of clusters.
            - ``k_values`` (list[int]): All k values tested.
            - ``scores`` (list[float]): Primary score per k (method-dependent).
            - ``method`` (str): Method used.
            - When method='all': also includes ``silhouette_scores``,
              ``davies_bouldin_scores``, ``calinski_harabasz_scores``,
              ``optimal_k_silhouette``, ``optimal_k_davies_bouldin``,
              ``optimal_k_calinski_harabasz``.

    Example:
        >>> result = find_optimal_clusters(sp, method='all', k_range=(2, 15))
        >>> print(f"Consensus optimal k: {result['optimal_k']}")
        >>> kmeans_clustering(sp, n_clusters=result['optimal_k'])
    """
    print(f"\nFinding optimal number of clusters (method={method})")
    print(f"  Testing k from {k_range[0]} to {k_range[1]}")

    backend = _resolve_device(device, "find_optimal_clusters")

    # PCA / feature matrix
    if use_pca:
        if hasattr(spatioloji_obj, "_embeddings") and "X_pca" in spatioloji_obj._embeddings:
            X_reduced = np.asarray(spatioloji_obj._embeddings["X_pca"])[:, :n_pcs]
            print(f"  Using stored PCA (first {n_pcs} PCs)")
        else:
            print(f"  No stored PCA found, computing PCA (n_pcs={n_pcs})...")
            X = _get_expression_matrix(spatioloji_obj, layer)
            pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]), random_state=random_state)
            X_reduced = pca.fit_transform(X)
            print(f"    Variance explained: {pca.explained_variance_ratio_.sum() * 100:.1f}%")
    else:
        X_reduced = _get_expression_matrix(spatioloji_obj, layer)

    k_values = range(k_range[0], k_range[1] + 1)
    scores = []

    if method == "elbow":
        for k in k_values:
            _, inertia = _kmeans_fit_predict(X_reduced, k, n_init=10, random_state=random_state, backend=backend)
            scores.append(inertia)
            print(f"    k={k}: inertia={inertia:.2f}")

        line = np.linspace(scores[0], scores[-1], len(scores))
        distances = np.abs(np.array(scores) - line)
        optimal_k = list(k_values)[int(np.argmax(distances))]

    elif method == "silhouette":
        from sklearn.metrics import silhouette_score

        for k in k_values:
            labels, _ = _kmeans_fit_predict(X_reduced, k, n_init=10, random_state=random_state, backend=backend)
            X_sil = X_reduced
            if sample_size is not None and X_reduced.shape[0] > sample_size:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(X_reduced.shape[0], size=sample_size, replace=False)
                X_sil, labels_sil = X_reduced[idx], labels[idx]
            else:
                X_sil, labels_sil = X_reduced, labels
            score = silhouette_score(X_sil, labels_sil)
            scores.append(score)
            print(f"    k={k}: silhouette={score:.4f}")

        optimal_k = list(k_values)[int(np.argmax(scores))]

    elif method == "gap":
        print("  Computing gap statistic (this may take a while)...")

        def compute_inertia(X, k):
            _, inertia = _kmeans_fit_predict(X, k, n_init=10, random_state=random_state, backend=backend)
            return inertia

        n_refs = 10
        gaps = []

        for k in k_values:
            ref_inertias = []
            obs_inertia = compute_inertia(X_reduced, k)
            for _ in range(n_refs):
                X_ref = np.random.uniform(X_reduced.min(axis=0), X_reduced.max(axis=0), size=X_reduced.shape)
                ref_inertias.append(compute_inertia(X_ref, k))
            gap = np.log(np.mean(ref_inertias)) - np.log(obs_inertia)
            gaps.append(gap)
            print(f"    k={k}: gap={gap:.4f}")

        scores = gaps
        optimal_k = list(k_values)[int(np.argmax(gaps))]

    elif method == "davies_bouldin":
        from sklearn.metrics import davies_bouldin_score

        for k in k_values:
            labels, _ = _kmeans_fit_predict(X_reduced, k, n_init=10, random_state=random_state, backend=backend)
            score = davies_bouldin_score(X_reduced, labels)
            scores.append(score)
            print(f"    k={k}: davies_bouldin={score:.4f}")

        optimal_k = list(k_values)[int(np.argmin(scores))]

    elif method == "calinski_harabasz":
        from sklearn.metrics import calinski_harabasz_score

        for k in k_values:
            labels, _ = _kmeans_fit_predict(X_reduced, k, n_init=10, random_state=random_state, backend=backend)
            score = calinski_harabasz_score(X_reduced, labels)
            scores.append(score)
            print(f"    k={k}: calinski_harabasz={score:.2f}")

        optimal_k = list(k_values)[int(np.argmax(scores))]

    elif method == "all":
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

        sil_scores, db_scores, ch_scores = [], [], []
        print("  Running silhouette + Davies-Bouldin + Calinski-Harabasz...")

        for k in k_values:
            labels, _ = _kmeans_fit_predict(X_reduced, k, n_init=10, random_state=random_state, backend=backend)

            if sample_size is not None and X_reduced.shape[0] > sample_size:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(X_reduced.shape[0], size=sample_size, replace=False)
                sil = silhouette_score(X_reduced[idx], labels[idx])
            else:
                sil = silhouette_score(X_reduced, labels)

            db = davies_bouldin_score(X_reduced, labels)
            ch = calinski_harabasz_score(X_reduced, labels)

            sil_scores.append(sil)
            db_scores.append(db)
            ch_scores.append(ch)
            print(f"    k={k}: silhouette={sil:.4f}  davies_bouldin={db:.4f}  calinski_harabasz={ch:.2f}")

        # Normalize each to [0,1]; invert DB so higher = better for all
        def _norm(arr):
            a = np.array(arr, dtype=float)
            rng_val = a.max() - a.min()
            return (a - a.min()) / (rng_val + 1e-12)

        sil_norm = _norm(sil_scores)
        db_norm = 1.0 - _norm(db_scores)  # invert: lower DB is better
        ch_norm = _norm(ch_scores)
        consensus = (sil_norm + db_norm + ch_norm) / 3.0

        k_list = list(k_values)
        optimal_k = k_list[int(np.argmax(consensus))]
        scores = consensus.tolist()

        print("\n  Per-metric optimal k:")
        print(f"    silhouette:         k={k_list[int(np.argmax(sil_scores))]}")
        print(f"    davies_bouldin:     k={k_list[int(np.argmin(db_scores))]}")
        print(f"    calinski_harabasz:  k={k_list[int(np.argmax(ch_scores))]}")
        print(f"  Consensus optimal k: {optimal_k}")

        return {
            "optimal_k": optimal_k,
            "k_values": k_list,
            "scores": scores,
            "method": "all",
            "silhouette_scores": sil_scores,
            "davies_bouldin_scores": db_scores,
            "calinski_harabasz_scores": ch_scores,
            "optimal_k_silhouette": k_list[int(np.argmax(sil_scores))],
            "optimal_k_davies_bouldin": k_list[int(np.argmin(db_scores))],
            "optimal_k_calinski_harabasz": k_list[int(np.argmax(ch_scores))],
        }

    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose from: elbow, silhouette, gap, davies_bouldin, calinski_harabasz, all"
        )

    print(f"\n  ✓ Optimal k: {optimal_k}")

    return {"optimal_k": optimal_k, "k_values": list(k_values), "scores": scores, "method": method}


def assess_clustering_quality(
    spatioloji_obj,
    cluster_col: str,
    layer: str | None = "normalized",
    n_pcs: int = 50,
    use_pca: bool = True,
    sample_size: int | None = 5000,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Evaluate the quality of an existing clustering result.

    Computes silhouette score, Davies-Bouldin index, and Calinski-Harabasz index
    for cluster labels already stored in ``spatioloji_obj.cell_meta``. Uses stored
    PCA embeddings when available to avoid redundant computation.

    Args:
        spatioloji_obj: Spatioloji object with expression data and cluster labels.
        cluster_col: Column in ``cell_meta`` containing cluster labels to evaluate.
        layer: Expression layer to use if PCA must be recomputed, by default 'normalized'.
        n_pcs: Number of PCs to use, by default 50.
        use_pca: Whether to project to PCA space before scoring, by default True.
        sample_size: Max cells for silhouette computation (subsampled if exceeded).
            Set to None to use all cells. By default 5000.
        random_state: Random seed for subsampling, by default 42.

    Returns:
        dict with keys:
            - ``n_clusters`` (int): Number of unique clusters found.
            - ``silhouette`` (float): Mean silhouette score [-1, 1]; higher = better.
            - ``davies_bouldin`` (float): Davies-Bouldin index; lower = better.
            - ``calinski_harabasz`` (float): Calinski-Harabasz index; higher = better.
            - ``cluster_sizes`` (dict): {cluster_label: cell_count}.

    Raises:
        ValueError: If ``cluster_col`` is not found in ``cell_meta``.

    Example:
        >>> leiden_clustering(sp, resolution=0.5)
        >>> metrics = assess_clustering_quality(sp, cluster_col='leiden')
        >>> print(metrics)
    """
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    if cluster_col not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in cell_meta.")

    labels = np.asarray(spatioloji_obj.cell_meta[cluster_col])
    # Encode non-numeric labels (including categorical)
    if not np.issubdtype(labels.dtype, np.integer):
        _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(int)

    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        raise ValueError(f"Need at least 2 clusters to compute quality metrics, found {n_clusters}.")

    # PCA / feature matrix
    if use_pca:
        if hasattr(spatioloji_obj, "_embeddings") and "X_pca" in spatioloji_obj._embeddings:
            X = np.asarray(spatioloji_obj._embeddings["X_pca"])[:, :n_pcs]
        else:
            X_raw = _get_expression_matrix(spatioloji_obj, layer)
            pca = PCA(n_components=min(n_pcs, X_raw.shape[0], X_raw.shape[1]), random_state=random_state)
            X = pca.fit_transform(X_raw)
    else:
        X = _get_expression_matrix(spatioloji_obj, layer)

    # Subsample for silhouette if needed
    if sample_size is not None and X.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_sil, labels_sil = X[idx], labels[idx]
    else:
        X_sil, labels_sil = X, labels

    sil = float(silhouette_score(X_sil, labels_sil))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))

    unique_labels, counts = np.unique(spatioloji_obj.cell_meta[cluster_col].values, return_counts=True)
    cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist(), strict=False))

    metrics = {
        "n_clusters": n_clusters,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "cluster_sizes": cluster_sizes,
    }

    print(f"\nClustering quality for '{cluster_col}' ({n_clusters} clusters):")
    print(f"  Silhouette score:        {sil:+.4f}  (higher = better, max 1.0)")
    print(f"  Davies-Bouldin index:    {db:.4f}   (lower = better, min 0.0)")
    print(f"  Calinski-Harabasz index: {ch:.2f}  (higher = better)")
    sizes = list(cluster_sizes.values())
    print(f"  Cluster sizes - min: {min(sizes)}, max: {max(sizes)}, mean: {np.mean(sizes):.1f}")

    return metrics


def leiden_resolution_sweep(
    spatioloji_obj,
    resolutions: list[float] | None = None,
    n_runs: int = 5,
    layer: str | None = "normalized",
    use_highly_variable: bool = True,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    graph_method: Literal["umap", "simple"] = "umap",
    random_state: int = 42,
    n_jobs: int = 1,
    device: Device = "auto",
) -> pd.DataFrame:
    """
    Sweep Leiden resolution parameter and report stability and quality metrics.

    For each resolution, runs Leiden ``n_runs`` times with different seeds and
    measures cluster stability (mean pairwise ARI), modularity, and the number
    of clusters found. Use the resulting DataFrame to pick a resolution in a
    stable plateau with the desired granularity.

    Args:
        spatioloji_obj: Spatioloji object with expression data.
        resolutions: List of resolution values to test. Defaults to
            ``[0.1, 0.2, ..., 2.0]`` if None.
        n_runs: Number of Leiden runs per resolution for stability estimation,
            by default 5.
        layer: Expression layer to use when no stored PCA is available,
            by default 'normalized'. Ignored if ``X_pca`` exists in embeddings.
        use_highly_variable: Use only highly variable genes when computing
            PCA on the fly, by default True. Ignored if ``X_pca`` exists.
        n_pcs: Number of PCs to use, by default 50.
        n_neighbors: k for the k-NN graph, by default 15.
        graph_method: Edge-weight method — ``'umap'`` (default, fuzzy simplicial
            set) or ``'simple'`` (inverse-distance).  See :func:`leiden_clustering`
            for details.
        random_state: Base random seed (seeds used are random_state, random_state+1, …),
            by default 42.
        n_jobs: Number of parallel threads for Leiden calls, by default 1.
            Set to -1 to use all available CPUs. Values > 1 parallelize the
            ``len(resolutions) × n_runs`` Leiden calls via threads.
            Ignored when ``device='gpu'`` (GPU runs serialise on the device).
        device: Compute backend (``'auto' | 'cpu' | 'gpu'``).  When GPU is
            used, the k-NN, σ search and per-resolution Leiden runs all
            execute on cuML/cupy/cuGraph; ``modularity_mean`` is the
            modularity scalar reported by ``cugraph.leiden`` and is directly
            comparable to the igraph value reported on the CPU path.

    Returns:
        pd.DataFrame with columns:
            - ``resolution``: Resolution parameter.
            - ``n_clusters_mean``: Mean number of clusters across runs.
            - ``n_clusters_min`` / ``n_clusters_max``: Range of cluster counts.
            - ``mean_ari``: Mean pairwise ARI across all run pairs (stability).
            - ``std_ari``: Std of pairwise ARIs.
            - ``modularity_mean``: Mean igraph modularity across runs.

    Raises:
        ImportError: If ``igraph`` or ``leidenalg`` are not installed.

    Example:
        >>> df = leiden_resolution_sweep(sp, resolutions=[0.2, 0.4, 0.6, 0.8, 1.0])
        >>> print(df)
        >>> # Pick resolution with high mean_ari and desired n_clusters
        >>> leiden_clustering(sp, resolution=0.6)
    """
    from sklearn.metrics import adjusted_rand_score

    backend = _resolve_device(device, "leiden_resolution_sweep")

    if backend == "cpu":
        try:
            import leidenalg
        except ImportError as err:
            raise ImportError(
                "leiden_resolution_sweep requires igraph and leidenalg on CPU. "
                "Install with: pip install igraph leidenalg"
            ) from err

    if resolutions is None:
        resolutions = [round(r, 1) for r in np.arange(0.1, 2.1, 0.1)]

    # --- PCA embeddings (prefer stored, fall back to computing) ---
    if hasattr(spatioloji_obj, "_embeddings") and "X_pca" in spatioloji_obj._embeddings:
        X_reduced = np.asarray(spatioloji_obj._embeddings["X_pca"])[:, :n_pcs]
        print(f"  Using stored PCA (first {n_pcs} PCs)")
    else:
        print(f"  No stored PCA found, computing PCA (n_pcs={n_pcs})...")
        if layer is None:
            X = spatioloji_obj.expression.get_dense()
        else:
            X = spatioloji_obj.get_layer(layer)
            if sparse.issparse(X):
                X = X.toarray()

        if use_highly_variable and "highly_variable" in spatioloji_obj.gene_meta.columns:
            hvg_mask = spatioloji_obj.gene_meta["highly_variable"].values
            if hvg_mask.sum() > 0:
                X = X[:, hvg_mask]

        pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]), random_state=random_state)
        X_reduced = pca.fit_transform(X)
        print(f"    Variance explained: {pca.explained_variance_ratio_.sum() * 100:.1f}%")

    # --- Build cell graph once (shared across all resolutions) ---
    print(f"  Building {graph_method} connectivity graph...")
    g, _, _, W_sym = _build_leiden_graph(X_reduced, n_neighbors, graph_method, device=backend)

    n_total = len(resolutions) * n_runs
    print(
        f"\nLeiden resolution sweep "
        f"({len(resolutions)} resolutions × {n_runs} runs = {n_total} calls, "
        f"graph={graph_method}, device={backend})"
    )
    print(f"  {'Resolution':>10}  {'n_clusters':>10}  {'mean_ARI':>9}  {'modularity':>10}")
    print("  " + "-" * 46)

    # --- Run all (resolution, seed) pairs ---
    import os
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    def _run_one_cpu(res: float, seed: int) -> tuple[float, int, np.ndarray, float]:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=res,
            seed=seed,
        )
        return res, seed, np.array(partition.membership), float(partition.modularity)

    def _run_one_gpu(res: float, seed: int) -> tuple[float, int, np.ndarray, float]:
        mem, mod = _leiden_cugraph(W_sym, resolution=res, random_state=seed)
        return res, seed, mem, mod

    runner = _run_one_gpu if backend == "gpu" else _run_one_cpu
    tasks = [(res, random_state + run) for res in resolutions for run in range(n_runs)]

    # GPU runs serialise on-device, so threading there gains nothing.
    if backend == "cpu" and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            results_list = list(pool.map(lambda args: runner(*args), tasks))
    else:
        results_list = [runner(*t) for t in tasks]

    # --- Group results by resolution ---
    grouped = defaultdict(lambda: {"memberships": [], "modularities": []})
    for res, _seed, mem, mod in results_list:
        grouped[res]["memberships"].append(mem)
        grouped[res]["modularities"].append(mod)

    rows = []
    for res in resolutions:
        memberships = grouped[res]["memberships"]
        modularities = grouped[res]["modularities"]
        n_clusters_list = [int(m.max()) + 1 for m in memberships]

        # Pairwise ARI across all run pairs
        ari_vals = [
            adjusted_rand_score(memberships[i], memberships[j]) for i in range(n_runs) for j in range(i + 1, n_runs)
        ]
        mean_ari = float(np.mean(ari_vals)) if ari_vals else 1.0
        std_ari = float(np.std(ari_vals)) if ari_vals else 0.0

        row = {
            "resolution": res,
            "n_clusters_mean": float(np.mean(n_clusters_list)),
            "n_clusters_min": int(min(n_clusters_list)),
            "n_clusters_max": int(max(n_clusters_list)),
            "mean_ari": mean_ari,
            "std_ari": std_ari,
            "modularity_mean": float(np.mean(modularities)),
        }
        rows.append(row)
        print(f"  {res:>10.2f}  {row['n_clusters_mean']:>10.1f}  {mean_ari:>9.4f}  {row['modularity_mean']:>10.4f}")

    df = pd.DataFrame(rows)
    best_res = df.loc[df["mean_ari"].idxmax(), "resolution"]
    print(f"\n  Most stable resolution: {best_res:.2f}  (mean ARI={df['mean_ari'].max():.4f})")
    return df
