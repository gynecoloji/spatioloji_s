"""Spatial motif discovery via neighborhood composition clustering.

Builds a per-cell neighbourhood composition feature vector from the spatial
graph, then clusters cells to identify recurring local tissue motifs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from sklearn.preprocessing import normalize

from spatioloji_s.spatial._motif_types import MotifCatalog, _get_cell_ids, _get_sparse_adjacency

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    from spatioloji_s.spatial.point.graph import PointSpatialGraph
    from spatioloji_s.spatial.polygon.graph import PolygonSpatialGraph


def _is_point_graph(graph: PolygonSpatialGraph | PointSpatialGraph) -> bool:
    """Return True if *graph* is a PointSpatialGraph (has cell_ids, not cell_index)."""
    return hasattr(graph, "cell_ids") and not hasattr(graph, "cell_index")


def _build_composition_matrix(
    adj: sp_sparse.csr_matrix,
    labels: pd.Series,
    k_hops: int = 1,
) -> sp_sparse.csr_matrix:
    """Build a (n_cells x n_types) neighbourhood composition matrix.

    Args:
        adj: Sparse binary adjacency matrix (n_cells x n_cells).
        labels: Cell-type labels aligned to adjacency rows.
        k_hops: Neighbourhood radius in graph hops.

    Returns:
        L2-normalised sparse composition matrix.
    """
    # One-hot encode cell types
    categories = pd.Categorical(labels)
    codes = categories.codes
    n_cells = len(labels)
    n_types = len(categories.categories)

    one_hot = sp_sparse.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (np.arange(n_cells), codes)),
        shape=(n_cells, n_types),
    )

    # Multi-hop adjacency
    if k_hops > 1:
        A = adj.copy().astype(bool).astype(np.float32)
        A_power = A.copy()
        for _ in range(k_hops - 1):
            A_power = A_power @ A
        # Binarise (any path of length <= k_hops)
        A_power = A_power.astype(bool).astype(np.float32)
        adj = sp_sparse.csr_matrix(A_power)

    composition = adj @ one_hot  # (n_cells x n_types)
    composition = normalize(composition, norm="l2", axis=1)
    return composition


def _auto_select_n_motifs(
    features: sp_sparse.csr_matrix,
    max_k: int = 20,
    subsample: int = 5000,
    random_state: int = 0,
) -> int:
    """Choose *n_motifs* via Calinski-Harabasz on a subsample.

    Args:
        features: Feature matrix (sparse or dense).
        max_k: Maximum number of clusters to try.
        subsample: Maximum number of cells to subsample.
        random_state: Random seed.

    Returns:
        Optimal number of clusters (>= 2).
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import calinski_harabasz_score

    rng = np.random.RandomState(random_state)
    n = features.shape[0]
    if n > subsample:
        idx = rng.choice(n, subsample, replace=False)
        X = features[idx]
    else:
        X = features

    if sp_sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    best_score = -1.0
    best_k = 2
    upper = min(max_k, X_dense.shape[0] - 1)
    for k in range(2, upper + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3)
        cluster_labels = km.fit_predict(X_dense)
        if len(set(cluster_labels)) < 2:
            continue
        score = calinski_harabasz_score(X_dense, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _cluster_kmeans(
    features: sp_sparse.csr_matrix,
    n_motifs: int,
    random_state: int = 0,
) -> np.ndarray:
    """Cluster feature matrix with MiniBatchKMeans.

    Args:
        features: (n_cells x n_features) matrix.
        n_motifs: Number of clusters.
        random_state: Random seed.

    Returns:
        Array of integer cluster labels (length n_cells).
    """
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=n_motifs, random_state=random_state, n_init=5)
    return km.fit_predict(features)


def _cluster_leiden(
    features: sp_sparse.csr_matrix,
    k_nn: int = 15,
    resolution: float = 1.0,
    random_state: int = 0,
) -> np.ndarray:
    """Cluster feature matrix with Leiden community detection.

    Args:
        features: (n_cells x n_features) matrix.
        k_nn: Number of nearest neighbours for the KNN graph.
        resolution: Leiden resolution parameter.
        random_state: Random seed.

    Returns:
        Array of integer cluster labels (length n_cells).

    Raises:
        ImportError: If leidenalg or igraph are not installed.
    """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError("Install igraph with: pip install igraph") from None

    try:
        import leidenalg
    except ImportError:
        raise ImportError("Install leidenalg with: pip install leidenalg") from None

    from sklearn.neighbors import NearestNeighbors

    k_nn = min(k_nn, features.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k_nn, algorithm="ball_tree")
    if sp_sparse.issparse(features):
        nn.fit(features.toarray())
    else:
        nn.fit(features)
    knn_graph = nn.kneighbors_graph(mode="connectivity")

    # Symmetrise
    knn_graph = knn_graph.maximum(knn_graph.T)

    sources, targets = knn_graph.nonzero()
    mask = sources < targets
    edges = list(zip(sources[mask].tolist(), targets[mask].tolist(), strict=True))

    g = ig.Graph(n=features.shape[0], edges=edges, directed=False)
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_state,
    )
    return np.array(partition.membership)


def discover_motifs(
    sp: spatioloji,
    graph: PolygonSpatialGraph | PointSpatialGraph,
    group_col: str = "cell_type",
    n_motifs: int | None = 5,
    method: Literal["kmeans", "leiden"] = "kmeans",
    k_hops: int = 1,
    use_morphology: bool = False,
    use_density: bool = False,
    keep_features: bool = False,
    store: bool = False,
    random_state: int = 0,
    leiden_resolution: float = 1.0,
    leiden_k_nn: int = 15,
) -> MotifCatalog:
    """Discover recurring local spatial motifs by clustering neighbourhood composition.

    Builds a per-cell feature vector from the neighbourhood composition of the
    spatial graph (optionally augmented with morphology and density), then
    clusters cells to identify motifs.

    Args:
        sp: spatioloji object with cell metadata containing *group_col*.
        graph: Pre-built spatial graph (PolygonSpatialGraph or PointSpatialGraph).
        group_col: Column in ``sp.cell_meta`` holding cell-type labels.
        n_motifs: Number of motifs to discover.  ``None`` triggers automatic
            selection via Calinski-Harabasz (kmeans only).
        method: Clustering algorithm — ``"kmeans"`` (default) or ``"leiden"``.
        k_hops: Neighbourhood radius in graph hops (1 = direct neighbours).
        use_morphology: Append polygon morphology stats (area, circularity,
            elongation) to the feature vector.  Requires a polygon graph.
        use_density: Append local cell density to the feature vector.
            Requires a polygon graph.
        keep_features: If True, store the feature matrix in the result.
        store: If True, write motif labels into ``sp.cell_meta["motif"]``.
        random_state: Random seed for reproducibility.
        leiden_resolution: Resolution parameter for Leiden clustering.
        leiden_k_nn: Number of nearest neighbours for the Leiden KNN graph.

    Returns:
        MotifCatalog with labels, signatures, counts, and parameters.

    Raises:
        ValueError: If *group_col* is not in ``sp.cell_meta``, if *method*
            is not recognised, or if morphology/density is requested with a
            point graph.

    Example:
        >>> from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        >>> from spatioloji_s.spatial.polygon.motifs import discover_motifs
        >>> graph = build_buffer_graph(sp, buffer_distance=30)
        >>> catalog = discover_motifs(sp, graph, group_col="cell_type", n_motifs=5)
        >>> catalog.signatures
    """
    # ---- validation --------------------------------------------------------
    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"Column '{group_col}' not found in sp.cell_meta. Available: {list(sp.cell_meta.columns)}")

    if method not in ("kmeans", "leiden"):
        raise ValueError(f"Unknown clustering method '{method}'. Choose 'kmeans' or 'leiden'.")

    is_point = _is_point_graph(graph)
    if use_morphology and is_point:
        raise ValueError("Morphology features require a polygon graph (PolygonSpatialGraph).")
    if use_density and is_point:
        raise ValueError("Density features require a polygon graph (PolygonSpatialGraph).")

    # ---- cell IDs & labels -------------------------------------------------
    cell_ids = _get_cell_ids(graph)
    adj = _get_sparse_adjacency(graph)
    labels = sp.cell_meta[group_col].reindex(cell_ids)

    # ---- composition feature matrix ----------------------------------------
    features = _build_composition_matrix(adj, labels, k_hops=k_hops)
    type_names = pd.Categorical(labels).categories.tolist()

    # ---- optional polygon features -----------------------------------------
    extra_cols: list[str] = []
    extra_parts: list[sp_sparse.csr_matrix] = []

    if use_morphology:
        from spatioloji_s.spatial.polygon.morphology import compute_morphology

        morph_df = compute_morphology(sp)
        morph_cols = ["area", "circularity", "elongation"]
        avail = [c for c in morph_cols if c in morph_df.columns]
        morph_vals = morph_df.reindex(cell_ids)[avail].fillna(0).values.astype(np.float32)
        morph_vals = normalize(morph_vals, norm="l2", axis=0)
        extra_parts.append(sp_sparse.csr_matrix(morph_vals))
        extra_cols.extend(avail)

    if use_density:
        degrees = np.array(adj.sum(axis=1)).flatten().astype(np.float32)
        degrees = degrees / (degrees.max() + 1e-12)
        extra_parts.append(sp_sparse.csr_matrix(degrees.reshape(-1, 1)))
        extra_cols.append("density")

    if extra_parts:
        features = sp_sparse.hstack([features] + extra_parts, format="csr")

    # ---- clustering --------------------------------------------------------
    if method == "kmeans":
        if n_motifs is None:
            n_motifs_resolved = _auto_select_n_motifs(features, random_state=random_state)
        else:
            n_motifs_resolved = n_motifs
        cluster_labels = _cluster_kmeans(features, n_motifs_resolved, random_state=random_state)
    else:
        cluster_labels = _cluster_leiden(
            features,
            k_nn=leiden_k_nn,
            resolution=leiden_resolution,
            random_state=random_state,
        )

    # ---- build MotifCatalog ------------------------------------------------
    labels_series = pd.Series(cluster_labels, index=cell_ids, name="motif")

    # Signatures: mean composition per motif (only composition columns)
    comp_dense = _build_composition_matrix(adj, labels, k_hops=k_hops).toarray()
    sig_df = pd.DataFrame(comp_dense, index=cell_ids, columns=type_names)
    sig_df["motif"] = cluster_labels
    signatures = sig_df.groupby("motif")[type_names].mean()

    counts = labels_series.value_counts().sort_index()
    counts.index.name = "motif"
    counts.name = "n_cells"

    params = {
        "method": method,
        "n_motifs": int(labels_series.nunique()),
        "k_hops": k_hops,
        "use_morphology": use_morphology,
        "use_density": use_density,
        "random_state": random_state,
    }

    catalog = MotifCatalog(
        labels=labels_series,
        signatures=signatures,
        counts=counts,
        group_col=group_col,
        feature_matrix=features if keep_features else None,
        params=params,
    )

    # ---- optional store ----------------------------------------------------
    if store:
        sp.cell_meta["motif"] = labels_series.reindex(sp.cell_meta.index)

    return catalog
