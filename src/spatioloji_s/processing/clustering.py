"""
clustering.py - Clustering methods for spatial transcriptomics data

Provides various clustering algorithms optimized for spatioloji objects,
including spatial-aware clustering methods.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def leiden_clustering(
    spatioloji_obj,
    layer: str | None = "normalized",
    use_highly_variable: bool = True,  # NEW PARAMETER
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    use_pca: bool = True,
    random_state: int = 42,
    output_column: str = "leiden",
    inplace: bool = True,
):
    """
    Leiden clustering for single-cell data.

    Graph-based clustering using the Leiden algorithm (improved Louvain).
    Requires `leidenalg` package.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which expression layer to use, by default 'normalized'
    use_highly_variable : bool, optional
        Use only highly variable genes if available, by default True
        Requires running highly_variable_genes() first
    resolution : float, optional
        Resolution parameter (higher = more clusters), by default 1.0
    ...

    Examples
    --------
    >>> # Standard workflow with HVG selection
    >>> sp.processing.highly_variable_genes(sp, layer='normalized_counts')
    >>> sp.processing.leiden_clustering(sp, use_highly_variable=True)
    """
    print(f"\nLeiden clustering (resolution={resolution}, n_neighbors={n_neighbors})")

    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    # SUBSET TO HVGs if requested
    if use_highly_variable:
        if "highly_variable" not in spatioloji_obj.gene_meta.columns:
            warnings.warn(
                "No 'highly_variable' column in gene_meta. "
                "Run highly_variable_genes() first or set use_highly_variable=False. "
                "Using all genes.",
                stacklevel=2,
            )
        else:
            hvg_mask = spatioloji_obj.gene_meta["highly_variable"].values
            n_hvg = hvg_mask.sum()

            if n_hvg == 0:
                warnings.warn("No highly variable genes found. Using all genes.", stacklevel=2)
            else:
                X = X[:, hvg_mask]
                print(f"  Using {n_hvg} highly variable genes (out of {hvg_mask.shape[0]} total)")

    try:
        import igraph as ig
        import leidenalg
    except ImportError as err:
        raise ImportError(
            "Leiden clustering requires igraph and leidenalg. " "Install with: pip install igraph leidenalg"
        ) from err

    print(f"\nLeiden clustering (resolution={resolution}, n_neighbors={n_neighbors})")

    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    # Apply PCA if requested
    if use_pca:
        print(f"  Computing PCA (n_pcs={n_pcs})...")
        pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]), random_state=random_state)
        X_reduced = pca.fit_transform(X)
        print(f"    Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    else:
        X_reduced = X

    # Build k-NN graph
    print("  Building k-NN graph...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X_reduced)
    knn_distances, knn_indices = nbrs.kneighbors(X_reduced)

    # Create igraph
    n_cells = X.shape[0]
    edges = []
    weights = []

    for i in range(n_cells):
        for j, dist in zip(knn_indices[i][1:], knn_distances[i][1:], strict=True):  # Skip self
            edges.append((i, j))
            weights.append(1.0 / (1.0 + dist))  # Convert distance to weight

    g = ig.Graph(n=n_cells, edges=edges, directed=False)
    g.es["weight"] = weights

    # Run Leiden
    print("  Running Leiden algorithm...")
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state,
    )

    clusters = np.array(partition.membership)
    n_clusters = len(np.unique(clusters))

    print(f"  ✓ Found {n_clusters} clusters")

    # Calculate cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"    Cluster sizes - min: {counts.min()}, max: {counts.max()}, " f"mean: {counts.mean():.1f}")

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

    Returns
    -------
    np.ndarray or None
        Cluster labels if inplace=False, otherwise None

    Examples
    --------
    >>> # K-means with 10 clusters
    >>> sp.processing.kmeans_clustering(sp, n_clusters=10)
    >>>
    >>> # K-means without PCA (use full expression)
    >>> sp.processing.kmeans_clustering(sp, n_clusters=5, use_pca=False)
    """
    print(f"\nK-means clustering (n_clusters={n_clusters})")

    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    # Apply PCA if requested
    if use_pca:
        print(f"  Computing PCA (n_pcs={n_pcs})...")
        pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]), random_state=random_state)
        X_reduced = pca.fit_transform(X)
        print(f"    Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    else:
        X_reduced = X

    # Run k-means
    print("  Running k-means...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=random_state)
    clusters = kmeans.fit_predict(X_reduced)

    # Calculate cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"  ✓ Clustered into {n_clusters} groups")
    print(f"    Cluster sizes - min: {counts.min()}, max: {counts.max()}, " f"mean: {counts.mean():.1f}")
    print(f"    Inertia: {kmeans.inertia_:.2f}")

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

    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    # Apply PCA if requested
    if use_pca:
        print(f"  Computing PCA (n_pcs={n_pcs})...")
        pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]))
        X_reduced = pca.fit_transform(X)
        print(f"    Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    else:
        X_reduced = X

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
    print(f"    Cluster sizes - min: {counts.min()}, max: {counts.max()}, " f"mean: {counts.mean():.1f}")

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
        print(f"    Noise points: {n_noise} ({n_noise/len(clusters)*100:.1f}%)")

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
        print(f"    Cluster sizes - min: {counts.min()}, max: {counts.max()}, " f"mean: {counts.mean():.1f}")

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

    # Get expression data
    if layer is None:
        X_expr = spatioloji_obj.expression.get_dense()
    else:
        X_expr = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X_expr):
            X_expr = X_expr.toarray()

    # Apply PCA if requested
    if use_pca:
        print(f"  Computing PCA (n_pcs={n_pcs})...")
        pca = PCA(n_components=min(n_pcs, X_expr.shape[0], X_expr.shape[1]), random_state=random_state)
        X_expr_reduced = pca.fit_transform(X_expr)
    else:
        X_expr_reduced = X_expr

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
    print(f"    Expression: {X_expr_norm.shape[1]} dims × {1-spatial_weight:.2f} weight")
    print(f"    Spatial: {X_spatial_norm.shape[1]} dims × {spatial_weight:.2f} weight")

    # Run k-means on combined space
    print("  Running k-means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_combined)

    # Calculate cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"  ✓ Found {n_clusters} clusters")
    print(f"    Cluster sizes - min: {counts.min()}, max: {counts.max()}, " f"mean: {counts.mean():.1f}")

    if inplace:
        spatioloji_obj._cell_meta[output_column] = pd.Categorical(clusters)
        return None
    else:
        return clusters


def find_optimal_clusters(
    spatioloji_obj,
    layer: str | None = "normalized",
    method: Literal["elbow", "silhouette", "gap"] = "silhouette",
    k_range: tuple[int, int] = (2, 20),
    n_pcs: int = 50,
    use_pca: bool = True,
    random_state: int = 42,
) -> dict[str, int | list[float]]:
    """
    Find optimal number of clusters.

    Uses elbow method, silhouette score, or gap statistic.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which expression layer to use, by default 'normalized'
    method : {'elbow', 'silhouette', 'gap'}, optional
        Method for finding optimal k, by default 'silhouette'
    k_range : tuple, optional
        Range of k values to test (min, max), by default (2, 20)
    n_pcs : int, optional
        Number of PCs to use if use_pca=True, by default 50
    use_pca : bool, optional
        Whether to use PCA, by default True
    random_state : int, optional
        Random seed, by default 42

    Returns
    -------
    dict
        Dictionary with 'optimal_k' and 'scores' for each k

    Examples
    --------
    >>> # Find optimal k using silhouette score
    >>> result = sp.processing.find_optimal_clusters(sp, method='silhouette')
    >>> print(f"Optimal k: {result['optimal_k']}")
    >>>
    >>> # Then cluster with optimal k
    >>> sp.processing.kmeans_clustering(sp, n_clusters=result['optimal_k'])
    """
    print(f"\nFinding optimal number of clusters (method={method})")
    print(f"  Testing k from {k_range[0]} to {k_range[1]}")

    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    # Apply PCA if requested
    if use_pca:
        pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]), random_state=random_state)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    k_values = range(k_range[0], k_range[1] + 1)
    scores = []

    if method == "elbow":
        # Elbow method (inertia)
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(X_reduced)
            scores.append(kmeans.inertia_)
            print(f"    k={k}: inertia={kmeans.inertia_:.2f}")

        # Find elbow (maximum curvature)
        # Simple heuristic: maximum distance from line connecting first and last points
        line = np.linspace(scores[0], scores[-1], len(scores))
        distances = np.abs(np.array(scores) - line)
        optimal_k = k_values[np.argmax(distances)]

    elif method == "silhouette":
        # Silhouette score
        from sklearn.metrics import silhouette_score

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X_reduced)
            score = silhouette_score(X_reduced, labels)
            scores.append(score)
            print(f"    k={k}: silhouette={score:.3f}")

        optimal_k = k_values[np.argmax(scores)]

    elif method == "gap":
        # Gap statistic
        print("  Computing gap statistic (this may take a while)...")

        def compute_inertia(X, k):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(X)
            return kmeans.inertia_

        n_refs = 10  # Number of reference datasets
        gaps = []

        for k in k_values:
            # Observed inertia
            ref_inertias = []
            obs_inertia = compute_inertia(X_reduced, k)

            # Reference inertias
            for _ in range(n_refs):
                # Generate random reference data
                X_ref = np.random.uniform(X_reduced.min(axis=0), X_reduced.max(axis=0), size=X_reduced.shape)
                ref_inertias.append(compute_inertia(X_ref, k))

            gap = np.log(np.mean(ref_inertias)) - np.log(obs_inertia)
            gaps.append(gap)
            print(f"    k={k}: gap={gap:.3f}")

        scores = gaps
        optimal_k = k_values[np.argmax(gaps)]

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"\n  ✓ Optimal k: {optimal_k}")

    return {"optimal_k": optimal_k, "k_values": list(k_values), "scores": scores, "method": method}
