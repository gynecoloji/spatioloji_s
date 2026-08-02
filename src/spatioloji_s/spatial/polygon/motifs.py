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

from spatioloji_s.spatial._motif_types import (
    AssemblyCatalog,
    MotifCatalog,
    MotifResult,
    StructureMatches,
    _get_cell_ids,
    _get_sparse_adjacency,
)

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


def detect_assemblies(
    sp: spatioloji,
    graph: PolygonSpatialGraph | PointSpatialGraph,
    motif_catalog: MotifCatalog,
    method: str = "leiden",
    resolution: float = 0.5,
    n_assemblies: int | None = None,
    min_assembly_cells: int = 10,
    random_state: int = 42,
    store: bool = True,
) -> AssemblyCatalog:
    """Detect mesoscale tissue assemblies by clustering spatially contiguous motif instances.

    Groups cells sharing the same motif label into connected components (motif
    instances), builds a region-level graph over those instances, and clusters
    it to identify assemblies — recurring mesoscale tissue structures.

    Args:
        sp: spatioloji object with spatial coordinates.
        graph: Pre-built spatial graph (PolygonSpatialGraph or PointSpatialGraph).
        motif_catalog: Result of :func:`discover_motifs`.
        method: Clustering algorithm — ``"kmeans"`` or ``"leiden"`` (default).
        resolution: Leiden resolution parameter (ignored for kmeans).
        n_assemblies: Number of assemblies for kmeans.  If ``None`` and method
            is ``"kmeans"``, auto-selects via Calinski-Harabasz.
        min_assembly_cells: Assemblies with fewer total cells are labelled -1.
        random_state: Random seed for reproducibility.
        store: If True, write ``assembly_label`` into ``sp.cell_meta``.

    Returns:
        AssemblyCatalog with per-cell labels, instance table, composition,
        and adjacency pattern DataFrames.

    Raises:
        ValueError: If *method* is not ``"kmeans"`` or ``"leiden"``.

    Example:
        >>> from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        >>> from spatioloji_s.spatial.polygon.motifs import discover_motifs, detect_assemblies
        >>> graph = build_buffer_graph(sp, buffer_distance=30)
        >>> catalog = discover_motifs(sp, graph, group_col="cell_type", n_motifs=5)
        >>> assemblies = detect_assemblies(sp, graph, catalog)
        >>> assemblies.instances.head()
    """
    from scipy.sparse.csgraph import connected_components as _connected_components

    # ---- validation --------------------------------------------------------
    if method not in ("kmeans", "leiden"):
        raise ValueError(f"Unknown clustering method '{method}'. Choose 'kmeans' or 'leiden'.")

    cell_ids = _get_cell_ids(graph)
    adj = _get_sparse_adjacency(graph)
    motif_labels = motif_catalog.labels.reindex(cell_ids)
    n_cells = len(cell_ids)
    unique_motifs = sorted(motif_labels.unique())

    # ---- connected components per motif ------------------------------------
    instance_records: list[dict] = []
    cell_instance_map = np.full(n_cells, -1, dtype=np.int64)
    instance_id_counter = 0

    # Align coordinates to graph ordering
    idx_in_sp = sp.cell_index.get_indexer(cell_ids)
    x_coords = np.asarray(sp.spatial.x_global)[idx_in_sp]
    y_coords = np.asarray(sp.spatial.y_global)[idx_in_sp]

    for motif_id in unique_motifs:
        mask = motif_labels.values == motif_id
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # Extract subgraph for this motif
        sub_adj = adj[np.ix_(indices, indices)]
        n_comp, comp_labels = _connected_components(sub_adj, directed=False)

        for comp in range(n_comp):
            comp_mask = comp_labels == comp
            comp_cell_indices = indices[comp_mask]
            n_comp_cells = len(comp_cell_indices)

            cx = float(x_coords[comp_cell_indices].mean())
            cy = float(y_coords[comp_cell_indices].mean())

            instance_records.append(
                {
                    "instance_id": instance_id_counter,
                    "motif_id": motif_id,
                    "n_cells": n_comp_cells,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "_cell_indices": comp_cell_indices,
                }
            )

            cell_instance_map[comp_cell_indices] = instance_id_counter
            instance_id_counter += 1

    n_instances = len(instance_records)

    # ---- build region graph ------------------------------------------------
    # Edges: two instances are connected if any of their cells are neighbours
    # in the original graph
    instance_adj = np.zeros((n_instances, n_instances), dtype=np.float32)
    for i, rec_i in enumerate(instance_records):
        neighbours_of_i = set()
        for ci in rec_i["_cell_indices"]:
            row = adj.getrow(ci)
            neighbours_of_i.update(row.indices.tolist())
        for j in range(i + 1, n_instances):
            rec_j_set = set(instance_records[j]["_cell_indices"].tolist())
            if neighbours_of_i & rec_j_set:
                instance_adj[i, j] = 1.0
                instance_adj[j, i] = 1.0

    # ---- featurise instances -----------------------------------------------
    n_motif_types = len(unique_motifs)
    motif_to_idx = {m: i for i, m in enumerate(unique_motifs)}

    features_list = []
    for rec in instance_records:
        # One-hot motif identity
        one_hot = np.zeros(n_motif_types, dtype=np.float32)
        one_hot[motif_to_idx[rec["motif_id"]]] = 1.0

        # Log(size) normalised
        log_size = np.array([np.log1p(rec["n_cells"])], dtype=np.float32)

        # Neighbour motif composition
        neigh_comp = np.zeros(n_motif_types, dtype=np.float32)
        inst_idx = rec["instance_id"]
        for j in range(n_instances):
            if instance_adj[inst_idx, j] > 0:
                neigh_comp[motif_to_idx[instance_records[j]["motif_id"]]] += 1.0
        total = neigh_comp.sum()
        if total > 0:
            neigh_comp /= total

        features_list.append(np.concatenate([one_hot, log_size, neigh_comp]))

    feature_matrix = np.vstack(features_list) if features_list else np.empty((0, 0))

    # Normalise log_size column across instances
    if n_instances > 0:
        log_col = n_motif_types  # index of log_size column
        max_log = feature_matrix[:, log_col].max()
        if max_log > 0:
            feature_matrix[:, log_col] /= max_log

    # ---- cluster region graph ----------------------------------------------
    if n_instances < 2:
        instance_cluster_labels = np.zeros(n_instances, dtype=np.int64)
    elif method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans

        if n_assemblies is None:
            n_assemblies_resolved = _auto_select_n_motifs(
                sp_sparse.csr_matrix(feature_matrix), random_state=random_state
            )
        else:
            n_assemblies_resolved = min(n_assemblies, n_instances)
        km = MiniBatchKMeans(n_clusters=n_assemblies_resolved, random_state=random_state, n_init=5)
        instance_cluster_labels = km.fit_predict(feature_matrix)
    else:
        # Leiden on the instance feature matrix
        instance_cluster_labels = _cluster_leiden(
            sp_sparse.csr_matrix(feature_matrix),
            resolution=resolution,
            random_state=random_state,
        )

    # Assign assembly labels to instances
    for i, rec in enumerate(instance_records):
        rec["assembly_id"] = int(instance_cluster_labels[i])

    # ---- filter small assemblies -------------------------------------------
    # Count total cells per assembly
    assembly_cell_counts: dict[int, int] = {}
    for rec in instance_records:
        aid = rec["assembly_id"]
        assembly_cell_counts[aid] = assembly_cell_counts.get(aid, 0) + rec["n_cells"]

    small_assemblies = {aid for aid, cnt in assembly_cell_counts.items() if cnt < min_assembly_cells}
    for rec in instance_records:
        if rec["assembly_id"] in small_assemblies:
            rec["assembly_id"] = -1

    # ---- remap to contiguous IDs -------------------------------------------
    valid_aids = sorted({rec["assembly_id"] for rec in instance_records if rec["assembly_id"] >= 0})
    aid_remap = {old: new for new, old in enumerate(valid_aids)}
    aid_remap[-1] = -1

    for rec in instance_records:
        rec["assembly_id"] = aid_remap[rec["assembly_id"]]

    # ---- build per-cell assembly labels ------------------------------------
    cell_assembly = np.full(n_cells, -1, dtype=np.int64)
    for rec in instance_records:
        cell_assembly[rec["_cell_indices"]] = rec["assembly_id"]

    labels_series = pd.Series(cell_assembly, index=cell_ids, name="assembly_label")

    # Extend to full sp.cell_index (cells not in graph get -1)
    full_labels = labels_series.reindex(sp.cell_index, fill_value=-1)

    # ---- instances DataFrame -----------------------------------------------
    instances_df = pd.DataFrame(
        [
            {
                "instance_id": rec["instance_id"],
                "assembly_id": rec["assembly_id"],
                "motif_id": rec["motif_id"],
                "n_cells": rec["n_cells"],
                "centroid_x": rec["centroid_x"],
                "centroid_y": rec["centroid_y"],
            }
            for rec in instance_records
        ]
    )

    # ---- composition DataFrame ---------------------------------------------
    if instances_df.empty or (instances_df["assembly_id"] == -1).all():
        composition = pd.DataFrame()
    else:
        valid_inst = instances_df[instances_df["assembly_id"] >= 0]
        comp_rows = []
        for aid, grp in valid_inst.groupby("assembly_id"):
            total_cells = grp["n_cells"].sum()
            motif_props = {}
            for motif_id in unique_motifs:
                motif_cells = grp.loc[grp["motif_id"] == motif_id, "n_cells"].sum()
                motif_props[motif_id] = motif_cells / total_cells if total_cells > 0 else 0.0
            motif_props["assembly_id"] = aid
            comp_rows.append(motif_props)
        composition = pd.DataFrame(comp_rows).set_index("assembly_id")

    # ---- adjacency_pattern DataFrame ---------------------------------------
    adj_rows = []
    if not instances_df.empty:
        valid_inst = instances_df[instances_df["assembly_id"] >= 0]
        for aid, grp in valid_inst.groupby("assembly_id"):
            inst_ids = grp["instance_id"].values
            for i_idx in range(len(inst_ids)):
                for j_idx in range(i_idx + 1, len(inst_ids)):
                    ii = inst_ids[i_idx]
                    jj = inst_ids[j_idx]
                    if instance_adj[ii, jj] > 0:
                        ma = instance_records[ii]["motif_id"]
                        mb = instance_records[jj]["motif_id"]
                        adj_rows.append(
                            {
                                "assembly_id": aid,
                                "motif_a": min(ma, mb),
                                "motif_b": max(ma, mb),
                                "frequency": 1,
                            }
                        )

    if adj_rows:
        adjacency_pattern = pd.DataFrame(adj_rows)
        adjacency_pattern = adjacency_pattern.groupby(["assembly_id", "motif_a", "motif_b"], as_index=False)[
            "frequency"
        ].sum()
    else:
        adjacency_pattern = pd.DataFrame(columns=["assembly_id", "motif_a", "motif_b", "frequency"])

    # ---- store -------------------------------------------------------------
    if store:
        sp.cell_meta["assembly_label"] = full_labels.reindex(sp.cell_meta.index)

    params = {
        "method": method,
        "resolution": resolution,
        "n_assemblies": n_assemblies,
        "min_assembly_cells": min_assembly_cells,
        "random_state": random_state,
    }

    return AssemblyCatalog(
        labels=full_labels,
        composition=composition,
        instances=instances_df,
        adjacency_pattern=adjacency_pattern,
        params=params,
    )


# ---------------------------------------------------------------------------
# Built-in structure signatures
# ---------------------------------------------------------------------------

_BUILTIN_SIGNATURES: dict[str, dict[str, dict[str, float]]] = {
    "TME": {
        "TLS": {"B_cell": 0.25, "T_cell": 0.15, "DC": 0.05},
        "immune_aggregate": {"CD8_T": 0.2, "CD4_T": 0.15, "Macrophage": 0.15},
        "tumor_bud": {"Tumor": 0.8},
        "perivascular_niche": {"Endothelial": 0.2, "Pericyte": 0.15},
        "immune_desert": {"Tumor": 0.9, "T_cell": 0.0},
    },
}


# ---------------------------------------------------------------------------
# match_known_structures
# ---------------------------------------------------------------------------


def match_known_structures(
    sp: spatioloji,
    motif_catalog: MotifCatalog,
    assembly_catalog: AssemblyCatalog | None = None,
    signatures: dict[str, dict[str, float]] | None = None,
    builtin: str | None = None,
    threshold: float = 0.5,
    absence_threshold: float = 0.05,
    coord_type: Literal["global", "local"] = "global",
) -> StructureMatches:
    """Match discovered motifs (and optionally assemblies) to known tissue structure signatures.

    Computes cosine similarity between each discovered motif/assembly composition
    and a set of reference structure signatures.  Signatures with a ``0.0`` value
    for a cell type enforce an *absence* constraint: the motif is rejected if that
    type exceeds *absence_threshold*.

    Args:
        sp: spatioloji object with spatial coordinates.
        motif_catalog: Result of :func:`discover_motifs`.
        assembly_catalog: Optional result of :func:`detect_assemblies`.
        signatures: User-provided dict ``{structure_name: {cell_type: proportion}}``.
        builtin: Name of a built-in signature set (e.g. ``"TME"``).
        threshold: Minimum cosine similarity for a match.
        absence_threshold: Maximum allowed proportion for cell types with ``0.0``
            in the reference signature.
        coord_type: Which coordinates to use for centroids (``"global"`` or ``"local"``).

    Returns:
        StructureMatches with per-match rows and per-cell labels.

    Raises:
        ValueError: If neither *signatures* nor *builtin* is provided, or if
            *builtin* is not a recognised name.

    Example:
        >>> from spatioloji_s.spatial.polygon.motifs import discover_motifs, match_known_structures
        >>> catalog = discover_motifs(sp, graph, group_col="cell_type", n_motifs=5)
        >>> sigs = {"tumor_core": {"Tumor": 0.6}}
        >>> result = match_known_structures(sp, catalog, signatures=sigs)
        >>> result.matches
    """
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity

    # ---- merge signatures --------------------------------------------------
    merged: dict[str, dict[str, float]] = {}
    if builtin is not None:
        if builtin not in _BUILTIN_SIGNATURES:
            raise ValueError(
                f"Unknown builtin signature set '{builtin}'. Available: {list(_BUILTIN_SIGNATURES.keys())}"
            )
        merged.update(_BUILTIN_SIGNATURES[builtin])
    if signatures is not None:
        merged.update(signatures)
    if not merged:
        raise ValueError("Provide at least one of 'signatures' or 'builtin'.")

    # ---- align reference vectors to motif signature columns ----------------
    columns = motif_catalog.signatures.columns.tolist()

    def _ref_vector(sig: dict[str, float]) -> np.ndarray:
        return np.array([sig.get(c, 0.0) for c in columns], dtype=np.float64)

    def _absence_mask(sig: dict[str, float]) -> list[int]:
        """Return column indices where the signature demands absence (value == 0.0)."""
        return [i for i, c in enumerate(columns) if c in sig and sig[c] == 0.0]

    # ---- match against motifs ----------------------------------------------
    match_rows: list[dict] = []

    motif_sigs = motif_catalog.signatures.values  # (n_motifs, n_types)

    for struct_name, sig in merged.items():
        ref = _ref_vector(sig).reshape(1, -1)
        ref_norm = np.linalg.norm(ref)
        if ref_norm == 0:
            continue
        absent_cols = _absence_mask(sig)

        sims = _cosine_similarity(ref, motif_sigs).flatten()

        for motif_id, sim_val in zip(motif_catalog.signatures.index, sims, strict=True):
            # Absence filter
            reject = False
            for ci in absent_cols:
                if motif_sigs[motif_catalog.signatures.index.get_loc(motif_id), ci] > absence_threshold:
                    reject = True
                    break
            if reject:
                continue
            if sim_val < threshold:
                continue

            # Centroid computation
            cell_idx = motif_catalog.labels[motif_catalog.labels == motif_id].index
            pos_idx = sp.cell_index.get_indexer(cell_idx)
            valid = pos_idx >= 0
            pos_idx = pos_idx[valid]
            if coord_type == "global":
                cx = float(np.asarray(sp.spatial.x_global)[pos_idx].mean()) if len(pos_idx) > 0 else 0.0
                cy = float(np.asarray(sp.spatial.y_global)[pos_idx].mean()) if len(pos_idx) > 0 else 0.0
            else:
                cx = float(np.asarray(sp.spatial.x_local)[pos_idx].mean()) if len(pos_idx) > 0 else 0.0
                cy = float(np.asarray(sp.spatial.y_local)[pos_idx].mean()) if len(pos_idx) > 0 else 0.0

            match_rows.append(
                {
                    "structure_name": struct_name,
                    "target_type": "motif",
                    "target_id": motif_id,
                    "similarity": float(sim_val),
                    "n_cells": int((motif_catalog.labels == motif_id).sum()),
                    "centroid_x": cx,
                    "centroid_y": cy,
                }
            )

    # ---- match against assemblies (optional) -------------------------------
    if assembly_catalog is not None and not assembly_catalog.composition.empty:
        # Weighted-average assembly composition aligned to columns
        assembly_comp = assembly_catalog.composition.reindex(columns=columns, fill_value=0.0).values

        for struct_name, sig in merged.items():
            ref = _ref_vector(sig).reshape(1, -1)
            ref_norm = np.linalg.norm(ref)
            if ref_norm == 0:
                continue
            absent_cols = _absence_mask(sig)

            sims = _cosine_similarity(ref, assembly_comp).flatten()

            for aid, sim_val in zip(assembly_catalog.composition.index, sims, strict=True):
                reject = False
                for ci in absent_cols:
                    if assembly_comp[assembly_catalog.composition.index.get_loc(aid), ci] > absence_threshold:
                        reject = True
                        break
                if reject:
                    continue
                if sim_val < threshold:
                    continue

                cell_idx = assembly_catalog.labels[assembly_catalog.labels == aid].index
                pos_idx = sp.cell_index.get_indexer(cell_idx)
                valid = pos_idx >= 0
                pos_idx = pos_idx[valid]
                if coord_type == "global":
                    cx = float(np.asarray(sp.spatial.x_global)[pos_idx].mean()) if len(pos_idx) > 0 else 0.0
                    cy = float(np.asarray(sp.spatial.y_global)[pos_idx].mean()) if len(pos_idx) > 0 else 0.0
                else:
                    cx = float(np.asarray(sp.spatial.x_local)[pos_idx].mean()) if len(pos_idx) > 0 else 0.0
                    cy = float(np.asarray(sp.spatial.y_local)[pos_idx].mean()) if len(pos_idx) > 0 else 0.0

                match_rows.append(
                    {
                        "structure_name": struct_name,
                        "target_type": "assembly",
                        "target_id": aid,
                        "similarity": float(sim_val),
                        "n_cells": int((assembly_catalog.labels == aid).sum()),
                        "centroid_x": cx,
                        "centroid_y": cy,
                    }
                )

    matches_df = pd.DataFrame(
        match_rows,
        columns=["structure_name", "target_type", "target_id", "similarity", "n_cells", "centroid_x", "centroid_y"],
    )

    # ---- per-cell labels ---------------------------------------------------
    per_cell = pd.Series("unmatched", index=sp.cell_index, name="structure_label")

    if not matches_df.empty:
        # For each cell, pick the highest-similarity match
        # First build cell → (structure_name, similarity) from motif matches
        motif_matches = matches_df[matches_df["target_type"] == "motif"]
        for _, row in motif_matches.iterrows():
            cell_mask = motif_catalog.labels == row["target_id"]
            cells = motif_catalog.labels[cell_mask].index
            for c in cells:
                if c in per_cell.index:
                    if per_cell[c] == "unmatched" or row["similarity"] > 0:
                        per_cell[c] = row["structure_name"]

        # Assembly matches override if higher similarity
        if assembly_catalog is not None:
            assembly_matches = matches_df[matches_df["target_type"] == "assembly"]
            for _, row in assembly_matches.iterrows():
                cell_mask = assembly_catalog.labels == row["target_id"]
                cells = assembly_catalog.labels[cell_mask].index
                for c in cells:
                    if c in per_cell.index:
                        per_cell[c] = row["structure_name"]

    return StructureMatches(
        matches=matches_df,
        per_cell=per_cell,
        signatures_used=merged,
    )


# ---------------------------------------------------------------------------
# run_motif_pipeline
# ---------------------------------------------------------------------------


def run_motif_pipeline(
    sp: spatioloji,
    graph: PolygonSpatialGraph | PointSpatialGraph,
    group_col: str = "cell_type",
    method: Literal["kmeans", "leiden"] = "kmeans",
    n_motifs: int | None = None,
    resolution: float = 1.0,
    k_hops: int = 1,
    include_morphology: bool = False,
    include_density: bool = False,
    detect_assemblies_flag: bool = True,
    assembly_method: str = "leiden",
    assembly_resolution: float = 0.5,
    n_assemblies: int | None = None,
    min_assembly_cells: int = 10,
    match_signatures: dict[str, dict[str, float]] | None = None,
    match_builtin: str | None = None,
    match_threshold: float = 0.5,
    coord_type: Literal["global", "local"] = "global",
    random_state: int = 42,
    n_jobs: int = 1,
    store: bool = True,
) -> MotifResult:
    """Run the full spatial motif discovery pipeline in one call.

    Convenience wrapper that chains :func:`discover_motifs`,
    :func:`detect_assemblies`, and :func:`match_known_structures`.

    Args:
        sp: spatioloji object.
        graph: Pre-built spatial graph (polygon or point).
        group_col: Cell-type column in ``sp.cell_meta``.
        method: Clustering method for motif discovery (``"kmeans"`` or ``"leiden"``).
        n_motifs: Number of motifs (``None`` for auto-selection).
        resolution: Leiden resolution for motif discovery.
        k_hops: Neighbourhood radius in graph hops.
        include_morphology: Append morphology features (polygon graph only).
        include_density: Append density features (polygon graph only).
        detect_assemblies_flag: Whether to run assembly detection.
        assembly_method: Clustering method for assemblies.
        assembly_resolution: Leiden resolution for assemblies.
        n_assemblies: Number of assemblies for kmeans.
        min_assembly_cells: Minimum cells per assembly.
        match_signatures: User-provided structure signatures for matching.
        match_builtin: Built-in signature set name for matching.
        match_threshold: Minimum cosine similarity for structure matching.
        coord_type: Coordinate type for centroids (``"global"`` or ``"local"``).
        random_state: Random seed.
        n_jobs: Number of parallel jobs (reserved for future use).
        store: Write labels into ``sp.cell_meta``.

    Returns:
        MotifResult containing motif_catalog, assembly_catalog, and structure_matches.

    Example:
        >>> from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        >>> from spatioloji_s.spatial.polygon.motifs import run_motif_pipeline
        >>> graph = build_buffer_graph(sp, buffer_distance=30)
        >>> result = run_motif_pipeline(sp, graph, group_col="cell_type", n_motifs=5)
    """
    # ---- Step 1: discover motifs -------------------------------------------
    motif_catalog = discover_motifs(
        sp,
        graph,
        group_col=group_col,
        n_motifs=n_motifs,
        method=method,
        k_hops=k_hops,
        use_morphology=include_morphology,
        use_density=include_density,
        store=store,
        random_state=random_state,
        leiden_resolution=resolution,
    )

    # Store motif labels with a clearer column name
    if store:
        sp.cell_meta["motif_label"] = motif_catalog.labels.reindex(sp.cell_meta.index)

    # ---- Step 2: detect assemblies -----------------------------------------
    assembly_catalog: AssemblyCatalog | None = None
    if detect_assemblies_flag:
        assembly_catalog = detect_assemblies(
            sp,
            graph,
            motif_catalog,
            method=assembly_method,
            resolution=assembly_resolution,
            n_assemblies=n_assemblies,
            min_assembly_cells=min_assembly_cells,
            random_state=random_state,
            store=store,
        )

    # ---- Step 3: match known structures ------------------------------------
    structure_matches: StructureMatches | None = None
    if match_signatures is not None or match_builtin is not None:
        structure_matches = match_known_structures(
            sp,
            motif_catalog,
            assembly_catalog=assembly_catalog,
            signatures=match_signatures,
            builtin=match_builtin,
            threshold=match_threshold,
            coord_type=coord_type,
        )

    params = {
        "group_col": group_col,
        "method": method,
        "n_motifs": n_motifs,
        "resolution": resolution,
        "k_hops": k_hops,
        "detect_assemblies": detect_assemblies_flag,
        "assembly_method": assembly_method,
        "random_state": random_state,
    }

    return MotifResult(
        motif_catalog=motif_catalog,
        assembly_catalog=assembly_catalog,
        structure_matches=structure_matches,
        params=params,
    )
