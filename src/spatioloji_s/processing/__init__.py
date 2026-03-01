"""
processing/__init__.py - Processing subpackage for spatioloji

Provides normalization, feature selection, dimensionality reduction,
clustering, batch correction, and imputation methods.

Usage
-----
    import spatioloji as sj
    sj.processing.normalize_total(sp)
    sj.processing.highly_variable_genes(sp)
    sj.processing.pca(sp)
    sj.processing.leiden_clustering(sp)
"""

from .batch_correction import (
    combat,
    evaluate_batch_correction,
    harmony,
    mnn_correct,
    regress_out,
    scale_by_batch,
)
from .clustering import (
    find_optimal_clusters,
    hierarchical_clustering,
    kmeans_clustering,
    leiden_clustering,
    spatial_clustering,
    spatially_constrained_clustering,
)
from .dimension_reduction import (
    diffusion_map,
    pca,
    plot_pca_variance,
    tsne,
    umap,
)
from .feature_selection import (
    highly_variable_genes,
    select_genes_by_pattern,
)
from .imputation import (
    alra_impute,
    compare_imputation_methods,
    dca_impute,
    knn_smooth,
    magic_impute,
)
from .normalization import (
    log_transform,
    normalize_pearson_residuals,
    normalize_standard_workflow,
    normalize_total,
    scale,
)

__all__ = [
    # Normalization
    "normalize_total",
    "log_transform",
    "scale",
    "normalize_pearson_residuals",
    "normalize_standard_workflow",
    # Feature selection
    "highly_variable_genes",
    "select_genes_by_pattern",
    # Dimensionality reduction
    "pca",
    "tsne",
    "umap",
    "diffusion_map",
    "plot_pca_variance",
    # Clustering
    "leiden_clustering",
    "kmeans_clustering",
    "hierarchical_clustering",
    "spatial_clustering",
    "spatially_constrained_clustering",
    "find_optimal_clusters",
    # Batch correction
    "combat",
    "harmony",
    "regress_out",
    "scale_by_batch",
    "mnn_correct",
    "evaluate_batch_correction",
    # Imputation
    "magic_impute",
    "knn_smooth",
    "alra_impute",
    "dca_impute",
    "compare_imputation_methods",
]
