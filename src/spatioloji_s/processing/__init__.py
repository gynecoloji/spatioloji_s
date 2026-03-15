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
    cca_integrate,
    combat,
    evaluate_batch_correction,
    harmony,
    regress_out,
    rpca_integrate,
    scale_by_batch,
    scvi_integrate,
)
from .clustering import (
    assess_clustering_quality,
    find_optimal_clusters,
    hierarchical_clustering,
    kmeans_clustering,
    leiden_clustering,
    leiden_resolution_sweep,
    spatial_clustering,
    spatially_constrained_clustering,
)
from .DEG import (
    deg_deseq2,
    deg_mast,
    deg_nb_glm,
    deg_ttest,
    deg_wilcoxon,
    run_deg,
)
from .dimension_reduction import (
    diffusion_map,
    pca,
    plot_pca_variance,
    tsne,
    umap,
)
from .feature_selection import (
    compare_hvg_methods,
    highly_variable_genes,
    select_genes_by_pattern,
)
from .imputation import (
    alra_impute,
    compare_imputation_methods,
    knn_smooth,
    magic_impute,
    scvi_impute,
)
from .normalization import (
    log_transform,
    normalize_pearson_residuals,
    normalize_standard_workflow,
    normalize_total,
    scale,
    scale_by_batch_normalization,
)

__all__ = [
    # Normalization
    "normalize_total",
    "log_transform",
    "scale",
    "scale_by_batch_normalization",
    "normalize_pearson_residuals",
    "normalize_standard_workflow",
    # Feature selection
    "highly_variable_genes",
    "compare_hvg_methods",
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
    "assess_clustering_quality",
    "leiden_resolution_sweep",
    # Batch correction
    "combat",
    "harmony",
    "regress_out",
    "scale_by_batch",
    "scvi_integrate",
    "cca_integrate",
    "rpca_integrate",
    "evaluate_batch_correction",
    # Imputation
    "magic_impute",
    "knn_smooth",
    "alra_impute",
    "scvi_impute",
    "compare_imputation_methods",
    # DEG
    "run_deg",
    "deg_wilcoxon",
    "deg_ttest",
    "deg_mast",
    "deg_nb_glm",
    "deg_deseq2",
]
