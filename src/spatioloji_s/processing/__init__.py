"""
spatioloji.processing - Processing methods for spatial transcriptomics

Comprehensive data processing tools for spatioloji objects.
"""

__version__ = '0.1.0'

# Import all functions
from .normalization import *
from .feature_selection import *
from .dimension_reduction import *
from .clustering import *
from .batch_correction import *
from .imputation import *

# Define what gets imported with "from spatioloji.processing import *"
__all__ = [
    # Normalization
    'normalize_total', 'log_transform', 'scale',
    'normalize_pearson_residuals', 'normalize_standard_workflow',
    
    # Feature selection
    'highly_variable_genes', 'select_genes_by_pattern',
    
    # Dimension reduction
    'pca', 'tsne', 'umap', 'diffusion_map', 'plot_pca_variance',
    
    # Clustering
    'leiden_clustering', 'kmeans_clustering', 'hierarchical_clustering',
    'spatial_clustering', 'spatially_constrained_clustering', 'find_optimal_clusters',
    
    # Batch correction
    'combat', 'harmony', 'mnn_correct', 'simple_mnn_correct',
    'regress_out', 'scale_by_batch', 'evaluate_batch_correction',
    
    # Imputation
    'magic_impute', 'knn_smooth', 'alra_impute', 'compare_imputation_methods',
]