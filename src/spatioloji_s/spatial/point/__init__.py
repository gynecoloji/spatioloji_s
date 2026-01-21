# src/spatioloji_s/spatial/point/__init__.py

"""
Point-based spatial analysis using cell centroids.

All methods use global coordinates exclusively. For FOV-specific analysis,
subset your spatioloji object first using sp.subset_by_fovs().

Modules
-------
- graph: Spatial graph construction (k-NN, radius, Delaunay)
- neighborhoods: Neighborhood queries and composition analysis
- statistics: Spatial statistics (Moran's I, Geary's C, Local Moran's I)
- patterns: Spatial pattern detection and domain identification
- ripley: Ripley's K/L functions for point pattern analysis

Quick Start
-----------
All functions now use global coordinates by default:

>>> import spatioloji as sj
>>> 
>>> # Build spatial graph (uses global coordinates automatically)
>>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
>>> 
>>> # Compute Moran's I
>>> expr = sp.get_expression(gene_names='CD4', as_dataframe=True)['CD4']
>>> result = sj.spatial.point.morans_i(sp, graph, expr)
>>> 
>>> # For single FOV analysis, subset first
>>> sp_fov = sp.subset_by_fovs(['fov1'])
>>> graph_fov = sj.spatial.point.build_knn_graph(sp_fov, k=10)
>>> result_fov = sj.spatial.point.morans_i(sp_fov, graph_fov, expr)

Graph Construction
------------------
build_knn_graph
    Build k-nearest neighbors graph
build_radius_graph
    Build radius-based graph
build_delaunay_graph
    Build Delaunay triangulation graph
compute_distance_matrix
    Compute pairwise distance matrix
compute_pairwise_distances
    Compute distances between two sets of cells
compute_distance_to_region
    Distance from each cell to nearest cell in target region
filter_graph_by_fov
    Filter graph to remove/keep inter-FOV edges
filter_graph_by_distance
    Filter graph by distance threshold

Neighborhood Analysis
---------------------
get_neighbors
    Get neighbors for a specific cell
get_cells_in_radius
    Get cells within radius of a center cell (no graph needed)
get_neighborhood_composition
    Get neighborhood composition by cell type
get_neighborhood_expression
    Get aggregated gene expression in neighborhoods
compute_neighborhood_diversity
    Compute diversity of cell types in neighborhoods
compute_spatial_lag
    Compute spatial lag (average of neighbors' values)

Spatial Statistics
------------------
morans_i
    Moran's I statistic for global spatial autocorrelation
gearys_c
    Geary's C statistic for local dissimilarity
local_morans_i
    Local Moran's I (LISA) for identifying clusters
test_spatial_autocorrelation_genes
    Test multiple genes for spatial autocorrelation

Pattern Detection
-----------------
identify_spatial_domains
    Identify spatially coherent domains/niches
find_spatially_variable_genes
    Identify genes with spatial expression patterns
detect_colocalization
    Test for spatial co-localization between cell groups
detect_spatial_gradients
    Detect spatial gradients in expression or features
identify_hotspots
    Identify hotspots and coldspots (Getis-Ord Gi*)

Point Pattern Analysis
----------------------
ripleys_k
    Ripley's K function for clustering analysis
ripleys_l
    Ripley's L function (variance-stabilized K)
cross_k_function
    Cross-type K function for spatial association
cross_l_function
    Cross-type L function
test_csr
    Test Complete Spatial Randomness using Monte Carlo

Examples
--------
Complete workflow for spatial transcriptomics analysis:

>>> import spatioloji as sj
>>> 
>>> # 1. Build spatial graph
>>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
>>> print(graph.summary())
>>> 
>>> # 2. Analyze neighborhood composition
>>> comp = sj.spatial.point.get_neighborhood_composition(
...     sp, graph, groupby='cell_type'
... )
>>> 
>>> # 3. Find spatially variable genes
>>> svg = sj.spatial.point.find_spatially_variable_genes(
...     sp, graph, n_genes=500, permutations=99
... )
>>> 
>>> # 4. Test spatial autocorrelation
>>> expr = sp.get_expression(gene_names='CD4', as_dataframe=True)['CD4']
>>> moran = sj.spatial.point.morans_i(sp, graph, expr)
>>> 
>>> # 5. Identify spatial domains
>>> domains = sj.spatial.point.identify_spatial_domains(
...     sp, graph, method='leiden', resolution=1.0
... )
>>> sp.cell_meta['spatial_domain'] = domains
>>> 
>>> # 6. Local cluster analysis
>>> local_i = sj.spatial.point.local_morans_i(sp, graph, expr)
>>> sp.cell_meta['cluster_type'] = local_i['cluster_type']
>>> 
>>> # 7. Co-localization analysis
>>> coloc = sj.spatial.point.detect_colocalization(
...     sp, graph, group1='T_cell', group2='Tumor'
... )
>>> 
>>> # 8. Point pattern analysis
>>> k_result = sj.spatial.point.ripleys_k(sp, n_radii=50)

FOV-Specific Analysis
---------------------
For analyzing individual FOVs, always subset first:

>>> # Analyze each FOV separately
>>> for fov_id in sp.fov_index:
...     sp_fov = sp.subset_by_fovs([fov_id])
...     graph_fov = sj.spatial.point.build_knn_graph(sp_fov, k=10)
...     
...     # Compute metrics for this FOV
...     comp = sj.spatial.point.get_neighborhood_composition(
...         sp_fov, graph_fov, groupby='cell_type'
...     )
...     
...     # Store FOV-specific results
...     results[fov_id] = comp

Migration Guide
---------------
If you were using coord_type='local' previously:

OLD CODE:
>>> graph = sj.spatial.point.build_knn_graph(sp, k=10, coord_type='local')
>>> result = sj.spatial.point.morans_i(sp, graph, values)

NEW CODE (global for all cells):
>>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
>>> result = sj.spatial.point.morans_i(sp, graph, values)

NEW CODE (for FOV-specific analysis):
>>> sp_fov = sp.subset_by_fovs(['fov1'])
>>> graph = sj.spatial.point.build_knn_graph(sp_fov, k=10)
>>> result = sj.spatial.point.morans_i(sp_fov, graph, values)

Notes
-----
- All functions use global coordinates exclusively
- Graphs built on subsetted data automatically use the correct coordinate system
- FOV-specific analysis is cleaner: subset once, then use all functions normally
- No risk of coordinate system mismatch between graph and analysis functions
"""

# Graph construction
from .graph import (
    SpatialGraph,
    build_knn_graph,
    build_radius_graph,
    build_delaunay_graph,
    compute_distance_matrix,
    compute_pairwise_distances,
    compute_distance_to_region,
    filter_graph_by_fov,
    filter_graph_by_distance
)

# Neighborhood analysis
from .neighborhoods import (
    get_neighbors,
    get_cells_in_radius,
    get_neighborhood_composition,
    get_neighborhood_expression,
    compute_neighborhood_diversity,
    compute_spatial_lag
)

# Spatial statistics
from .statistics import (
    morans_i,
    gearys_c,
    local_morans_i,
    test_spatial_autocorrelation_genes
)

# Pattern detection
from .patterns import (
    identify_spatial_domains,
    find_spatially_variable_genes,
    detect_colocalization,
    detect_spatial_gradients,
    identify_hotspots
)

# Point pattern analysis (Ripley's K/L)
from .ripley import (
    ripleys_k,
    ripleys_l,
    cross_k_function,
    cross_l_function,
    test_csr
)

__all__ = [
    # Classes
    'SpatialGraph',
    
    # Graph construction
    'build_knn_graph',
    'build_radius_graph',
    'build_delaunay_graph',
    'compute_distance_matrix',
    'compute_pairwise_distances',
    'compute_distance_to_region',
    'filter_graph_by_fov',
    'filter_graph_by_distance',
    
    # Neighborhood analysis
    'get_neighbors',
    'get_cells_in_radius',
    'get_neighborhood_composition',
    'get_neighborhood_expression',
    'compute_neighborhood_diversity',
    'compute_spatial_lag',
    
    # Spatial statistics
    'morans_i',
    'gearys_c',
    'local_morans_i',
    'test_spatial_autocorrelation_genes',
    
    # Pattern detection
    'identify_spatial_domains',
    'find_spatially_variable_genes',
    'detect_colocalization',
    'detect_spatial_gradients',
    'identify_hotspots',
    
    # Point pattern analysis
    'ripleys_k',
    'ripleys_l',
    'cross_k_function',
    'cross_l_function',
    'test_csr',
]