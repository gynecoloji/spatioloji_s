"""
polygon - Polygon-based spatial analysis for spatioloji

Analyzes spatial relationships using actual cell boundary polygons
rather than centroid approximations. All distance calculations use
polygon-to-polygon (boundary-to-boundary) distances.

Modules
-------
graph : Polygon adjacency graph construction
    build_contact_graph, build_buffer_graph, build_knn_graph
morphology : Cell shape feature extraction
    compute_morphology, classify_morphology, morphology_by_group
boundaries : Cell-cell contact quantification
    contact_length, contact_fraction, free_boundary_fraction, contact_summary
neighborhoods : Neighborhood composition and niche identification
    neighborhood_composition, neighborhood_enrichment,
    niche_identification, boundary_cells
patterns : Spatial pattern detection
    cell_density_map, hotspot_detection, spatial_autocorrelation, colocalization
statistics : Formal hypothesis testing
    contact_permutation_test, morphology_association_test,
    spatial_autocorrelation_test, boundary_enrichment_test

Typical workflow
----------------
>>> import spatioloji as sj
>>>
>>> # 1. Build polygon adjacency graph
>>> graph = sj.spatial.polygon.build_contact_graph(sp)
>>>
>>> # 2. Compute morphology features
>>> sj.spatial.polygon.compute_morphology(sp)
>>>
>>> # 3. Quantify cell-cell contacts
>>> sj.spatial.polygon.contact_fraction(sp, graph)
>>>
>>> # 4. Identify spatial niches
>>> sj.spatial.polygon.niche_identification(sp, graph, group_col='cell_type')
>>>
>>> # 5. Detect hotspots
>>> sj.spatial.polygon.hotspot_detection(sp, graph, metric='total_counts')
>>>
>>> # 6. Statistical testing
>>> sj.spatial.polygon.contact_permutation_test(sp, graph, group_col='cell_type')
"""

# Graph construction
from .graph import (
    PolygonSpatialGraph,
    build_contact_graph,
    build_buffer_graph,
    build_knn_graph,
)

# Morphology
from .morphology import (
    compute_morphology,
    classify_morphology,
    morphology_by_group,
)

# Boundaries
from .boundaries import (
    contact_length,
    contact_fraction,
    free_boundary_fraction,
    contact_summary,
)

# Neighborhoods
from .neighborhoods import (
    neighborhood_composition,
    neighborhood_enrichment,
    niche_identification,
    boundary_cells,
)

# Patterns
from .patterns import (
    cell_density_map,
    hotspot_detection,
    spatial_autocorrelation,
    colocalization,
)

# Statistics
from .statistics import (
    contact_permutation_test,
    morphology_association_test,
    spatial_autocorrelation_test,
    boundary_enrichment_test,
)

__all__ = [
    # Graph
    'PolygonSpatialGraph',
    'build_contact_graph',
    'build_buffer_graph',
    'build_knn_graph',
    # Morphology
    'compute_morphology',
    'classify_morphology',
    'morphology_by_group',
    # Boundaries
    'contact_length',
    'contact_fraction',
    'free_boundary_fraction',
    'contact_summary',
    # Neighborhoods
    'neighborhood_composition',
    'neighborhood_enrichment',
    'niche_identification',
    'boundary_cells',
    # Patterns
    'cell_density_map',
    'hotspot_detection',
    'spatial_autocorrelation',
    'colocalization',
    # Statistics
    'contact_permutation_test',
    'morphology_association_test',
    'spatial_autocorrelation_test',
    'boundary_enrichment_test',
]