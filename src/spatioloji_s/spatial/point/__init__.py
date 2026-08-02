"""
point - Spatial analysis using point coordinates (centroids)

Provides spatial graph construction, neighborhood analysis,
pattern detection, Ripley's statistics, and distance-based
analyses using cell centroid coordinates.

Modules
-------
graph          : Spatial graph construction (KNN, radius, Delaunay)
neighborhoods  : Cell type neighborhood analysis
patterns       : Spatial autocorrelation, hotspots, co-occurrence
ripley         : Ripley's K/L functions and simulation envelopes
statistics     : Distance-based spatial statistics

Examples
--------
>>> import spatioloji_s as sj
>>>
>>> # Build a graph
>>> graph = sj.spatial.point.build_knn_graph(my_sj, k=10)
>>>
>>> # Neighborhood enrichment
>>> enrich = sj.spatial.point.neighborhood_enrichment(
...     my_sj, graph, 'cell_type')
>>>
>>> # Find spatially variable genes
>>> svg = sj.spatial.point.spatially_variable_genes(my_sj, graph)
>>>
>>> # Ripley's L with significance envelope
>>> result = sj.spatial.point.simulation_envelope(
...     my_sj, function='L', n_simulations=99)
>>>
>>> # Distance from tumor to nearest T cell
>>> dists = sj.spatial.point.cross_type_distances(
...     my_sj, 'cell_type', 'Tumor', 'T cell')
"""

# --- Graph construction ---
from spatioloji_s.spatial._distance_utils import classify_regions

# Gradient
from spatioloji_s.spatial._gradient import compute_gradient

# Infiltration
from spatioloji_s.spatial._infiltration import score_infiltration

# Interface
from spatioloji_s.spatial._interface import filter_interface, identify_interface

from .._gradient_types import GradientResult
from .._infiltration_types import InfiltrationResult
from .._interface_types import InterfaceResult

# Motifs
from .._motif_types import AssemblyCatalog, MotifCatalog, MotifResult, StructureMatches
from .graph import (
    PointSpatialGraph,
    build_delaunay_graph,
    build_knn_graph,
    build_radius_graph,
    build_weighted_knn_graph,
)
from .motifs import detect_assemblies, discover_motifs, match_known_structures, run_motif_pipeline

# --- Neighborhood analysis ---
from .neighborhoods import (
    identify_niches,
    neighborhood_composition,
    neighborhood_diversity,
    neighborhood_enrichment,
)

# --- Spatial patterns ---
from .patterns import (
    co_occurrence,
    getis_ord_gi,
    morans_i,
    spatially_variable_genes,
)

# --- Ripley's statistics ---
from .ripley import (
    RipleyResult,
    cross_k,
    cross_l,
    ripleys_k,
    ripleys_l,
    simulation_envelope,
)

# --- Distance-based statistics ---
from .statistics import (
    cross_type_distances,
    nearest_neighbor_distances,
    permutation_test,
    proximity_score,
)

__all__ = [
    # Graph
    "PointSpatialGraph",
    "build_knn_graph",
    "build_weighted_knn_graph",
    "build_radius_graph",
    "build_delaunay_graph",
    # Neighborhoods
    "neighborhood_composition",
    "neighborhood_enrichment",
    "identify_niches",
    "neighborhood_diversity",
    # Patterns
    "morans_i",
    "getis_ord_gi",
    "co_occurrence",
    "spatially_variable_genes",
    # Ripley
    "RipleyResult",
    "ripleys_k",
    "ripleys_l",
    "cross_k",
    "cross_l",
    "simulation_envelope",
    # Statistics
    "nearest_neighbor_distances",
    "cross_type_distances",
    "proximity_score",
    "permutation_test",
    # Interface
    "identify_interface",
    "filter_interface",
    "classify_regions",
    "InterfaceResult",
    # Gradient
    "compute_gradient",
    "GradientResult",
    # Infiltration
    "score_infiltration",
    "InfiltrationResult",
    # Motifs
    "discover_motifs",
    "detect_assemblies",
    "match_known_structures",
    "run_motif_pipeline",
    "MotifCatalog",
    "AssemblyCatalog",
    "MotifResult",
    "StructureMatches",
]
