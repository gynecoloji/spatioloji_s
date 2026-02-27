"""
spatial - Spatial analysis for spatioloji

Two complementary approaches to spatial analysis:

point : Centroid-based spatial analysis
    Uses cell centroids (x, y coordinates) for fast distance
    calculations. Best for large datasets and initial exploration.

polygon : Polygon-based spatial analysis
    Uses actual cell boundary polygons for true cell-cell topology.
    Captures contact interfaces, cell shape, and boundary geometry.
    More accurate but computationally heavier.

shared : Utilities shared across both approaches

Usage
-----
>>> import spatioloji as sj
>>>
>>> # Point-based analysis (centroid distances)
>>> graph = sj.spatial.point.build_knn_graph(sp, k=6)
>>> sj.spatial.point.neighborhood_composition(sp, graph, group_col='cell_type')
>>>
>>> # Polygon-based analysis (boundary distances)
>>> graph = sj.spatial.polygon.build_contact_graph(sp)
>>> sj.spatial.polygon.contact_fraction(sp, graph)
>>> sj.spatial.polygon.compute_morphology(sp)
"""

from . import point
from . import polygon

__all__ = [
    'point',
    'polygon',
]