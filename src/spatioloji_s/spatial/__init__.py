# src/spatioloji_s/spatial/__init__.py

"""
Spatial analysis module for spatioloji.

Provides two categories of spatial analysis:
- Point-based: Uses cell centroids for distance-based analysis
- Polygon-based: Uses cell boundaries for contact and shape analysis

Usage
-----
import spatioloji as sj

# Point-based analysis
graph = sj.spatial.point.build_knn_graph(sp, k=10)

# Polygon-based analysis
contact_graph = sj.spatial.polygon.build_contact_graph(sp)
"""

# Import submodules
from . import point
from . import polygon
from . import shared

__version__ = '0.1.0'

# Define what gets imported with "from spatioloji.spatial import *"
__all__ = [
    'point',
    'polygon',
    'shared',
]