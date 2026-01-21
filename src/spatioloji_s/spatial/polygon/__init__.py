# src/spatioloji_s/spatial/polygon/__init__.py

"""
Polygon-based spatial analysis using cell boundaries.

All methods use actual polygon coordinates for more accurate
spatial relationships. Requires sp.polygons to be available.

Modules
-------
- contact: Direct cell-cell contact analysis
- adjacency: True adjacency graphs (touching cells)
- overlap: Polygon overlap analysis
- morphology: Shape metrics and descriptors
- boundaries: Interface and border analysis
- voronoi: Voronoi tessellation generation
"""

# Import submodules (they exist but are empty for now)
from . import contact
from . import adjacency
from . import overlap
from . import morphology
from . import boundaries
from . import voronoi

# Will import specific functions as we implement them
__all__ = [
    'contact',
    'adjacency',
    'overlap',
    'morphology',
    'boundaries',
    'voronoi',
]