# src/spatioloji_s/spatial/shared/__init__.py

"""
Shared utilities for spatial analysis.

Common functions used by both point-based and polygon-based methods.
"""

from .utils import (
    # Validation
    validate_spatial_data,
    
    # Coordinate utilities
    get_centroid_coords,
    normalize_coordinates,
    calculate_spatial_extent,
    compute_spatial_bins,
    
    # Graph utilities
    convert_graph_to_networkx,
    compute_graph_metrics,
    create_spatial_index,
    
    # Spatial utilities
    get_boundary_cells,
    safe_divide,
)

__all__ = [
    # Validation
    'validate_spatial_data',
    
    # Coordinate utilities
    'get_centroid_coords',
    'normalize_coordinates',
    'calculate_spatial_extent',
    'compute_spatial_bins',
    
    # Graph utilities
    'convert_graph_to_networkx',
    'compute_graph_metrics',
    'create_spatial_index',
    
    # Spatial utilities
    'get_boundary_cells',
    'safe_divide',
]