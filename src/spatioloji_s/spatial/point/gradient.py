"""Spatial gradient analysis for point-based spatial data.

Thin wrapper — re-exports ``compute_gradient`` from the polygon module.
Both modes use centroid-based distances, so the logic is identical.
"""

from spatioloji_s.spatial.polygon.gradient import compute_gradient

__all__ = ["compute_gradient"]
