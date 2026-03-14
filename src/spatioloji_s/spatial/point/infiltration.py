"""Immune infiltration scoring for point-based spatial data.

Thin wrapper — re-exports ``score_infiltration`` from the polygon module.
Both modes use centroid-based distances, so the logic is identical.
"""

from spatioloji_s.spatial.polygon.infiltration import score_infiltration

__all__ = ["score_infiltration"]
