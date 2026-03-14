"""Spatial motif discovery for point-based spatial data.

Thin wrapper — re-exports from the polygon module.
Both modes use the same graph adjacency interface.
"""

from spatioloji_s.spatial.polygon.motifs import discover_motifs

__all__ = ["discover_motifs"]
