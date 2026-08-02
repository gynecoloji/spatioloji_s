"""Spatial motif discovery for point-based spatial data.

Thin wrapper — re-exports from the polygon module.
Both modes use the same graph adjacency interface.
"""

from spatioloji_s.spatial.polygon.motifs import (
    detect_assemblies,
    discover_motifs,
    match_known_structures,
    run_motif_pipeline,
)

__all__ = ["discover_motifs", "detect_assemblies", "match_known_structures", "run_motif_pipeline"]
