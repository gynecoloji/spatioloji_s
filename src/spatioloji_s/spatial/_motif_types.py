"""Shared data structures and helpers for spatial motif analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import scipy.sparse


@dataclass
class MotifCatalog:
    """Container for local motif discovery results.

    Attributes:
        labels: Series mapping cell_id to motif_id (int).
        signatures: DataFrame (motif_id x cell_type) with mean composition.
        counts: Series mapping motif_id to number of cells.
        group_col: Cell-type column name used for composition.
        feature_matrix: Sparse feature matrix used for clustering.
            Retained only if ``keep_features=True``.
        params: Parameters used for discovery.
    """

    labels: pd.Series
    signatures: pd.DataFrame
    counts: pd.Series
    group_col: str
    feature_matrix: scipy.sparse.csr_matrix | None
    params: dict


@dataclass
class AssemblyCatalog:
    """Container for mesoscale assembly detection results.

    Attributes:
        labels: Series mapping cell_id to assembly_id (int, -1 = unassigned).
        composition: DataFrame (assembly_id x motif_id) with mean motif proportions.
        instances: DataFrame with one row per motif instance.
            Columns: instance_id, assembly_id, motif_id, n_cells, centroid_x, centroid_y.
        adjacency_pattern: Long-form DataFrame of motif-pair adjacency frequencies
            per assembly type. Columns: assembly_id, motif_a, motif_b, frequency.
        params: Parameters used for detection.
    """

    labels: pd.Series
    composition: pd.DataFrame
    instances: pd.DataFrame
    adjacency_pattern: pd.DataFrame
    params: dict


@dataclass
class StructureMatches:
    """Container for known structure matching results.

    Attributes:
        matches: DataFrame with columns structure_name, target_type
            ("motif"/"assembly"), target_id, similarity, n_cells,
            centroid_x, centroid_y.
        per_cell: Series mapping cell_id to matched structure name or "unmatched".
        signatures_used: Dict of signatures that were queried.
    """

    matches: pd.DataFrame
    per_cell: pd.Series
    signatures_used: dict


@dataclass
class MotifResult:
    """Top-level container for the full motif pipeline.

    Attributes:
        motif_catalog: Local motif discovery results.
        assembly_catalog: Mesoscale assembly results (None if skipped).
        structure_matches: Known structure matches (None if skipped).
        params: Pipeline parameters.
    """

    motif_catalog: MotifCatalog
    assembly_catalog: AssemblyCatalog | None
    structure_matches: StructureMatches | None
    params: dict


def _get_cell_ids(graph) -> pd.Index:
    """Return cell IDs from either PolygonSpatialGraph or PointSpatialGraph."""
    if hasattr(graph, "cell_index"):
        return graph.cell_index
    return graph.cell_ids


def _get_sparse_adjacency(graph) -> scipy.sparse.csr_matrix:
    """Return sparse adjacency matrix from either graph type."""
    return graph.adjacency
