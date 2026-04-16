"""
halo.py - Halo/buffer strategy for CCC edge-cell correction.

When an ROI is cropped from a larger dataset, cells at the ROI boundary
lose neighbors that were cut away.  This biases CCC scores downward for
edge cells.

The halo strategy solves this by:
  1. Temporarily enlarging the ROI to include nearby "halo" cells from
     the full dataset.
  2. Running the CCC pipeline on the enlarged set (so edge cells see
     their true neighbors).
  3. Filtering results back to the core ROI cells only.

Typical usage
-------------
>>> from spatioloji_s.ccc import CCCConfig, run_ccc
>>> from spatioloji_s.ccc.halo import create_halo_subset, filter_ccc_to_core
>>>
>>> config = CCCConfig(group_col="cell_type", secreted_radius=200.0)
>>> sp_halo, core_ids = create_halo_subset(sp_full, roi_cell_ids, config=config)
>>> result = run_ccc(sp_halo, config)
>>> result_roi = filter_ccc_to_core(result, core_ids, sp_core=sp_ROI)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from .run import CCCConfig, CCCResult

# ── Public API ───────────────────────────────────────────────────────────────


def create_halo_subset(
    sp_full: spatioloji,
    core_cell_ids: list[str] | np.ndarray | pd.Index,
    halo_distance: float | None = None,
    config: CCCConfig | None = None,
    coord_type: str = "global",
) -> tuple[spatioloji, pd.Index]:
    """Create an enlarged spatioloji that includes core cells plus nearby halo cells.

    Finds all cells in ``sp_full`` within ``halo_distance`` of any core cell
    and returns a subset containing both core and halo cells.  The halo cells
    provide proper neighbor context so that edge cells are not penalised by
    the ROI boundary.

    Args:
        sp_full: Full dataset containing all cells.
        core_cell_ids: Cell IDs defining the ROI (core cells).  Must be a
            subset of ``sp_full.cell_index``.
        halo_distance: Maximum distance from any core cell to include halo
            cells.  If ``None`` and ``config`` is provided, defaults to
            ``max(config.secreted_radius, config.ecm_radius)``.  If both
            are ``None``, raises ``ValueError``.
        config: CCC configuration.  Used to infer ``halo_distance`` when
            not explicitly set.
        coord_type: Which coordinates to use for distance calculation:
            ``'global'`` (default) or ``'local'``.

    Returns:
        Tuple of ``(sp_halo, core_ids)`` where ``sp_halo`` is a
        spatioloji containing core + halo cells, and ``core_ids`` is a
        ``pd.Index`` of validated core cell IDs for downstream filtering.

    Raises:
        ValueError: If ``halo_distance`` cannot be determined, or if no
            core cells are found in ``sp_full``.

    Example:
        >>> config = CCCConfig(secreted_radius=200.0, ecm_radius=150.0)
        >>> sp_halo, core_ids = create_halo_subset(sp_full, roi_ids, config=config)
        >>> # halo_distance = max(200, 150) = 200
        >>> print(f"Core: {len(core_ids)}, Total: {sp_halo.n_cells}")
    """
    from scipy.spatial import cKDTree

    # ── Resolve halo distance ────────────────────────────────────────────
    if halo_distance is None:
        if config is not None:
            halo_distance = max(config.secreted_radius, config.ecm_radius)
        else:
            raise ValueError(
                "halo_distance must be specified explicitly or inferred "
                "from a CCCConfig (pass config=)."
            )

    # ── Validate core cell IDs ───────────────────────────────────────────
    core_ids = pd.Index(core_cell_ids)
    core_ids = core_ids.intersection(sp_full.cell_index)

    if len(core_ids) == 0:
        raise ValueError("No core_cell_ids found in sp_full.cell_index.")

    n_dropped = len(core_cell_ids) - len(core_ids)
    if n_dropped > 0:
        print(f"[halo] Warning: {n_dropped} core cell IDs not found in sp_full — ignored.")

    # ── Get coordinates ──────────────────────────────────────────────────
    if coord_type == "global":
        x_all = sp_full.spatial.x_global
        y_all = sp_full.spatial.y_global
    elif coord_type == "local":
        x_all = sp_full.spatial.x_local
        y_all = sp_full.spatial.y_local
    else:
        raise ValueError(f"Unknown coord_type: {coord_type!r}. Use 'global' or 'local'.")

    coords_all = np.column_stack([x_all, y_all])

    # ── Find core indices in sp_full ─────────────────────────────────────
    full_index = sp_full.cell_index
    core_mask = full_index.isin(core_ids)
    core_indices = np.where(core_mask)[0]
    core_coords = coords_all[core_indices]

    # ── KDTree query for halo neighbours ─────────────────────────────────
    tree = cKDTree(coords_all)
    neighbor_lists = tree.query_ball_point(core_coords, r=halo_distance)

    # Flatten and deduplicate
    halo_indices = set()
    for nbrs in neighbor_lists:
        halo_indices.update(nbrs)

    # Combined = core ∪ halo (halo already includes core cells)
    combined_indices = sorted(halo_indices)
    combined_ids = full_index[combined_indices]

    n_halo = len(combined_ids) - len(core_ids)
    print(
        f"[halo] Core: {len(core_ids)} cells, "
        f"Halo: {n_halo} cells added (distance={halo_distance:.1f}), "
        f"Total: {len(combined_ids)} cells"
    )

    # ── Subset sp_full ───────────────────────────────────────────────────
    sp_halo = sp_full.subset_by_cells(combined_ids)

    return sp_halo, core_ids


def filter_ccc_to_core(
    result: CCCResult,
    core_cell_ids: pd.Index | list[str] | np.ndarray,
    sp_core: spatioloji | None = None,
) -> CCCResult:
    """Filter a CCCResult to retain only core-cell information.

    After running CCC on a halo-enlarged dataset, this function removes
    halo-cell contributions and recomputes summary statistics for the
    core ROI cells.

    An edge is retained if **at least one** endpoint (sender or receiver)
    is a core cell.  This ensures that core cells' scores include their
    communication with halo neighbors, which is the correct behaviour:
    a core cell at the ROI boundary *should* score interactions with
    its true (halo) neighbors.

    Args:
        result: Output of :func:`~spatioloji_s.ccc.run.run_ccc` on
            the halo-enlarged spatioloji.  ``result.edge_df`` must not
            be ``None``.
        core_cell_ids: Cell IDs of the core ROI (as returned by
            :func:`create_halo_subset`).
        sp_core: spatioloji object containing **only** core cells.
            When provided, significance testing is recomputed with
            correct cell-type counts.  When ``None``, significance
            columns (pvalue, fdr) are dropped with a warning.

    Returns:
        New :class:`~spatioloji_s.ccc.run.CCCResult` filtered to core
        cells, with recomputed scores, cell_scores, and (optionally)
        significance statistics.

    Raises:
        ValueError: If ``result.edge_df`` is ``None``.

    Example:
        >>> result = run_ccc(sp_halo, config)
        >>> result_roi = filter_ccc_to_core(result, core_ids, sp_core=sp_ROI)
    """
    from .scoring import aggregate_scores, test_significance

    if result.edge_df is None:
        raise ValueError(
            "CCCResult.edge_df is None — cannot filter. "
            "Re-run run_ccc (v0.3+) to populate edge_df."
        )

    core_set = set(pd.Index(core_cell_ids))
    edge_df = result.edge_df

    # ── Filter edges: keep if sender OR receiver is core ─────────────────
    mask = edge_df["sender"].isin(core_set) | edge_df["receiver"].isin(core_set)
    filtered_edges = edge_df[mask].copy()

    n_edges = len(filtered_edges)

    # ── Recompute summary and cell_scores ────────────────────────────────
    if sp_core is not None:
        summary_df, cell_scores_df = aggregate_scores(filtered_edges, sp_core, group_col=result.config.group_col)
    else:
        # Without sp_core, aggregate using core_cell_ids as the index
        core_index = pd.Index(core_cell_ids)
        summary_df, cell_scores_df = _aggregate_scores_from_index(filtered_edges, core_index)

    # ── Significance testing ─────────────────────────────────────────────
    if sp_core is not None and result.config is not None:
        cfg = result.config
        summary_df = test_significance(
            summary_df,
            filtered_edges,
            sp_core,
            group_col=cfg.group_col,
            method=cfg.test_method,
            n_subsample=cfg.n_subsample,
            n_permutations=cfg.n_permutations,
            seed=cfg.seed,
            alpha=cfg.alpha,
        )
    else:
        print("[halo] Warning: sp_core not provided — pvalue/fdr columns omitted.")

    # ── Filter optional zone/morphology DataFrames ───────────────────────
    zone_comparison = result.zone_comparison
    zone_gradient = result.zone_gradient
    morphology_comparison = result.morphology_comparison

    return CCCResult(
        scores=summary_df,
        cell_scores=cell_scores_df,
        lr_pairs=result.lr_pairs,
        edge_df=filtered_edges,
        zone_comparison=zone_comparison,
        zone_gradient=zone_gradient,
        morphology_comparison=morphology_comparison,
        config=result.config,
        n_cells=len(core_set),
        n_edges=n_edges,
        runtime_seconds=result.runtime_seconds,
    )


# ── Private helpers ──────────────────────────────────────────────────────────


def _aggregate_scores_from_index(
    edge_df: pd.DataFrame,
    core_index: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate edge scores using a bare cell index (no spatioloji required).

    Args:
        edge_df: Filtered edge DataFrame.
        core_index: Cell IDs to include in cell_scores.

    Returns:
        Tuple of (summary_df, cell_scores_df).
    """
    if edge_df.empty:
        cols = ["lr_name", "sender_type", "receiver_type", "mean_score", "sum_score", "n_edges"]
        return pd.DataFrame(columns=cols), pd.DataFrame(index=core_index)

    # Type-pair summary
    grouped = edge_df.groupby(["lr_name", "sender_type", "receiver_type"])["score"]
    summary = pd.DataFrame(
        {
            "mean_score": grouped.mean(),
            "sum_score": grouped.sum(),
            "n_edges": grouped.count(),
        }
    ).reset_index()

    # Per-cell scores (only core cells)
    sender_agg = edge_df.groupby(["lr_name", "sender"])["score"].sum()
    receiver_agg = edge_df.groupby(["lr_name", "receiver"])["score"].sum()

    sender_wide = sender_agg.unstack(level=0, fill_value=0.0)
    receiver_wide = receiver_agg.unstack(level=0, fill_value=0.0)

    sender_wide = sender_wide.reindex(core_index, fill_value=0.0)
    receiver_wide = receiver_wide.reindex(core_index, fill_value=0.0)

    sender_wide.columns = [f"{lr}_sender" for lr in sender_wide.columns]
    receiver_wide.columns = [f"{lr}_receiver" for lr in receiver_wide.columns]

    cell_scores = pd.concat([sender_wide, receiver_wide], axis=1)
    cell_scores.index = core_index

    return summary, cell_scores
