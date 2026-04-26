"""
run.py - Full CCC pipeline orchestrator.

Single entry point for the spatioloji CCC workflow:

    result = run_ccc(sp, config)

Executes in order:
  1. Load LR database and filter to expressed pairs
  2. Build spatial graphs (juxtacrine contact, secreted/ECM radius)
  3. Score edges (score_edges)
  4. Filter by sender/receiver types (optional)
  5. Aggregate scores (aggregate_scores)
  6. Test significance (test_significance)
  7. Interface zone comparison and gradient (optional)
  8. Morphology stratification (optional)

Typical usage
-------------
>>> from spatioloji_s.ccc.run import run_ccc, CCCConfig
>>>
>>> config = CCCConfig(
...     group_col="cell_type",
...     db_source="builtin",
...     secreted_radius=200.0,
...     test_method="analytical",
... )
>>> result = run_ccc(sp_fov, config)
>>> result.scores.head()
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from .database import (
    LRPair,
    filter_to_expressed,
    load_from_cellchatdb_csv,
    load_lr_database,
)
from .scoring import aggregate_scores, score_edges, test_significance
from .zones import communication_gradient, compare_morphology, compare_zones

try:
    from spatioloji_s.spatial._interface_types import InterfaceResult
except ImportError:
    InterfaceResult = None  # type: ignore[assignment,misc]


# ── Config dataclass ──────────────────────────────────────────────────────────


@dataclass
class CCCConfig:
    """Configuration for CCC analysis.

    Attributes:
        db_source: Database source. ``'builtin'`` uses curated built-in
            LR pairs; ``'cellchatdb'`` loads from CSV (requires
            ``db_csv_path``); ``'custom'`` requires ``custom_df`` passed
            to :func:`load_lr_database`.
        db_csv_path: Path to CellChatDB CSV file. Required when
            ``db_source='cellchatdb'``.
        lr_types: Filter to specific LR types (``'juxtacrine'``,
            ``'secreted'``, ``'ecm'``). None = all types.
        min_pct: Minimum fraction of cells expressing a gene for a pair to
            be kept after expression filtering. Default 0.05.
        secreted_radius: Radius (in coordinate units) for secreted
            signaling graph. Default 200.0.
        ecm_radius: Radius for ECM signaling graph. Default 200.0.
        buffer_distance: Buffer distance for juxtacrine contact graph.
            None = use 0 (touching cells only).
        sigma_secreted: Distance decay sigma for secreted pairs. None =
            estimated as median edge distance.
        sigma_ecm: Distance decay sigma for ECM pairs. None = estimated
            as median edge distance.
        layer: Expression layer name. None = raw counts.
        test_method: Significance test method. ``'analytical'`` or
            ``'permutation'``.
        n_subsample: Maximum cells for permutation test (stratified
            subsample). Default 10000.
        n_permutations: Number of permutations. Default 1000.
        alpha: Significance threshold for reporting. Default 0.05.
        group_col: Column in ``sp.cell_meta`` for cell type labels.
        sender_types: If set, filter edges to these sender cell types.
        receiver_types: If set, filter edges to these receiver cell types.
        interface_result: Optional
            :class:`~spatioloji_s.spatial._interface_types.InterfaceResult`
            for zone comparison and gradient analysis.
        n_distance_bins: Number of bins for gradient analysis. Default 20.
        morphology_col: Column in ``sp.cell_meta`` for morphology categories.
            If set, :func:`~spatioloji_s.ccc.zones.compare_morphology` is run.
        coord_type: Coordinate type for graph building. ``'global'`` or
            ``'local'``.
        include_autocrine: If True, also score self-edges (i, i) for
            cells co-expressing both ligand and receptor of an LR pair.
            Autocrine edges use weight = 1.0, are tagged
            ``interaction_mode='autocrine'`` in ``edge_df`` / ``scores``,
            and are tested separately from paracrine.  Default True.
        seed: Random seed. Default 42.
        verbose: Print progress messages. Default True.

    Example:
        >>> config = CCCConfig(group_col="cell_type", test_method="permutation")
    """

    # Database
    db_source: str = "builtin"
    db_csv_path: str | None = None
    lr_types: list[str] | None = None
    min_pct: float = 0.05

    # Graph
    secreted_radius: float = 200.0
    ecm_radius: float = 200.0
    buffer_distance: float | None = None  # None = contact only (0 = touching)

    # Scoring
    sigma_secreted: float | None = None
    sigma_ecm: float | None = None
    layer: str | None = None

    # Significance
    test_method: str = "analytical"
    n_subsample: int = 10000
    n_permutations: int = 1000
    alpha: float = 0.05

    # Cell types
    group_col: str = "cell_type"
    sender_types: list[str] | None = None
    receiver_types: list[str] | None = None

    # Interface (optional)
    interface_result: object | None = None  # InterfaceResult | None
    n_distance_bins: int = 20

    # Morphology (optional)
    morphology_col: str | None = None

    # Autocrine
    include_autocrine: bool = True

    # General
    coord_type: str = "global"
    seed: int = 42
    verbose: bool = True


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class CCCResult:
    """Results from CCC analysis.

    Attributes:
        scores: Summary DataFrame with columns lr_name, sender_type,
            receiver_type, mean_score, sum_score, n_edges, pvalue, fdr.
        cell_scores: Per-cell sender/receiver scores indexed by cell_id.
        lr_pairs: List of :class:`~spatioloji_s.ccc.database.LRPair`
            objects used in the analysis.
        zone_comparison: Zone-stratified scores (set when
            ``config.interface_result`` is provided).
        zone_gradient: Communication gradient across the interface axis
            (set when ``config.interface_result`` is provided).
        morphology_comparison: Morphology-stratified scores (set when
            ``config.morphology_col`` is provided).
        edge_df: Raw edge-level scores from
            :func:`~spatioloji_s.ccc.scoring.score_edges`, retained for
            post-hoc filtering (e.g. halo cell removal via
            :func:`~spatioloji_s.ccc.halo.filter_ccc_to_core`).
        config: The :class:`CCCConfig` used to produce this result.
        n_cells: Number of cells in the analysis.
        n_edges: Number of scored edges.
        runtime_seconds: Wall-clock runtime of :func:`run_ccc`.

    Example:
        >>> result = run_ccc(sp, config)
        >>> result.scores[result.scores["fdr"] < 0.05]
    """

    scores: pd.DataFrame
    cell_scores: pd.DataFrame
    lr_pairs: list

    edge_df: pd.DataFrame | None = None
    zone_comparison: pd.DataFrame | None = None
    zone_gradient: pd.DataFrame | None = None
    morphology_comparison: pd.DataFrame | None = None

    config: CCCConfig | None = None
    n_cells: int = 0
    n_edges: int = 0
    runtime_seconds: float = 0.0


# ── Public entry point ────────────────────────────────────────────────────────


def run_ccc(
    sp: spatioloji,
    config: CCCConfig | None = None,
    lr_pairs: list[LRPair] | None = None,
) -> CCCResult:
    """Run the full CCC pipeline on a spatioloji object.

    Steps:
        1. Load LR database and filter to expressed pairs (skipped when
           ``lr_pairs`` is provided directly).
        2. Build spatial graphs (buffer for juxtacrine; radius for
           secreted / ECM).
        3. Score edges with :func:`~spatioloji_s.ccc.scoring.score_edges`.
        4. Filter edges by ``sender_types`` / ``receiver_types`` (optional).
        5. Aggregate with :func:`~spatioloji_s.ccc.scoring.aggregate_scores`.
        6. Test significance with
           :func:`~spatioloji_s.ccc.scoring.test_significance`.
        7. Zone analysis with
           :func:`~spatioloji_s.ccc.zones.compare_zones` and
           :func:`~spatioloji_s.ccc.zones.communication_gradient`
           (when ``config.interface_result`` is not None).
        8. Morphology analysis with
           :func:`~spatioloji_s.ccc.zones.compare_morphology`
           (when ``config.morphology_col`` is not None).

    Args:
        sp: spatioloji object.
        config: Pipeline configuration. If None, defaults are used.
        lr_pairs: If provided, skip database loading entirely and use
            these pairs directly. Useful in tests or when you already
            have pre-filtered pairs.

    Returns:
        :class:`CCCResult` with all analysis results.

    Raises:
        ValueError: If ``db_source`` is unknown or required parameters
            are missing.

    Example:
        >>> config = CCCConfig(group_col="cell_type")
        >>> result = run_ccc(sp_fov, config)
        >>> sig = result.scores[result.scores["fdr"] < 0.05]
    """
    start = time.time()

    if config is None:
        config = CCCConfig()

    def _log(msg: str) -> None:
        if config.verbose:
            print(f"[run_ccc] {msg}")

    # ── Step 1: Load / filter LR pairs ───────────────────────────────────
    if lr_pairs is not None:
        _log(f"Using {len(lr_pairs)} user-supplied LR pairs (database loading skipped)")
        pairs = list(lr_pairs)
    else:
        _log(f"Loading LR database (source={config.db_source!r})")
        if config.db_source == "cellchatdb" and config.db_csv_path is not None:
            pairs = load_from_cellchatdb_csv(config.db_csv_path, lr_types=config.lr_types)
        else:
            pairs = load_lr_database(config.db_source, lr_types=config.lr_types)

        _log(f"Filtering to expressed pairs (min_pct={config.min_pct})")
        pairs = filter_to_expressed(pairs, sp, min_pct=config.min_pct, layer=config.layer)

    if not pairs:
        _log("No expressed LR pairs found — returning empty result")
        empty_scores = pd.DataFrame(
            columns=["lr_name", "sender_type", "receiver_type", "mean_score", "sum_score", "n_edges"]
        )
        return CCCResult(
            scores=empty_scores,
            cell_scores=pd.DataFrame(index=sp.cell_index),
            lr_pairs=pairs,
            config=config,
            n_cells=len(sp.cell_index),
            n_edges=0,
            runtime_seconds=time.time() - start,
        )

    _log(f"{len(pairs)} LR pairs will be scored")

    # ── Step 2: Separate by signaling type ───────────────────────────────
    juxtacrine_pairs = [p for p in pairs if p.lr_type == "juxtacrine"]
    secreted_pairs = [p for p in pairs if p.lr_type == "secreted"]
    ecm_pairs = [p for p in pairs if p.lr_type == "ecm"]

    graph_jux = None
    graph_secreted = None
    graph_ecm = None

    # ── Step 3: Build graphs ──────────────────────────────────────────────
    if juxtacrine_pairs:
        from spatioloji_s.spatial.polygon.boundaries import contact_fraction
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        buf_dist = config.buffer_distance if config.buffer_distance is not None else 0
        _log(f"Building juxtacrine contact graph (buffer_distance={buf_dist})")
        graph_jux = build_buffer_graph(sp, buffer_distance=buf_dist, coord_type=config.coord_type)
        graph_jux.contact_frac_df = contact_fraction(sp, graph_jux, coord_type=config.coord_type)

    if secreted_pairs:
        from spatioloji_s.spatial.point.graph import build_radius_graph

        _log(f"Building secreted radius graph (radius={config.secreted_radius})")
        graph_secreted = build_radius_graph(sp, radius=config.secreted_radius, coord_type=config.coord_type)

    if ecm_pairs:
        from spatioloji_s.spatial.point.graph import build_radius_graph

        # Reuse secreted graph when radii are the same to avoid redundant work
        if graph_secreted is not None and config.ecm_radius == config.secreted_radius:
            _log("Reusing secreted graph for ECM pairs (same radius)")
            graph_ecm = graph_secreted
        else:
            _log(f"Building ECM radius graph (radius={config.ecm_radius})")
            graph_ecm = build_radius_graph(sp, radius=config.ecm_radius, coord_type=config.coord_type)

    # Pick the diffusible graph (secreted takes priority; fall back to ECM-only)
    graph_diffusible = graph_secreted if graph_secreted is not None else graph_ecm

    # ── Step 4: Score edges ───────────────────────────────────────────────
    if config.include_autocrine:
        _log("Scoring edges (incl. autocrine self-edges)")
    else:
        _log("Scoring edges")
    edge_df = score_edges(
        sp,
        pairs,
        graph_juxtacrine=graph_jux,
        graph_diffusible=graph_diffusible,
        group_col=config.group_col,
        layer=config.layer,
        sigma_secreted=config.sigma_secreted,
        sigma_ecm=config.sigma_ecm,
        include_autocrine=config.include_autocrine,
    )

    # ── Step 5: Filter by sender/receiver types ───────────────────────────
    if not edge_df.empty:
        if config.sender_types is not None:
            edge_df = edge_df[edge_df["sender_type"].isin(config.sender_types)]
            _log(f"Filtered to sender types: {config.sender_types} → {len(edge_df)} edges")
        if config.receiver_types is not None:
            edge_df = edge_df[edge_df["receiver_type"].isin(config.receiver_types)]
            _log(f"Filtered to receiver types: {config.receiver_types} → {len(edge_df)} edges")

    n_edges = len(edge_df)
    _log(f"{n_edges} edges after filtering")

    # ── Step 6: Aggregate ─────────────────────────────────────────────────
    _log("Aggregating scores")
    summary_df, cell_scores_df = aggregate_scores(edge_df, sp, group_col=config.group_col)

    # ── Step 7: Significance testing ──────────────────────────────────────
    _log(f"Testing significance (method={config.test_method!r})")
    summary_df = test_significance(
        summary_df,
        edge_df,
        sp,
        group_col=config.group_col,
        method=config.test_method,
        n_subsample=config.n_subsample,
        n_permutations=config.n_permutations,
        seed=config.seed,
        alpha=config.alpha,
    )

    # ── Step 8: Interface zone analysis (optional) ────────────────────────
    zone_comparison = None
    zone_gradient = None
    if config.interface_result is not None:
        _log("Running interface zone comparison")
        zone_comparison = compare_zones(
            edge_df,
            sp,
            config.interface_result,
            group_col=config.group_col,
            coord_type=config.coord_type,
        )
        _log("Computing communication gradient")
        zone_gradient = communication_gradient(
            edge_df,
            sp,
            config.interface_result,
            group_col=config.group_col,
            n_bins=config.n_distance_bins,
            coord_type=config.coord_type,
        )

    # ── Step 9: Morphology analysis (optional) ────────────────────────────
    morphology_comparison = None
    if config.morphology_col is not None:
        _log(f"Running morphology comparison (col={config.morphology_col!r})")
        morphology_comparison = compare_morphology(
            edge_df,
            sp,
            morphology_col=config.morphology_col,
            group_col=config.group_col,
        )

    elapsed = time.time() - start
    _log(f"Done in {elapsed:.2f}s")

    return CCCResult(
        scores=summary_df,
        cell_scores=cell_scores_df,
        lr_pairs=pairs,
        edge_df=edge_df,
        zone_comparison=zone_comparison,
        zone_gradient=zone_gradient,
        morphology_comparison=morphology_comparison,
        config=config,
        n_cells=len(sp.cell_index),
        n_edges=n_edges,
        runtime_seconds=elapsed,
    )
