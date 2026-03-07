"""
run.py - Full CCC pipeline orchestrator.

Single entry point for the entire spatioloji CCC workflow:

    ccc_results = run_ccc(sp_fov, config)

Executes in order:
  Prerequisites  — build_contact_graph, contact_fraction,
                   free_boundary_fraction, compute_morphology
  Database       — load LR pairs, filter to expressed
  Layer 1        — Bivariate Moran + Spatial Lag (discovery)
  Layer 2        — Polygon OT + Message Passing (scoring)
  Layer 3        — Contrastive scoring + NMF programs (patterns)

Prerequisites can be supplied pre-computed to avoid redundant work
when running multiple FOVs (pass via the `precomputed` argument).

Results can be saved and reloaded with save_ccc_results() /
load_ccc_results() for downstream visualization or comparison.

Typical usage
-------------
>>> from spatioloji_s.ccc.run import run_ccc, CCCConfig, save_ccc_results
>>>
>>> config = CCCConfig(
...     cell_type_col   = 'cell_type',
...     layer           = 'log_normalized',
...     db_source       = 'cellchatdb',
...     db_csv_path     = 'CellChatDB.csv',
...     lr_types        = ['juxtacrine', 'secreted'],
...     n_permutations  = 1000,
...     K               = None,   # auto-select
... )
>>>
>>> ccc_results = run_ccc(sp_fov, config)
>>> save_ccc_results(ccc_results, 'fov01_ccc.pkl')
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    from spatioloji_s.spatial.polygon.graph import PolygonSpatialGraph

from dataclasses import dataclass, field
import pickle
import time

import numpy as np
import pandas as pd

from .database import (
    LRPair,
    load_lr_database,
    load_from_cellchatdb_csv,
    filter_to_expressed,
    lr_pairs_to_dataframe,
)
from .layer1 import run_layer1, summarize_layer1, get_top_pairs
from .layer2 import run_layer2
from .layer3 import (
    run_layer3,
    get_program_cells,
    get_program_lr_pairs,
)


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class CCCConfig:
    """
    Configuration for the full CCC pipeline.

    All parameters have sensible defaults — only cell_type_col and
    layer typically need to be changed for a new dataset.

    Parameters
    ----------
    cell_type_col : str
        Column in sp.cell_meta with cell type labels.
    layer : str | None
        Expression layer for CCC scoring. None → raw counts.
        Recommended: 'log_normalized'.

    Database settings
    -----------------
    db_source : str
        'builtin'    — use curated built-in LR pairs (no file needed)
        'cellchatdb' — load from CellChatDB CSV (requires db_csv_path)
        'custom'     — pass a custom DataFrame via db_custom_df
    db_csv_path : str | None
        Path to CellChatDB CSV (required when db_source='cellchatdb').
    db_custom_df : pd.DataFrame | None
        Custom LR DataFrame (required when db_source='custom').
        Required columns: lr_name, ligand, receptor, pathway, lr_type.
    lr_types : list[str] | None
        Which signaling types to include.
        Options: 'juxtacrine', 'secreted', 'ecm'. None = all.
    min_pct : float
        Minimum fraction of cells expressing a gene to include it.
        Default 0.05 (5%).

    Layer 1 settings
    ----------------
    n_permutations : int
        Permutations for Moran significance test. Default 1000.
    alpha_moran : float
        Moran p-value threshold. Default 0.05.
    alpha_lag : float
        Spatial lag FDR threshold. Default 0.05.
    min_rho : float
        Minimum spatial lag effect size. Default 0.05.
    n_jobs : int
        Parallel workers for Moran permutation test. -1 = all cores. Default 1.

    Layer 2 settings
    ----------------
    ot_reg : float
        Sinkhorn entropy regularization. Default 0.1.
    ot_iter : int
        Max Sinkhorn iterations. Default 50.
    hub_percentile : float
        Hub cell detection threshold (percentile). Default 90.
    top_k_l2 : int | None
        Limit Layer 2 to top-K pairs from Layer 1. None = all.

    Layer 3 settings
    ----------------
    n_permutations_contrast : int
        Permutations for contrastive null models. Default 1000.
    z_thresh : float
        Z-score threshold for driver classification. Default 1.96.
    K : int | None
        Number of NMF programs. None = auto-select via elbow.
    k_range : range
        K values to test when K=None. Default range(2, 11).
    lambda_reg : float
        Spatial regularization for NMF. Default 0.1.

    General
    -------
    coord_type : str
        'global' or 'local'. Default 'global'.
    seed : int
        Random seed for reproducibility. Default 42.
    verbose : bool
        Print progress. Default True.
    """
    # Required
    cell_type_col : str         = 'cell_type'
    layer         : str | None  = 'log_normalized'

    # Database
    db_source     : str         = 'builtin'
    db_csv_path   : str | None  = None
    db_custom_df  : pd.DataFrame | None = field(default=None, repr=False)
    lr_types      : list[str] | None    = None
    min_pct       : float       = 0.05

    # Layer 1
    n_permutations  : int   = 1000
    alpha_moran     : float = 0.05
    alpha_lag       : float = 0.05
    min_rho         : float = 0.05
    n_jobs          : int   = 1

    # Layer 2
    ot_reg          : float     = 0.1
    ot_iter         : int       = 50
    hub_percentile  : float     = 90.0
    top_k_l2        : int | None = None

    # Layer 3
    n_permutations_contrast : int   = 1000
    z_thresh                : float = 1.96
    K                       : int | None = None
    k_range                 : range = field(default_factory=lambda: range(2, 11))
    lambda_reg              : float = 0.1

    # General
    coord_type : str  = 'global'
    seed       : int  = 42
    verbose    : bool = True

    def __post_init__(self):
        if self.db_source == 'cellchatdb' and self.db_csv_path is None:
            raise ValueError(
                "db_csv_path is required when db_source='cellchatdb'."
            )
        if self.db_source == 'custom' and self.db_custom_df is None:
            raise ValueError(
                "db_custom_df is required when db_source='custom'."
            )
        valid_sources = {'builtin', 'cellchatdb', 'custom'}
        if self.db_source not in valid_sources:
            raise ValueError(
                f"db_source must be one of {valid_sources}, "
                f"got {self.db_source!r}."
            )


# ── Prerequisites helper ──────────────────────────────────────────────────────

def _run_prerequisites(
    sp:         spatioloji,
    coord_type: str,
    verbose:    bool,
) -> tuple:
    """
    Build polygon contact graph and compute all required cell_meta columns.

    Required by Layer 1:
      sp.cell_meta['free_boundary_fraction']
      sp.cell_meta['total_contact_fraction']
      sp.cell_meta['morph_solidity']
      sp.cell_meta['morph_circularity']

    Returns
    -------
    graph           : PolygonSpatialGraph
    contact_frac_df : pd.DataFrame  (cell_a, cell_b, fraction_a, fraction_b)
    free_boundary_ser : pd.Series   (indexed by cell_id)
    """
    from spatioloji_s.spatial.polygon.graph import build_contact_graph
    from spatioloji_s.spatial.polygon.boundaries import (
        contact_fraction,
        free_boundary_fraction,
    )
    from spatioloji_s.spatial.polygon.morphology import compute_morphology

    if verbose:
        print("\n[Prerequisites] Building contact graph...")
    graph = build_contact_graph(sp, coord_type=coord_type)

    if verbose:
        print("[Prerequisites] Computing contact fractions...")
    cf_df = contact_fraction(sp, graph, coord_type=coord_type, store=True)

    if verbose:
        print("[Prerequisites] Computing free boundary fractions...")
    fb_ser = free_boundary_fraction(sp, graph, coord_type=coord_type, store=True)

    if verbose:
        print("[Prerequisites] Computing morphology metrics...")
    compute_morphology(sp, store=True)

    return graph, cf_df, fb_ser


def _check_prerequisites(sp: spatioloji) -> list[str]:
    """Return list of missing prerequisite columns in sp.cell_meta."""
    required = [
        'free_boundary_fraction',
        'total_contact_fraction',
        'morph_solidity',
        'morph_circularity',
    ]
    return [c for c in required if c not in sp.cell_meta.columns]


# ── Database loader ───────────────────────────────────────────────────────────

def _load_database(
    config: CCCConfig,
    sp:     spatioloji,
    verbose: bool,
) -> list[LRPair]:
    """Load LR database and filter to expressed genes."""
    if verbose:
        print(f"\n[Database] Loading LR pairs (source={config.db_source!r})...")

    if config.db_source == 'builtin':
        lr_all = load_lr_database(
            source   = 'builtin',
            lr_types = config.lr_types,
        )
    elif config.db_source == 'cellchatdb':
        lr_all = load_from_cellchatdb_csv(
            csv_path = config.db_csv_path,
            lr_types = config.lr_types,
        )
    else:  # custom
        lr_all = load_lr_database(
            source      = 'custom',
            lr_types    = config.lr_types,
            custom_df   = config.db_custom_df,
        )

    if verbose:
        print(f"[Database] Filtering to expressed genes "
              f"(min_pct={config.min_pct:.0%})...")

    lr_expressed = filter_to_expressed(
        lr_all,
        sp,
        min_pct = config.min_pct,
        layer   = config.layer,
    )

    if not lr_expressed:
        raise ValueError(
            f"No LR pairs passed the expression filter "
            f"(min_pct={config.min_pct:.0%}). "
            f"Consider lowering min_pct or checking your gene panel."
        )

    return lr_expressed


# ── Main entry point ──────────────────────────────────────────────────────────

def run_ccc(
    sp:          spatioloji,
    config:      CCCConfig,
    precomputed: dict | None = None,
) -> dict:
    """
    Run the full spatioloji CCC pipeline on a single FOV.

    sp should already be subset to one FOV before calling run_ccc.
    Multi-FOV analysis: call run_ccc once per FOV and collect results.

    Parameters
    ----------
    sp : spatioloji
        spatioloji object (already subset to one FOV).
    config : CCCConfig
        Pipeline configuration. See CCCConfig docstring.
    precomputed : dict | None
        Optional pre-computed prerequisites to skip rebuilding.
        Accepted keys:
          'graph'           : PolygonSpatialGraph
          'contact_frac_df' : pd.DataFrame
          'free_boundary_ser': pd.Series
        Missing keys will be recomputed automatically.
        If None, all prerequisites are computed fresh.

    Returns
    -------
    dict with keys:
        'config'          : CCCConfig        — parameters used
        'sp_summary'      : dict             — FOV metadata
        'lr_pairs'        : list[LRPair]     — expressed LR pairs
        'lr_pairs_df'     : pd.DataFrame     — tabular summary of LR pairs
        'layer1_results'  : dict             — Layer 1 outputs
        'layer2_results'  : dict             — Layer 2 outputs
        'layer3_results'  : dict             — Layer 3 outputs
        'significant_pairs': pd.DataFrame   — shortcut to Layer 1 sig pairs
        'scores'          : pd.DataFrame    — shortcut to Layer 2 scores
        'contrastive'     : pd.DataFrame    — shortcut to Layer 3 contrastive
        'programs'        : pd.DataFrame    — shortcut to Layer 3 programs
        'timing'          : dict            — wall time per step (seconds)

    Examples
    --------
    >>> config = CCCConfig(
    ...     cell_type_col = 'cell_type',
    ...     layer         = 'log_normalized',
    ...     db_source     = 'cellchatdb',
    ...     db_csv_path   = 'CellChatDB.csv',
    ...     K             = 5,
    ... )
    >>> ccc_results = run_ccc(sp_fov, config)
    >>> print(ccc_results['programs'])
    """
    t_total = time.time()
    timing: dict[str, float] = {}
    verbose = config.verbose

    if verbose:
        print("\n" + "═" * 60)
        print("spatioloji CCC Pipeline")
        print("═" * 60)
        print(f"  Cells      : {sp.n_cells}")
        print(f"  Genes      : {sp.n_genes}")
        print(f"  Cell types : {sp.cell_meta[config.cell_type_col].nunique()}")
        print(f"  Layer      : {config.layer!r}")
        print(f"  db_source  : {config.db_source!r}")

    # ── Prerequisites ─────────────────────────────────────────────────────────
    t0 = time.time()

    precomputed = precomputed or {}
    graph            = precomputed.get('graph')
    contact_frac_df  = precomputed.get('contact_frac_df')
    free_boundary_ser = precomputed.get('free_boundary_ser')

    # Check which cell_meta columns are already present
    missing_meta = _check_prerequisites(sp)

    needs_compute = (
        graph is None or
        contact_frac_df is None or
        free_boundary_ser is None or
        bool(missing_meta)
    )

    if needs_compute:
        if verbose and missing_meta:
            print(f"\n  Missing prerequisite columns: {missing_meta}")
            print("  Running prerequisites automatically...")
        graph, contact_frac_df, free_boundary_ser = _run_prerequisites(
            sp, config.coord_type, verbose
        )
    else:
        if verbose:
            print("\n[Prerequisites] Using pre-computed graph and fractions.")

    timing['prerequisites'] = time.time() - t0

    # ── Database ──────────────────────────────────────────────────────────────
    t0 = time.time()
    lr_expressed = _load_database(config, sp, verbose)
    timing['database'] = time.time() - t0

    if verbose:
        print(f"\n  {len(lr_expressed)} LR pairs after expression filter")

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    t0 = time.time()
    layer1_results = run_layer1(
        sp              = sp,
        lr_pairs        = lr_expressed,
        contact_graph   = graph,
        contact_frac_df = contact_frac_df,
        free_boundary_ser = free_boundary_ser,
        cell_type_col   = config.cell_type_col,
        layer           = config.layer,
        n_permutations  = config.n_permutations,
        alpha_moran     = config.alpha_moran,
        alpha_lag       = config.alpha_lag,
        min_rho         = config.min_rho,
        seed            = config.seed,
        n_jobs          = config.n_jobs,
    )
    timing['layer1'] = time.time() - t0

    n_sig = len(layer1_results['significant_pairs'])
    if verbose:
        print(f"\n  Layer 1: {n_sig} significant (lr, type_pair) combinations")

    if n_sig == 0:
        if verbose:
            print("\n  ⚠ No significant pairs found after Layer 1.")
            print("  Consider loosening alpha_moran, alpha_lag, or min_rho.")
        return _build_results(
            config, sp, lr_expressed,
            layer1_results, {}, {},
            timing, t_total,
        )

    # ── Layer 2 ───────────────────────────────────────────────────────────────
    t0 = time.time()
    layer2_results = run_layer2(
        sp              = sp,
        layer1_results  = layer1_results,
        contact_frac_df = contact_frac_df,
        coord_type      = config.coord_type,
        ot_reg          = config.ot_reg,
        ot_iter         = config.ot_iter,
        hub_percentile  = config.hub_percentile,
        top_k           = config.top_k_l2,
    )
    timing['layer2'] = time.time() - t0

    if verbose:
        n_scored = len(layer2_results.get('scores', pd.DataFrame()))
        print(f"\n  Layer 2: {n_scored} combinations scored")

    if not layer2_results.get('edge_data'):
        if verbose:
            print("\n  ⚠ No edge data produced by Layer 2.")
        return _build_results(
            config, sp, lr_expressed,
            layer1_results, layer2_results, {},
            timing, t_total,
        )

    # ── Layer 3 ───────────────────────────────────────────────────────────────
    t0 = time.time()
    layer3_results = run_layer3(
        sp                      = sp,
        layer1_results          = layer1_results,
        layer2_results          = layer2_results,
        contact_graph           = graph,
        n_permutations          = config.n_permutations_contrast,
        z_thresh                = config.z_thresh,
        K                       = config.K,
        k_range                 = config.k_range,
        lambda_reg              = config.lambda_reg,
        seed                    = config.seed,
    )
    timing['layer3'] = time.time() - t0

    # ── Package and return ────────────────────────────────────────────────────
    results = _build_results(
        config, sp, lr_expressed,
        layer1_results, layer2_results, layer3_results,
        timing, t_total,
    )

    if verbose:
        _print_final_summary(results)

    return results


# ── Multi-FOV pipeline ────────────────────────────────────────────────────────

def run_ccc_multifov(
    sp:          spatioloji,
    config:      CCCConfig,
    fov_col:     str        = 'fov',
    fov_ids:     list | None = None,
) -> dict[str, dict]:
    """
    Run CCC pipeline independently on each FOV.

    Spatial graphs must be built per-FOV (FOVs are not physically
    connected — cross-FOV edges would be biologically meaningless).

    Parameters
    ----------
    sp : spatioloji
        Full spatioloji object (all FOVs).
    config : CCCConfig
        Shared configuration applied to every FOV.
    fov_col : str
        Column in sp.cell_meta with FOV labels.
    fov_ids : list | None
        Subset of FOV IDs to process. None = all FOVs.

    Returns
    -------
    dict[str, dict]
        Keys = FOV IDs, values = ccc_results dicts from run_ccc().

    Examples
    --------
    >>> all_results = run_ccc_multifov(sp, config, fov_col='fov')
    >>> fov01 = all_results['F001']
    >>> print(fov01['programs'])
    """
    if fov_col not in sp.cell_meta.columns:
        raise ValueError(
            f"fov_col={fov_col!r} not found in sp.cell_meta. "
            f"Available: {list(sp.cell_meta.columns)}"
        )

    all_fovs = sorted(sp.cell_meta[fov_col].unique())
    if fov_ids is not None:
        all_fovs = [f for f in all_fovs if f in set(fov_ids)]

    print(f"\n[Multi-FOV CCC] Processing {len(all_fovs)} FOVs...")

    results: dict[str, dict] = {}

    for fov_id in all_fovs:
        print(f"\n{'─' * 60}")
        print(f"FOV: {fov_id}")
        print(f"{'─' * 60}")

        # Subset to this FOV
        fov_cell_ids = sp.cell_meta.index[
            sp.cell_meta[fov_col].astype(str) == str(fov_id)
        ].tolist()

        if len(fov_cell_ids) < 20:
            print(f"  ⚠ Skipping FOV {fov_id}: only {len(fov_cell_ids)} cells")
            continue

        try:
            sp_fov = sp.subset_by_cells(fov_cell_ids)
            fov_results = run_ccc(sp_fov, config)
            results[str(fov_id)] = fov_results

        except Exception as e:
            print(f"  ✗ FOV {fov_id} failed: {e}")
            results[str(fov_id)] = {'error': str(e), 'fov_id': fov_id}

    n_success = sum(1 for v in results.values() if 'error' not in v)
    print(f"\n[Multi-FOV CCC] Completed: {n_success}/{len(all_fovs)} FOVs successful")

    return results


# ── Save / load ───────────────────────────────────────────────────────────────

def save_ccc_results(
    ccc_results: dict,
    path:        str,
) -> None:
    """
    Save ccc_results dict to disk as a pickle file.

    Parameters
    ----------
    ccc_results : dict
        Output of run_ccc() or run_ccc_multifov().
    path : str
        Output file path. Recommend .pkl extension.

    Examples
    --------
    >>> save_ccc_results(ccc_results, 'fov01_ccc.pkl')
    """
    with open(path, 'wb') as f:
        pickle.dump(ccc_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[CCC] Saved results to {path!r}")


def load_ccc_results(path: str) -> dict:
    """
    Load ccc_results dict from a pickle file.

    Parameters
    ----------
    path : str
        Path to pickle file saved by save_ccc_results().

    Returns
    -------
    dict
        ccc_results dict as returned by run_ccc().

    Examples
    --------
    >>> ccc_results = load_ccc_results('fov01_ccc.pkl')
    """
    with open(path, 'rb') as f:
        results = pickle.load(f)
    print(f"[CCC] Loaded results from {path!r}")
    return results


# ── Comparison helpers ────────────────────────────────────────────────────────

def compare_fov_results(
    multifov_results: dict[str, dict],
    metric:           str = 'combined_strength',
    top_n:            int = 20,
) -> pd.DataFrame:
    """
    Compare CCC results across FOVs.

    Builds a tidy DataFrame with one row per (lr_name, sender, receiver, fov).
    Useful for identifying LR pairs that are consistently significant
    or FOV-specific.

    Parameters
    ----------
    multifov_results : dict[str, dict]
        Output of run_ccc_multifov().
    metric : str
        Column from Layer 2 scores to compare. Default 'combined_strength'.
    top_n : int
        Top-N pairs per FOV to include. Default 20.

    Returns
    -------
    pd.DataFrame
        Columns: fov_id, lr_name, sender_type, receiver_type, <metric>
    """
    rows: list[dict] = []

    for fov_id, res in multifov_results.items():
        if 'error' in res:
            continue
        scores = res.get('scores', pd.DataFrame())
        if scores.empty:
            continue
        if metric not in scores.columns:
            continue

        top = scores.nlargest(top_n, metric)
        for _, row in top.iterrows():
            rows.append({
                'fov_id'       : fov_id,
                'lr_name'      : row['lr_name'],
                'sender_type'  : row['sender_type'],
                'receiver_type': row['receiver_type'],
                metric         : row[metric],
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ['lr_name', 'fov_id']
    ).reset_index(drop=True)


def summarize_ccc(ccc_results: dict, top_n: int = 10) -> None:
    """
    Print a compact human-readable summary of ccc_results.

    Parameters
    ----------
    ccc_results : dict   Output of run_ccc().
    top_n : int          Top pairs to display per section.
    """
    print("\n" + "═" * 60)
    print("CCC Pipeline Summary")
    print("═" * 60)

    # FOV info
    sp_sum = ccc_results.get('sp_summary', {})
    print(f"\n  FOV: {sp_sum.get('fov_id', 'unknown')}")
    print(f"  Cells: {sp_sum.get('n_cells', '?')}  |  "
          f"Genes: {sp_sum.get('n_genes', '?')}  |  "
          f"Cell types: {sp_sum.get('n_cell_types', '?')}")

    # LR pairs
    lr_df = ccc_results.get('lr_pairs_df', pd.DataFrame())
    if not lr_df.empty:
        print(f"\n  Expressed LR pairs: {len(lr_df)}")
        for t, grp in lr_df.groupby('lr_type'):
            print(f"    {t:12s}: {len(grp)}")

    # Layer 1
    sig = ccc_results.get('significant_pairs', pd.DataFrame())
    if not sig.empty:
        print(f"\n  Layer 1 significant: {len(sig)} (lr, type_pair) combinations")
        print(f"  Top {min(top_n, len(sig))} by layer1_score:")
        cols = [c for c in ['lr_name', 'sender_type', 'receiver_type',
                             'I_bivar', 'rho_LR', 'layer1_score']
                if c in sig.columns]
        print(sig[cols].head(top_n).to_string(index=False))

    # Layer 2
    scores = ccc_results.get('scores', pd.DataFrame())
    if not scores.empty:
        print(f"\n  Layer 2 scored: {len(scores)} combinations")
        cols = [c for c in ['lr_name', 'sender_type', 'receiver_type',
                             'ot_strength', 'mp_strength', 'combined_strength']
                if c in scores.columns]
        print(f"  Top {min(top_n, len(scores))} by combined_strength:")
        print(scores[cols].head(top_n).to_string(index=False))

    # Layer 3
    programs = ccc_results.get('programs', pd.DataFrame())
    if not programs.empty:
        K = ccc_results['layer3_results'].get('K', '?')
        print(f"\n  Layer 3 NMF programs: K={K}")
        for _, prow in programs.iterrows():
            print(f"    Program {prow['program']}: "
                  f"{prow['dominant_sender_type']} → "
                  f"{prow['dominant_recv_type']}  |  "
                  f"top LR: {prow['top_lr_pairs'][:3]}")

    # Timing
    timing = ccc_results.get('timing', {})
    if timing:
        print("\n  Timing:")
        for step, t in timing.items():
            print(f"    {step:20s}: {t:.1f}s")
        print(f"    {'total':20s}: {timing.get('total', 0):.1f}s")


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_results(
    config:          CCCConfig,
    sp:              spatioloji,
    lr_expressed:    list[LRPair],
    layer1_results:  dict,
    layer2_results:  dict,
    layer3_results:  dict,
    timing:          dict[str, float],
    t_total:         float,
) -> dict:
    """Package all outputs into the ccc_results dict."""
    timing['total'] = time.time() - t_total

    # FOV summary (best-effort: extract fov_id from cell_meta if available)
    fov_id = 'unknown'
    fov_col = 'fov'
    if fov_col in sp.cell_meta.columns:
        unique_fovs = sp.cell_meta[fov_col].unique()
        fov_id = str(unique_fovs[0]) if len(unique_fovs) == 1 else 'multi'

    sp_summary = {
        'fov_id'      : fov_id,
        'n_cells'     : sp.n_cells,
        'n_genes'     : sp.n_genes,
        'n_cell_types': (
            sp.cell_meta[config.cell_type_col].nunique()
            if config.cell_type_col in sp.cell_meta.columns else '?'
        ),
    }

    return {
        # Metadata
        'config'           : config,
        'sp_summary'       : sp_summary,

        # Database
        'lr_pairs'         : lr_expressed,
        'lr_pairs_df'      : lr_pairs_to_dataframe(lr_expressed),

        # Full layer results
        'layer1_results'   : layer1_results,
        'layer2_results'   : layer2_results,
        'layer3_results'   : layer3_results,

        # Shortcuts for common access patterns
        'significant_pairs': layer1_results.get('significant_pairs', pd.DataFrame()),
        'scores'           : layer2_results.get('scores', pd.DataFrame()),
        'contrastive'      : layer3_results.get('contrastive', pd.DataFrame()),
        'programs'         : layer3_results.get('programs', pd.DataFrame()),

        # Diagnostics
        'timing'           : timing,
    }


def _print_final_summary(results: dict) -> None:
    """Print compact completion summary."""
    timing = results['timing']
    print(f"\n{'═' * 60}")
    print("CCC Pipeline Complete")

    sig = results.get('significant_pairs', pd.DataFrame())
    scores = results.get('scores', pd.DataFrame())
    K = results.get('layer3_results', {}).get('K', 0)

    print(f"  Layer 1: {len(sig)} significant combinations")
    print(f"  Layer 2: {len(scores)} combinations scored")
    print(f"  Layer 3: {K} NMF programs")
    print(f"  Total time: {timing.get('total', 0):.1f}s  "
          f"(L1: {timing.get('layer1', 0):.1f}s  "
          f"L2: {timing.get('layer2', 0):.1f}s  "
          f"L3: {timing.get('layer3', 0):.1f}s)")
    print("═" * 60)
