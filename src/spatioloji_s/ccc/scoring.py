"""
scoring.py - Core CCC scoring engine.

Computes ligand-receptor communication scores on spatial graph edges,
aggregates them to cell-type pair summaries, and tests statistical
significance via analytical z-scores or label-permutation.

Signaling-type weighting
------------------------
juxtacrine : weight = fraction_a * fraction_b  (from contact_frac_df)
secreted   : weight = exp(-d / sigma)          (distance decay)
ecm        : same as secreted

Multi-subunit complexes
-----------------------
L_i = min(subunit expressions across ligand genes)
R_j = min(subunit expressions across receptor genes)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    from spatioloji_s.spatial.point.graph import PointSpatialGraph
    from spatioloji_s.spatial.polygon.graph import PolygonSpatialGraph

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .database import LRPair
from ._accel import score_edges_batch, permutation_test_accel

# ── Public API ───────────────────────────────────────────────────────────────


def score_edges(
    sp: spatioloji,
    lr_pairs: list[LRPair],
    graph_juxtacrine: PolygonSpatialGraph | None = None,
    graph_diffusible: PointSpatialGraph | PolygonSpatialGraph | None = None,
    group_col: str = "cell_type",
    layer: str | None = None,
    sigma_secreted: float | None = None,
    sigma_ecm: float | None = None,
) -> pd.DataFrame:
    """
    Score every graph edge for each LR pair.

    For each directed edge (i -> j) and LR pair::

        score = sqrt(L_i * R_j) * w_ij

    where L_i and R_j are expression values (min across subunits for
    complexes) and w_ij is a signaling-type-specific weight.

    Args:
        sp: spatioloji object.
        lr_pairs: List of LRPair objects to score.
        graph_juxtacrine: Polygon spatial graph with ``contact_frac_df``
            attribute.  Required when any LR pair has lr_type == 'juxtacrine'.
        graph_diffusible: Spatial graph with ``adjacency``, ``distances``,
            and ``cell_ids`` (PointSpatialGraph) or ``cell_index``
            (PolygonSpatialGraph).  Required when any LR pair has
            lr_type in ('secreted', 'ecm').
        group_col: Column in ``sp.cell_meta`` for cell type labels.
        layer: Expression layer name.  If None, uses
            ``sp.expression.to_dataframe()``.
        sigma_secreted: Distance decay parameter for secreted pairs.
            If None, estimated as median edge distance.
        sigma_ecm: Distance decay parameter for ECM pairs.
            If None, estimated as median edge distance.

    Returns:
        DataFrame with columns: sender, receiver, lr_name, lr_type,
        score, weight, sender_type, receiver_type.

    Raises:
        ValueError: If a required graph is missing for a signaling type.

    Example:
        >>> from spatioloji_s.ccc.database import LRPair
        >>> pairs = [LRPair("L1_R1", "gene_L1", "gene_R1", "test", "secreted")]
        >>> edges = score_edges(sp, pairs, graph_diffusible=graph)
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"'{group_col}' not found in cell_meta")

    has_juxtacrine = any(p.lr_type == "juxtacrine" for p in lr_pairs)
    has_diffusible = any(p.lr_type in ("secreted", "ecm") for p in lr_pairs)

    if has_juxtacrine and graph_juxtacrine is None:
        raise ValueError(
            "graph_juxtacrine is required for juxtacrine LR pairs. "
            "Build one with build_buffer_graph() + contact_fraction()."
        )
    if has_diffusible and graph_diffusible is None:
        raise ValueError(
            "graph_diffusible is required for secreted/ecm LR pairs. "
            "Build one with build_radius_graph() or build_knn_graph()."
        )

    # ── Expression matrix ────────────────────────────────────────────────
    expr_df = _get_expression_df(sp, layer)
    cell_type_series = sp.cell_meta[group_col]

    # Pre-compute NumPy expression matrix and index maps for fast lookups
    expr_mat = expr_df.values
    cell_id_to_row = {c: i for i, c in enumerate(expr_df.index)}
    gene_to_col = {g: i for i, g in enumerate(expr_df.columns)}

    # ── Pre-compute edge tables per signaling type ───────────────────────
    jux_edges = _build_juxtacrine_edge_table(graph_juxtacrine) if has_juxtacrine else None
    if has_diffusible:
        diff_edges, sigma_s, sigma_e = _build_diffusible_edge_table(graph_diffusible, sigma_secreted, sigma_ecm)
    else:
        diff_edges, sigma_s, sigma_e = None, None, None

    # Pre-compute distance-decay weights for diffusible edges (once, not per pair)
    if diff_edges is not None and len(diff_edges) > 0:
        diff_distances = diff_edges["distance"].values
        diff_weights_secreted = np.exp(-diff_distances / sigma_s) if sigma_s and sigma_s > 0 else np.ones(len(diff_edges))
        diff_weights_ecm = np.exp(-diff_distances / sigma_e) if sigma_e and sigma_e > 0 else np.ones(len(diff_edges))
    else:
        diff_weights_secreted = diff_weights_ecm = None

    # ── Score each LR pair (accelerated batch path) ───────────────────────
    # Separate pairs by edge table to batch within each table type
    jux_pairs = [p for p in lr_pairs if p.lr_type == "juxtacrine"] if jux_edges is not None else []
    diff_sec_pairs = [p for p in lr_pairs if p.lr_type == "secreted"] if diff_edges is not None else []
    diff_ecm_pairs = [p for p in lr_pairs if p.lr_type == "ecm"] if diff_edges is not None else []

    chunks: list[pd.DataFrame] = []

    for batch_pairs, edge_tbl, batch_weights in [
        (jux_pairs, jux_edges, jux_edges["weight"].values if jux_edges is not None and len(jux_pairs) > 0 else None),
        (diff_sec_pairs, diff_edges, diff_weights_secreted),
        (diff_ecm_pairs, diff_edges, diff_weights_ecm),
    ]:
        if not batch_pairs or edge_tbl is None or len(edge_tbl) == 0 or batch_weights is None:
            continue

        senders = edge_tbl["sender"].values
        receivers = edge_tbl["receiver"].values
        n_edges = len(senders)

        # Build integer index arrays for expression lookup
        sender_row_idx = np.array([cell_id_to_row[c] for c in senders], dtype=np.intp)
        receiver_row_idx = np.array([cell_id_to_row[c] for c in receivers], dtype=np.intp)

        # Build per-pair column index lists
        ligand_cols: list[list[int]] = []
        receptor_cols: list[list[int]] = []
        valid_pairs: list[LRPair] = []
        for pair in batch_pairs:
            lcols = [gene_to_col[g] for g in pair.ligand_genes if g in gene_to_col]
            rcols = [gene_to_col[g] for g in pair.receptor_genes if g in gene_to_col]
            ligand_cols.append(lcols)
            receptor_cols.append(rcols)
            valid_pairs.append(pair)

        if not valid_pairs:
            continue

        # Batch score all pairs at once (C++ or Python fallback)
        flat_scores = score_edges_batch(
            expr_mat, sender_row_idx, receiver_row_idx, batch_weights,
            ligand_cols, receptor_cols,
        )

        # Cell type lookup (once per edge table, reused for all pairs)
        sender_types = cell_type_series.reindex(senders).values
        receiver_types = cell_type_series.reindex(receivers).values

        # Split flat_scores into per-pair DataFrames
        for i, pair in enumerate(valid_pairs):
            offset = i * n_edges
            pair_scores = flat_scores[offset:offset + n_edges]

            chunks.append(
                pd.DataFrame(
                    {
                        "sender": senders,
                        "receiver": receivers,
                        "lr_name": pair.lr_name,
                        "lr_type": pair.lr_type,
                        "score": pair_scores,
                        "weight": batch_weights,
                        "sender_type": sender_types,
                        "receiver_type": receiver_types,
                    }
                )
            )

    if not chunks:
        return pd.DataFrame(
            columns=["sender", "receiver", "lr_name", "lr_type", "score", "weight", "sender_type", "receiver_type"]
        )

    return pd.concat(chunks, ignore_index=True)


def aggregate_scores(
    edge_df: pd.DataFrame,
    sp: spatioloji,
    group_col: str = "cell_type",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate edge-level scores to type-pair summaries and per-cell scores.

    Args:
        edge_df: Output of :func:`score_edges`.
        sp: spatioloji object (used only for cell index).
        group_col: Column in ``sp.cell_meta`` for cell type labels.

    Returns:
        Tuple of (summary_df, cell_scores_df).

        **summary_df** — one row per (lr_name, sender_type, receiver_type)
        with columns: lr_name, sender_type, receiver_type, mean_score,
        sum_score, n_edges.

        **cell_scores_df** — indexed by cell_id with columns
        ``{lr_name}_sender`` and ``{lr_name}_receiver`` for each LR pair.

    Example:
        >>> summary, cell_scores = aggregate_scores(edge_df, sp)
    """
    if edge_df.empty:
        cols = ["lr_name", "sender_type", "receiver_type", "mean_score", "sum_score", "n_edges"]
        summary = pd.DataFrame(columns=cols)
        cell_scores = pd.DataFrame(index=sp.cell_index)
        return summary, cell_scores

    # ── Type-pair summary ────────────────────────────────────────────────
    grouped = edge_df.groupby(["lr_name", "sender_type", "receiver_type"])["score"]
    summary = pd.DataFrame(
        {
            "mean_score": grouped.mean(),
            "sum_score": grouped.sum(),
            "n_edges": grouped.count(),
        }
    ).reset_index()

    # ── Per-cell scores (single groupby + pivot, no per-LR loop) ────────
    sender_agg = edge_df.groupby(["lr_name", "sender"])["score"].sum()
    receiver_agg = edge_df.groupby(["lr_name", "receiver"])["score"].sum()

    # Unstack LR names into columns: index=cell_id, columns=lr_name
    sender_wide = sender_agg.unstack(level=0, fill_value=0.0)
    receiver_wide = receiver_agg.unstack(level=0, fill_value=0.0)

    # Reindex to full cell set and rename columns
    sender_wide = sender_wide.reindex(sp.cell_index, fill_value=0.0)
    receiver_wide = receiver_wide.reindex(sp.cell_index, fill_value=0.0)

    sender_wide.columns = [f"{lr}_sender" for lr in sender_wide.columns]
    receiver_wide.columns = [f"{lr}_receiver" for lr in receiver_wide.columns]

    cell_scores = pd.concat([sender_wide, receiver_wide], axis=1)
    cell_scores.index = sp.cell_index

    return summary, cell_scores


def test_significance(
    summary_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    sp: spatioloji,
    group_col: str = "cell_type",
    method: str = "analytical",
    n_subsample: int = 10000,
    n_permutations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Test whether cell-type pair communication scores are significant.

    Args:
        summary_df: Output of :func:`aggregate_scores` (first element).
        edge_df: Output of :func:`score_edges`.
        sp: spatioloji object.
        group_col: Column in ``sp.cell_meta`` for cell type labels.
        method: ``'analytical'`` (z-score) or ``'permutation'``.
        n_subsample: Maximum cells for permutation (stratified subsample).
        n_permutations: Number of permutations.
        seed: Random seed for reproducibility.
        alpha: Significance threshold (used only for reporting).

    Returns:
        Updated summary_df with added columns: pvalue, fdr, and
        z_score (analytical only).

    Example:
        >>> result = test_significance(summary, edges, sp, method="analytical")
        >>> sig = result[result["fdr"] < 0.05]
    """
    try:
        from scipy.stats import norm
    except ImportError as err:
        raise ImportError("scipy is required for significance testing") from err

    if summary_df.empty:
        summary_df = summary_df.copy()
        summary_df["pvalue"] = pd.Series(dtype=float)
        summary_df["fdr"] = pd.Series(dtype=float)
        if method == "analytical":
            summary_df["z_score"] = pd.Series(dtype=float)
        return summary_df

    cell_types = sp.cell_meta[group_col]
    N = len(cell_types)
    type_counts = cell_types.value_counts().to_dict()

    result = summary_df.copy()

    if method == "analytical":
        result = _analytical_test(result, edge_df, N, type_counts, norm)
    elif method == "permutation":
        result = _permutation_test(result, edge_df, sp, group_col, n_subsample, n_permutations, seed)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'analytical' or 'permutation'.")

    # ── BH FDR correction ────────────────────────────────────────────────
    result["fdr"] = _benjamini_hochberg(result["pvalue"].values)

    return result


# ── Private helpers ──────────────────────────────────────────────────────────


def _get_expression_df(sp: spatioloji, layer: str | None) -> pd.DataFrame:
    """Return expression as a DataFrame (cell_id x gene)."""
    if layer is not None:
        mat = sp.get_layer(layer)
        if issparse(mat):
            mat = mat.toarray()
        return pd.DataFrame(mat, index=sp.cell_index, columns=sp.gene_index)
    return sp.expression.to_dataframe()


def _get_complex_expr(expr_df: pd.DataFrame, genes: list[str], cell_ids: np.ndarray) -> np.ndarray:
    """Min expression across subunits for a set of cells (complex handling)."""
    available = [g for g in genes if g in expr_df.columns]
    if not available:
        return np.zeros(len(cell_ids))
    vals = expr_df.loc[cell_ids, available].values
    return vals.min(axis=1) if vals.shape[1] > 1 else vals.ravel()


def _get_complex_expr_fast(
    expr_mat: np.ndarray,
    cell_id_to_row: dict,
    gene_to_col: dict,
    genes: list[str],
    cell_ids: np.ndarray,
) -> np.ndarray:
    """Min expression across subunits using NumPy integer indexing (fast path)."""
    col_idx = [gene_to_col[g] for g in genes if g in gene_to_col]
    if not col_idx:
        return np.zeros(len(cell_ids))
    row_idx = np.array([cell_id_to_row[c] for c in cell_ids], dtype=np.intp)
    vals = expr_mat[np.ix_(row_idx, col_idx)]
    return vals.min(axis=1) if len(col_idx) > 1 else vals.ravel()


def _get_cell_ids_from_graph(graph) -> np.ndarray:
    """Extract cell IDs from either PointSpatialGraph or PolygonSpatialGraph."""
    if hasattr(graph, "cell_ids"):
        return np.asarray(graph.cell_ids)
    if hasattr(graph, "cell_index"):
        return np.asarray(graph.cell_index)
    raise ValueError("Graph object has neither 'cell_ids' nor 'cell_index' attribute.")


def _build_juxtacrine_edge_table(graph) -> pd.DataFrame:
    """Build edge table from juxtacrine graph's contact_frac_df."""
    if not hasattr(graph, "contact_frac_df") or graph.contact_frac_df is None:
        raise ValueError("graph_juxtacrine must have contact_frac_df. Call contact_fraction(sp, graph) first.")
    cdf = graph.contact_frac_df
    # Each row is an undirected edge; create both directions
    forward = pd.DataFrame(
        {
            "sender": cdf["cell_a"].values,
            "receiver": cdf["cell_b"].values,
            "weight": (cdf["fraction_a"] * cdf["fraction_b"]).values,
        }
    )
    backward = pd.DataFrame(
        {
            "sender": cdf["cell_b"].values,
            "receiver": cdf["cell_a"].values,
            "weight": (cdf["fraction_b"] * cdf["fraction_a"]).values,
        }
    )
    return pd.concat([forward, backward], ignore_index=True)


def _build_diffusible_edge_table(graph, sigma_secreted, sigma_ecm):
    """Build edge table from diffusible graph's adjacency + distances."""
    cell_ids = _get_cell_ids_from_graph(graph)
    adj = graph.adjacency
    dist = graph.distances

    rows_idx, cols_idx = adj.nonzero()
    senders = cell_ids[rows_idx]
    receivers = cell_ids[cols_idx]

    dist_csr = dist.tocsr()
    distances = np.asarray(dist_csr[rows_idx, cols_idx], dtype=np.float64).ravel()

    median_dist = float(np.median(distances)) if len(distances) > 0 else 1.0
    if sigma_secreted is None:
        sigma_secreted = median_dist
    if sigma_ecm is None:
        sigma_ecm = median_dist

    edge_tbl = pd.DataFrame(
        {
            "sender": senders,
            "receiver": receivers,
            "distance": distances,
        }
    )
    return edge_tbl, sigma_secreted, sigma_ecm


def _analytical_test(result, edge_df, N, type_counts, norm):
    """Analytical z-score significance test."""
    n_rows = len(result)
    pvals = np.ones(n_rows)
    zscores = np.zeros(n_rows)

    if N < 2:
        result = result.copy()
        result["z_score"] = zscores
        result["pvalue"] = pvals
        return result

    # Pre-compute per-LR stats once (instead of filtering edge_df per row)
    lr_stats = edge_df.groupby("lr_name")["score"].agg(
        total_score="sum",
        mean_sq=lambda x: (x**2).mean(),
        n_total_edges="count",
    )

    # Vectorized lookup of n_s, n_r for each row
    n_s_arr = result["sender_type"].map(type_counts).fillna(0).values.astype(float)
    n_r_arr = result["receiver_type"].map(type_counts).fillna(0).values.astype(float)
    obs_sum_arr = result["sum_score"].values.astype(float)
    lr_names = result["lr_name"].values

    # Vectorized lookup of per-LR stats
    total_scores = lr_stats["total_score"].reindex(lr_names).values
    mean_sqs = lr_stats["mean_sq"].reindex(lr_names).values
    n_total_edges = lr_stats["n_total_edges"].reindex(lr_names).values

    # Compute p_pair, expected sum, and variance for all rows at once
    valid = (n_s_arr > 0) & (n_r_arr > 0) & np.isfinite(total_scores)
    p_pair = np.where(valid, (n_s_arr * n_r_arr) / (N * (N - 1)), 0.0)
    e_sum = p_pair * total_scores
    var_sum = p_pair * (1.0 - p_pair) * mean_sqs * n_total_edges

    # Compute z-scores where variance is sufficient
    computable = valid & (var_sum > 1e-30)
    z = np.where(computable, (obs_sum_arr - e_sum) / np.sqrt(np.maximum(var_sum, 1e-30)), 0.0)
    zscores = z
    pvals = np.where(computable, norm.sf(z), 1.0)

    result = result.copy()
    result["z_score"] = zscores
    result["pvalue"] = pvals
    return result


def _permutation_test(result, edge_df, sp, group_col, n_subsample, n_permutations, seed):
    """Permutation-based significance test."""
    rng = np.random.RandomState(seed)
    cell_types = sp.cell_meta[group_col].copy()
    all_cells = np.asarray(sp.cell_index)
    N = len(all_cells)

    # Stratified subsample
    n_sub = min(n_subsample, N)
    if n_sub < N:
        type_fracs = cell_types.value_counts(normalize=True)
        sub_indices = []
        for ctype, frac in type_fracs.items():
            ct_idx = np.where(cell_types.values == ctype)[0]
            n_take = max(1, int(round(frac * n_sub)))
            chosen = rng.choice(ct_idx, size=min(n_take, len(ct_idx)), replace=False)
            sub_indices.append(chosen)
        sub_indices = np.concatenate(sub_indices)
        sub_cells = set(all_cells[sub_indices])
    else:
        sub_cells = set(all_cells)

    # Subset edge_df to subsampled cells
    mask = edge_df["sender"].isin(sub_cells) & edge_df["receiver"].isin(sub_cells)
    sub_edge = edge_df[mask].copy()

    if sub_edge.empty:
        result = result.copy()
        result["pvalue"] = 1.0
        return result

    # ── Encode everything as integer arrays for fast NumPy operations ────
    cells_in_edges = np.unique(np.concatenate([sub_edge["sender"].values, sub_edge["receiver"].values]))
    sub_types = cell_types.reindex(cells_in_edges)

    # Integer-encode cell IDs
    cell_to_idx = {c: i for i, c in enumerate(cells_in_edges)}
    sender_idx = np.array([cell_to_idx[c] for c in sub_edge["sender"].values], dtype=np.intp)
    receiver_idx = np.array([cell_to_idx[c] for c in sub_edge["receiver"].values], dtype=np.intp)

    # Integer-encode cell types
    unique_types = sub_types.unique()
    type_to_int = {t: i for i, t in enumerate(unique_types)}
    type_array = np.array([type_to_int[t] for t in sub_types.values], dtype=np.intp)

    # Integer-encode LR names
    unique_lrs = sub_edge["lr_name"].unique()
    lr_to_int = {lr: i for i, lr in enumerate(unique_lrs)}
    lr_int = np.array([lr_to_int[lr] for lr in sub_edge["lr_name"].values], dtype=np.intp)

    score_arr = sub_edge["score"].values.astype(np.float64)

    # ── Build key index: each result row → (lr_int, st_int, rt_int) ──────
    n_keys = len(result)
    key_lr = np.empty(n_keys, dtype=np.intp)
    key_st = np.empty(n_keys, dtype=np.intp)
    key_rt = np.empty(n_keys, dtype=np.intp)
    obs_sums = np.empty(n_keys, dtype=np.float64)
    key_valid = np.ones(n_keys, dtype=bool)

    for i, (_, row) in enumerate(result.iterrows()):
        lr_i = lr_to_int.get(row["lr_name"])
        st_i = type_to_int.get(row["sender_type"])
        rt_i = type_to_int.get(row["receiver_type"])
        if lr_i is None or st_i is None or rt_i is None:
            key_valid[i] = False
            key_lr[i] = key_st[i] = key_rt[i] = 0
            obs_sums[i] = 0.0
            continue
        key_lr[i] = lr_i
        key_st[i] = st_i
        key_rt[i] = rt_i

    # Compute observed sums using integer matching
    for i in range(n_keys):
        if not key_valid[i]:
            obs_sums[i] = 0.0
            continue
        mask_i = (lr_int == key_lr[i]) & (type_array[sender_idx] == key_st[i]) & (type_array[receiver_idx] == key_rt[i])
        obs_sums[i] = score_arr[mask_i].sum()

    # ── Permutation (accelerated — C++ or Python fallback) ────────────────
    perm_counts = permutation_test_accel(
        sender_idx=sender_idx,
        receiver_idx=receiver_idx,
        type_array=type_array,
        lr_int=lr_int,
        score_arr=score_arr,
        key_lr=key_lr,
        key_st=key_st,
        key_rt=key_rt,
        key_valid=key_valid,
        obs_sums=obs_sums,
        n_types=len(unique_types),
        n_lrs=len(unique_lrs),
        n_permutations=n_permutations,
        seed=seed,
    )

    result = result.copy()
    pvals = np.where(key_valid, (perm_counts + 1) / (n_permutations + 1), 1.0)
    result["pvalue"] = pvals

    return result


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return pvalues.copy()
    order = np.argsort(pvalues)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    fdr = pvalues * n / ranked
    # Enforce monotonicity (step-down)
    fdr = np.minimum.accumulate(fdr[np.argsort(ranked)[::-1]])[::-1]
    # Restore original order
    out = np.empty(n)
    out[np.argsort(ranked).astype(int)] = fdr
    return np.clip(out, 0.0, 1.0)
