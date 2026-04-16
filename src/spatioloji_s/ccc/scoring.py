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
    cell_type_map = sp.cell_meta[group_col].to_dict()

    # ── Pre-compute edge tables per signaling type ───────────────────────
    jux_edges = _build_juxtacrine_edge_table(graph_juxtacrine) if has_juxtacrine else None
    if has_diffusible:
        diff_edges, sigma_s, sigma_e = _build_diffusible_edge_table(graph_diffusible, sigma_secreted, sigma_ecm)
    else:
        diff_edges, sigma_s, sigma_e = None, None, None

    # ── Score each LR pair ───────────────────────────────────────────────
    records: list[dict] = []

    for pair in lr_pairs:
        # Pick edge table and weights
        if pair.lr_type == "juxtacrine":
            edge_tbl = jux_edges
        else:
            edge_tbl = diff_edges

        if edge_tbl is None or len(edge_tbl) == 0:
            continue

        senders = edge_tbl["sender"].values
        receivers = edge_tbl["receiver"].values

        # Compute weights
        if pair.lr_type == "juxtacrine":
            weights = edge_tbl["weight"].values
        elif pair.lr_type == "secreted":
            weights = np.exp(-edge_tbl["distance"].values / sigma_s) if sigma_s > 0 else np.ones(len(edge_tbl))
        else:  # ecm
            weights = np.exp(-edge_tbl["distance"].values / sigma_e) if sigma_e > 0 else np.ones(len(edge_tbl))

        # Expression: min across subunits for complexes
        l_vals = _get_complex_expr(expr_df, pair.ligand_genes, senders)
        r_vals = _get_complex_expr(expr_df, pair.receptor_genes, receivers)

        scores = np.sqrt(np.maximum(l_vals, 0.0) * np.maximum(r_vals, 0.0)) * weights

        for k in range(len(senders)):
            records.append(
                {
                    "sender": senders[k],
                    "receiver": receivers[k],
                    "lr_name": pair.lr_name,
                    "lr_type": pair.lr_type,
                    "score": scores[k],
                    "weight": weights[k],
                    "sender_type": cell_type_map.get(senders[k], "unknown"),
                    "receiver_type": cell_type_map.get(receivers[k], "unknown"),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["sender", "receiver", "lr_name", "lr_type", "score", "weight", "sender_type", "receiver_type"]
        )

    return pd.DataFrame(records)


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

    # ── Per-cell scores ──────────────────────────────────────────────────
    lr_names = edge_df["lr_name"].unique()
    cell_scores = pd.DataFrame(index=sp.cell_index)

    for lr in lr_names:
        sub = edge_df[edge_df["lr_name"] == lr]
        sender_agg = sub.groupby("sender")["score"].sum()
        receiver_agg = sub.groupby("receiver")["score"].sum()
        cell_scores[f"{lr}_sender"] = sender_agg.reindex(sp.cell_index, fill_value=0.0).values
        cell_scores[f"{lr}_receiver"] = receiver_agg.reindex(sp.cell_index, fill_value=0.0).values

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
    distances = np.array([dist_csr[r, c] for r, c in zip(rows_idx, cols_idx, strict=True)], dtype=np.float64)

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
    pvals = np.ones(len(result))
    zscores = np.zeros(len(result))

    for i, row in result.iterrows():
        lr = row["lr_name"]
        st = row["sender_type"]
        rt = row["receiver_type"]
        obs_sum = row["sum_score"]

        n_s = type_counts.get(st, 0)
        n_r = type_counts.get(rt, 0)

        if n_s == 0 or n_r == 0 or N < 2:
            pvals[i] = 1.0
            continue

        # All edge scores for this LR pair (across all cell types)
        lr_edges = edge_df[edge_df["lr_name"] == lr]
        if lr_edges.empty:
            pvals[i] = 1.0
            continue

        all_scores = lr_edges["score"].values
        total_score = all_scores.sum()
        mean_sq = (all_scores**2).mean()
        n_total_edges = len(all_scores)

        # Expected sum under null (random label assignment)
        p_pair = (n_s * n_r) / (N * (N - 1)) if N > 1 else 0.0
        e_sum = p_pair * total_score

        # Variance of sum under null
        # Var = p*(1-p) * sum(s_e^2) for independent edge assignment
        var_sum = p_pair * (1.0 - p_pair) * mean_sq * n_total_edges

        if var_sum < 1e-30:
            pvals[i] = 1.0
            zscores[i] = 0.0
            continue

        z = (obs_sum - e_sum) / np.sqrt(var_sum)
        zscores[i] = z
        pvals[i] = float(norm.sf(z))  # one-sided

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

    # Cell IDs in subsampled edges
    cells_in_edges = np.unique(np.concatenate([sub_edge["sender"].values, sub_edge["receiver"].values]))
    sub_types = cell_types.reindex(cells_in_edges)

    # Observed sums per (lr, sender_type, receiver_type)
    obs_key_to_sum = {}
    for _, row in result.iterrows():
        key = (row["lr_name"], row["sender_type"], row["receiver_type"])
        sub_lr = sub_edge[sub_edge["lr_name"] == row["lr_name"]]
        mask_st = sub_lr["sender"].map(lambda c, st=row["sender_type"]: sub_types.get(c) == st)
        mask_rt = sub_lr["receiver"].map(lambda c, rt=row["receiver_type"]: sub_types.get(c) == rt)
        obs_key_to_sum[key] = sub_lr.loc[mask_st & mask_rt, "score"].sum()

    # Permutations
    perm_counts = {k: 0 for k in obs_key_to_sum}

    for _ in range(n_permutations):
        shuffled = sub_types.values.copy()
        rng.shuffle(shuffled)
        shuffled_map = dict(zip(cells_in_edges, shuffled, strict=True))

        perm_sender_types = sub_edge["sender"].map(shuffled_map)
        perm_receiver_types = sub_edge["receiver"].map(shuffled_map)

        for key in obs_key_to_sum:
            lr, st, rt = key
            lr_mask = sub_edge["lr_name"] == lr
            mask_st = perm_sender_types == st
            mask_rt = perm_receiver_types == rt
            perm_sum = sub_edge.loc[lr_mask & mask_st & mask_rt, "score"].sum()
            if perm_sum >= obs_key_to_sum[key]:
                perm_counts[key] += 1

    result = result.copy()
    pvals = []
    for _, row in result.iterrows():
        key = (row["lr_name"], row["sender_type"], row["receiver_type"])
        p = (perm_counts.get(key, n_permutations) + 1) / (n_permutations + 1)
        pvals.append(p)
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
