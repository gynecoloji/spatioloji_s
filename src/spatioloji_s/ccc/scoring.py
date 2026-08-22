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

Autocrine (optional)
--------------------
When ``include_autocrine=True``, self-edges (i, i) are added for every
cell co-expressing a pair's ligand and receptor.  Autocrine edges:
  - use weight = 1.0 (no spatial decay)
  - apply to all lr_types (juxtacrine, secreted, ecm)
  - are tagged ``interaction_mode='autocrine'`` so they remain
    separable from same-type paracrine in aggregation and significance.

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
    include_autocrine: bool = True,
) -> pd.DataFrame:
    """
    Score every graph edge for each LR pair.

    For each directed edge (i -> j) and LR pair::

        score = sqrt(L_i * R_j) * w_ij

    where L_i and R_j are expression values (min across subunits for
    complexes) and w_ij is a signaling-type-specific weight.

    When ``include_autocrine=True``, self-edges (i, i) are additionally
    scored for every cell co-expressing the pair's ligand and receptor,
    using weight = 1.0.  Autocrine rows are tagged
    ``interaction_mode='autocrine'``; intercellular rows are tagged
    ``interaction_mode='paracrine'``.

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
        include_autocrine: If True, also score self-edges (i, i) for
            every cell co-expressing both the ligand and the receptor
            of a given LR pair.  Applies to all lr_types.  Default
            True — pass ``include_autocrine=False`` to recover
            intercellular-only behavior.

    Returns:
        DataFrame with columns: sender, receiver, lr_name, lr_type,
        score, weight, sender_type, receiver_type, interaction_mode.

    Raises:
        ValueError: If a required graph is missing for a signaling type.

    Example:
        >>> from spatioloji_s.ccc.database import LRPair
        >>> pairs = [LRPair("L1_R1", "gene_L1", "gene_R1", "test", "secreted")]
        >>> edges = score_edges(sp, pairs, graph_diffusible=graph)
        >>> edges = score_edges(sp, pairs, graph_diffusible=graph,
        ...                     include_autocrine=True)
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

        if edge_tbl is not None and len(edge_tbl) > 0:
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
                        "interaction_mode": "paracrine",
                    }
                )

        # ── Autocrine self-edges (opt-in) ────────────────────────────
        if include_autocrine:
            auto_records = _score_autocrine_for_pair(pair, expr_df, cell_type_map)
            records.extend(auto_records)

    if not records:
        return pd.DataFrame(
            columns=[
                "sender",
                "receiver",
                "lr_name",
                "lr_type",
                "score",
                "weight",
                "sender_type",
                "receiver_type",
                "interaction_mode",
            ]
        )

    return pd.DataFrame(records)


def _score_autocrine_for_pair(
    pair: LRPair,
    expr_df: pd.DataFrame,
    cell_type_map: dict,
) -> list[dict]:
    """Build autocrine self-edge records for a single LR pair.

    For every cell co-expressing all ligand subunits AND all receptor
    subunits at non-zero level, emit an (i, i) edge with weight = 1.0
    and ``interaction_mode='autocrine'``.
    """
    cell_ids = np.asarray(expr_df.index)
    if len(cell_ids) == 0:
        return []

    l_vals = _get_complex_expr(expr_df, pair.ligand_genes, cell_ids)
    r_vals = _get_complex_expr(expr_df, pair.receptor_genes, cell_ids)

    coexpr_mask = (l_vals > 0.0) & (r_vals > 0.0)
    if not coexpr_mask.any():
        return []

    sel_cells = cell_ids[coexpr_mask]
    sel_l = np.maximum(l_vals[coexpr_mask], 0.0)
    sel_r = np.maximum(r_vals[coexpr_mask], 0.0)
    scores = np.sqrt(sel_l * sel_r)  # weight = 1.0

    out: list[dict] = []
    for k in range(len(sel_cells)):
        ctype = cell_type_map.get(sel_cells[k], "unknown")
        out.append(
            {
                "sender": sel_cells[k],
                "receiver": sel_cells[k],
                "lr_name": pair.lr_name,
                "lr_type": pair.lr_type,
                "score": float(scores[k]),
                "weight": 1.0,
                "sender_type": ctype,
                "receiver_type": ctype,
                "interaction_mode": "autocrine",
            }
        )
    return out


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

        **summary_df** — one row per
        (lr_name, sender_type, receiver_type, interaction_mode) with
        columns: lr_name, sender_type, receiver_type, interaction_mode,
        mean_score, sum_score, n_edges.  When the input ``edge_df`` has
        no ``interaction_mode`` column (legacy callers), all rows are
        treated as ``'paracrine'``.

        **cell_scores_df** — indexed by cell_id with columns
        ``{lr_name}_sender`` and ``{lr_name}_receiver`` for each LR pair.

    Example:
        >>> summary, cell_scores = aggregate_scores(edge_df, sp)
    """
    if edge_df.empty:
        cols = [
            "lr_name",
            "sender_type",
            "receiver_type",
            "interaction_mode",
            "mean_score",
            "sum_score",
            "n_edges",
        ]
        summary = pd.DataFrame(columns=cols)
        cell_scores = pd.DataFrame(index=sp.cell_index)
        return summary, cell_scores

    # Backfill interaction_mode for callers that produced edge_df with an
    # older score_edges (no autocrine support) — treat every row as paracrine.
    if "interaction_mode" not in edge_df.columns:
        edge_df = edge_df.assign(interaction_mode="paracrine")

    # ── Type-pair summary ────────────────────────────────────────────────
    grouped = edge_df.groupby(["lr_name", "sender_type", "receiver_type", "interaction_mode"])["score"]
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
        z_score (analytical only).  Autocrine summary rows (where the
        input has ``interaction_mode == 'autocrine'``) are tested
        against a null built only from paracrine edges, so paracrine
        and autocrine p-values are directly comparable.

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

    # Backfill interaction_mode for legacy callers (treat all as paracrine)
    if "interaction_mode" not in edge_df.columns:
        edge_df = edge_df.assign(interaction_mode="paracrine")
    if "interaction_mode" not in summary_df.columns:
        summary_df = summary_df.assign(interaction_mode="paracrine")

    cell_types = sp.cell_meta[group_col]
    N = len(cell_types)
    type_counts = cell_types.value_counts().to_dict()

    result = summary_df.copy()

    if method == "analytical":
        # Paracrine-only edges as the null distribution
        paracrine_edges = edge_df[edge_df["interaction_mode"] == "paracrine"]
        result = _analytical_test(result, paracrine_edges, N, type_counts, norm)
    elif method == "permutation":
        # Permutation handles paracrine and autocrine separately
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
    """Analytical z-score significance test.

    For paracrine rows the pair-coincidence probability is
    ``n_s * n_r / (N * (N - 1))`` (ordered pair of distinct cells).
    For autocrine rows it is ``n_s / N`` (single cell of type s).
    The null is built from ``edge_df``, which the caller has already
    restricted to paracrine edges.
    """
    n_rows = len(result)
    pvals = np.ones(n_rows)
    zscores = np.zeros(n_rows)

    if n_rows == 0:
        result = result.copy()
        result["z_score"] = zscores
        result["pvalue"] = pvals
        return result

    modes = (
        result["interaction_mode"].to_numpy()
        if "interaction_mode" in result.columns
        else np.array(["paracrine"] * n_rows)
    )

    # Per-LR null statistics, computed once over the whole edge table rather
    # than re-filtering it for every summary row. The naive form is
    # O(rows x edges) and dominates runtime once the edge table reaches
    # millions of rows; this is O(edges) + O(rows) for identical output.
    if edge_df is None or len(edge_df) == 0:
        result = result.copy()
        result["z_score"] = zscores
        result["pvalue"] = pvals
        return result

    grouped = edge_df.groupby("lr_name")["score"]
    total_by_lr = grouped.sum().to_dict()
    count_by_lr = grouped.size().to_dict()
    # mean of squared edge scores, per LR pair
    meansq_by_lr = (
        edge_df.assign(_sq=edge_df["score"].to_numpy() ** 2)
        .groupby("lr_name")["_sq"]
        .mean()
        .to_dict()
    )

    lr_names = result["lr_name"].to_numpy()
    senders = result["sender_type"].to_numpy()
    receivers = result["receiver_type"].to_numpy()
    obs_sums = result["sum_score"].to_numpy(dtype=float)

    for i in range(n_rows):
        lr = lr_names[i]
        n_s = type_counts.get(senders[i], 0)
        n_r = type_counts.get(receivers[i], 0)

        if n_s == 0 or n_r == 0 or N < 2:
            continue

        # Null edge scores for this LR pair (paracrine only)
        total_score = total_by_lr.get(lr)
        if total_score is None:
            continue
        mean_sq = meansq_by_lr[lr]
        n_total_edges = count_by_lr[lr]

        # Expected sum under null (random label assignment)
        if modes[i] == "autocrine":
            # Self-edge: a single cell drawn at random has type s with
            # prob n_s / N; sender_type == receiver_type by construction
            # so n_r is ignored.
            p_pair = n_s / N if N > 0 else 0.0
        else:
            p_pair = (n_s * n_r) / (N * (N - 1)) if N > 1 else 0.0
        e_sum = p_pair * total_score

        # Variance of sum under null
        # Var = p*(1-p) * sum(s_e^2) for independent edge assignment
        var_sum = p_pair * (1.0 - p_pair) * mean_sq * n_total_edges

        if var_sum < 1e-30:
            continue

        z = (obs_sums[i] - e_sum) / np.sqrt(var_sum)
        zscores[i] = z
        pvals[i] = float(norm.sf(z))  # one-sided

    result = result.copy()
    result["z_score"] = zscores
    result["pvalue"] = pvals
    return result


def _permutation_test(result, edge_df, sp, group_col, n_subsample, n_permutations, seed):
    """Permutation-based significance test.

    Vectorised 2026-08-22. The previous form re-scanned the whole edge table
    once per result row *inside* every permutation, costing
    O(n_permutations x n_rows x n_edges): on a LymphNode ROI that was ~39,555 s
    at n_permutations=100, against 7.3 s for squidpy doing the same 100
    permutations, and six ROIs would have needed ~660 h.

    The statistic being computed is only a group-sum of edge scores keyed by
    (lr_name, interaction_mode, sender_type, receiver_type), so the entire inner
    double loop collapses to one integer-keyed ``np.bincount`` per permutation --
    O(n_permutations x n_edges), every group updated in a single pass.

    **The RNG call sequence is preserved deliberately**, so this returns p-values
    identical to the naive form rather than merely equivalent ones.
    ``RandomState.shuffle`` is Fisher-Yates and draws once per position
    regardless of dtype, so shuffling the integer-coded label array consumes the
    same stream as shuffling the original object array. That premise is asserted
    directly in tests/unit/test_ccc_permutation_equivalence.py; if it ever fails,
    the equivalence claim fails with it.
    """
    rng = np.random.RandomState(seed)
    cell_types = sp.cell_meta[group_col].copy()
    all_cells = np.asarray(sp.cell_index)
    N = len(all_cells)

    # Stratified subsample -- unchanged, and it must stay first so the rng
    # stream reaching the permutation loop is the same as before.
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

    mask = edge_df["sender"].isin(sub_cells) & edge_df["receiver"].isin(sub_cells)
    sub_edge = edge_df[mask].copy()

    if sub_edge.empty:
        result = result.copy()
        result["pvalue"] = 1.0
        return result

    if "interaction_mode" not in sub_edge.columns:
        sub_edge["interaction_mode"] = "paracrine"

    cells_in_edges = np.unique(
        np.concatenate([sub_edge["sender"].values, sub_edge["receiver"].values])
    )
    sub_types = cell_types.reindex(cells_in_edges)
    has_mode = "interaction_mode" in result.columns

    # ---- integer coding, computed once ------------------------------------
    # Labels -> codes. Shuffling `type_codes` is what keeps the rng in lockstep
    # with the old `rng.shuffle(sub_types.values.copy())`.
    uniq_types, type_codes = np.unique(sub_types.values.astype(object),
                                       return_inverse=True)
    n_types = len(uniq_types)
    type_to_code = {t: i for i, t in enumerate(uniq_types)}

    cell_pos = {c: i for i, c in enumerate(cells_in_edges)}
    send_idx = sub_edge["sender"].map(cell_pos).to_numpy(dtype=np.int64)
    recv_idx = sub_edge["receiver"].map(cell_pos).to_numpy(dtype=np.int64)

    lm_codes, lm_uniques = pd.factorize(
        pd.Series(list(zip(sub_edge["lr_name"], sub_edge["interaction_mode"])))
    )
    lm_codes = lm_codes.astype(np.int64)
    lm_to_code = {lm: i for i, lm in enumerate(lm_uniques)}
    n_flat = max(len(lm_uniques), 1) * n_types * n_types
    score = sub_edge["score"].to_numpy(dtype=np.float64)

    def _group_sums(labels_by_cell):
        st = labels_by_cell[send_idx]
        rt = labels_by_cell[recv_idx]
        flat = (lm_codes * n_types + st) * n_types + rt
        return np.bincount(flat, weights=score, minlength=n_flat)

    obs_sums = _group_sums(type_codes)

    # ---- where each result row lands in the flat array ---------------------
    # A row whose (lr, mode) or whose cell type never appears in the edge table
    # has no edges at all. The naive form gave it observed 0 and permuted 0 every
    # iteration, so 0 >= 0 counted every time and p was 1.0; -1 reproduces that.
    flat_of_row = np.full(len(result), -1, dtype=np.int64)
    for i, row in enumerate(result.itertuples(index=False)):
        mode = row.interaction_mode if has_mode else "paracrine"
        lm = lm_to_code.get((row.lr_name, mode))
        st = type_to_code.get(row.sender_type)
        rt = type_to_code.get(row.receiver_type)
        if lm is not None and st is not None and rt is not None:
            flat_of_row[i] = (lm * n_types + st) * n_types + rt

    # ---- permutations ------------------------------------------------------
    counts = np.zeros(n_flat, dtype=np.int64)
    for _ in range(n_permutations):
        # The copy MUST be inside the loop. Fisher-Yates applied to an
        # already-shuffled array consumes the same rng draws but yields a
        # different permutation, so hoisting this out silently changes the null
        # -- it shifted p-values by up to 8 permutation counts before the
        # equivalence test caught it.
        shuffled = type_codes.copy()
        rng.shuffle(shuffled)
        counts += _group_sums(shuffled) >= obs_sums

    result = result.copy()
    present = flat_of_row >= 0
    row_counts = np.where(present, counts[np.clip(flat_of_row, 0, None)],
                          n_permutations)
    result["pvalue"] = (row_counts + 1) / (n_permutations + 1)

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
