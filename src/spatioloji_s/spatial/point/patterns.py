"""
patterns.py - Spatial pattern detection

Detects spatial structure in gene expression and cell type distributions,
including autocorrelation, hotspots, and co-occurrence patterns.

Layer support
-------------
All gene-related functions (morans_i, getis_ord_gi, spatially_variable_genes)
accept a `layer` parameter. When provided, gene expression is pulled from
sp.layers[layer] instead of the raw expression matrix. This lets you run
spatial analysis on normalized, log-transformed, or scaled data.

Example
-------
>>> sp.add_layer('lognorm', lognorm_matrix)
>>> graph = spoint.build_knn_graph(sp, k=10)
>>>
>>> # Moran's I on log-normalized expression
>>> spoint.morans_i(sp, graph, values='MyGene', layer='lognorm')
>>>
>>> # SVG screening on normalized data
>>> spoint.spatially_variable_genes(sp, graph, layer='lognorm')
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from typing import Optional, List, Union
import numpy as np
import pandas as pd
from scipy import sparse

from .graph import PointSpatialGraph


# ========== Shared helpers ==========

def _row_normalize(W: sparse.spmatrix) -> sparse.csr_matrix:
    """Row-normalize a sparse weight matrix."""
    W = W.astype(np.float64).tocsr()
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # avoid division by zero
    diag_inv = sparse.diags(1.0 / row_sums)
    return diag_inv.dot(W)


def _resolve_values(
    sj: 'spatioloji',
    values: Union[str, np.ndarray],
    layer: Optional[str] = None,
) -> np.ndarray:
    """
    Resolve values to a 1D numpy array for spatial analysis.

    Priority order when `values` is a string:
    1. If layer is given → pull gene from sp.layers[layer]
    2. If gene found in cell_meta → use that column directly
    3. Otherwise → pull gene from raw expression (sp.expression)

    Parameters
    ----------
    sj : spatioloji
    values : str or np.ndarray
        Gene name or pre-computed array.
    layer : str, optional
        Layer name to use for gene lookup (e.g. 'lognorm', 'scaled').

    Returns
    -------
    np.ndarray (1D, float64)
    """
    if isinstance(values, np.ndarray):
        return values.flatten().astype(np.float64)

    # String: gene name or cell_meta column
    if layer is not None:
        # Pull from specified layer
        layer_data = sj.get_layer(layer)   # (n_cells × n_genes)
        gene_indices = sj._get_gene_indices([values])
        if len(gene_indices) == 0:
            raise ValueError(f"Gene '{values}' not found in gene index")
        if sparse.issparse(layer_data):
            return np.array(layer_data[:, gene_indices[0]].todense()).flatten().astype(np.float64)
        return layer_data[:, gene_indices[0]].flatten().astype(np.float64)

    # No layer: check cell_meta first, then raw expression
    if values in sj.cell_meta.columns:
        return sj.cell_meta[values].values.astype(np.float64)

    expr = sj.get_expression(gene_names=values)
    return expr.flatten().astype(np.float64)


def _classify_lisa(
    z: np.ndarray,
    Wz: np.ndarray,
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Classify LISA results into spot types.

    HH: high value, high neighbors  → hot cluster
    LL: low value,  low neighbors   → cold cluster
    HL: high value, low neighbors   → high outlier
    LH: low value,  high neighbors  → low outlier
    ns: not significant
    """
    spot_type = np.full(len(z), 'ns', dtype=object)
    sig = pvalues < alpha
    spot_type[(z > 0) & (Wz > 0) & sig] = 'HH'
    spot_type[(z < 0) & (Wz < 0) & sig] = 'LL'
    spot_type[(z > 0) & (Wz < 0) & sig] = 'HL'
    spot_type[(z < 0) & (Wz > 0) & sig] = 'LH'
    return spot_type


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    ranked_idx = np.argsort(pvalues)
    ranked_pvalues = pvalues[ranked_idx]
    fdr = np.zeros(n)
    fdr[ranked_idx] = ranked_pvalues * n / np.arange(1, n + 1)
    fdr[ranked_idx] = np.minimum.accumulate(fdr[ranked_idx][::-1])[::-1]
    return np.clip(fdr, 0, 1)


# ========== morans_i ==========

def morans_i(
    sj: 'spatioloji',
    graph: PointSpatialGraph,
    values: Union[str, np.ndarray],
    layer: Optional[str] = None,
    local: bool = False,
    n_permutations: int = 999,
    seed: Optional[int] = None,
) -> Union[dict, pd.DataFrame]:
    """
    Compute Moran's I spatial autocorrelation.

    Global: single statistic testing whether values are spatially
    clustered across the entire tissue.
    Local (LISA): per-cell statistic identifying WHERE clustering occurs.

    Parameters
    ----------
    sj : spatioloji
    graph : PointSpatialGraph
    values : str or np.ndarray
        Gene name or pre-computed array.
        If str and layer is given → pulled from sp.layers[layer].
        If str and no layer → pulled from cell_meta or raw expression.
    layer : str, optional
        Layer to use for gene expression lookup.
        Example: 'lognorm', 'scaled', 'normalized'.
        If None, uses raw expression (or cell_meta if gene is a column there).
    local : bool
        If True, compute local Moran's I (LISA) per cell.
    n_permutations : int
    seed : int, optional

    Returns
    -------
    dict (global) with keys: I, expected, zscore, pvalue
    pd.DataFrame (local) with columns: local_i, zscore, pvalue, spot_type

    Examples
    --------
    >>> # Global, raw expression
    >>> spoint.morans_i(sp, graph, values='CD8A')
    >>>
    >>> # Global, log-normalized layer
    >>> spoint.morans_i(sp, graph, values='CD8A', layer='lognorm')
    >>>
    >>> # Local LISA on normalized data
    >>> spoint.morans_i(sp, graph, values='CD8A', layer='lognorm', local=True)
    >>>
    >>> # From cell_meta column (e.g. total_counts) — layer ignored
    >>> spoint.morans_i(sp, graph, values='total_counts')
    """
    rng = np.random.default_rng(seed)

    x = _resolve_values(sj, values, layer=layer)
    n = len(x)

    W = _row_normalize(graph.adjacency)
    z = x - x.mean()
    denom = np.sum(z ** 2)

    label = values if isinstance(values, str) else 'custom'
    layer_str = f", layer='{layer}'" if layer else ''

    if denom == 0:
        print(f"  ⚠ All values identical — Moran's I undefined ({label}{layer_str})")
        if local:
            return pd.DataFrame(
                {'local_i': 0, 'zscore': 0, 'pvalue': 1, 'spot_type': 'ns'},
                index=graph.cell_ids,
            )
        return {'I': 0, 'expected': -1 / (n - 1), 'zscore': 0, 'pvalue': 1}

    if not local:
        # ── Global Moran's I ──────────────────────────────────────────────────
        Wz = W.dot(z)
        observed_I = n * z.dot(Wz) / (W.sum() * denom)
        expected_I = -1 / (n - 1)

        perm_Is = np.empty(n_permutations)
        for i in range(n_permutations):
            z_perm = rng.permutation(z)
            Wz_perm = W.dot(z_perm)
            perm_Is[i] = n * z_perm.dot(Wz_perm) / (W.sum() * np.sum(z_perm ** 2))

        perm_mean = perm_Is.mean()
        perm_std = perm_Is.std()
        zscore = (observed_I - perm_mean) / perm_std if perm_std > 0 else 0
        pvalue = (
            np.sum(np.abs(perm_Is - perm_mean) >= np.abs(observed_I - perm_mean)) + 1
        ) / (n_permutations + 1)

        print(f"  ✓ Global Moran's I ({label}{layer_str}): "
              f"I={observed_I:.4f}, z={zscore:.2f}, p={pvalue:.4f}")

        return {'I': observed_I, 'expected': expected_I, 'zscore': zscore, 'pvalue': pvalue}

    else:
        # ── Local Moran's I (LISA) ────────────────────────────────────────────
        Wz = np.array(W.dot(z)).flatten()
        variance = denom / n
        local_i = (z / variance) * Wz

        perm_local = np.empty((n_permutations, n))
        for i in range(n_permutations):
            z_perm = rng.permutation(z)
            Wz_perm = np.array(W.dot(z_perm)).flatten()
            perm_local[i] = (z / variance) * Wz_perm

        perm_mean = perm_local.mean(axis=0)
        perm_std = perm_local.std(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            zscores = (local_i - perm_mean) / perm_std
            zscores = np.nan_to_num(zscores, nan=0.0)

        pvalues = np.zeros(n)
        for c in range(n):
            extreme = np.sum(
                np.abs(perm_local[:, c] - perm_mean[c])
                >= np.abs(local_i[c] - perm_mean[c])
            )
            pvalues[c] = (extreme + 1) / (n_permutations + 1)

        spot_type = _classify_lisa(z, Wz, pvalues, alpha=0.05)

        result = pd.DataFrame({
            'local_i': local_i,
            'zscore': zscores,
            'pvalue': pvalues,
            'spot_type': spot_type,
        }, index=graph.cell_ids)

        counts = result['spot_type'].value_counts()
        print(f"  ✓ Local Moran's I / LISA ({label}{layer_str}): "
              f"HH={counts.get('HH', 0)}, LL={counts.get('LL', 0)}, "
              f"HL={counts.get('HL', 0)}, LH={counts.get('LH', 0)}, "
              f"ns={counts.get('ns', 0)}")

        return result


# ========== getis_ord_gi ==========

def getis_ord_gi(
    sj: 'spatioloji',
    graph: PointSpatialGraph,
    values: Union[str, np.ndarray],
    layer: Optional[str] = None,
    star: bool = True,
    permutation_pvalue: bool = False,
    n_permutations: int = 999,
    seed: Optional[int] = None,
    store: bool = True,
) -> pd.DataFrame:
    """
    Compute Getis-Ord Gi(*) hotspot statistic.

    Identifies spatial clusters of HIGH values (hotspots) and
    LOW values (coldspots). Unlike Moran's I, Gi* distinguishes
    the direction of clustering.

    Parameters
    ----------
    sj : spatioloji
    graph : PointSpatialGraph
    values : str or np.ndarray
        Gene name or pre-computed array.
    layer : str, optional
        Layer to use for gene expression lookup.
        Example: 'lognorm', 'scaled'.
        If None, uses raw expression or cell_meta column.
    star : bool
        If True, use Gi* (includes self-weight). Recommended.
    permutation_pvalue : bool
        If True, compute p-values via permutation (more robust for
        non-normal data). If False, uses analytical z-score.
    n_permutations : int
    seed : int, optional
    store : bool
        If True, adds gi_{label} and gi_{label}_spot to sj.cell_meta.

    Returns
    -------
    pd.DataFrame with columns: gi, zscore, pvalue, spot_type

    Examples
    --------
    >>> # Hotspots on raw expression
    >>> spoint.getis_ord_gi(sp, graph, values='FOXP3')
    >>>
    >>> # Hotspots on log-normalized layer
    >>> spoint.getis_ord_gi(sp, graph, values='FOXP3', layer='lognorm')
    >>>
    >>> # On cell_meta numeric column (layer ignored)
    >>> spoint.getis_ord_gi(sp, graph, values='total_counts')
    """
    rng = np.random.default_rng(seed)

    x = _resolve_values(sj, values, layer=layer)
    n = len(x)

    label = values if isinstance(values, str) else 'custom'
    layer_str = f", layer='{layer}'" if layer else ''

    W = graph.adjacency.astype(np.float64).tocsr()
    if star:
        W = W + sparse.eye(n, format='csr')

    x_mean = x.mean()
    x_var = x.var()
    S = np.sqrt(x_var)

    if S == 0:
        print(f"  ⚠ All values identical — Gi* undefined ({label}{layer_str})")
        return pd.DataFrame(
            {'gi': 0, 'zscore': 0, 'pvalue': 1, 'spot_type': 'ns'},
            index=graph.cell_ids,
        )

    Wi  = np.array(W.sum(axis=1)).flatten()
    Wi2 = np.array(W.multiply(W).sum(axis=1)).flatten()
    Wx  = np.array(W.dot(x)).flatten()

    numerator   = Wx - x_mean * Wi
    denominator = S * np.sqrt((n * Wi2 - Wi ** 2) / (n - 1))

    with np.errstate(divide='ignore', invalid='ignore'):
        gi = numerator / denominator
        gi = np.nan_to_num(gi, nan=0.0)

    if not permutation_pvalue:
        from scipy.stats import norm
        pvalues = 2 * norm.sf(np.abs(gi))
        zscores = gi
    else:
        perm_gi = np.empty((n_permutations, n))
        for i in range(n_permutations):
            x_perm = rng.permutation(x)
            Wx_perm = np.array(W.dot(x_perm)).flatten()
            num_perm  = Wx_perm - x_perm.mean() * Wi
            S_perm    = np.sqrt(x_perm.var())
            denom_perm = S_perm * np.sqrt((n * Wi2 - Wi ** 2) / (n - 1))
            with np.errstate(divide='ignore', invalid='ignore'):
                perm_gi[i] = np.nan_to_num(num_perm / denom_perm, nan=0.0)

        perm_mean = perm_gi.mean(axis=0)
        perm_std  = perm_gi.std(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            zscores = (gi - perm_mean) / perm_std
            zscores = np.nan_to_num(zscores, nan=0.0)

        pvalues = np.zeros(n)
        for c in range(n):
            extreme = np.sum(
                np.abs(perm_gi[:, c] - perm_mean[c])
                >= np.abs(gi[c] - perm_mean[c])
            )
            pvalues[c] = (extreme + 1) / (n_permutations + 1)

    spot_type = np.full(n, 'ns', dtype=object)
    sig = pvalues < 0.05
    spot_type[(gi > 0) & sig] = 'hotspot'
    spot_type[(gi < 0) & sig] = 'coldspot'

    result = pd.DataFrame({
        'gi': gi, 'zscore': zscores, 'pvalue': pvalues, 'spot_type': spot_type,
    }, index=graph.cell_ids)

    if store:
        sj.cell_meta[f'gi_{label}'] = gi
        sj.cell_meta[f'gi_{label}_spot'] = spot_type

    n_hot  = (spot_type == 'hotspot').sum()
    n_cold = (spot_type == 'coldspot').sum()
    print(f"  ✓ Getis-Ord Gi{'*' if star else ''} ({label}{layer_str}): "
          f"{n_hot} hotspot cells, {n_cold} coldspot cells")

    return result


# ========== spatially_variable_genes ==========

def spatially_variable_genes(
    sj: 'spatioloji',
    graph: PointSpatialGraph,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    n_top: int = 50,
    fdr_threshold: float = 0.05,
    method: str = 'morans_i',
) -> pd.DataFrame:
    """
    Screen genes for spatial variability.

    Ranks all (or selected) genes by spatial autocorrelation using an
    analytical (fast) Moran's I approximation. Use for discovery, then
    validate top hits with full permutation via morans_i().

    Parameters
    ----------
    sj : spatioloji
    graph : PointSpatialGraph
    genes : list of str, optional
        Genes to test. If None, tests all genes in sj.gene_index.
    layer : str, optional
        Layer to use for expression matrix.
        Example: 'lognorm', 'normalized', 'scaled'.
        If None, uses raw expression (sp.expression).
        Recommended: use a normalized layer for fair comparison across genes.
    n_top : int
        Number of top genes to print in summary.
    fdr_threshold : float
        FDR cutoff for significance reporting.
    method : str
        'morans_i' (only option currently).

    Returns
    -------
    pd.DataFrame, indexed by gene name, columns:
        I, expected, variance, zscore, pvalue, fdr
    Sorted by I descending (most spatially variable first).

    Examples
    --------
    >>> # Screen all genes on raw expression
    >>> svg = spoint.spatially_variable_genes(sp, graph)
    >>>
    >>> # Screen on log-normalized layer (recommended)
    >>> svg = spoint.spatially_variable_genes(sp, graph, layer='lognorm')
    >>>
    >>> # Screen a gene subset
    >>> svg = spoint.spatially_variable_genes(
    ...     sp, graph, genes=['CD8A', 'FOXP3', 'MKI67'], layer='lognorm'
    ... )
    """
    from scipy.stats import norm

    if method != 'morans_i':
        raise ValueError(f"Unknown method: {method}. Use 'morans_i'.")

    if genes is None:
        genes = sj.gene_index.tolist()

    n_genes  = len(genes)
    n_cells  = sj.n_cells
    layer_str = f"layer='{layer}'" if layer else 'raw expression'
    print(f"\n[SVG] Screening {n_genes} genes ({layer_str})...")

    # ── Get expression matrix (cells × genes) ────────────────────────────────
    if layer is not None:
        layer_data = sj.get_layer(layer)     # (n_cells × n_genes_total)
        gene_indices = sj._get_gene_indices(genes)
        if sparse.issparse(layer_data):
            X = layer_data[:, gene_indices].toarray()
        else:
            X = layer_data[:, gene_indices]
    else:
        X = sj.get_expression(gene_names=genes)   # (n_cells, n_genes)

    X = X.astype(np.float64)

    # ── Analytical Moran's I for all genes at once ────────────────────────────
    W     = _row_normalize(graph.adjacency)
    W_sum = W.sum()

    means = X.mean(axis=0)
    Z     = X - means                          # (n_cells, n_genes)
    denom = (Z ** 2).sum(axis=0)              # (n_genes,)

    WZ        = W.dot(Z)                       # (n_cells, n_genes)
    numerator = (Z * WZ).sum(axis=0)          # (n_genes,)

    with np.errstate(divide='ignore', invalid='ignore'):
        I_values = (n_cells / W_sum) * numerator / denom
        I_values = np.nan_to_num(I_values, nan=0.0)

    # ── Analytical variance ───────────────────────────────────────────────────
    expected_I = -1.0 / (n_cells - 1)

    W_sym    = W + W.T
    S1       = 0.5 * W_sym.multiply(W_sym).sum()
    row_sums = np.array(W.sum(axis=1)).flatten()
    col_sums = np.array(W.sum(axis=0)).flatten()
    S2       = np.sum((row_sums + col_sums) ** 2)

    variance_I = (
        (n_cells ** 2 * S1 - n_cells * S2 + 3 * W_sum ** 2)
        / ((n_cells ** 2 - 1) * W_sum ** 2)
        - expected_I ** 2
    )
    variance_I = max(variance_I, 1e-10)

    zscores = (I_values - expected_I) / np.sqrt(variance_I)
    pvalues = 2 * norm.sf(np.abs(zscores))
    fdr     = _benjamini_hochberg(pvalues)

    result = pd.DataFrame({
        'I': I_values, 'expected': expected_I,
        'variance': variance_I, 'zscore': zscores,
        'pvalue': pvalues, 'fdr': fdr,
    }, index=genes).sort_values('I', ascending=False)

    n_sig = (result['fdr'] < fdr_threshold).sum()
    print(f"  ✓ {n_sig}/{n_genes} significant SVGs (FDR < {fdr_threshold})")
    print(f"    Top genes:")
    for gene_name, row in result.head(min(n_top, 10)).iterrows():
        print(f"      {gene_name:>15s}: I={row['I']:.4f}, "
              f"z={row['zscore']:.1f}, FDR={row['fdr']:.1e}")

    return result


# ========== co_occurrence (unchanged — no gene analysis) ==========

def co_occurrence(
    sj: 'spatioloji',
    cell_type_col: str,
    coord_type: str = 'global',
    intervals: Optional[np.ndarray] = None,
    n_intervals: int = 10,
    max_distance: Optional[float] = None,
    n_permutations: int = 100,
    seed: Optional[int] = None,
    type_pairs: Optional[List[tuple]] = None,
) -> dict:
    """
    Compute distance-resolved co-occurrence between cell types.

    For each distance interval, measures whether cell type pairs
    appear together more or less than expected by chance.
    Score > 1 → co-localized, Score < 1 → segregated.

    Parameters
    ----------
    sj : spatioloji
    cell_type_col : str
    coord_type : str
    intervals : np.ndarray, optional
    n_intervals : int
    max_distance : float, optional
    n_permutations : int
    seed : int, optional
    type_pairs : list of tuples, optional

    Returns
    -------
    dict with keys: score, observed, expected, intervals, midpoints
    """
    from scipy.spatial import KDTree

    rng = np.random.default_rng(seed)

    if cell_type_col not in sj.cell_meta.columns:
        raise ValueError(f"'{cell_type_col}' not found in cell_meta")

    coords  = sj.get_spatial_coords(coord_type=coord_type)
    labels  = sj.cell_meta[cell_type_col].values
    types   = np.unique(labels)
    n_cells = len(labels)
    tree    = KDTree(coords)

    if intervals is None:
        if max_distance is None:
            sample_idx = rng.choice(n_cells, size=min(1000, n_cells), replace=False)
            sample_dists = tree.query(coords[sample_idx], k=20)[0][:, 1:]
            max_distance = np.percentile(sample_dists, 75)
        intervals = np.linspace(0, max_distance, n_intervals + 1)

    midpoints = (intervals[:-1] + intervals[1:]) / 2
    n_bins    = len(midpoints)

    if type_pairs is None:
        type_pairs = [
            (types[i], types[j])
            for i in range(len(types))
            for j in range(i, len(types))
        ]

    # Precompute pairs per distance bin
    print(f"  Precomputing distance pairs across {n_bins} bins...")
    bin_pairs = []
    for bin_idx in range(n_bins):
        r_inner = intervals[bin_idx]
        r_outer = intervals[bin_idx + 1]
        pairs_outer = tree.query_pairs(r_outer)
        pairs_in_bin = (pairs_outer - tree.query_pairs(r_inner)
                        if r_inner > 0 else pairs_outer)
        bin_pairs.append(list(pairs_in_bin))

    def _count_from_precomputed(labs):
        pair_counts = {pair: np.zeros(n_bins) for pair in type_pairs}
        for bin_idx in range(n_bins):
            for i, j in bin_pairs[bin_idx]:
                ti, tj = labs[i], labs[j]
                if (ti, tj) in pair_counts:
                    pair_counts[(ti, tj)][bin_idx] += 1
                elif (tj, ti) in pair_counts:
                    pair_counts[(tj, ti)][bin_idx] += 1
        return pair_counts

    observed = _count_from_precomputed(labels)

    print(f"  Running {n_permutations} permutations...")
    perm_counts = {pair: np.zeros((n_permutations, n_bins)) for pair in type_pairs}
    for p in range(n_permutations):
        shuffled = rng.permutation(labels)
        perm_result = _count_from_precomputed(shuffled)
        for pair in type_pairs:
            perm_counts[pair][p] = perm_result[pair]

    expected = {pair: perm_counts[pair].mean(axis=0) for pair in type_pairs}
    scores   = {}
    for pair in type_pairs:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = observed[pair] / expected[pair]
            scores[pair] = np.nan_to_num(ratio, nan=1.0, posinf=1.0)

    pair_labels = [f"{a}-{b}" for a, b in type_pairs]
    cols = [f"{m:.1f}" for m in midpoints]

    score_df    = pd.DataFrame([scores[p]   for p in type_pairs], index=pair_labels, columns=cols)
    observed_df = pd.DataFrame([observed[p] for p in type_pairs], index=pair_labels, columns=cols)
    expected_df = pd.DataFrame([expected[p] for p in type_pairs], index=pair_labels, columns=cols)

    print(f"  ✓ Co-occurrence: {len(type_pairs)} pairs × {n_bins} bins")
    for pair, lbl in zip(type_pairs, pair_labels):
        mean_score = scores[pair].mean()
        trend = "attract" if mean_score > 1.1 else "avoid" if mean_score < 0.9 else "neutral"
        print(f"    {lbl}: mean score={mean_score:.2f} ({trend})")

    return {
        'score': score_df, 'observed': observed_df,
        'expected': expected_df, 'intervals': intervals, 'midpoints': midpoints,
    }