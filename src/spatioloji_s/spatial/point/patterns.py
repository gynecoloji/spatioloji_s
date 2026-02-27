"""
patterns.py - Spatial pattern detection

Detects spatial structure in gene expression and cell type distributions,
including autocorrelation, hotspots, and co-occurrence patterns.
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


def morans_i(
    sj: 'spatioloji',
    graph: PointSpatialGraph,
    values: Union[str, np.ndarray],
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
        spatioloji object.
    graph : PointSpatialGraph
        Pre-built spatial graph.
    values : str or np.ndarray
        Gene name (looked up in expression) or pre-computed array.
    local : bool
        If True, compute local Moran's I (LISA) per cell.
        If False, compute global Moran's I.
    n_permutations : int
        Permutations for significance testing.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    dict (global) with keys:
        'I'       : float, Moran's I statistic
        'expected': float, expected I under randomness (-1/(N-1))
        'zscore'  : float, z-score from permutation test
        'pvalue'  : float, pseudo p-value
    
    pd.DataFrame (local) with columns:
        'local_i'  : per-cell local Moran's I
        'zscore'   : per-cell z-score
        'pvalue'   : per-cell pseudo p-value
        'spot_type': 'HH', 'LL', 'HL', 'LH', or 'ns'
    """
    rng = np.random.default_rng(seed)
    
    # Resolve values
    x = _resolve_values(sj, values)
    n = len(x)
    
    # Row-normalize the weight matrix
    W = _row_normalize(graph.adjacency)
    
    # Mean-center
    z = x - x.mean()
    denom = np.sum(z ** 2)
    
    if denom == 0:
        print("  ⚠ All values identical — Moran's I undefined")
        if local:
            return pd.DataFrame(
                {'local_i': 0, 'zscore': 0, 'pvalue': 1, 'spot_type': 'ns'},
                index=graph.cell_ids,
            )
        return {'I': 0, 'expected': -1 / (n - 1), 'zscore': 0, 'pvalue': 1}
    
    if not local:
        # --- Global Moran's I ---
        # I = (N / W_sum) * (z' W z) / (z' z)
        # With row-normalized W, simplifies to: N * (z' W z) / (W_sum * z'z)
        Wz = W.dot(z)
        observed_I = n * z.dot(Wz) / (W.sum() * denom)
        expected_I = -1 / (n - 1)
        
        # Permutation test
        perm_Is = np.empty(n_permutations)
        for i in range(n_permutations):
            z_perm = rng.permutation(z)
            Wz_perm = W.dot(z_perm)
            perm_Is[i] = n * z_perm.dot(Wz_perm) / (W.sum() * np.sum(z_perm ** 2))
        
        perm_mean = perm_Is.mean()
        perm_std = perm_Is.std()
        zscore = (observed_I - perm_mean) / perm_std if perm_std > 0 else 0
        pvalue = (np.sum(np.abs(perm_Is - perm_mean)
                         >= np.abs(observed_I - perm_mean)) + 1) / (n_permutations + 1)
        
        label = values if isinstance(values, str) else 'custom'
        print(f"  ✓ Global Moran's I ({label}): I={observed_I:.4f}, "
              f"z={zscore:.2f}, p={pvalue:.4f}")
        
        return {
            'I': observed_I,
            'expected': expected_I,
            'zscore': zscore,
            'pvalue': pvalue,
        }
    
    else:
        # --- Local Moran's I (LISA) ---
        # Ii = (zi / var) * sum_j(wij * zj)
        Wz = np.array(W.dot(z)).flatten()
        variance = denom / n
        local_i = (z / variance) * Wz
        
        # Permutation for each cell
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
        
        # Per-cell p-values
        pvalues = np.zeros(n)
        for c in range(n):
            extreme = np.sum(
                np.abs(perm_local[:, c] - perm_mean[c])
                >= np.abs(local_i[c] - perm_mean[c])
            )
            pvalues[c] = (extreme + 1) / (n_permutations + 1)
        
        # Classify spots: HH, LL, HL, LH
        spot_type = _classify_lisa(z, Wz, pvalues, alpha=0.05)
        
        result = pd.DataFrame({
            'local_i': local_i,
            'zscore': zscores,
            'pvalue': pvalues,
            'spot_type': spot_type,
        }, index=graph.cell_ids)
        
        counts = result['spot_type'].value_counts()
        print(f"  ✓ Local Moran's I (LISA): "
              f"HH={counts.get('HH', 0)}, LL={counts.get('LL', 0)}, "
              f"HL={counts.get('HL', 0)}, LH={counts.get('LH', 0)}, "
              f"ns={counts.get('ns', 0)}")
        
        return result


def _classify_lisa(
    z: np.ndarray,
    Wz: np.ndarray,
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Classify LISA results into spot types.
    
    HH: high value, high neighbors (hot cluster)
    LL: low value, low neighbors (cold cluster)
    HL: high value, low neighbors (high outlier)
    LH: low value, high neighbors (low outlier)
    ns: not significant
    """
    spot_type = np.full(len(z), 'ns', dtype=object)
    sig = pvalues < alpha
    
    spot_type[(z > 0) & (Wz > 0) & sig] = 'HH'
    spot_type[(z < 0) & (Wz < 0) & sig] = 'LL'
    spot_type[(z > 0) & (Wz < 0) & sig] = 'HL'
    spot_type[(z < 0) & (Wz > 0) & sig] = 'LH'
    
    return spot_type


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
) -> np.ndarray:
    """Resolve values: gene name -> expression array, or pass through."""
    if isinstance(values, str):
        # Look up gene expression
        if values in sj.cell_meta.columns:
            return sj.cell_meta[values].values.astype(np.float64)
        else:
            expr = sj.get_expression(gene_names=values)
            return expr.flatten().astype(np.float64)
    return np.asarray(values, dtype=np.float64)

def getis_ord_gi(
    sj: 'spatioloji',
    graph: PointSpatialGraph,
    values: Union[str, np.ndarray],
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
        spatioloji object.
    graph : PointSpatialGraph
        Pre-built spatial graph.
    values : str or np.ndarray
        Gene name or pre-computed array.
    star : bool
        If True, use Gi* (includes self-weight). Recommended.
    permutation_pvalue : bool
        If True, compute p-values via permutation instead of
        analytical z-score. More robust for non-normal data.
    n_permutations : int
        Number of permutations (if permutation_pvalue=True).
    seed : int, optional
        Random seed.
    store : bool
        If True, add results to sj.cell_meta.
    
    Returns
    -------
    pd.DataFrame with columns:
        'gi'       : Gi(*) statistic per cell
        'zscore'   : z-score (analytical or permutation-based)
        'pvalue'   : p-value
        'spot_type': 'hotspot', 'coldspot', or 'ns'
    """
    rng = np.random.default_rng(seed)
    
    x = _resolve_values(sj, values)
    n = len(x)
    
    # Prepare weight matrix
    W = graph.adjacency.astype(np.float64).tocsr()
    
    if star:
        # Gi*: add self-connections (diagonal = 1)
        W = W + sparse.eye(n, format='csr')
    
    # Global statistics
    x_mean = x.mean()
    x_var = x.var()  # population variance
    S = np.sqrt(x_var)
    
    if S == 0:
        print("  ⚠ All values identical — Gi* undefined")
        return pd.DataFrame(
            {'gi': 0, 'zscore': 0, 'pvalue': 1, 'spot_type': 'ns'},
            index=graph.cell_ids,
        )
    
    # Per-cell weight sums
    Wi = np.array(W.sum(axis=1)).flatten()       # Σⱼ wᵢⱼ
    Wi2 = np.array(W.multiply(W).sum(axis=1)).flatten()  # Σⱼ wᵢⱼ²
    
    # Numerator: Σⱼ wᵢⱼ xⱼ  -  x̄ Σⱼ wᵢⱼ
    Wx = np.array(W.dot(x)).flatten()
    numerator = Wx - x_mean * Wi
    
    # Denominator: S * sqrt((N * Σⱼ wᵢⱼ² - (Σⱼ wᵢⱼ)²) / (N - 1))
    denominator = S * np.sqrt((n * Wi2 - Wi ** 2) / (n - 1))
    
    # Gi statistic (already a z-score under analytical null)
    with np.errstate(divide='ignore', invalid='ignore'):
        gi = numerator / denominator
        gi = np.nan_to_num(gi, nan=0.0)
    
    # P-values
    if not permutation_pvalue:
        # Analytical: Gi is approximately standard normal
        from scipy.stats import norm
        pvalues = 2 * norm.sf(np.abs(gi))  # two-sided
        zscores = gi
    else:
        # Permutation-based
        perm_gi = np.empty((n_permutations, n))
        for i in range(n_permutations):
            x_perm = rng.permutation(x)
            Wx_perm = np.array(W.dot(x_perm)).flatten()
            num_perm = Wx_perm - x_perm.mean() * Wi
            S_perm = np.sqrt(x_perm.var())
            denom_perm = S_perm * np.sqrt((n * Wi2 - Wi ** 2) / (n - 1))
            with np.errstate(divide='ignore', invalid='ignore'):
                perm_gi[i] = np.nan_to_num(num_perm / denom_perm, nan=0.0)
        
        perm_mean = perm_gi.mean(axis=0)
        perm_std = perm_gi.std(axis=0)
        
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
    
    # Classify spots
    spot_type = np.full(n, 'ns', dtype=object)
    sig = pvalues < 0.05
    spot_type[(gi > 0) & sig] = 'hotspot'
    spot_type[(gi < 0) & sig] = 'coldspot'
    
    result = pd.DataFrame({
        'gi': gi,
        'zscore': zscores,
        'pvalue': pvalues,
        'spot_type': spot_type,
    }, index=graph.cell_ids)
    
    # Store
    if store:
        label = values if isinstance(values, str) else 'custom'
        sj.cell_meta[f'gi_{label}'] = gi
        sj.cell_meta[f'gi_{label}_spot'] = spot_type
    
    # Report
    n_hot = (spot_type == 'hotspot').sum()
    n_cold = (spot_type == 'coldspot').sum()
    label = values if isinstance(values, str) else 'custom'
    print(f"  ✓ Getis-Ord Gi{'*' if star else ''} ({label}): "
          f"{n_hot} hotspot cells, {n_cold} coldspot cells")
    
    return result

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
    
    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    cell_type_col : str
        Column in sj.cell_meta with cell type labels.
    coord_type : str
        'global' or 'local'.
    intervals : np.ndarray, optional
        Distance bin edges. If None, auto-computed from data.
    n_intervals : int
        Number of distance bins (if intervals is None).
    max_distance : float, optional
        Maximum distance to consider. If None, uses 25th percentile
        of all pairwise distances (keeps computation feasible).
    n_permutations : int
        Number of label shuffles for null distribution.
    seed : int, optional
        Random seed.
    type_pairs : list of tuples, optional
        Specific (typeA, typeB) pairs to compute. If None, all pairs.
    
    Returns
    -------
    dict with keys:
        'score'     : pd.DataFrame, co-occurrence score per pair per interval
                      (observed/expected ratio). Columns = interval midpoints.
        'observed'  : pd.DataFrame, observed pair counts
        'expected'  : pd.DataFrame, mean expected pair counts
        'intervals' : np.ndarray, distance bin edges
        'midpoints' : np.ndarray, distance bin centers
    """
    from scipy.spatial import KDTree
    
    rng = np.random.default_rng(seed)
    
    if cell_type_col not in sj.cell_meta.columns:
        raise ValueError(f"'{cell_type_col}' not found in cell_meta")
    
    coords = sj.get_spatial_coords(coord_type=coord_type)
    labels = sj.cell_meta[cell_type_col].values
    types = np.unique(labels)
    n_cells = len(labels)
    
    # Build KDTree
    tree = KDTree(coords)
    
    # Auto-compute distance intervals
    if intervals is None:
        if max_distance is None:
            # Sample pairwise distances to estimate range
            sample_idx = rng.choice(n_cells, size=min(1000, n_cells), replace=False)
            sample_dists = tree.query(coords[sample_idx], k=20)[0][:, 1:]
            max_distance = np.percentile(sample_dists, 75)
        
        intervals = np.linspace(0, max_distance, n_intervals + 1)
    
    midpoints = (intervals[:-1] + intervals[1:]) / 2
    n_bins = len(midpoints)
    
    # Determine type pairs
    if type_pairs is None:
        type_pairs = [
            (types[i], types[j])
            for i in range(len(types))
            for j in range(i, len(types))
        ]
    
    # --- Helper: count pairs per distance bin for given labels ---
    def _count_pairs(labs):
        pair_counts = {pair: np.zeros(n_bins) for pair in type_pairs}
        
        for bin_idx in range(n_bins):
            r_inner = intervals[bin_idx]
            r_outer = intervals[bin_idx + 1]
            
            # Find all pairs within r_outer
            pairs_outer = tree.query_pairs(r_outer)
            if r_inner > 0:
                pairs_inner = tree.query_pairs(r_inner)
                pairs_in_bin = pairs_outer - pairs_inner
            else:
                pairs_in_bin = pairs_outer
            
            # Count by type pair
            for i, j in pairs_in_bin:
                ti, tj = labs[i], labs[j]
                # Check both orders since pairs are unordered
                if (ti, tj) in pair_counts:
                    pair_counts[(ti, tj)] += np.zeros(n_bins)
                    pair_counts[(ti, tj)][bin_idx] += 1
                elif (tj, ti) in pair_counts:
                    pair_counts[(tj, ti)][bin_idx] += 1
        
        return pair_counts
    
    # Optimized: precompute pairs per bin, then just look up labels
    print(f"  Precomputing distance pairs across {n_bins} bins...")
    bin_pairs = []
    for bin_idx in range(n_bins):
        r_inner = intervals[bin_idx]
        r_outer = intervals[bin_idx + 1]
        pairs_outer = tree.query_pairs(r_outer)
        if r_inner > 0:
            pairs_inner = tree.query_pairs(r_inner)
            pairs_in_bin = pairs_outer - pairs_inner
        else:
            pairs_in_bin = pairs_outer
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
    
    # Observed counts
    observed = _count_from_precomputed(labels)
    
    # Permutations
    print(f"  Running {n_permutations} permutations...")
    perm_counts = {pair: np.zeros((n_permutations, n_bins)) for pair in type_pairs}
    
    for p in range(n_permutations):
        shuffled = rng.permutation(labels)
        perm_result = _count_from_precomputed(shuffled)
        for pair in type_pairs:
            perm_counts[pair][p] = perm_result[pair]
    
    # Compute scores: observed / expected
    expected = {pair: perm_counts[pair].mean(axis=0) for pair in type_pairs}
    
    scores = {}
    for pair in type_pairs:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = observed[pair] / expected[pair]
            ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0)
        scores[pair] = ratio
    
    # Build DataFrames
    pair_labels = [f"{a}-{b}" for a, b in type_pairs]
    
    score_df = pd.DataFrame(
        [scores[p] for p in type_pairs],
        index=pair_labels,
        columns=[f"{m:.1f}" for m in midpoints],
    )
    observed_df = pd.DataFrame(
        [observed[p] for p in type_pairs],
        index=pair_labels,
        columns=[f"{m:.1f}" for m in midpoints],
    )
    expected_df = pd.DataFrame(
        [expected[p] for p in type_pairs],
        index=pair_labels,
        columns=[f"{m:.1f}" for m in midpoints],
    )
    
    # Report
    print(f"  ✓ Co-occurrence: {len(type_pairs)} pairs × {n_bins} distance bins")
    for pair, pair_label in zip(type_pairs, pair_labels):
        mean_score = scores[pair].mean()
        trend = "attract" if mean_score > 1.1 else "avoid" if mean_score < 0.9 else "neutral"
        print(f"    {pair_label}: mean score={mean_score:.2f} ({trend})")
    
    return {
        'score': score_df,
        'observed': observed_df,
        'expected': expected_df,
        'intervals': intervals,
        'midpoints': midpoints,
    }

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
    
    Ranks all genes by spatial autocorrelation using an analytical
    (fast) approximation. Use for discovery, then validate top hits
    with full permutation tests via morans_i().
    
    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    graph : PointSpatialGraph
        Pre-built spatial graph.
    genes : list of str, optional
        Genes to test. If None, tests all genes.
    layer : str, optional
        Expression layer to use. If None, uses raw expression.
    n_top : int
        Number of top genes to report in summary.
    fdr_threshold : float
        FDR cutoff for significance.
    method : str
        'morans_i' (only option currently; extensible).
    
    Returns
    -------
    pd.DataFrame with columns:
        'I'       : Moran's I statistic
        'expected': expected I under randomness
        'variance': analytical variance
        'zscore'  : z-score (analytical)
        'pvalue'  : p-value (analytical, two-sided)
        'fdr'     : Benjamini-Hochberg adjusted p-value
    Sorted by I descending (most spatially variable first).
    """
    from scipy.stats import norm
    
    if method != 'morans_i':
        raise ValueError(f"Unknown method: {method}. Use 'morans_i'.")
    
    # Resolve genes
    if genes is None:
        genes = sj.gene_index.tolist()
    
    n_genes = len(genes)
    n_cells = sj.n_cells
    
    print(f"  Screening {n_genes} genes for spatial variability...")
    
    # Get expression matrix (cells x genes)
    if layer is not None:
        expr = sj.get_layer(layer)
        gene_indices = sj._get_gene_indices(genes)
        if sparse.issparse(expr):
            X = expr[:, gene_indices].toarray()
        else:
            X = expr[:, gene_indices]
    else:
        X = sj.get_expression(gene_names=genes)  # (n_cells, n_genes)
    
    X = X.astype(np.float64)
    
    # Row-normalize weight matrix
    W = _row_normalize(graph.adjacency)
    W_sum = W.sum()
    
    # Mean-center each gene
    means = X.mean(axis=0)
    Z = X - means  # (n_cells, n_genes)
    
    # Denominator: sum of squared deviations per gene
    denom = (Z ** 2).sum(axis=0)  # (n_genes,)
    
    # Numerator: for each gene g, z_g.T @ W @ z_g
    # Efficient: compute W @ Z first, then element-wise multiply with Z
    WZ = W.dot(Z)  # (n_cells, n_genes) — sparse @ dense
    numerator = (Z * WZ).sum(axis=0)  # (n_genes,) — diagonal of Z.T @ W @ Z
    
    # Global Moran's I per gene
    with np.errstate(divide='ignore', invalid='ignore'):
        I_values = (n_cells / W_sum) * numerator / denom
        I_values = np.nan_to_num(I_values, nan=0.0)
    
    # Analytical expected value and variance
    expected_I = -1.0 / (n_cells - 1)
    
    # Analytical variance (under normality assumption)
    # Var(I) = (N^2 * S1 - N * S2 + 3 * W_sum^2) / 
    #          ((N^2 - 1) * W_sum^2)  -  expected_I^2
    # where S1 = 0.5 * sum((wij + wji)^2), S2 = sum((wi. + w.i)^2)
    W_sym = W + W.T
    S1 = 0.5 * W_sym.multiply(W_sym).sum()
    
    row_sums = np.array(W.sum(axis=1)).flatten()
    col_sums = np.array(W.sum(axis=0)).flatten()
    S2 = np.sum((row_sums + col_sums) ** 2)
    
    variance_I = (
        (n_cells ** 2 * S1 - n_cells * S2 + 3 * W_sum ** 2)
        / ((n_cells ** 2 - 1) * W_sum ** 2)
        - expected_I ** 2
    )
    variance_I = max(variance_I, 1e-10)  # avoid zero
    
    # Z-scores and p-values
    zscores = (I_values - expected_I) / np.sqrt(variance_I)
    pvalues = 2 * norm.sf(np.abs(zscores))  # two-sided
    
    # FDR correction (Benjamini-Hochberg)
    fdr = _benjamini_hochberg(pvalues)
    
    # Build result DataFrame
    result = pd.DataFrame({
        'I': I_values,
        'expected': expected_I,
        'variance': variance_I,
        'zscore': zscores,
        'pvalue': pvalues,
        'fdr': fdr,
    }, index=genes)
    
    result = result.sort_values('I', ascending=False)
    
    # Report
    n_sig = (result['fdr'] < fdr_threshold).sum()
    print(f"  ✓ Spatially variable genes: {n_sig}/{n_genes} significant "
          f"(FDR < {fdr_threshold})")
    print(f"    Top {min(n_top, n_sig)} genes:")
    
    top = result.head(min(n_top, 10))
    for gene_name, row in top.iterrows():
        print(f"      {gene_name:>15s}: I={row['I']:.4f}, "
              f"z={row['zscore']:.1f}, FDR={row['fdr']:.1e}")
    
    return result


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    
    Parameters
    ----------
    pvalues : np.ndarray
        Raw p-values.
    
    Returns
    -------
    np.ndarray
        Adjusted p-values (FDR).
    """
    n = len(pvalues)
    ranked_idx = np.argsort(pvalues)
    ranked_pvalues = pvalues[ranked_idx]
    
    # BH formula: adjusted_p[i] = p[i] * n / rank[i]
    fdr = np.zeros(n)
    fdr[ranked_idx] = ranked_pvalues * n / np.arange(1, n + 1)
    
    # Enforce monotonicity (from largest to smallest rank)
    fdr[ranked_idx] = np.minimum.accumulate(
        fdr[ranked_idx][::-1]
    )[::-1]
    
    # Clip to [0, 1]
    fdr = np.clip(fdr, 0, 1)
    
    return fdr