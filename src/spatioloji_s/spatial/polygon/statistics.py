"""
statistics.py - Statistical tests for polygon spatial relationships

Provides formal hypothesis testing on polygon-derived spatial features:
- Contact frequency significance between cell types (permutation)
- Association between cell morphology and cell type / expression
- Spatial autocorrelation significance on polygon graph (multiple metrics)
- Boundary enrichment of specific cell types

All tests use polygon adjacency (not centroid distances) and return
standardized results with test statistic, p-value, and effect size.
Multiple testing correction is applied where appropriate.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

import numpy as np
import pandas as pd

from .graph import PolygonSpatialGraph

# ========== Contact Significance ==========

def contact_permutation_test(sp: spatioloji,
                             graph: PolygonSpatialGraph,
                             group_col: str,
                             n_permutations: int = 1000,
                             seed: int = 42,
                             correction: str = 'bonferroni') -> pd.DataFrame:
    """
    Test significance of contact frequency for all cell type pairs.

    Runs a permutation test for every pair of cell types simultaneously,
    with multiple testing correction.

    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : PolygonSpatialGraph
        Pre-built polygon graph
    group_col : str
        Column in cell_meta defining cell types
    n_permutations : int
        Number of label permutations
    seed : int
        Random seed
    correction : str
        Multiple testing correction: 'bonferroni', 'fdr', or 'none'

    Returns
    -------
    pd.DataFrame
        One row per cell type pair with columns:
        type_a, type_b, observed, expected, z_score,
        p_value, p_adjusted, significant, direction

    Examples
    --------
    >>> graph = build_contact_graph(sp)
    >>> results = contact_permutation_test(sp, graph, group_col='cell_type')
    >>> # Show significant interactions
    >>> results[results['significant']]
    """
    from .neighborhoods import neighborhood_enrichment

    print(f"\n[Statistics] Contact permutation test "
          f"(correction='{correction}')...")

    # Run enrichment analysis (does the heavy permutation work)
    enrichment = neighborhood_enrichment(
        sp, graph, group_col,
        n_permutations=n_permutations,
        seed=seed
    )

    z_scores = enrichment['z_scores']
    p_values = enrichment['p_values']
    observed = enrichment['observed']
    expected = enrichment['expected']
    unique_types = z_scores.index.tolist()

    # Build results table (upper triangle only)
    records = []
    for i, ta in enumerate(unique_types):
        for j, tb in enumerate(unique_types):
            if j < i:
                continue
            records.append({
                'type_a': ta,
                'type_b': tb,
                'observed': observed.loc[ta, tb],
                'expected': expected.loc[ta, tb],
                'z_score': z_scores.loc[ta, tb],
                'p_value': p_values.loc[ta, tb]
            })

    results = pd.DataFrame(records)

    # Multiple testing correction
    raw_pvals = results['p_value'].values
    n_tests = len(raw_pvals)

    if correction == 'bonferroni':
        adjusted = np.minimum(raw_pvals * n_tests, 1.0)
    elif correction == 'fdr':
        # Benjamini-Hochberg
        sorted_idx = np.argsort(raw_pvals)
        adjusted = np.ones(n_tests)
        for rank, idx in enumerate(sorted_idx, 1):
            adjusted[idx] = raw_pvals[idx] * n_tests / rank
        # Enforce monotonicity
        adjusted[sorted_idx] = np.minimum.accumulate(
            adjusted[sorted_idx][::-1]
        )[::-1]
        adjusted = np.minimum(adjusted, 1.0)
    elif correction == 'none':
        adjusted = raw_pvals.copy()
    else:
        raise ValueError(
            f"Unknown correction: '{correction}'. "
            "Use 'bonferroni', 'fdr', or 'none'."
        )

    results['p_adjusted'] = adjusted
    results['significant'] = adjusted < 0.05
    results['direction'] = np.where(
        results['z_score'] > 0, 'enriched', 'depleted'
    )

    results = results.sort_values('p_adjusted')

    # Report
    n_sig = results['significant'].sum()
    print(f"  ✓ {n_sig}/{n_tests} pairs significant (p_adj < 0.05)")
    if n_sig > 0:
        print("\n  Significant pairs:")
        for _, row in results[results['significant']].iterrows():
            print(f"    {row.type_a} – {row.type_b}: "
                  f"z={row.z_score:+.2f}, p_adj={row.p_adjusted:.3f} "
                  f"({row.direction})")

    return results


# ========== Morphology Association ==========

def morphology_association_test(sp: spatioloji,
                                metric: str,
                                group_col: str | None = None,
                                expression_gene: str | None = None,
                                test: str = 'kruskal') -> dict:
    """
    Test association between a morphology metric and cell type or gene expression.

    Two modes:
    - group_col provided: tests if morphology differs across cell types
      (Kruskal-Wallis or ANOVA)
    - expression_gene provided: tests correlation between morphology
      and gene expression (Spearman or Pearson)

    Parameters
    ----------
    sp : spatioloji
        spatioloji object with morph_ columns in cell_meta
    metric : str
        Morphology metric name (without 'morph_' prefix)
    group_col : str, optional
        Column in cell_meta for group comparison
    expression_gene : str, optional
        Gene name for correlation test
    test : str
        For groups: 'kruskal' or 'anova'
        For correlation: 'spearman' or 'pearson'

    Returns
    -------
    dict
        Group mode: metric, group_col, test, statistic, p_value,
            n_groups, n_cells, effect_size_eta_sq, significant
        Correlation mode: metric, gene, test, correlation, p_value,
            n_cells, significant

    Examples
    --------
    >>> # Is circularity associated with cell type?
    >>> result = morphology_association_test(
    ...     sp, metric='circularity', group_col='cell_type'
    ... )
    >>>
    >>> # Does elongation correlate with a gene?
    >>> result = morphology_association_test(
    ...     sp, metric='elongation', expression_gene='VIM', test='spearman'
    ... )
    """
    from scipy import stats as scipy_stats

    morph_col = f'morph_{metric}'

    if morph_col not in sp.cell_meta.columns:
        raise ValueError(
            f"'{morph_col}' not found in cell_meta. "
            "Run compute_morphology(sp) first."
        )

    if group_col is None and expression_gene is None:
        raise ValueError(
            "Provide either group_col or expression_gene"
        )

    morph_values = sp.cell_meta[morph_col]

    # Mode 1: Group comparison
    if group_col is not None:
        print(f"\n[Statistics] Morphology association: "
              f"{metric} ~ {group_col} ({test})...")

        if group_col not in sp.cell_meta.columns:
            raise ValueError(f"'{group_col}' not found in cell_meta")

        data = sp.cell_meta[[group_col, morph_col]].dropna()
        groups = [
            grp[morph_col].values
            for _, grp in data.groupby(group_col)
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            raise ValueError("Need at least 2 groups with data")

        if test == 'kruskal':
            stat, pval = scipy_stats.kruskal(*groups)
        elif test == 'anova':
            stat, pval = scipy_stats.f_oneway(*groups)
        else:
            raise ValueError(f"Use 'kruskal' or 'anova', got '{test}'")

        # Effect size: eta-squared (for KW: H / (n-1))
        n_total = sum(len(g) for g in groups)
        eta_sq = stat / (n_total - 1) if test == 'kruskal' else None

        result = {
            'metric': metric,
            'group_col': group_col,
            'test': test,
            'statistic': float(stat),
            'p_value': float(pval),
            'n_groups': len(groups),
            'n_cells': n_total,
            'effect_size_eta_sq': float(eta_sq) if eta_sq is not None else None,
            'significant': pval < 0.05
        }

        sig = "significant" if pval < 0.05 else "not significant"
        print(f"  ✓ {test}: stat={stat:.2f}, p={pval:.2e} ({sig})")

        return result

    # Mode 2: Correlation with gene expression
    if expression_gene is not None:
        print(f"\n[Statistics] Morphology correlation: "
              f"{metric} ~ {expression_gene} ({test})...")

        if expression_gene not in sp.gene_index:
            raise ValueError(f"Gene '{expression_gene}' not found")

        # Get expression values
        gene_idx = sp._gene_name_to_idx[expression_gene]
        if sp.expression.is_sparse:
            expr_values = np.array(
                sp.expression._data[:, gene_idx].toarray()
            ).flatten()
        else:
            expr_values = sp.expression._data[:, gene_idx]

        expr_series = pd.Series(expr_values, index=sp.cell_index)

        # Combine and drop NaN
        combined = pd.DataFrame({
            'morph': morph_values,
            'expr': expr_series
        }).dropna()

        if len(combined) < 3:
            raise ValueError("Need at least 3 cells with both values")

        if test == 'spearman':
            corr, pval = scipy_stats.spearmanr(
                combined['morph'], combined['expr']
            )
        elif test == 'pearson':
            corr, pval = scipy_stats.pearsonr(
                combined['morph'], combined['expr']
            )
        else:
            raise ValueError(f"Use 'spearman' or 'pearson', got '{test}'")

        result = {
            'metric': metric,
            'gene': expression_gene,
            'test': test,
            'correlation': float(corr),
            'p_value': float(pval),
            'n_cells': len(combined),
            'significant': pval < 0.05
        }

        sig = "significant" if pval < 0.05 else "not significant"
        print(f"  ✓ {test}: r={corr:.3f}, p={pval:.2e} ({sig})")

        return result


# ========== Spatial Autocorrelation (Multi-Metric) ==========

def spatial_autocorrelation_test(sp: spatioloji,
                                 graph: PolygonSpatialGraph,
                                 metrics: list[str] | None = None,
                                 method: str = 'moran',
                                 correction: str = 'fdr') -> pd.DataFrame:
    """
    Test spatial autocorrelation for multiple metrics at once.

    Runs Moran's I or Geary's C for each metric and applies
    multiple testing correction.

    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : PolygonSpatialGraph
        Pre-built polygon graph
    metrics : list of str, optional
        Columns in cell_meta to test. If None, tests all numeric columns.
    method : str
        'moran' or 'geary'
    correction : str
        'bonferroni', 'fdr', or 'none'

    Returns
    -------
    pd.DataFrame
        One row per metric with columns:
        metric, statistic, expected, z_score, p_value,
        p_adjusted, significant, method

    Examples
    --------
    >>> graph = build_contact_graph(sp)
    >>> results = spatial_autocorrelation_test(
    ...     sp, graph, metrics=['total_counts', 'pct_counts_mt']
    ... )
    >>> results[results['significant']]
    """
    from .patterns import spatial_autocorrelation

    print(f"\n[Statistics] Spatial autocorrelation test "
          f"({method}, correction='{correction}')...")

    # Select metrics
    if metrics is None:
        metrics = sp.cell_meta.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        print(f"  → Testing all {len(metrics)} numeric columns")

    # Run for each metric
    records = []
    for metric in metrics:
        if metric not in sp.cell_meta.columns:
            print(f"  ⚠ '{metric}' not found, skipping")
            continue

        # Check if numeric and has variance
        vals = sp.cell_meta[metric].dropna()
        if len(vals) < 3 or vals.var() == 0:
            records.append({
                'metric': metric,
                'statistic': np.nan,
                'expected': np.nan,
                'z_score': np.nan,
                'p_value': np.nan
            })
            continue

        result = spatial_autocorrelation(
            sp, graph, metric, method=method, store=False
        )

        records.append({
            'metric': metric,
            'statistic': result['statistic'],
            'expected': result['expected'],
            'z_score': result['z_score'],
            'p_value': result['p_value']
        })

    results = pd.DataFrame(records)

    if len(results) == 0:
        print("  ⚠ No valid metrics to test")
        return results

    # Multiple testing correction
    raw_pvals = results['p_value'].values
    n_tests = len(raw_pvals)
    valid_p = ~np.isnan(raw_pvals)
    adjusted = np.full(n_tests, np.nan)

    if valid_p.sum() > 0:
        valid_raw = raw_pvals[valid_p]
        n_valid = valid_p.sum()

        if correction == 'bonferroni':
            valid_adj = np.minimum(valid_raw * n_valid, 1.0)
        elif correction == 'fdr':
            # Benjamini-Hochberg
            sorted_idx = np.argsort(valid_raw)
            valid_adj = np.ones(n_valid)
            for rank, idx in enumerate(sorted_idx, 1):
                valid_adj[idx] = valid_raw[idx] * n_valid / rank
            valid_adj[sorted_idx] = np.minimum.accumulate(
                valid_adj[sorted_idx][::-1]
            )[::-1]
            valid_adj = np.minimum(valid_adj, 1.0)
        elif correction == 'none':
            valid_adj = valid_raw.copy()
        else:
            raise ValueError(
                f"Unknown correction: '{correction}'. "
                "Use 'bonferroni', 'fdr', or 'none'."
            )

        adjusted[valid_p] = valid_adj

    results['p_adjusted'] = adjusted
    results['significant'] = results['p_adjusted'] < 0.05
    results['method'] = method

    results = results.sort_values('p_adjusted')

    # Report
    n_sig = results['significant'].sum()
    print(f"\n  ✓ {n_sig}/{n_tests} metrics spatially autocorrelated "
          f"(p_adj < 0.05)")
    if n_sig > 0:
        for _, row in results[results['significant']].head(10).iterrows():
            direction = "clustered" if row.z_score > 0 else "dispersed"
            print(f"    {row.metric}: {method}={row.statistic:.4f}, "
                  f"p_adj={row.p_adjusted:.2e} ({direction})")

    return results


# ========== Boundary Enrichment ==========

def boundary_enrichment_test(sp: spatioloji,
                             graph: PolygonSpatialGraph,
                             group_col: str,
                             min_different_fraction: float = 0.3,
                             n_permutations: int = 1000,
                             seed: int = 42) -> pd.DataFrame:
    """
    Test if specific cell types are enriched at tissue boundaries.

    Identifies boundary cells (cells with many different-type neighbors),
    then tests whether each cell type is over-/under-represented among
    boundary cells compared to random expectation.

    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : PolygonSpatialGraph
        Pre-built polygon graph
    group_col : str
        Column in cell_meta defining cell types
    min_different_fraction : float
        Threshold for boundary cell classification
    n_permutations : int
        Number of permutations
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        One row per cell type with columns:
        cell_type, n_total, n_boundary, observed_fraction,
        expected_fraction, fold_change, z_score, p_value, significant

    Examples
    --------
    >>> graph = build_contact_graph(sp)
    >>> results = boundary_enrichment_test(sp, graph, group_col='cell_type')
    >>> # Which cell types are enriched at boundaries?
    >>> results[results['significant'] & (results['fold_change'] > 1)]
    """
    from .neighborhoods import boundary_cells

    print("\n[Statistics] Boundary enrichment test...")

    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"'{group_col}' not found in cell_meta")

    # Detect boundary cells
    is_boundary = boundary_cells(
        sp, graph, group_col,
        min_different_fraction=min_different_fraction,
        store=False
    )

    labels = sp.cell_meta[group_col]
    unique_types = sorted(labels.dropna().unique())

    # Observed: count of each type among boundary cells
    boundary_labels = labels[is_boundary].dropna()
    total_boundary = len(boundary_labels)

    if total_boundary == 0:
        print("  ⚠ No boundary cells found")
        return pd.DataFrame(
            columns=['cell_type', 'n_total', 'n_boundary',
                     'observed_fraction', 'expected_fraction',
                     'fold_change', 'z_score', 'p_value', 'significant']
        )

    # Count per type
    observed_counts = boundary_labels.value_counts()
    total_per_type = labels.value_counts()
    total_labeled = labels.notna().sum()

    # Permutation null: randomly assign "boundary" status
    print(f"  → Running {n_permutations} permutations...")
    rng = np.random.default_rng(seed)

    labeled_indices = np.where(labels.notna())[0]

    perm_counts = {t: np.zeros(n_permutations) for t in unique_types}

    for p in range(n_permutations):
        # Randomly select same number of "boundary" cells
        perm_boundary = rng.choice(
            labeled_indices, size=total_boundary, replace=False
        )
        perm_labels = labels.iloc[perm_boundary]
        perm_vc = perm_labels.value_counts()

        for t in unique_types:
            perm_counts[t][p] = perm_vc.get(t, 0)

    # Build results
    records = []
    for t in unique_types:
        n_total = int(total_per_type.get(t, 0))
        n_boundary = int(observed_counts.get(t, 0))

        obs_frac = n_boundary / total_boundary if total_boundary > 0 else 0
        exp_frac = n_total / total_labeled if total_labeled > 0 else 0

        # Permutation stats
        perm_vals = perm_counts[t]
        perm_mean = perm_vals.mean()
        perm_std = perm_vals.std()

        if perm_std > 0:
            z_score = (n_boundary - perm_mean) / perm_std
        else:
            z_score = 0.0

        # Two-sided p-value
        n_extreme = np.sum(
            np.abs(perm_vals - perm_mean) >= np.abs(n_boundary - perm_mean)
        )
        p_value = (n_extreme + 1) / (n_permutations + 1)

        fold_change = obs_frac / exp_frac if exp_frac > 0 else np.inf

        records.append({
            'cell_type': t,
            'n_total': n_total,
            'n_boundary': n_boundary,
            'observed_fraction': obs_frac,
            'expected_fraction': exp_frac,
            'fold_change': fold_change,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    results = pd.DataFrame(records)
    results = results.sort_values('p_value')

    # Report
    n_sig = results['significant'].sum()
    print(f"\n  ✓ {n_sig}/{len(unique_types)} types significantly "
          f"enriched/depleted at boundaries")
    for _, row in results[results['significant']].iterrows():
        direction = "enriched" if row.fold_change > 1 else "depleted"
        print(f"    {row.cell_type}: FC={row.fold_change:.2f}, "
              f"p={row.p_value:.3f} ({direction})")

    return results
