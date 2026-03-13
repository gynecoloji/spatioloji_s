"""
DEG.py - Differentially Expressed Gene analysis for spatial transcriptomics

Supports five statistical methods:
  - Wilcoxon rank-sum test (scipy.stats.mannwhitneyu)
  - Student's t-test (scipy.stats.ttest_ind)
  - MAST-inspired hurdle model (statsmodels; optional [deg])
  - Negative-binomial GLM (statsmodels; optional [deg])
  - DESeq2 pseudobulk (pydeseq2; optional [deg])

Scalability design
------------------
  Wilcoxon / t-test   Gene-chunked loops; t-test is fully vectorized via
                      scipy axis=0. Wilcoxon uses per-gene ThreadPoolExecutor.
  NB-GLM / MAST       Per-gene model fitting parallelized via ThreadPoolExecutor.
  DESeq2              Pseudobulk aggregation collapses millions of cells to
                      n_replicates × n_genes before calling pydeseq2.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _n_workers(n_jobs: int) -> int:
    """Resolve n_jobs to a positive thread count (matches normalization.py pattern)."""
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return n_jobs


def _get_X(spatioloji_obj, layer: str | None):
    """Return expression matrix without unnecessary copies.

    This is an intentional per-module copy of the pattern established in
    normalization.py. Each processing module defines its own private `_get_X`
    rather than importing across module boundaries, following the project's
    convention of keeping private helpers module-local.
    """
    if layer is None:
        if spatioloji_obj.expression.is_sparse:
            return spatioloji_obj.expression.get_sparse()
        return spatioloji_obj.expression.get_dense()
    return spatioloji_obj.get_layer(layer)


def _build_cell_mask(
    spatioloji_obj,
    groupby: str,
    group_fg: str | list[str],
    group_bg: str | list[str],
    spatial_filter: dict | None,
    min_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build positional (0-based) fg and bg cell index arrays.

    Args:
        spatioloji_obj: spatioloji object.
        groupby: Column in cell_meta used to define groups.
        group_fg: One or more group labels for the foreground.
        group_bg: One or more group labels for the background, or ``"rest"``.
        spatial_filter: Optional dict with keys ``x_range``, ``y_range``, or
            ``polygon`` (shapely geometry).  Restricts the cell universe before
            group assignment.
        min_cells: Minimum required cells in fg and bg; raises ValueError if not met.

    Returns:
        Tuple of (fg_idx, bg_idx) — positional integer arrays. Use
        ``cell_meta.iloc[fg_idx]`` (not ``.loc``) to access metadata.

    Raises:
        ValueError: If groupby column missing, group labels not found, or
            fewer cells than min_cells in either group.
    """
    cell_meta = spatioloji_obj.cell_meta

    if groupby not in cell_meta.columns:
        raise ValueError(f"'{groupby}' not found in cell_meta columns: {list(cell_meta.columns)}")

    # Normalise to lists
    if isinstance(group_fg, str):
        group_fg = [group_fg]
    if isinstance(group_bg, str) and group_bg != "rest":
        group_bg = [group_bg]

    col_values = set(cell_meta[groupby].unique())
    missing_fg = [g for g in group_fg if g not in col_values]
    if missing_fg:
        raise ValueError(
            f"group_fg values {missing_fg} not found in '{groupby}' column. " f"Available: {sorted(col_values)}"
        )

    n_cells = len(cell_meta)
    universe = np.ones(n_cells, dtype=bool)

    if spatial_filter is not None:
        x = spatioloji_obj.spatial.x_global
        y = spatioloji_obj.spatial.y_global

        if "polygon" in spatial_filter:
            from shapely.geometry import Point

            poly = spatial_filter["polygon"]
            universe = np.array([poly.contains(Point(float(xi), float(yi))) for xi, yi in zip(x, y)])
        elif "x_range" in spatial_filter or "y_range" in spatial_filter:
            if "x_range" in spatial_filter:
                x0, x1 = spatial_filter["x_range"]
                universe &= (x >= x0) & (x <= x1)
            if "y_range" in spatial_filter:
                y0, y1 = spatial_filter["y_range"]
                universe &= (y >= y0) & (y <= y1)
        else:
            raise ValueError("spatial_filter must contain 'polygon', 'x_range', or 'y_range'")

    labels = cell_meta[groupby].values
    fg_mask = universe & np.isin(labels, group_fg)

    if group_bg == "rest":
        bg_mask = universe & ~np.isin(labels, group_fg)
    else:
        missing_bg = [g for g in group_bg if g not in col_values]
        if missing_bg:
            raise ValueError(f"group_bg values {missing_bg} not found in '{groupby}' column.")
        bg_mask = universe & np.isin(labels, group_bg)

    fg_idx = np.where(fg_mask)[0]
    bg_idx = np.where(bg_mask)[0]

    if len(fg_idx) < min_cells:
        raise ValueError(
            f"Foreground has {len(fg_idx)} cells after filtering; "
            f"need >= {min_cells} (min_cells={min_cells})"
        )
    if len(bg_idx) < min_cells:
        raise ValueError(
            f"Background has {len(bg_idx)} cells after filtering; "
            f"need >= {min_cells} (min_cells={min_cells})"
        )

    return fg_idx, bg_idx


def _apply_correction(pvals: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
    """Apply multiple-testing correction to an array of p-values.

    Handles NaN values by carrying them through unchanged.

    Args:
        pvals: Raw p-values, shape (n_genes,). May contain NaN.
        method: Correction method. ``'fdr_bh'`` (Benjamini-Hochberg) and
            ``'bonferroni'`` are implemented without extra dependencies.
            Any other value is forwarded to ``statsmodels.stats.multitest.multipletests``.

    Returns:
        Adjusted p-values, same shape as *pvals*. NaN entries stay NaN.

    Raises:
        ImportError: If *method* requires statsmodels and it is not installed.
    """
    valid = ~np.isnan(pvals)
    padj = np.full(len(pvals), np.nan)

    if valid.sum() == 0:
        return padj

    p_valid = pvals[valid]
    n = len(p_valid)

    if method == "fdr_bh":
        order = np.argsort(p_valid)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)
        adjusted = np.minimum(1.0, p_valid * n / ranks)
        # Enforce monotonicity: take cumulative min from largest to smallest rank
        adjusted_sorted = adjusted[order]
        for i in range(n - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
        result = np.empty(n)
        result[order] = adjusted_sorted
        padj[valid] = result

    elif method == "bonferroni":
        padj[valid] = np.minimum(p_valid * n, 1.0)

    else:
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            raise ImportError(
                f"Correction method '{method}' requires statsmodels. "
                "Install with: pip install spatioloji_s[deg]"
            )
        _, padj_valid, _, _ = multipletests(p_valid, method=method)
        padj[valid] = padj_valid

    return padj


def _build_result_df(
    gene_names: np.ndarray,
    stats: dict[str, np.ndarray],
    padj: np.ndarray,
    n_fg: int,
    n_bg: int,
) -> pd.DataFrame:
    """Assemble the final result DataFrame from backend stats arrays.

    Args:
        gene_names: Gene name array, shape (n_genes,).
        stats: Dict with keys ``pval``, ``mean_fg``, ``mean_bg``, ``pct_fg``,
            ``pct_bg`` — all shape (n_genes,), already computed by the backend.
        padj: Adjusted p-values, shape (n_genes,).
        n_fg: Number of foreground cells (scalar, stored in every row).
        n_bg: Number of background cells (scalar, stored in every row).

    Returns:
        pd.DataFrame sorted by ``padj`` ascending (NaN last).
    """
    log2fc = np.log2(
        (stats["mean_fg"].astype(np.float64) + 1e-9) / (stats["mean_bg"].astype(np.float64) + 1e-9)
    )
    df = pd.DataFrame(
        {
            "gene": gene_names,
            "log2fc": log2fc,
            "mean_fg": stats["mean_fg"].astype(np.float64),
            "mean_bg": stats["mean_bg"].astype(np.float64),
            "pct_fg": stats["pct_fg"].astype(np.float64),
            "pct_bg": stats["pct_bg"].astype(np.float64),
            "pval": stats["pval"].astype(np.float64),
            "padj": padj.astype(np.float64),
            "n_fg": n_fg,
            "n_bg": n_bg,
        }
    )
    return df.sort_values("padj", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Backend functions
# ---------------------------------------------------------------------------


def _wilcoxon_backend(
    X_fg: np.ndarray,
    X_bg: np.ndarray,
    n_jobs: int = 1,
    **_kwargs,
) -> dict[str, np.ndarray]:
    """Per-gene Mann-Whitney U test (Wilcoxon rank-sum).

    Args:
        X_fg: Foreground expression, shape (n_fg, chunk_genes). Dense float32.
        X_bg: Background expression, shape (n_bg, chunk_genes). Dense float32.
        n_jobs: Worker threads for parallelism across genes in this chunk.

    Returns:
        Dict with keys ``pval``, ``mean_fg``, ``mean_bg``, ``pct_fg``, ``pct_bg``.
    """
    from scipy.stats import mannwhitneyu

    chunk_genes = X_fg.shape[1]
    mean_fg = X_fg.mean(axis=0)
    mean_bg = X_bg.mean(axis=0)
    pct_fg = (X_fg > 0).mean(axis=0)
    pct_bg = (X_bg > 0).mean(axis=0)
    pvals = np.empty(chunk_genes, dtype=np.float64)

    def _test_gene(j: int) -> float:
        try:
            _, p = mannwhitneyu(X_fg[:, j], X_bg[:, j], alternative="two-sided")
        except Exception:
            p = np.nan
        return p

    if n_jobs == 1:
        for j in range(chunk_genes):
            pvals[j] = _test_gene(j)
    else:
        with ThreadPoolExecutor(max_workers=_n_workers(n_jobs)) as ex:
            for j, p in zip(range(chunk_genes), ex.map(_test_gene, range(chunk_genes))):
                pvals[j] = p

    return {
        "pval": pvals,
        "mean_fg": mean_fg.astype(np.float64),
        "mean_bg": mean_bg.astype(np.float64),
        "pct_fg": pct_fg.astype(np.float64),
        "pct_bg": pct_bg.astype(np.float64),
    }


def _ttest_backend(
    X_fg: np.ndarray,
    X_bg: np.ndarray,
    n_jobs: int = 1,
    **_kwargs,
) -> dict[str, np.ndarray]:
    """Welch's t-test, fully vectorized across genes via scipy axis=0.

    Args:
        X_fg: Foreground expression, shape (n_fg, chunk_genes). Dense float32.
        X_bg: Background expression, shape (n_bg, chunk_genes). Dense float32.
        n_jobs: Accepted for API consistency; ignored (vectorized path).

    Returns:
        Dict with keys ``pval``, ``mean_fg``, ``mean_bg``, ``pct_fg``, ``pct_bg``.
    """
    from scipy.stats import ttest_ind

    _, pvals = ttest_ind(X_fg, X_bg, axis=0, equal_var=False)

    return {
        "pval": np.asarray(pvals, dtype=np.float64),
        "mean_fg": X_fg.mean(axis=0).astype(np.float64),
        "mean_bg": X_bg.mean(axis=0).astype(np.float64),
        "pct_fg": (X_fg > 0).mean(axis=0).astype(np.float64),
        "pct_bg": (X_bg > 0).mean(axis=0).astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_VALID_METHODS = frozenset({"wilcoxon", "ttest", "mast", "nb_glm", "deseq2"})

# Populated incrementally as backends are implemented.
# Finalized to the full four-method definition after all backends exist.
_BACKEND_MAP: dict = {
    "wilcoxon": _wilcoxon_backend,
    "ttest": _ttest_backend,
}


def run_deg(
    spatioloji_obj,
    groupby: str,
    group_fg: str | list[str],
    group_bg: str | list[str] = "rest",
    methods: list[str] | None = None,
    layer: str | None = None,
    spatial_filter: dict | None = None,
    replicate_key: str | None = None,
    min_cells: int = 10,
    n_jobs: int = 1,
    gene_chunk_size: int = 500,
    correction: str = "fdr_bh",
) -> dict[str, pd.DataFrame]:
    """Run differentially expressed gene analysis.

    Args:
        spatioloji_obj: spatioloji object with expression data.
        groupby: Column in ``cell_meta`` defining cell groups.
        group_fg: Foreground group label(s).  Single string or list of strings.
        group_bg: Background group label(s), or ``"rest"`` for all non-fg cells.
        methods: Statistical methods to run.  Any subset of
            ``["wilcoxon", "ttest", "mast", "nb_glm", "deseq2"]``.
            Default: ``["wilcoxon", "ttest"]``.
        layer: Expression layer to use.  None uses the main matrix.
        spatial_filter: Optional dict restricting the cell universe spatially.
            Keys: ``x_range`` (tuple), ``y_range`` (tuple), or ``polygon``
            (shapely geometry).
        replicate_key: Column in ``cell_meta`` identifying pseudobulk replicates.
            Required when ``"deseq2"`` is in *methods*.
        min_cells: Minimum cells required in fg and bg after filtering.
        n_jobs: Worker threads for parallelism (``-1`` = all cores).
        gene_chunk_size: Genes processed per backend call.
        correction: Multiple-testing correction method (``'fdr_bh'``,
            ``'bonferroni'``, or any statsmodels method string).

    Returns:
        Dict mapping method name -> pd.DataFrame with columns:
        ``gene``, ``log2fc``, ``mean_fg``, ``mean_bg``, ``pct_fg``, ``pct_bg``,
        ``pval``, ``padj``, ``n_fg``, ``n_bg``.  Sorted by ``padj`` ascending,
        NaN last.

    Raises:
        ValueError: Invalid method names, missing replicate_key, insufficient cells.
        ImportError: If statsmodels or pydeseq2 not installed for the chosen method.

    Examples:
        >>> results = run_deg(sp, "leiden", "0", group_bg="rest",
        ...                    methods=["wilcoxon", "ttest"])
        >>> results["wilcoxon"].head()
    """
    if methods is None:
        methods = ["wilcoxon", "ttest"]

    invalid = set(methods) - _VALID_METHODS
    if invalid:
        raise ValueError(f"Unknown method(s): {invalid}. Valid methods: {sorted(_VALID_METHODS)}")

    if "deseq2" in methods and replicate_key is None:
        raise ValueError("'deseq2' requires replicate_key to specify pseudobulk replicates.")
    if replicate_key is not None and replicate_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(
            f"replicate_key '{replicate_key}' not found in cell_meta columns: "
            f"{list(spatioloji_obj.cell_meta.columns)}"
        )

    # -- Build cell masks --
    fg_idx, bg_idx = _build_cell_mask(spatioloji_obj, groupby, group_fg, group_bg, spatial_filter, min_cells)
    n_fg, n_bg = len(fg_idx), len(bg_idx)

    # -- Get expression matrix and densify fg/bg slices --
    X = _get_X(spatioloji_obj, layer)
    gene_names = np.asarray(spatioloji_obj.gene_index)
    n_genes = len(gene_names)

    if n_genes == 0:
        return {}

    if sparse.issparse(X):
        X_fg = X[fg_idx, :].toarray().astype(np.float32)
        X_bg = X[bg_idx, :].toarray().astype(np.float32)
    else:
        X_fg = X[fg_idx, :].astype(np.float32, copy=False)
        X_bg = X[bg_idx, :].astype(np.float32, copy=False)

    results: dict[str, pd.DataFrame] = {}

    for method in methods:
        print(f"\nRunning DEG [{method}]: {n_fg} fg vs {n_bg} bg, {n_genes} genes")

        # -- DESeq2 pseudobulk path --
        if method == "deseq2":
            counts_fg, counts_bg, _, _ = _aggregate_pseudobulk(
                X, fg_idx, bg_idx, replicate_key, spatioloji_obj.cell_meta
            )
            results["deseq2"] = _deseq2_backend(counts_fg, counts_bg, gene_names)
            n_sig = int((results["deseq2"]["padj"] < 0.05).sum())
            print(f"  ✓ deseq2: {n_sig} significant genes (padj < 0.05)")
            # IMPORTANT: `continue` skips _apply_correction — pydeseq2's own
            # BH-corrected padj is used verbatim, not recomputed.
            continue

        # -- Gene-chunked path for Wilcoxon / t-test / MAST / NB-GLM --
        backend_fn = _BACKEND_MAP[method]

        # MAST needs CDR computed from the full gene set before chunking
        extra_kwargs: dict = {}
        if method == "mast":
            extra_kwargs["cdr_fg"] = (X_fg > 0).mean(axis=1).astype(np.float32)
            extra_kwargs["cdr_bg"] = (X_bg > 0).mean(axis=1).astype(np.float32)

        chunk_stats: list[dict[str, np.ndarray]] = []
        for s in range(0, n_genes, gene_chunk_size):
            e = min(s + gene_chunk_size, n_genes)
            chunk = backend_fn(X_fg[:, s:e], X_bg[:, s:e], n_jobs=n_jobs, **extra_kwargs)
            chunk_stats.append(chunk)

        # Concatenate chunks
        stats = {k: np.concatenate([c[k] for c in chunk_stats]) for k in chunk_stats[0]}
        padj = _apply_correction(stats["pval"], method=correction)
        result_df = _build_result_df(gene_names, stats, padj, n_fg, n_bg)
        results[method] = result_df

        n_sig = int((result_df["padj"] < 0.05).sum())
        print(f"  ✓ {method}: {n_sig} significant genes (padj < 0.05)")

    return results
