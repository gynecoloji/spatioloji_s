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
