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
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------
# Per-method layer conventions
# ---------------------------------------------------------------------------
# Each statistical model expects a particular kind of input.  ``layer="auto"``
# in run_deg() maps each method to its expected layer using this table; the
# convenience wrappers (deg_wilcoxon, deg_ttest, ...) inherit the default by
# leaving ``layer`` unset.  Override on a per-method basis with the
# ``layer={...}`` dict form.
#
#   value=None          → main expression matrix (raw counts)
#   value=str           → name of a derived layer expected to exist on sp
_DEFAULT_LAYER_PER_METHOD: dict[str, str | None] = {
    "wilcoxon": "log_normalized",   # rank-sum: log scale stabilises ties
    "ttest":    "log_normalized",   # parametric: assumes ~normal residuals
    "mast":     "log_normalized",   # hurdle on log(TPM+1) per the paper
    "nb_glm":   None,               # negative-binomial: requires raw counts
    "deseq2":   None,               # pseudobulk DESeq2: raw counts only
}

# Methods that *must* see raw counts; emit a warning when the user passes a
# layer whose name suggests it has been log-transformed or scaled.
_RAW_COUNT_DEG_METHODS = frozenset({"nb_glm", "deseq2"})

# Methods that *expect* log-normalized input; warn when raw counts seem to
# have been passed (layer=None and the main matrix looks like raw integers
# is the typical case but cannot be verified without inspecting data, so we
# only warn on the layer-name heuristic).
_LOG_NORM_DEG_METHODS = frozenset({"wilcoxon", "ttest", "mast"})

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
            f"group_fg values {missing_fg} not found in '{groupby}' column. Available: {sorted(col_values)}"
        )

    n_cells = len(cell_meta)
    universe = np.ones(n_cells, dtype=bool)

    if spatial_filter is not None:
        x = spatioloji_obj.spatial.x_global
        y = spatioloji_obj.spatial.y_global

        if "polygon" in spatial_filter:
            from shapely.geometry import Point

            poly = spatial_filter["polygon"]
            universe = np.array([poly.contains(Point(float(xi), float(yi))) for xi, yi in zip(x, y, strict=False)])
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
            f"Foreground has {len(fg_idx)} cells after filtering; need >= {min_cells} (min_cells={min_cells})"
        )
    if len(bg_idx) < min_cells:
        raise ValueError(
            f"Background has {len(bg_idx)} cells after filtering; need >= {min_cells} (min_cells={min_cells})"
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
                f"Correction method '{method}' requires statsmodels. Install with: pip install spatioloji_s[deg]"
            ) from None
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
    log2fc = np.log2((stats["mean_fg"].astype(np.float64) + 1e-9) / (stats["mean_bg"].astype(np.float64) + 1e-9))
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
            for j, p in zip(range(chunk_genes), ex.map(_test_gene, range(chunk_genes)), strict=True):
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


def _mast_one_gene(
    x_fg: np.ndarray,
    x_bg: np.ndarray,
    cdr_fg: np.ndarray,
    cdr_bg: np.ndarray,
) -> float:
    """Fit the MAST-inspired two-part hurdle model for a single gene.

    Args:
        x_fg: Expression values for fg cells, shape (n_fg,).
        x_bg: Expression values for bg cells, shape (n_bg,).
        cdr_fg: Cellular detection rate per fg cell (fraction of all genes > 0).
        cdr_bg: Cellular detection rate per bg cell.

    Returns:
        Combined p-value (Fisher's method). NaN if fitting fails.
    """
    try:
        import statsmodels.api as sm
        from scipy.stats import combine_pvalues
    except ImportError:
        raise ImportError("MAST requires statsmodels. Install with: pip install spatioloji_s[deg]") from None

    x = np.concatenate([x_fg, x_bg]).astype(np.float64)
    cdr = np.concatenate([cdr_fg, cdr_bg]).astype(np.float64)
    group = np.array([1.0] * len(x_fg) + [0.0] * len(x_bg))

    # --- Discrete component: logistic regression on expressed/not-expressed ---
    y_disc = (x > 0).astype(np.float64)
    if y_disc.sum() == 0 or y_disc.sum() == len(y_disc):
        p_disc = np.nan
    else:
        X_disc = sm.add_constant(np.column_stack([group, cdr]))
        try:
            res_disc = sm.Logit(y_disc, X_disc).fit(disp=False, maxiter=100)
            p_disc = float(res_disc.pvalues[1])
        except Exception:
            p_disc = np.nan

    # --- Continuous component: OLS on expressed cells only ---
    expr_mask = x > 0
    p_cont = np.nan
    fg_expr = (x_fg > 0).sum()
    bg_expr = (x_bg > 0).sum()
    if expr_mask.sum() >= 5 and fg_expr >= 2 and bg_expr >= 2:
        y_cont = x[expr_mask]
        X_cont = sm.add_constant(np.column_stack([group[expr_mask], cdr[expr_mask]]))
        try:
            res_cont = sm.OLS(y_cont, X_cont).fit()
            p_cont = float(res_cont.pvalues[1])
        except Exception:
            p_cont = np.nan

    # --- Combine via Fisher's method ---
    valid_pvals = [p for p in [p_disc, p_cont] if not np.isnan(p)]
    if not valid_pvals:
        return np.nan
    if len(valid_pvals) == 1:
        return valid_pvals[0]
    # Floor at smallest positive float so log(p) stays finite in Fisher's sum.
    floor = np.finfo(np.float64).tiny
    clipped = [max(p, floor) for p in valid_pvals]
    _, combined = combine_pvalues(clipped, method="fisher")
    return float(combined)


def _mast_backend(
    X_fg: np.ndarray,
    X_bg: np.ndarray,
    cdr_fg: np.ndarray,
    cdr_bg: np.ndarray,
    n_jobs: int = 1,
    **_kwargs,
) -> dict[str, np.ndarray]:
    """MAST-inspired hurdle model backend.

    Fits a two-part logistic + OLS model per gene with CDR as covariate.
    P-values from the two components are combined via Fisher's method.
    This is a Python approximation of MAST, not a faithful port.

    Args:
        X_fg: Foreground expression, shape (n_fg, chunk_genes).
        X_bg: Background expression, shape (n_bg, chunk_genes).
        cdr_fg: Cellular detection rate per fg cell, shape (n_fg,).
        cdr_bg: Cellular detection rate per bg cell, shape (n_bg,).
        n_jobs: Worker threads for per-gene fitting.

    Returns:
        Dict with keys ``pval``, ``mean_fg``, ``mean_bg``, ``pct_fg``, ``pct_bg``.
    """
    chunk_genes = X_fg.shape[1]
    mean_fg = X_fg.mean(axis=0).astype(np.float64)
    mean_bg = X_bg.mean(axis=0).astype(np.float64)
    pct_fg = (X_fg > 0).mean(axis=0).astype(np.float64)
    pct_bg = (X_bg > 0).mean(axis=0).astype(np.float64)
    pvals = np.empty(chunk_genes, dtype=np.float64)

    def _fit(j: int) -> float:
        return _mast_one_gene(X_fg[:, j], X_bg[:, j], cdr_fg, cdr_bg)

    if n_jobs == 1:
        for j in range(chunk_genes):
            pvals[j] = _fit(j)
    else:
        with ThreadPoolExecutor(max_workers=_n_workers(n_jobs)) as ex:
            pvals[:] = list(ex.map(_fit, range(chunk_genes)))

    return {
        "pval": pvals,
        "mean_fg": mean_fg,
        "mean_bg": mean_bg,
        "pct_fg": pct_fg,
        "pct_bg": pct_bg,
    }


def _nb_glm_one_gene(x_fg: np.ndarray, x_bg: np.ndarray) -> float:
    """Fit a negative-binomial GLM for a single gene.

    Args:
        x_fg: Raw count values for fg cells, shape (n_fg,).
        x_bg: Raw count values for bg cells, shape (n_bg,).

    Returns:
        Two-sided p-value for the group coefficient. NaN on convergence failure.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("NB-GLM requires statsmodels. Install with: pip install spatioloji_s[deg]") from None

    y = np.concatenate([x_fg, x_bg]).astype(np.float64)
    group = np.array([1.0] * len(x_fg) + [0.0] * len(x_bg))
    X = sm.add_constant(group)

    try:
        res = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit(disp=False, maxiter=100)
        return float(res.pvalues[1])
    except Exception:
        return np.nan


def _nb_glm_backend(
    X_fg: np.ndarray,
    X_bg: np.ndarray,
    n_jobs: int = 1,
    **_kwargs,
) -> dict[str, np.ndarray]:
    """Negative-binomial GLM backend, per gene.

    Args:
        X_fg: Foreground expression, shape (n_fg, chunk_genes). Dense float32.
        X_bg: Background expression, shape (n_bg, chunk_genes). Dense float32.
        n_jobs: Worker threads for per-gene model fitting.

    Returns:
        Dict with keys ``pval``, ``mean_fg``, ``mean_bg``, ``pct_fg``, ``pct_bg``.
        Genes where the GLM fails to converge return NaN in ``pval``.
    """
    chunk_genes = X_fg.shape[1]
    mean_fg = X_fg.mean(axis=0).astype(np.float64)
    mean_bg = X_bg.mean(axis=0).astype(np.float64)
    pct_fg = (X_fg > 0).mean(axis=0).astype(np.float64)
    pct_bg = (X_bg > 0).mean(axis=0).astype(np.float64)
    pvals = np.empty(chunk_genes, dtype=np.float64)

    def _fit(j: int) -> float:
        return _nb_glm_one_gene(X_fg[:, j], X_bg[:, j])

    if n_jobs == 1:
        for j in range(chunk_genes):
            pvals[j] = _fit(j)
    else:
        with ThreadPoolExecutor(max_workers=_n_workers(n_jobs)) as ex:
            pvals[:] = list(ex.map(_fit, range(chunk_genes)))

    return {
        "pval": pvals,
        "mean_fg": mean_fg,
        "mean_bg": mean_bg,
        "pct_fg": pct_fg,
        "pct_bg": pct_bg,
    }


# ---------------------------------------------------------------------------
# Pseudobulk / DESeq2
# ---------------------------------------------------------------------------


def _aggregate_pseudobulk(
    X: np.ndarray,
    fg_idx: np.ndarray,
    bg_idx: np.ndarray,
    replicate_key: str,
    cell_meta: pd.DataFrame,
    min_replicates: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum raw counts per replicate per group for pseudobulk DESeq2 analysis.

    Args:
        X: Dense expression matrix, shape (n_cells, n_genes). Must be dense (not sparse).
        fg_idx: Positional indices of foreground cells.
        bg_idx: Positional indices of background cells.
        replicate_key: Column in *cell_meta* identifying replicates.
        cell_meta: Full cell metadata DataFrame (use ``iloc`` indexing).
        min_replicates: Minimum unique replicates required per group.
            Enforced independently for fg and bg.

    Returns:
        Tuple (counts_fg, counts_bg, rep_fg_labels, rep_bg_labels):
            - counts_fg: shape (n_rep_fg, n_genes), float64 aggregated counts
            - counts_bg: shape (n_rep_bg, n_genes), float64 aggregated counts
            - rep_fg_labels: unique replicate labels for fg
            - rep_bg_labels: unique replicate labels for bg

    Raises:
        ValueError: If either group has fewer than *min_replicates* replicates.
    """
    rep_fg = cell_meta.iloc[fg_idx][replicate_key].values
    rep_bg = cell_meta.iloc[bg_idx][replicate_key].values

    unique_rep_fg = np.unique(rep_fg)
    unique_rep_bg = np.unique(rep_bg)

    if len(unique_rep_fg) < min_replicates:
        raise ValueError(
            f"Foreground has only {len(unique_rep_fg)} replicate(s) in '{replicate_key}'; "
            f"DESeq2 requires >= {min_replicates} per group."
        )
    if len(unique_rep_bg) < min_replicates:
        raise ValueError(
            f"Background has only {len(unique_rep_bg)} replicate(s) in '{replicate_key}'; "
            f"DESeq2 requires >= {min_replicates} per group."
        )

    n_genes = X.shape[1]

    def _aggregate(idx_array: np.ndarray, rep_labels: np.ndarray, unique_reps: np.ndarray):
        out = np.zeros((len(unique_reps), n_genes), dtype=np.float64)
        for i, rep in enumerate(unique_reps):
            mask = rep_labels == rep
            rows = idx_array[mask]
            chunk = X[rows, :]
            out[i] = chunk.sum(axis=0)
        return out

    counts_fg = _aggregate(fg_idx, rep_fg, unique_rep_fg)
    counts_bg = _aggregate(bg_idx, rep_bg, unique_rep_bg)

    return counts_fg, counts_bg, unique_rep_fg, unique_rep_bg


def _deseq2_backend(
    counts_fg: np.ndarray,
    counts_bg: np.ndarray,
    gene_names: np.ndarray,
) -> pd.DataFrame:
    """Run DESeq2 via pydeseq2 on pseudobulk-aggregated count matrices.

    Args:
        counts_fg: Pseudobulk counts for fg, shape (n_rep_fg, n_genes). Float64.
        counts_bg: Pseudobulk counts for bg, shape (n_rep_bg, n_genes). Float64.
        gene_names: Gene name array, shape (n_genes,).

    Returns:
        pd.DataFrame with the standard DEG output schema. padj comes from
        pydeseq2's own Benjamini-Hochberg correction (not re-applied externally).

    Raises:
        ImportError: If pydeseq2 is not installed.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError("DESeq2 requires pydeseq2. Install with: pip install spatioloji_s[deg]") from None

    n_fg = counts_fg.shape[0]
    n_bg = counts_bg.shape[0]

    counts = np.vstack([counts_fg, counts_bg]).round().astype(int)
    condition = ["fg"] * n_fg + ["bg"] * n_bg
    sample_df = pd.DataFrame({"condition": condition})
    counts_df = pd.DataFrame(counts, columns=gene_names)

    dds = DeseqDataSet(counts=counts_df, metadata=sample_df, design_factors="condition")
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=["condition", "fg", "bg"])
    stat_res.summary()

    res = stat_res.results_df.reset_index().rename(columns={"index": "gene"})

    # Merge on gene name to handle pydeseq2's internal gene reordering.
    gene_df = pd.DataFrame({"gene": gene_names})
    merged = gene_df.merge(res, on="gene", how="left")

    result_df = pd.DataFrame(
        {
            "gene": merged["gene"].values,
            "log2fc": merged["log2FoldChange"].values.astype(np.float64),
            "mean_fg": counts_fg.mean(axis=0).astype(np.float64),
            "mean_bg": counts_bg.mean(axis=0).astype(np.float64),
            "pct_fg": (counts_fg > 0).mean(axis=0).astype(np.float64),
            "pct_bg": (counts_bg > 0).mean(axis=0).astype(np.float64),
            "pval": merged["pvalue"].values.astype(np.float64),
            "padj": merged["padj"].values.astype(np.float64),
            "n_fg": n_fg,
            "n_bg": n_bg,
        }
    )
    return result_df.sort_values("padj", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_VALID_METHODS = frozenset({"wilcoxon", "ttest", "mast", "nb_glm", "deseq2"})

# Populated incrementally as backends are implemented.
_BACKEND_MAP: dict = {
    "wilcoxon": _wilcoxon_backend,
    "ttest": _ttest_backend,
    "mast": _mast_backend,
    "nb_glm": _nb_glm_backend,
}


def _validate_layer_for_method(method: str, lyr: str | None) -> None:
    """Warn when *lyr* looks incompatible with *method*'s expected input.

    This is a best-effort name-based check — it cannot inspect the actual
    matrix contents.  Triggers a UserWarning when:

    - a count-based method (``nb_glm``, ``deseq2``) is given a layer whose
      name contains ``"log"`` or ``"scaled"``;
    - a distribution-based method (``wilcoxon``, ``ttest``, ``mast``) is
      given ``layer=None`` (often the raw matrix on real data).
    """
    name = (lyr or "").lower()
    is_log = "log" in name
    is_scaled = "scaled" in name

    if method in _RAW_COUNT_DEG_METHODS and (is_log or is_scaled):
        warnings.warn(
            f"DEG method '{method}' expects raw counts but layer='{lyr}' "
            "appears to be log-transformed or scaled. Use layer=None or pass "
            "the raw-counts layer explicitly. Pass layer='auto' to let "
            "run_deg pick the right layer per method.",
            UserWarning,
            stacklevel=3,
        )
    elif method in _LOG_NORM_DEG_METHODS and lyr is None:
        warnings.warn(
            f"DEG method '{method}' expects log-normalized data but layer=None "
            "(main matrix is typically raw counts). Pass layer='log_normalized' "
            "or layer='auto' for per-method routing.",
            UserWarning,
            stacklevel=3,
        )


def _resolve_layer_per_method(
    layer: str | dict[str, str | None] | None,
    methods: list[str],
) -> dict[str, str | None]:
    """Expand the user-facing ``layer`` argument to a {method: layer} map.

    Args:
        layer: One of:

            - ``None`` — use the main expression matrix for every method.
            - ``str`` — a layer name applied to every method, **or** the
              special string ``"auto"`` which routes each method to its
              recommended layer (see ``_DEFAULT_LAYER_PER_METHOD``).  ``"auto"``
              is the recommended setting when running multiple methods that
              expect different inputs (e.g. Wilcoxon on log-normalized,
              DESeq2 on raw counts).
            - ``dict`` — explicit per-method mapping, e.g.
              ``{"wilcoxon": "log_normalized", "deseq2": None}``.  Must
              cover every method in *methods*.
        methods: Methods being run.

    Returns:
        Dict with an entry for every method in *methods*.

    Raises:
        ValueError: If *layer* is a dict missing entries for some methods, or
            contains keys not in *methods*.
    """
    # 1) "auto" → per-method default table
    if isinstance(layer, str) and layer == "auto":
        return {m: _DEFAULT_LAYER_PER_METHOD.get(m) for m in methods}

    # 2) Explicit dict → validate coverage, then warn per method
    if isinstance(layer, dict):
        extra = set(layer) - set(methods)
        if extra:
            raise ValueError(
                f"layer dict contains entries for methods not being run: {sorted(extra)}. "
                f"methods={methods}"
            )
        missing = [m for m in methods if m not in layer]
        if missing:
            raise ValueError(
                f"layer dict is missing entries for methods: {missing}. "
                f"Provide a layer (or None) for every method in `methods`."
            )
        resolved = {m: layer[m] for m in methods}
        for m, lyr in resolved.items():
            _validate_layer_for_method(m, lyr)
        return resolved

    # 3) Single value (None or str) → broadcast and validate per method
    resolved = {m: layer for m in methods}
    for m, lyr in resolved.items():
        _validate_layer_for_method(m, lyr)
    return resolved


def run_deg(
    spatioloji_obj,
    groupby: str,
    group_fg: str | list[str],
    group_bg: str | list[str] = "rest",
    methods: list[str] | None = None,
    layer: str | dict[str, str | None] | None = "auto",
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
        layer: Expression layer(s) to use.  Default ``"auto"`` routes each
            method to its expected input via the per-method table:

            ============= ================
            Method        Default layer
            ============= ================
            ``wilcoxon``  ``log_normalized``
            ``ttest``     ``log_normalized``
            ``mast``      ``log_normalized``
            ``nb_glm``    ``None`` (raw counts)
            ``deseq2``    ``None`` (raw counts)
            ============= ================

            Other accepted forms:

            - ``None`` — main expression matrix for every method.
            - ``str`` (other than ``"auto"``) — that layer for every method.
            - ``dict`` — per-method mapping, e.g.
              ``{"wilcoxon": "log_normalized", "deseq2": None, "nb_glm": None}``.
              Every method in ``methods`` must have an entry (``None`` is
              allowed as a value).  When you need distribution-based methods
              on normalized data **and** count-based methods on raw counts in
              a single call, prefer ``layer="auto"`` over hand-rolling the
              dict.

            A ``UserWarning`` is emitted when an explicit layer name looks
            incompatible with a method (e.g. ``layer="scaled"`` for DESeq2,
            or ``layer=None`` for Wilcoxon).
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
        ValueError: Invalid method names, missing replicate_key, insufficient
            cells, or a per-method ``layer`` dict that does not cover every
            method in *methods*.
        ImportError: If statsmodels or pydeseq2 not installed for the chosen method.

    Examples:
        >>> # Single method (uses its default layer automatically)
        >>> results = run_deg(sp, "leiden", "0", methods=["wilcoxon"])
        >>>
        >>> # Multiple methods, auto-routed to the right layer per method:
        >>> # wilcoxon / ttest → 'log_normalized', deseq2 → raw counts.
        >>> results = run_deg(
        ...     sp, "leiden", "0",
        ...     methods=["wilcoxon", "ttest", "deseq2"],
        ...     layer="auto",
        ...     replicate_key="fov",
        ... )
        >>>
        >>> # Explicit per-method override (e.g. compare on a custom layer)
        >>> results = run_deg(
        ...     sp, "leiden", "0",
        ...     methods=["wilcoxon", "deseq2"],
        ...     layer={"wilcoxon": "pearson_residuals", "deseq2": None},
        ...     replicate_key="fov",
        ... )
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
            f"replicate_key '{replicate_key}' not found in cell_meta columns: {list(spatioloji_obj.cell_meta.columns)}"
        )

    layer_map = _resolve_layer_per_method(layer, methods)

    # If layer='auto' picked a layer name that doesn't exist on this object
    # (e.g. user hasn't run normalize/log_transform yet), fall back to the
    # main matrix for that method.  Explicit user-supplied layer names that
    # are missing still raise downstream — this fallback only relaxes the
    # auto-routed case to keep "auto" safe as a default.
    if layer == "auto":
        try:
            existing = set(spatioloji_obj.list_layers())
        except AttributeError:
            existing = set()
        for m, lyr in list(layer_map.items()):
            if lyr is not None and lyr not in existing:
                print(
                    f"  [layer=auto] '{lyr}' not found on spatioloji object — "
                    f"falling back to main matrix for method '{m}'."
                )
                layer_map[m] = None

    # -- Build cell masks --
    fg_idx, bg_idx = _build_cell_mask(spatioloji_obj, groupby, group_fg, group_bg, spatial_filter, min_cells)
    n_fg, n_bg = len(fg_idx), len(bg_idx)

    gene_names = np.asarray(spatioloji_obj.gene_index)
    n_genes = len(gene_names)

    if n_genes == 0:
        return {}

    # Cache so the same layer isn't loaded and densified twice when two
    # methods share it (e.g. Wilcoxon + t-test both on 'log_normalized').
    # Keyed by layer name (None means main matrix).
    matrix_cache: dict[str | None, tuple] = {}

    def _load_matrices(lyr: str | None):
        if lyr in matrix_cache:
            return matrix_cache[lyr]
        X_full = _get_X(spatioloji_obj, lyr)
        if sparse.issparse(X_full):
            X_fg_ = X_full[fg_idx, :].toarray().astype(np.float32)
            X_bg_ = X_full[bg_idx, :].toarray().astype(np.float32)
        else:
            X_fg_ = X_full[fg_idx, :].astype(np.float32, copy=False)
            X_bg_ = X_full[bg_idx, :].astype(np.float32, copy=False)
        matrix_cache[lyr] = (X_full, X_fg_, X_bg_)
        return matrix_cache[lyr]

    results: dict[str, pd.DataFrame] = {}

    for method in methods:
        this_layer = layer_map[method]
        X, X_fg, X_bg = _load_matrices(this_layer)
        layer_label = this_layer if this_layer is not None else "main"
        print(f"\nRunning DEG [{method}]: {n_fg} fg vs {n_bg} bg, {n_genes} genes, layer={layer_label}")

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


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def deg_wilcoxon(
    spatioloji_obj,
    groupby: str,
    group_fg: str | list[str],
    group_bg: str | list[str] = "rest",
    layer: str | None = "auto",
    spatial_filter: dict | None = None,
    min_cells: int = 10,
    n_jobs: int = 1,
    gene_chunk_size: int = 500,
    correction: str = "fdr_bh",
    **_ignored,
) -> dict[str, pd.DataFrame]:
    """Run Wilcoxon rank-sum DEG test.

    ``layer`` defaults to ``"auto"`` which selects ``'log_normalized'`` per
    the per-method table; pass an explicit layer name (or ``None`` for the
    main matrix) to override. See ``run_deg`` for full parameter docs.
    """
    return run_deg(
        spatioloji_obj,
        groupby,
        group_fg,
        group_bg,
        methods=["wilcoxon"],
        layer=layer,
        spatial_filter=spatial_filter,
        min_cells=min_cells,
        n_jobs=n_jobs,
        gene_chunk_size=gene_chunk_size,
        correction=correction,
    )


def deg_ttest(
    spatioloji_obj,
    groupby: str,
    group_fg: str | list[str],
    group_bg: str | list[str] = "rest",
    layer: str | None = "auto",
    spatial_filter: dict | None = None,
    min_cells: int = 10,
    n_jobs: int = 1,
    gene_chunk_size: int = 500,
    correction: str = "fdr_bh",
    **_ignored,
) -> dict[str, pd.DataFrame]:
    """Run Student's t-test DEG analysis.

    ``layer`` defaults to ``"auto"`` (resolves to ``'log_normalized'``).
    See ``run_deg`` for full parameter docs.
    """
    return run_deg(
        spatioloji_obj,
        groupby,
        group_fg,
        group_bg,
        methods=["ttest"],
        layer=layer,
        spatial_filter=spatial_filter,
        min_cells=min_cells,
        n_jobs=n_jobs,
        gene_chunk_size=gene_chunk_size,
        correction=correction,
    )


def deg_mast(
    spatioloji_obj,
    groupby: str,
    group_fg: str | list[str],
    group_bg: str | list[str] = "rest",
    layer: str | None = "auto",
    spatial_filter: dict | None = None,
    min_cells: int = 10,
    n_jobs: int = 1,
    gene_chunk_size: int = 500,
    correction: str = "fdr_bh",
    **_ignored,
) -> dict[str, pd.DataFrame]:
    """Run MAST-inspired hurdle model DEG analysis. Requires ``statsmodels``.

    ``layer`` defaults to ``"auto"`` (resolves to ``'log_normalized'``).
    See ``run_deg`` for full parameter docs.
    """
    return run_deg(
        spatioloji_obj,
        groupby,
        group_fg,
        group_bg,
        methods=["mast"],
        layer=layer,
        spatial_filter=spatial_filter,
        min_cells=min_cells,
        n_jobs=n_jobs,
        gene_chunk_size=gene_chunk_size,
        correction=correction,
    )


def deg_nb_glm(
    spatioloji_obj,
    groupby: str,
    group_fg: str | list[str],
    group_bg: str | list[str] = "rest",
    layer: str | None = "auto",
    spatial_filter: dict | None = None,
    min_cells: int = 10,
    n_jobs: int = 1,
    gene_chunk_size: int = 500,
    correction: str = "fdr_bh",
    **_ignored,
) -> dict[str, pd.DataFrame]:
    """Run negative-binomial GLM DEG analysis. Requires ``statsmodels``.

    ``layer`` defaults to ``"auto"`` (resolves to ``None`` — raw counts).
    See ``run_deg`` for full parameter docs.
    """
    return run_deg(
        spatioloji_obj,
        groupby,
        group_fg,
        group_bg,
        methods=["nb_glm"],
        layer=layer,
        spatial_filter=spatial_filter,
        min_cells=min_cells,
        n_jobs=n_jobs,
        gene_chunk_size=gene_chunk_size,
        correction=correction,
    )


def deg_deseq2(
    spatioloji_obj,
    groupby: str,
    group_fg: str | list[str],
    group_bg: str | list[str] = "rest",
    layer: str | None = "auto",
    spatial_filter: dict | None = None,
    replicate_key: str | None = None,
    min_cells: int = 10,
    correction: str = "fdr_bh",
    **_ignored,
) -> dict[str, pd.DataFrame]:
    """Run DESeq2 pseudobulk DEG analysis. Requires ``pydeseq2``.

    ``layer`` defaults to ``"auto"`` (resolves to ``None`` — raw counts).
    See ``run_deg`` for full parameter docs. ``replicate_key`` is required.
    """
    return run_deg(
        spatioloji_obj,
        groupby,
        group_fg,
        group_bg,
        methods=["deseq2"],
        layer=layer,
        spatial_filter=spatial_filter,
        replicate_key=replicate_key,
        min_cells=min_cells,
        correction=correction,
    )
