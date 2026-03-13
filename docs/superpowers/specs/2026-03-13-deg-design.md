# DEG.py Design Spec

**Date:** 2026-03-13
**Module:** `src/spatioloji_s/processing/DEG.py`
**Status:** Approved

---

## Overview

Add a differentially expressed gene (DEG) analysis module to the `processing` subpackage of `spatioloji_s`. The module supports five mainstream statistical methods, scales to millions of cells via chunked gene-wise computation, and integrates with the existing `spatioloji` object pattern.

---

## Public API

### Primary entry point

```python
run_deg(
    spatioloji_obj,
    groupby: str,
    group_fg: str | list[str],
    group_bg: str | list[str] = "rest",
    methods: list[str] = ["wilcoxon", "ttest"],
    layer: str | None = None,
    spatial_filter: dict | None = None,
    replicate_key: str | None = None,
    min_cells: int = 10,
    n_jobs: int = 1,
    gene_chunk_size: int = 500,
    correction: str = "fdr_bh",
) -> dict[str, pd.DataFrame]
```

**Note on `layer`:** `None` uses the main expression matrix (same as `_get_X(spatioloji_obj, None)` in the existing codebase). There is no `.raw` attribute on the `spatioloji` object; `layer` follows the identical pattern used by all existing processing functions.

**Note on return value:** `run_deg` always returns a `dict[str, pd.DataFrame]`. There are no `inplace` / `copy` flags — DEG results are tabular outputs, not expression layers, so they are always returned to the caller and never stored on the `spatioloji` object.

### Convenience wrappers

Five thin wrappers that call `run_deg` with `methods` fixed to one method. Their signatures are identical to `run_deg` with the `methods` parameter removed. Parameters irrelevant to a specific method (e.g., `replicate_key` for `deg_wilcoxon`) are still accepted but documented as ignored, for API consistency.

- `deg_wilcoxon(spatioloji_obj, groupby, group_fg, group_bg="rest", layer=None, spatial_filter=None, min_cells=10, n_jobs=1, gene_chunk_size=500, correction="fdr_bh") -> dict[str, pd.DataFrame]`
- `deg_ttest(...)` — same signature as above
- `deg_mast(...)` — same signature as above
- `deg_nb_glm(...)` — same signature as above
- `deg_deseq2(spatioloji_obj, groupby, group_fg, group_bg="rest", layer=None, spatial_filter=None, replicate_key=None, min_cells=10, correction="fdr_bh") -> dict[str, pd.DataFrame]`

### Output format

Each value in the returned `dict[str, pd.DataFrame]` has columns:

| Column | Description |
|--------|-------------|
| `gene` | Gene name |
| `log2fc` | Log2 fold-change: `log2((mean_fg + 1e-9) / (mean_bg + 1e-9))` in linear expression space |
| `mean_fg` | Mean expression in foreground (in the units of the input layer) |
| `mean_bg` | Mean expression in background |
| `pct_fg` | Fraction of fg cells with expression > 0 |
| `pct_bg` | Fraction of bg cells with expression > 0 |
| `pval` | Raw p-value |
| `padj` | Adjusted p-value (BH by default). NaN values sorted last (pandas default). |
| `n_fg` | Number of foreground cells |
| `n_bg` | Number of background cells |

Sorted by `padj` ascending (NaN last). Dict keys are method names (e.g., `"wilcoxon"`, `"ttest"`).

**`log2fc` formula:** Computed in linear expression space as `log2((mean_fg + 1e-9) / (mean_bg + 1e-9))`. The pseudocount `1e-9` prevents division-by-zero when one group has zero mean. If the input layer is already log-normalized, the user should interpret `log2fc` accordingly — the module does not infer the input space.

---

## Contrast / Grouping Modes

`groupby` is a column in `spatioloji_obj.cell_meta`.

| `group_fg` | `group_bg` | Contrast type |
|------------|------------|---------------|
| `"ClusterA"` | `"rest"` | One-vs-rest |
| `"ClusterA"` | `"ClusterB"` | Pairwise |
| `["A", "B"]` | `["C", "D"]` | Custom multi-group |

---

## Spatial Filter

`spatial_filter` is an optional dict. Two forms:

```python
# Bounding box
spatial_filter = {"x_range": (x0, x1), "y_range": (y0, y1)}

# Polygon (shapely)
spatial_filter = {"polygon": shapely_polygon}
```

Applied inside `_build_cell_mask` before group mask construction. Restricts the entire cell universe to cells within the spatial region; both fg and bg are selected from this restricted universe.

---

## Architecture: Shared Pipeline + Pluggable Backends

### Private pipeline (inside `run_deg`)

```
1. _build_cell_mask(spatioloji_obj, groupby, group_fg, group_bg, spatial_filter, min_cells)
   → fg_idx (int array), bg_idx (int array)
   [fg_idx and bg_idx are POSITIONAL (0-based row indices into the n_cells axis of X).
    Use cell_meta.iloc[fg_idx] — NOT .loc[fg_idx] — when accessing metadata by these indices.
    min_cells guard enforced here: raises ValueError if len(fg_idx) < min_cells
    or len(bg_idx) < min_cells]

2. _get_X(spatioloji_obj, layer)   [reuse existing helper pattern]
   → X (sparse or dense, n_cells × n_genes)

3. Slice fg/bg:
   X_fg = X[fg_idx, :]   X_bg = X[bg_idx, :]

4. For each method in methods:
   └── if method == "deseq2":
         _aggregate_pseudobulk(X[fg_idx | bg_idx], fg_idx, bg_idx,
                                replicate_key, cell_meta, min_replicates=2)
         [min_replicates=2 enforced PER GROUP: fg must have >= 2 unique replicate
          values AND bg must have >= 2 unique replicate values. Test both cases separately.]
         → counts_fg (n_rep_fg × n_genes), counts_bg (n_rep_bg × n_genes)
         result_df = _deseq2_backend(counts_fg, counts_bg, gene_names)
         [DESeq2 path uses pydeseq2's own padj; _apply_correction is SKIPPED
          for this method. The returned DataFrame includes pydeseq2's padj directly.]
       else:
         Gene-chunked loop (gene_chunk_size steps over n_genes):
           stats_chunk = _<method>_backend(X_fg_chunk, X_bg_chunk, n_jobs)
           [each backend returns a dict with keys: pval, mean_fg, mean_bg,
            pct_fg, pct_bg — computed within the backend to avoid double-computation]
         Concatenate chunk dicts → arrays of length n_genes
         padj = _apply_correction(pvals, method=correction)
         result_df = _build_result_df(gene_names, stats, padj, n_fg, n_bg)

5. Return dict[method_name, result_df]
```

### Backend return contract

Each non-DESeq2 backend receives `(X_fg_chunk, X_bg_chunk, **kwargs)` and returns a dict:

```python
{
    "pval":    np.ndarray,   # shape (chunk_genes,)
    "mean_fg": np.ndarray,
    "mean_bg": np.ndarray,
    "pct_fg":  np.ndarray,
    "pct_bg":  np.ndarray,
}
```

`_build_result_df` assembles the final DataFrame from these arrays plus `gene_names`, `padj`, `n_fg`, `n_bg` — no recomputation of means or pct values.

### Backend functions

| Backend | Implementation | Parallelism |
|---------|---------------|-------------|
| `_wilcoxon_backend` | `scipy.stats.mannwhitneyu`, vectorized per gene chunk | Gene-chunk ThreadPoolExecutor |
| `_ttest_backend` | `scipy.stats.ttest_ind`, vectorized per gene chunk | Gene-chunk ThreadPoolExecutor |
| `_mast_backend` | Two-part hurdle model via `statsmodels` (logistic + linear with CDR covariate); p-values combined via Fisher's method | Per-gene ThreadPoolExecutor |
| `_nb_glm_backend` | `statsmodels.genmod.GLM` with NB family, per gene | Per-gene ThreadPoolExecutor |
| `_deseq2_backend` | `pydeseq2` on pseudobulk-aggregated counts matrix | pydeseq2 internal |

**MAST implementation detail:** The hurdle model is a simplified Python equivalent of MAST (not a full MAST port). For each gene:
1. Discrete component: logistic regression predicting non-zero vs zero (binary), with CDR as covariate. Coefficient p-value for the group indicator is extracted.
2. Continuous component: OLS on expressed cells only (expression > 0), with CDR as covariate. Coefficient p-value for the group indicator is extracted.
3. Combined p-value: Fisher's combined probability method (`scipy.stats.combine_pvalues(method='fisher')`).

This is a deliberate approximation. The spec explicitly documents it as a "MAST-inspired hurdle model" rather than a faithful MAST reproduction.

---

## Scalability Design

- **Wilcoxon / t-test:** Gene-chunked vectorized computation. O(n_cells) per chunk. No per-gene Python loop.
- **NB-GLM / MAST:** Per-gene model fitting parallelized across genes via `ThreadPoolExecutor` (n_jobs).
- **DESeq2:** Pseudobulk aggregation reduces millions of cells to `n_replicates × n_genes` (small), then `pydeseq2` runs on that small matrix.
- All backends receive pre-sliced `X_fg` / `X_bg` arrays — never the full matrix.

---

## Dependencies

| Package | Required | Install |
|---------|----------|---------|
| `numpy` | Always | core |
| `scipy` | Always | core |
| `pandas` | Always | core |
| `shapely` | Always (spatial filter) | core |
| `statsmodels` | MAST + NB-GLM only | `pip install spatioloji_s[deg]` |
| `pydeseq2` | DESeq2 only | `pip install spatioloji_s[deg]` |

Both `statsmodels` and `pydeseq2` are lazy-imported inside their respective backend functions with `ImportError` hints directing users to `pip install spatioloji_s[deg]`.

---

## Error Handling

| Condition | Exception |
|-----------|-----------|
| `groupby` not in `cell_meta` | `ValueError` |
| `group_fg` values not found in groupby column | `ValueError` |
| fg or bg cell count < `min_cells` (enforced in `_build_cell_mask`) | `ValueError` |
| `"deseq2"` in methods but `replicate_key` is None | `ValueError` |
| `replicate_key` not in `cell_meta` | `ValueError` |
| Either fg or bg has fewer than 2 replicates (DESeq2 path) | `ValueError` |
| `pydeseq2` not installed | `ImportError` with `pip install spatioloji_s[deg]` |
| `statsmodels` not installed (MAST or NB-GLM) | `ImportError` with `pip install spatioloji_s[deg]` |
| Unknown method name | `ValueError` |

---

## Testing (`tests/unit/test_deg.py`)

- Group mask construction: one-vs-rest, pairwise, custom multi-group
- Spatial filter: bounding box and polygon (point-in-polygon)
- `_build_cell_mask` enforces `min_cells` before group slicing
- Each backend (Wilcoxon, t-test, MAST, NB-GLM) on 500-cell × 100-gene synthetic sparse matrix
- DESeq2 pseudobulk: sum correctness, replicate count, `n_replicates < 2` raises `ValueError`
- Output DataFrame schema: all required columns present, correct dtypes, sorted by `padj` (NaN last)
- `min_cells` guard raises `ValueError`
- `pydeseq2` / `statsmodels` missing raises `ImportError` with hint (mock import to test)
- `n_jobs` parallelism: Wilcoxon and t-test results identical between n_jobs=1 and n_jobs=4; NB-GLM and MAST results numerically close within tolerance (rtol=1e-5) due to optimizer variability
- DESeq2 path: padj comes from pydeseq2 output, not re-applied BH correction
- `log2fc` formula: verified against manual calculation with pseudocount

---

## Integration

- Add to `processing/__init__.py`: export `run_deg`, `deg_wilcoxon`, `deg_ttest`, `deg_mast`, `deg_nb_glm`, `deg_deseq2`
- Add `statsmodels` and `pydeseq2` to `[deg]` optional dependency group in `pyproject.toml`
- No changes to core dependencies
