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
    use_raw: bool = True,
    spatial_filter: dict | None = None,
    replicate_key: str | None = None,
    min_cells: int = 10,
    n_jobs: int = 1,
    gene_chunk_size: int = 500,
    correction: str = "fdr_bh",
) -> dict[str, pd.DataFrame]
```

### Convenience wrappers

Five thin wrappers that call `run_deg` with a single method fixed:

- `deg_wilcoxon(...)`
- `deg_ttest(...)`
- `deg_mast(...)`
- `deg_nb_glm(...)`
- `deg_deseq2(...)`

### Output format

Each value in the returned `dict[str, pd.DataFrame]` has columns:

| Column | Description |
|--------|-------------|
| `gene` | Gene name |
| `log2fc` | Log2 fold-change (fg / bg) |
| `mean_fg` | Mean expression in foreground |
| `mean_bg` | Mean expression in background |
| `pct_fg` | Fraction of fg cells expressing the gene (> 0) |
| `pct_bg` | Fraction of bg cells expressing the gene (> 0) |
| `pval` | Raw p-value |
| `padj` | Adjusted p-value (BH by default) |
| `n_fg` | Number of foreground cells |
| `n_bg` | Number of background cells |

Sorted by `padj` ascending. Dict keys are method names (e.g., `"wilcoxon"`, `"ttest"`).

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

Applied before group mask construction. Restricts the cell universe to cells within the spatial region before any DEG comparison.

---

## Architecture: Shared Pipeline + Pluggable Backends

### Private pipeline (inside `run_deg`)

```
1. _build_cell_mask(spatioloji_obj, groupby, group_fg, group_bg, spatial_filter)
   → fg_idx (int array), bg_idx (int array)

2. _get_expr(spatioloji_obj, layer, use_raw)
   → X (sparse or dense, n_cells × n_genes)

3. For each method in methods:
   └── if method == "deseq2":
         _aggregate_pseudobulk(X, fg_idx, bg_idx, replicate_key, cell_meta)
         → counts_fg (n_replicates × n_genes), counts_bg
         _deseq2_backend(counts_fg, counts_bg, gene_names)
       else:
         Gene-chunked loop (gene_chunk_size):
           _<method>_backend(X_fg_chunk, X_bg_chunk) → stats chunk
         Concatenate chunks → raw stats DataFrame

4. _apply_correction(pvals, method=correction) → padj

5. _build_result_df(gene_names, stats, padj, X_fg, X_bg)
   → DataFrame with all output columns, sorted by padj
```

### Backend functions

| Backend | Implementation | Parallelism |
|---------|---------------|-------------|
| `_wilcoxon_backend` | `scipy.stats.mannwhitneyu`, vectorized per gene chunk | Gene-chunk ThreadPoolExecutor |
| `_ttest_backend` | `scipy.stats.ttest_ind`, vectorized per gene chunk | Gene-chunk ThreadPoolExecutor |
| `_mast_backend` | Two-part hurdle model via `statsmodels` (logistic + linear), CDR covariate | Per-gene ThreadPoolExecutor |
| `_nb_glm_backend` | `statsmodels.genmod.GLM` with NB family, per gene | Per-gene ThreadPoolExecutor |
| `_deseq2_backend` | `pydeseq2` on pseudobulk-aggregated counts matrix | pydeseq2 internal |

**MAST note:** Implemented as a pure-Python hurdle model equivalent. CDR (cellular detection rate = fraction of genes with non-zero counts per cell) is added as a covariate. This captures the core MAST concept without requiring R or an R bridge.

---

## Scalability Design

- **Wilcoxon / t-test:** Gene-chunked vectorized computation. O(n_cells) per chunk. No per-gene Python loop.
- **NB-GLM / MAST:** Per-gene model fitting parallelized across genes via `ThreadPoolExecutor` (n_jobs).
- **DESeq2:** Pseudobulk aggregation reduces millions of cells to n_replicates × n_genes (small), then `pydeseq2` runs on that small matrix.
- All backends receive pre-sliced numpy arrays (fg/bg separately), never the full matrix.

---

## Dependencies

| Package | Required | Install |
|---------|----------|---------|
| `numpy` | Always | core |
| `scipy` | Always | core |
| `pandas` | Always | core |
| `statsmodels` | Always | core |
| `shapely` | Always (spatial filter) | core |
| `pydeseq2` | DESeq2 method only | `pip install spatioloji_s[deg]` |

`pydeseq2` is lazy-imported inside `_deseq2_backend` with an `ImportError` hint.

---

## Error Handling

| Condition | Exception |
|-----------|-----------|
| `groupby` not in `cell_meta` | `ValueError` |
| `group_fg` values not found in groupby column | `ValueError` |
| fg or bg cell count < `min_cells` | `ValueError` |
| `"deseq2"` in methods but `replicate_key` is None | `ValueError` |
| `replicate_key` not in `cell_meta` | `ValueError` |
| `pydeseq2` not installed | `ImportError` with install hint |
| Unknown method name | `ValueError` |

---

## Testing (`tests/unit/test_deg.py`)

- Group mask construction: one-vs-rest, pairwise, custom multi-group
- Spatial filter: bounding box and polygon
- Each backend on 500-cell × 100-gene synthetic sparse matrix
- Pseudobulk aggregation: sum correctness, replicate count
- Output DataFrame schema: all required columns present, sorted by padj
- `min_cells` guard raises `ValueError`
- `pydeseq2` missing raises `ImportError` with hint
- `n_jobs` parallelism: results identical between n_jobs=1 and n_jobs=4

---

## Integration

- Add to `processing/__init__.py`: export `run_deg`, `deg_wilcoxon`, `deg_ttest`, `deg_mast`, `deg_nb_glm`, `deg_deseq2`
- Add `pydeseq2` to `[deg]` optional dependency group in `pyproject.toml`
- `statsmodels` added to core dependencies if not already present
