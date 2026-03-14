# Spatial Gradient Analysis & Immune Infiltration Scoring — Design Spec

> **Date:** 2026-03-13
> **Status:** Approved
> **Depends on:** `identify_interface` + `InterfaceResult` (implemented)

## Overview

Two new analysis modules for spatioloji_s that build on `InterfaceResult` from the interface detection system:

1. **Spatial Gradient Analysis** — quantify how gene expression and gene programs change with distance from the interface
2. **Immune Infiltration Scoring** — measure how deeply immune cells penetrate into a target region across the interface

Both modules support polygon-based and point-based spatial modes.

## Architecture: Two Independent Modules (Approach 1)

- `spatial/polygon/gradient.py` and `spatial/polygon/infiltration.py` as separate modules
- `spatial/point/gradient.py` and `spatial/point/infiltration.py` mirror modules
- Shared signed-distance utility in `spatial/_distance_utils.py`
- Result dataclasses in `spatial/_gradient_types.py` and `spatial/_infiltration_types.py`
- Both take `InterfaceResult` as input — explicit dependency, no hidden coupling

---

## Component 1: Shared Distance Utility

**File:** `src/spatioloji_s/spatial/_distance_utils.py`

### Function

```python
def signed_distance_to_interface(
    sp,
    interface_result: InterfaceResult,
    coord_type: str = "global",
    unsigned: bool = False,
) -> pd.Series:
```

### Behavior

- **Coordinate access:** uses `coord_type` parameter to select coordinates, matching `identify_interface` convention:
  - `"global"` → `sp.spatial.x_global`, `sp.spatial.y_global`
  - `"local"` → `sp.spatial.x_local`, `sp.spatial.y_local`
- Computes distance from each cell centroid to `interface_result.contour` (MultiLineString) using `shapely.geometry.Point(x, y).distance(contour)`
- **Signed distance:** positive = region A side, negative = region B side
  - Side determined by checking cell's region label from `interface_result.cell_labels`
  - Cells labeled as `region_a_interface` or `interior_a` → positive
  - Cells labeled as `region_b_interface` or `interior_b` → negative
  - Cells labeled `other` → positive (unsigned distance)
- If `unsigned=True`, returns absolute distances
- If `contour is None`, raises `ValueError` (contour required for distance computation)
- Returns `pd.Series` indexed by cell ID

---

## Component 2: Gradient Analysis

### Result Dataclass

**File:** `src/spatioloji_s/spatial/_gradient_types.py`

```python
@dataclass
class GradientResult:
    distances: pd.Series          # signed distance per cell
    gene_gradients: pd.DataFrame  # rows=genes, cols=[coef, pvalue, r2, trend]
    program_gradients: pd.DataFrame  # rows=programs, cols=[coef, pvalue, r2, trend]
    program_scores: pd.DataFrame  # cells × programs matrix (mean expression of gene set)
    bins: pd.DataFrame            # long-form: cols=[distance_bin, gene, mean_expr, std_expr]
    region_a: str | list[str]     # matches InterfaceResult.region_a type
    region_b: str | list[str]     # matches InterfaceResult.region_b type
```

### Main Function

**File:** `src/spatioloji_s/spatial/polygon/gradient.py` (and `point/gradient.py`)

```python
def compute_gradient(
    sp,
    interface_result: InterfaceResult,
    genes: list[str] | None = None,
    programs: dict[str, list[str]] | None = None,
    n_bins: int = 20,
    method: str = "ols",
    auto_programs: str | None = None,  # "nmf" or "pca"
    n_auto_programs: int = 5,
    coord_type: str = "global",
    unsigned: bool = False,
) -> GradientResult:
```

### Behavior

- **Expression access:** gene expression is retrieved via `sp.expression.to_dataframe()` which returns a DataFrame (cells × genes). For a subset of genes, index with `expr_df[genes]`.
- **genes**: list of gene names to analyze, or `None` for all genes
- **programs**: dict of `{name: [gene_list]}` for user-defined gene modules, or `None` to skip
- **auto_programs**: `"nmf"` or `"pca"` — auto-discover gene programs from expression matrix
  - NMF: `sklearn.decomposition.NMF` on `sp.expression.get_dense()`, top genes per component become programs (guarded with `try/except ImportError`)
  - PCA: `sklearn.decomposition.PCA`, loadings define programs (guarded with `try/except ImportError`)
- **n_auto_programs**: number of programs to discover (default 5)
- **n_bins**: number of equal-width distance bins for expression-vs-distance curves
- **method**: `"ols"` — linear regression via `scipy.stats.linregress`
- **coord_type**: `"global"` or `"local"` — passed through to `signed_distance_to_interface`
- For each gene/program, fits expression ~ distance:
  - `coef`: slope of regression
  - `pvalue`: p-value of slope
  - `r2`: R-squared
  - `trend`: "increasing_toward_a" (positive slope), "increasing_toward_b" (negative slope), or "flat" (p > 0.05)
- **program_scores**: per-cell mean expression of genes in each program
- **bins**: long-form DataFrame with columns `[distance_bin, gene, mean_expr, std_expr]` — one row per bin-gene combination

### Point Mode

`src/spatioloji_s/spatial/point/gradient.py` — thin wrapper that imports and calls the shared implementation from `spatial/_distance_utils.py` + gradient logic from polygon module. Both polygon and point modes use centroid distances (the distance utility is coordinate-based, not polygon-specific), so the core logic is shared. The point module re-exports `compute_gradient` from the polygon module directly.

---

## Component 3: Infiltration Scoring

### Result Dataclass

**File:** `src/spatioloji_s/spatial/_infiltration_types.py`

```python
@dataclass
class InfiltrationResult:
    distances: pd.Series              # signed distance per cell
    cell_classifications: pd.Series   # "infiltrating", "resident", "other"
    per_type_metrics: pd.DataFrame    # rows=immune cell types, cols below
    # cols: median_depth, max_depth, density_slope, density_pvalue,
    #        infiltration_fraction, n_infiltrating, n_resident
    region_a: str | list[str]         # matches InterfaceResult.region_a type
    region_b: str | list[str]         # matches InterfaceResult.region_b type
    target_region: str                # which region immune cells infiltrate into
```

### Main Function

**File:** `src/spatioloji_s/spatial/polygon/infiltration.py` (and `point/infiltration.py`)

```python
def score_infiltration(
    sp,
    interface_result: InterfaceResult,
    immune_col: str,
    immune_types: list[str],
    target_region: str | None = None,
    depth_bins: int = 10,
    coord_type: str = "global",
) -> InfiltrationResult:
```

Note: `unsigned` parameter removed from infiltration — signed distance is fundamental to distinguishing "infiltrating" from "resident" cells. Infiltration scoring always uses signed distance internally.

### Behavior

- **immune_col**: column in `sp.cell_meta` identifying cell types (e.g., `"cell_type"`)
- **immune_types**: list of cell type labels considered "immune" (required parameter, no default).
- **target_region**: which region the immune cells are infiltrating into (`region_a` or `region_b`). If `None`, auto-detect as the region with fewer immune cells.
- **coord_type**: `"global"` or `"local"` — passed through to `signed_distance_to_interface`
- **depth_bins**: number of equal-width distance bins for density gradient computation. Immune cells are binned into `depth_bins` bins by distance from interface, then linear regression is fit on cell count per bin ~ bin distance.
- **Penetration depth**: for each immune type found in the target region, compute median and max distance past the interface (absolute distance into target region)
- **Density gradient**: bin immune cells into `depth_bins` equal-width distance bins, fit linear regression on cell count ~ distance, report slope + p-value
- **Infiltration fraction**: `n_immune_cells_in_target_region / total_immune_cells_of_that_type`
- **cell_classifications**:
  - Immune cells past the interface (in target region) → "infiltrating"
  - Immune cells on their home side → "resident"
  - All other cells → "other"

### Point Mode

`src/spatioloji_s/spatial/point/infiltration.py` — thin wrapper that re-exports `score_infiltration` from the polygon module. Both modes use centroid distances via the shared utility, so the core logic is identical.

---

## Component 4: Visualization

Three plot functions added to `visualization/polygon_plots.py` and `visualization/point_plots.py`.

### `plot_gradient_curve`

```python
def plot_gradient_curve(
    gradient_result: GradientResult,
    genes: list[str] | None = None,
    programs: list[str] | None = None,
    n_cols: int = 3,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
```

- Expression (y) vs signed distance (x) line plot
- Uses `gradient_result.bins` for smooth curves
- Vertical dashed line at x=0 (interface)
- Shaded band for ±1 std per bin
- Subplots: one per gene/program, wrapped in n_cols grid
- Annotates each subplot with slope, R², p-value

### `plot_spatial_distance`

```python
def plot_spatial_distance(
    sp,
    distances: pd.Series,
    interface_result: InterfaceResult | None = None,
    coord_type: str = "global",
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
```

- Spatial scatter/polygon map colored by signed distance
- Diverging colormap centered at 0 (interface)
- Optionally overlays interface contour if `interface_result` provided
- Works for both gradient and infiltration (takes any distance Series)

### `plot_infiltration_summary`

```python
def plot_infiltration_summary(
    infiltration_result: InfiltrationResult,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
```

- Three side-by-side subplots, one per metric:
  - Panel 1: median penetration depth per immune type (horizontal bars)
  - Panel 2: density slope per immune type
  - Panel 3: infiltration fraction per immune type
- Each panel independently scaled

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/spatioloji_s/spatial/_distance_utils.py` | Create | Signed distance computation |
| `src/spatioloji_s/spatial/_gradient_types.py` | Create | GradientResult dataclass |
| `src/spatioloji_s/spatial/_infiltration_types.py` | Create | InfiltrationResult dataclass |
| `src/spatioloji_s/spatial/polygon/gradient.py` | Create | Polygon gradient analysis |
| `src/spatioloji_s/spatial/polygon/infiltration.py` | Create | Polygon infiltration scoring |
| `src/spatioloji_s/spatial/point/gradient.py` | Create | Thin wrapper, re-exports from polygon |
| `src/spatioloji_s/spatial/point/infiltration.py` | Create | Thin wrapper, re-exports from polygon |
| `src/spatioloji_s/spatial/polygon/__init__.py` | Modify | Export new functions + types |
| `src/spatioloji_s/spatial/point/__init__.py` | Modify | Export new functions + types |
| `src/spatioloji_s/visualization/polygon_plots.py` | Modify | Add 3 plot functions |
| `src/spatioloji_s/visualization/point_plots.py` | Modify | Add 3 plot functions |
| `src/spatioloji_s/visualization/__init__.py` | Modify | Export new plot functions |
| `tests/unit/test_gradient.py` | Create | Gradient analysis tests |
| `tests/unit/test_infiltration.py` | Create | Infiltration scoring tests |
| `tests/conftest.py` | Modify | Add fixtures for gradient/infiltration |

---

## Dependencies

- **Required:** numpy, pandas, scipy (linregress), shapely, matplotlib
- **Optional:** scikit-learn (NMF, PCA for auto_programs) — guarded with try/except ImportError
- **Internal:** `InterfaceResult` from `spatial/_interface_types.py`

## Design Decisions

1. **Approach 1 (two independent modules)** chosen over combined module or extending InterfaceResult — clean separation, follows existing package patterns
2. **Signed distance** by default with unsigned option — enables asymmetric gradient analysis
3. **Both gene-level and program-level gradients** — user-provided gene sets + auto-discovery (NMF/PCA)
4. **All three infiltration metrics** — penetration depth, density gradient, infiltration fraction
5. **Result dataclasses** (not sp.obs columns) — consistent with InterfaceResult pattern
6. **InterfaceResult as explicit input** — no auto-detection of interface, user must run identify_interface first
