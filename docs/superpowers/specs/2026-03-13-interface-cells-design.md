# Interface Cells Analysis — Design Spec

## Goal

Identify and characterize the interface (boundary zone) between two specific
cell regions (e.g., tumor vs stroma) using polygon-contact or point-KNN graphs,
with an alternative KDE density-boundary method. Produce cell labels, geometric
contour lines, per-segment metrics, and two visualization functions.

## Scope

- New file: `spatial/polygon/interface.py` (polygon-based)
- New file: `spatial/point/interface.py` (point-based)
- Shared dataclass: `spatial/_interface_types.py` (canonical location for
  `InterfaceResult`, imported by both submodules)
- New plots in: `visualization/polygon_plots.py` and `visualization/point_plots.py`
- Exports added to `spatial/polygon/__init__.py`, `spatial/point/__init__.py`,
  `visualization/__init__.py`
- Functions are accessed via fully-qualified paths:
  `sj.spatial.polygon.interface.identify_interface(...)` and
  `sj.spatial.point.interface.identify_interface(...)`.
  Neither is re-exported at the `spatial` top level to avoid name collisions.

Not in scope: expression-level analysis of interface cells (users can feed
interface cell labels into the existing DEG module).

---

## Data structures

### `InterfaceResult` (shared dataclass)

**Canonical location:** `spatial/_interface_types.py`. Imported by both
`polygon/interface.py` and `point/interface.py`.

```python
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiLineString

@dataclass
class InterfaceResult:
    """Container for interface analysis results."""

    cell_labels: pd.Series
    # Values: "region_a_interface", "region_b_interface",
    #         "interior_a", "interior_b", "other"
    # Index: cell IDs (matching sp.cell_index)

    contour: MultiLineString | None
    # Combined interface geometry (union of all segment lines).
    # None when no interface is found (empty result).

    segments: gpd.GeoDataFrame
    # Columns: segment_id (int), geometry (LineString), length (float),
    #          tortuosity (float), n_cells_a (int), n_cells_b (int)
    # One row per disconnected interface segment.
    # CRS is always None (pixel/micron coordinates, no EPSG).
    # Empty GeoDataFrame (0 rows, correct columns) when no interface found.

    summary: dict
    # Keys: total_length (float), n_segments (int), mean_tortuosity (float),
    #       n_interface_a (int), n_interface_b (int)
    # When empty: all values 0 or 0.0.

    region_a: str | list[str]  # region A label(s) used
    region_b: str | list[str]  # region B label(s) used
    method: str                # "graph" or "density"
```

**Empty result** (returned when no interface is found):
- `cell_labels`: all cells labelled `"interior_a"`, `"interior_b"`, or `"other"`
  (no interface labels)
- `contour`: `None`
- `segments`: empty `GeoDataFrame` with correct column schema
- `summary`: `{"total_length": 0.0, "n_segments": 0, "mean_tortuosity": 0.0,
  "n_interface_a": 0, "n_interface_b": 0}`

---

## Core function: `identify_interface`

Defined separately in `polygon/interface.py` and `point/interface.py`.
Same signature; internal implementation differs.

### Graph attribute abstraction

`PolygonSpatialGraph` exposes cell IDs as `cell_index` (pd.Index).
`PointSpatialGraph` exposes cell IDs as `cell_ids` (pd.Index).

Each module accesses the correct attribute directly — `polygon/interface.py`
uses `graph.cell_index`, `point/interface.py` uses `graph.cell_ids`. No
shared base class or duck-typing needed.

```python
def identify_interface(
    sp: spatioloji,
    graph=None,                       # PolygonSpatialGraph or PointSpatialGraph
    group_col: str,                   # cell_meta column, e.g. "cell_type"
    region_a: str | list[str],        # e.g. "Tumor" or ["Tumor_A", "Tumor_B"]
    region_b: str | list[str],        # e.g. "Stromal"
    method: str = "graph",            # "graph" or "density"
    min_interface_cells: int = 3,     # min cells per segment to retain
    bandwidth: float | None = None,   # KDE bandwidth (density method only)
    distance_threshold: float | None = None,  # proximity to contour (density only)
    coord_type: str = "global",       # "global" or "local"
    store: bool = True,               # store cell_labels in sp.cell_meta
) -> InterfaceResult
```

### Parameters

| Parameter | Description |
|---|---|
| `sp` | spatioloji object |
| `graph` | Pre-built spatial graph. **Required** for `method="graph"`. **Optional** for `method="density"` — if provided, used to auto-estimate `distance_threshold` via median nearest-neighbor distance; if `None`, `distance_threshold` must be given explicitly. |
| `group_col` | Column in `cell_meta` defining regions |
| `region_a` | Label(s) for region A. If list, all are treated as region A. |
| `region_b` | Label(s) for region B. If list, all are treated as region B. |
| `method` | `"graph"` (default) — adjacency-based; `"density"` — KDE boundary |
| `min_interface_cells` | **Per-segment** minimum: segments with fewer than this many cells on either side are dropped from the result. |
| `bandwidth` | KDE bandwidth for density method. Auto-estimated via Scott's rule if None. |
| `distance_threshold` | Max distance from KDE contour to label a cell as interface. Auto-estimated as median nearest-neighbor distance from `graph` if None. |
| `coord_type` | Which spatial coordinates to use. Coordinates are pulled via `sp.spatial.x_global / y_global` (or `x_local / y_local`). |
| `store` | If True, add `"interface_label"` column to `sp.cell_meta` |

### Validation

- `group_col` must exist in `cell_meta`
- All labels in `region_a` and `region_b` must exist in the column
- `region_a` and `region_b` must not overlap
- Each region must contain at least 1 cell (raise `ValueError` otherwise)
- `method="graph"` requires `graph` to be provided (raise `ValueError` if None)
- `method="density"` with `graph=None` requires `distance_threshold` to be set
  explicitly (raise `ValueError` if both are None)

---

## Method A: Graph-based (`method="graph"`)

### Step 1 — Find cross-region edges

Scan the adjacency matrix (CSR) for cell pairs `(i, j)` where `i ∈ region_a`
and `j ∈ region_b` (or vice versa). Vectorized: build boolean masks for
region_a and region_b cell indices, then extract cross-region edges from the
COO representation.

### Step 2 — Label cells

- Cells in region_a with ≥1 region_b neighbor → `"region_a_interface"`
- Cells in region_b with ≥1 region_a neighbor → `"region_b_interface"`
- Remaining cells in region_a → `"interior_a"`
- Remaining cells in region_b → `"interior_b"`
- All other cells → `"other"`

### Step 3 — Build contour

**Polygon mode** (`polygon/interface.py`):
- For each cross-region pair `(i, j)`, compute `polygon_i.intersection(polygon_j)`.
  Keep results that are `LineString` or `MultiLineString` (shared boundary edges).
- Union all shared edges via `shapely.ops.unary_union` → `MultiLineString`.

**Point mode** (`point/interface.py`):
- For each cross-region pair `(i, j)`, compute the midpoint of their centroids.
- Order midpoints along the interface by connecting midpoints that share a cell
  into sequential `LineString` segments (walk the shared-cell adjacency).
- If midpoint ordering produces disconnected short fragments (< 3 midpoints),
  connect them with straight segments between nearest endpoints. No alpha-shape
  fallback — keep it simple.

### Step 4 — Segment detection

Build a subgraph of only the cross-region edges. Find connected components
(using `scipy.sparse.csgraph.connected_components`). Each component is one
interface segment.

**Assigning geometry to segments:** After `unary_union`, iterate over
`contour.geoms` (the individual `LineString` components of the
`MultiLineString`). Assign each component to the segment whose interface cells
have the closest centroid (by minimum distance from the LineString to the
component's cell centroids). In practice, the connected components of the
cell graph and the disconnected LineStrings of the geometry will correspond
1-to-1 in most cases.

Per segment:
- `length`: total length of the segment geometry (`.length` property)
- `tortuosity`: `length / euclidean_distance(start_point, end_point)`.
  - Tortuosity ≥ 1.0 (straight line = 1.0, more winding = higher).
  - **Degenerate cases**: if endpoints coincide (closed loop) or the segment is
    a single point, set tortuosity to `np.inf`.
- `n_cells_a`, `n_cells_b`: interface cell counts on each side for that component
- Drop segments where `n_cells_a < min_interface_cells` or
  `n_cells_b < min_interface_cells`

---

## Method B: Density-based (`method="density"`)

### Step 1 — Compute KDE surfaces

Extract x/y coordinates for cells in region_a and region_b using
`sp.spatial.x_global / y_global` (or `x_local / y_local` per `coord_type`).
Fit `scipy.stats.gaussian_kde` for each region. If `bandwidth` is None,
use scipy's default (Scott's rule).

Evaluate both KDEs on a shared 2D grid (resolution auto-scaled to dataset
extent, e.g., 200×200 bins).

### Step 2 — Find decision boundary

Compute `diff = density_a - density_b` on the grid. Extract the zero-contour
using `skimage.measure.find_contours(diff, level=0.0)`. Convert pixel
coordinates back to spatial coordinates. Each contour becomes a `LineString`.

`scikit-image` is an optional dependency guarded with `try/except ImportError`.

### Step 3 — Label cells

For each cell, compute its minimum distance to the nearest contour line.
Cells within `distance_threshold` and belonging to region_a →
`"region_a_interface"`. Same for region_b. Cells outside the threshold or not
in either region → `"interior_a"`, `"interior_b"`, or `"other"`.

If `distance_threshold` is None:
- If `graph` is provided, auto-estimate as the median nearest-neighbor distance.
- If `graph` is None, raise `ValueError` (caught at validation).

### Step 4 — Segment detection & metrics

Each disconnected contour line is a separate segment. Compute length,
tortuosity (same formula and degenerate-case handling as Method A), and count
interface cells on each side per segment. Drop segments with fewer than
`min_interface_cells`.

---

## Visualization

### `plot_interface_map`

Added to `visualization/polygon_plots.py` and `visualization/point_plots.py`.

```python
def plot_interface_map(
    sp: spatioloji,
    interface_result: InterfaceResult,
    coord_type: str = "global",
    colors: dict | None = None,
    contour_color: str = "black",
    contour_width: float = 2.0,
    poly_alpha: float = 0.7,
    show_interior: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (9, 8),
    title: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure | None
```

**Behavior:**
- Polygon mode: filled polygons coloured by `cell_labels`, interface contour
  overlaid as a thick line.
- Point mode: scatter dots coloured by `cell_labels`, contour overlaid.
- Default colours: `region_a_interface` = warm red (`#e74c3c`),
  `region_b_interface` = warm blue (`#3498db`), `interior_a` = light red
  (`#fadbd8`), `interior_b` = light blue (`#d6eaf8`), `other` = light grey
  (`#e8e8e8`).
- If `show_interior=False`, interior cells and `other` cells rendered in grey.
- Legend with cell counts per label.
- If `interface_result.contour` is None (empty result), plot cells with
  interior/other labels only; no contour drawn.
- If `ax` is provided, draw into that axes (for multi-panel figures).

### `plot_interface_metrics`

Added to both visualization files (shared logic).

```python
def plot_interface_metrics(
    interface_result: InterfaceResult,
    metric: str = "length",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure | None
```

**Behavior:**
- Horizontal bar chart. Each bar = one interface segment ("Segment 1", etc.)
- `metric` selects which column: `"length"`, `"tortuosity"`, `"n_cells_a"`,
  `"n_cells_b"`.
- Bars coloured by metric intensity (sequential colormap).
- If `segments` is empty (no interface found), display an empty plot with a
  "No interface segments found" annotation.

---

## Module exports

### `spatial/_interface_types.py`
```python
# Canonical location for InterfaceResult
from dataclasses import dataclass
...
```

### `spatial/polygon/__init__.py`
```python
from ..\_interface_types import InterfaceResult
from .interface import identify_interface
```

### `spatial/point/__init__.py`
```python
from .._interface_types import InterfaceResult
from .interface import identify_interface
```

### `visualization/__init__.py`
```python
from .polygon_plots import plot_interface_map, plot_interface_metrics
# point versions: visualization.point_plots.plot_interface_map, etc.
```

Neither `identify_interface` function is re-exported at the `spatial` top
level. Users access them via `sj.spatial.polygon.interface.identify_interface`
or `sj.spatial.point.interface.identify_interface`.

---

## Usage examples

```python
import spatioloji_s as sj

sp = sj.read_cosmx("path/to/data/")

# --- Polygon-based (graph method) ---
g = sj.spatial.polygon.graph.build_contact_graph(sp)
result = sj.spatial.polygon.interface.identify_interface(
    sp, g,
    group_col="cell_type",
    region_a="Tumor",
    region_b="Stromal",
    method="graph",
)

# Access results
print(result.summary)
# {'total_length': 1234.5, 'n_segments': 3, 'mean_tortuosity': 1.42, ...}

# Interface cells for downstream DEG
interface_tumor = sp.cell_meta[sp.cell_meta["interface_label"] == "region_a_interface"]

# Visualize
sj.visualization.plot_interface_map(sp, result)
sj.visualization.plot_interface_metrics(result, metric="length")

# --- Density method (smooth contours, with graph for auto-threshold) ---
result_kde = sj.spatial.polygon.interface.identify_interface(
    sp, g,
    group_col="cell_type",
    region_a="Tumor",
    region_b="Stromal",
    method="density",
    bandwidth=50.0,
)

# --- Density method without graph (explicit threshold) ---
result_kde2 = sj.spatial.polygon.interface.identify_interface(
    sp,
    graph=None,
    group_col="cell_type",
    region_a="Tumor",
    region_b="Stromal",
    method="density",
    bandwidth=50.0,
    distance_threshold=20.0,
)

# --- Point-based (no polygons needed) ---
pg = sj.spatial.point.graph.build_knn_graph(sp, k=10)
result_point = sj.spatial.point.interface.identify_interface(
    sp, pg,
    group_col="cell_type",
    region_a="Tumor",
    region_b="Stromal",
)

# --- Multiple labels as one region ---
result = sj.spatial.polygon.interface.identify_interface(
    sp, g,
    group_col="cell_type",
    region_a=["Tumor_A", "Tumor_B"],
    region_b="Stromal",
)
```

---

## Dependencies

- **Required**: numpy, pandas, scipy, shapely, geopandas (already in core deps)
- **Density method**: scikit-image (`find_contours`) — optional dep guarded with
  `try/except ImportError`
- **No new required dependencies**

## Error handling

- `ValueError` if `group_col` not in `cell_meta`
- `ValueError` if any label in `region_a`/`region_b` not found in column
- `ValueError` if `region_a` and `region_b` overlap
- `ValueError` if either region has 0 cells
- `ValueError` if `method="graph"` and `graph is None`
- `ValueError` if `method="density"`, `graph is None`, and
  `distance_threshold is None`
- `ImportError` if `method="density"` and `scikit-image` not installed
- `UserWarning` if no cross-region edges found (graph method) — returns empty result
- `UserWarning` if KDE fails to find a contour (density method) — returns empty result
