# Interface Cells Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add interface cell identification between two named regions (e.g., tumor vs stroma) with graph-based and density-based methods, geometric contour output, per-segment metrics, and two visualization functions.

**Architecture:** Shared `InterfaceResult` dataclass in `spatial/_interface_types.py`. Separate `interface.py` modules in `spatial/polygon/` and `spatial/point/` with identical public API but different contour construction internals. Two plot functions added to existing visualization files.

**Tech Stack:** numpy, pandas, scipy (sparse, stats, csgraph), shapely, geopandas, matplotlib. Optional: scikit-image (density method contours).

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/spatioloji_s/spatial/_interface_types.py` | `InterfaceResult` dataclass |
| Create | `src/spatioloji_s/spatial/polygon/interface.py` | Polygon-based `identify_interface` |
| Create | `src/spatioloji_s/spatial/point/interface.py` | Point-based `identify_interface` |
| Create | `tests/unit/test_interface.py` | All interface tests |
| Modify | `src/spatioloji_s/spatial/polygon/__init__.py` | Add interface exports |
| Modify | `src/spatioloji_s/spatial/point/__init__.py` | Add interface exports |
| Modify | `src/spatioloji_s/visualization/polygon_plots.py` | `plot_interface_map`, `plot_interface_metrics` |
| Modify | `src/spatioloji_s/visualization/__init__.py` | Add viz exports |

---

## Chunk 1: Shared types, fixture, and polygon graph method

### Task 1: InterfaceResult dataclass

**Files:**
- Create: `src/spatioloji_s/spatial/_interface_types.py`
- Test: `tests/unit/test_interface.py`

- [ ] **Step 1: Write failing test for InterfaceResult**

```python
# tests/unit/test_interface.py
"""Tests for interface cell identification."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString

from spatioloji_s.spatial._interface_types import InterfaceResult


class TestInterfaceResult:
    """Tests for the InterfaceResult dataclass."""

    def test_dataclass_fields(self):
        """InterfaceResult has all required fields."""
        import geopandas as gpd

        labels = pd.Series(["interior_a", "interior_b"], index=["c0", "c1"])
        segs = gpd.GeoDataFrame(
            {"segment_id": pd.Series(dtype=int), "length": pd.Series(dtype=float),
             "tortuosity": pd.Series(dtype=float),
             "n_cells_a": pd.Series(dtype=int), "n_cells_b": pd.Series(dtype=int)},
            geometry=[],
        )
        result = InterfaceResult(
            cell_labels=labels,
            contour=None,
            segments=segs,
            summary={"total_length": 0.0, "n_segments": 0, "mean_tortuosity": 0.0,
                     "n_interface_a": 0, "n_interface_b": 0},
            region_a="Tumor",
            region_b="Stromal",
            method="graph",
        )
        assert result.contour is None
        assert result.method == "graph"
        assert result.summary["n_segments"] == 0

    def test_contour_accepts_multilinestring(self):
        """contour field accepts a MultiLineString."""
        import geopandas as gpd

        line = LineString([(0, 0), (1, 1)])
        contour = MultiLineString([line])
        labels = pd.Series(["region_a_interface", "region_b_interface"], index=["c0", "c1"])
        segs = gpd.GeoDataFrame(
            {"segment_id": [0], "length": [1.414], "tortuosity": [1.0],
             "n_cells_a": [1], "n_cells_b": [1]},
            geometry=[line],
        )
        result = InterfaceResult(
            cell_labels=labels, contour=contour, segments=segs,
            summary={"total_length": 1.414, "n_segments": 1,
                     "mean_tortuosity": 1.0, "n_interface_a": 1, "n_interface_b": 1},
            region_a="Tumor", region_b="Stromal", method="graph",
        )
        assert isinstance(result.contour, MultiLineString)
        assert len(result.segments) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONIOENCODING=utf-8 pytest tests/unit/test_interface.py::TestInterfaceResult -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spatioloji_s.spatial._interface_types'`

- [ ] **Step 3: Implement InterfaceResult**

```python
# src/spatioloji_s/spatial/_interface_types.py
"""Shared data structures for interface analysis."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiLineString


@dataclass
class InterfaceResult:
    """Container for interface analysis results.

    Attributes:
        cell_labels: Series indexed by cell ID with values
            ``"region_a_interface"``, ``"region_b_interface"``,
            ``"interior_a"``, ``"interior_b"``, or ``"other"``.
        contour: Combined interface geometry (union of all segment lines).
            ``None`` when no interface is found.
        segments: GeoDataFrame with one row per disconnected interface
            segment. Columns: ``segment_id``, ``geometry`` (LineString),
            ``length``, ``tortuosity``, ``n_cells_a``, ``n_cells_b``.
            CRS is always ``None`` (pixel/micron coordinates).
        summary: Dict with keys ``total_length``, ``n_segments``,
            ``mean_tortuosity``, ``n_interface_a``, ``n_interface_b``.
        region_a: Region A label(s) used.
        region_b: Region B label(s) used.
        method: ``"graph"`` or ``"density"``.
    """

    cell_labels: pd.Series
    contour: MultiLineString | None
    segments: gpd.GeoDataFrame
    summary: dict
    region_a: str | list[str]
    region_b: str | list[str]
    method: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONIOENCODING=utf-8 pytest tests/unit/test_interface.py::TestInterfaceResult -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/spatioloji_s/spatial/_interface_types.py tests/unit/test_interface.py
git commit -m "feat(spatial): add InterfaceResult dataclass"
```

---

### Task 2: Test fixture — sp_interface

**Files:**
- Modify: `tests/conftest.py`

Add a fixture with two clearly separated spatial regions (like `sp_deg` but with
polygon data). TypeA cells cluster in x ∈ [0, 500], TypeB in x ∈ [500, 1000].
Each cell gets a 4×4 square polygon. Cells near x=500 will be interface cells.

- [ ] **Step 1: Add sp_interface fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def sp_interface():
    """100-cell spatioloji with two spatially separated regions and polygons.

    TypeA (cells 0-49):  x_global in [50, 450], scattered in y [0, 1000].
    TypeB (cells 50-99): x_global in [550, 950], scattered in y [0, 1000].
    Interface cells: ~10 cells near x=500 on each side (x in [460, 540]).
    Each cell gets a 4×4 square polygon around its centroid.
    """
    np.random.seed(99)
    n_cells = 100
    n_genes = 10
    n_per_region = 50

    expression = np.random.poisson(2.0, (n_cells, n_genes)).astype(float)
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # TypeA: x in [50, 450] with a few cells near 490-500
    # TypeB: x in [550, 950] with a few cells near 500-510
    x_a = np.concatenate([
        np.random.uniform(50, 450, n_per_region - 5),
        np.random.uniform(460, 500, 5),  # interface cells
    ])
    x_b = np.concatenate([
        np.random.uniform(550, 950, n_per_region - 5),
        np.random.uniform(500, 540, 5),  # interface cells
    ])
    x_global = np.concatenate([x_a, x_b])
    y_global = np.random.uniform(0, 1000, n_cells)

    cell_meta = pd.DataFrame(
        {"cell_type": ["TypeA"] * n_per_region + ["TypeB"] * n_per_region},
        index=cell_ids,
    )

    spatial = {
        "x_global": x_global,
        "y_global": y_global,
        "x_local": x_global,
        "y_local": y_global,
    }

    # Build 4×4 square polygons
    rows = []
    for cid, cx, cy in zip(cell_ids, x_global, y_global, strict=True):
        for vx, vy in [
            (cx - 2, cy - 2), (cx + 2, cy - 2),
            (cx + 2, cy + 2), (cx - 2, cy + 2),
            (cx - 2, cy - 2),
        ]:
            rows.append({"cell": cid, "x_global_px": vx, "y_global_px": vy})
    polygons = pd.DataFrame(rows)

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        spatial_coords=spatial,
        polygons=polygons,
    )
```

- [ ] **Step 2: Verify fixture works**

Run: `PYTHONIOENCODING=utf-8 python -c "from tests.conftest import *; print('ok')"`
(Or add a trivial test that uses `sp_interface` and run it.)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add sp_interface fixture for interface analysis tests"
```

---

### Task 3: Polygon graph-method — validation and cross-region edges

**Files:**
- Create: `src/spatioloji_s/spatial/polygon/interface.py`
- Test: `tests/unit/test_interface.py`

- [ ] **Step 1: Write failing tests for validation and edge finding**

Add to `tests/unit/test_interface.py`:

```python
from spatioloji_s.spatial.polygon.interface import identify_interface
from spatioloji_s.spatial.polygon.graph import build_contact_graph


class TestPolygonValidation:
    """Tests for input validation in polygon identify_interface."""

    def test_invalid_group_col_raises(self, sp_interface):
        g = build_contact_graph(sp_interface)
        with pytest.raises(ValueError, match="not found in cell_meta"):
            identify_interface(sp_interface, g, group_col="nonexistent",
                               region_a="TypeA", region_b="TypeB")

    def test_invalid_region_label_raises(self, sp_interface):
        g = build_contact_graph(sp_interface)
        with pytest.raises(ValueError, match="not found"):
            identify_interface(sp_interface, g, group_col="cell_type",
                               region_a="Tumor", region_b="TypeB")

    def test_overlapping_regions_raises(self, sp_interface):
        g = build_contact_graph(sp_interface)
        with pytest.raises(ValueError, match="overlap"):
            identify_interface(sp_interface, g, group_col="cell_type",
                               region_a=["TypeA", "TypeB"], region_b="TypeA")

    def test_graph_required_for_graph_method(self, sp_interface):
        with pytest.raises(ValueError, match="graph.*required"):
            identify_interface(sp_interface, graph=None, group_col="cell_type",
                               region_a="TypeA", region_b="TypeB", method="graph")

    def test_density_without_graph_needs_threshold(self, sp_interface):
        with pytest.raises(ValueError, match="distance_threshold"):
            identify_interface(sp_interface, graph=None, group_col="cell_type",
                               region_a="TypeA", region_b="TypeB", method="density")


class TestPolygonGraphMethod:
    """Tests for the graph-based interface identification (polygon)."""

    def test_returns_interface_result(self, sp_interface):
        g = build_contact_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        assert isinstance(result, InterfaceResult)
        assert result.method == "graph"

    def test_cell_labels_values(self, sp_interface):
        g = build_contact_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        valid = {"region_a_interface", "region_b_interface",
                 "interior_a", "interior_b", "other"}
        assert set(result.cell_labels.unique()).issubset(valid)

    def test_cell_labels_index_matches_cells(self, sp_interface):
        g = build_contact_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        assert len(result.cell_labels) == len(sp_interface.cell_index)

    def test_interface_cells_detected(self, sp_interface):
        """With buffer_distance=50, cells near x=500 should be interface."""
        g = build_contact_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        n_a = (result.cell_labels == "region_a_interface").sum()
        n_b = (result.cell_labels == "region_b_interface").sum()
        assert n_a > 0, "Should detect TypeA interface cells"
        assert n_b > 0, "Should detect TypeB interface cells"

    def test_store_writes_to_cell_meta(self, sp_interface):
        g = build_contact_graph(sp_interface, buffer_distance=50)
        identify_interface(sp_interface, g, group_col="cell_type",
                           region_a="TypeA", region_b="TypeB", store=True)
        assert "interface_label" in sp_interface.cell_meta.columns

    def test_store_false_no_modification(self, sp_interface):
        g = build_contact_graph(sp_interface, buffer_distance=50)
        identify_interface(sp_interface, g, group_col="cell_type",
                           region_a="TypeA", region_b="TypeB", store=False)
        assert "interface_label" not in sp_interface.cell_meta.columns

    def test_list_region_labels(self, sp_interface):
        """region_a as a list should work."""
        g = build_contact_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a=["TypeA"], region_b="TypeB")
        assert isinstance(result, InterfaceResult)

    def test_no_interface_returns_empty(self, sp_interface):
        """With no buffer (contact only), far-apart cells have no interface."""
        g = build_contact_graph(sp_interface)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    min_interface_cells=1)
        # May or may not find interface depending on polygon proximity
        # But must return valid InterfaceResult either way
        assert isinstance(result, InterfaceResult)
        assert isinstance(result.summary, dict)
        assert "n_segments" in result.summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONIOENCODING=utf-8 pytest tests/unit/test_interface.py::TestPolygonValidation -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement polygon identify_interface (graph method)**

```python
# src/spatioloji_s/spatial/polygon/interface.py
"""Interface cell identification for polygon-based spatial analysis."""

from __future__ import annotations

import warnings
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

from spatioloji_s.spatial._interface_types import InterfaceResult


def _validate_inputs(
    sp,
    graph,
    group_col: str,
    region_a: str | list[str],
    region_b: str | list[str],
    method: str,
    distance_threshold: float | None,
) -> tuple[list[str], list[str]]:
    """Validate inputs and normalize region labels to lists.

    Args:
        sp: spatioloji object.
        graph: Spatial graph or None.
        group_col: Column in cell_meta.
        region_a: Region A label(s).
        region_b: Region B label(s).
        method: "graph" or "density".
        distance_threshold: Distance threshold for density method.

    Returns:
        Tuple of (region_a_list, region_b_list).

    Raises:
        ValueError: On invalid inputs.
    """
    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"'{group_col}' not found in cell_meta. "
                         f"Available: {list(sp.cell_meta.columns)}")

    a_list = [region_a] if isinstance(region_a, str) else list(region_a)
    b_list = [region_b] if isinstance(region_b, str) else list(region_b)

    col_vals = set(sp.cell_meta[group_col].dropna().unique())
    for label in a_list + b_list:
        if label not in col_vals:
            raise ValueError(f"Label '{label}' not found in '{group_col}'. "
                             f"Available: {sorted(col_vals)}")

    overlap = set(a_list) & set(b_list)
    if overlap:
        raise ValueError(f"region_a and region_b overlap on: {overlap}")

    labels = sp.cell_meta[group_col]
    n_a = labels.isin(a_list).sum()
    n_b = labels.isin(b_list).sum()
    if n_a == 0:
        raise ValueError(f"region_a {a_list} has 0 cells in '{group_col}'")
    if n_b == 0:
        raise ValueError(f"region_b {b_list} has 0 cells in '{group_col}'")

    if method == "graph" and graph is None:
        raise ValueError("graph is required for method='graph'")

    if method == "density" and graph is None and distance_threshold is None:
        raise ValueError(
            "distance_threshold must be set when method='density' and graph=None"
        )

    return a_list, b_list


def _empty_result(
    sp, a_list: list[str], b_list: list[str], group_col: str, method: str,
) -> InterfaceResult:
    """Build an empty InterfaceResult when no interface is found.

    Args:
        sp: spatioloji object.
        a_list: Region A labels.
        b_list: Region B labels.
        group_col: Column in cell_meta.
        method: "graph" or "density".

    Returns:
        InterfaceResult with no interface cells.
    """
    labels = sp.cell_meta[group_col]
    cell_labels = pd.Series("other", index=sp.cell_index)
    cell_labels[labels.isin(a_list)] = "interior_a"
    cell_labels[labels.isin(b_list)] = "interior_b"

    segs = gpd.GeoDataFrame(
        {"segment_id": pd.Series(dtype=int), "length": pd.Series(dtype=float),
         "tortuosity": pd.Series(dtype=float),
         "n_cells_a": pd.Series(dtype=int), "n_cells_b": pd.Series(dtype=int)},
        geometry=[],
    )
    return InterfaceResult(
        cell_labels=cell_labels, contour=None, segments=segs,
        summary={"total_length": 0.0, "n_segments": 0, "mean_tortuosity": 0.0,
                 "n_interface_a": 0, "n_interface_b": 0},
        region_a=a_list if len(a_list) > 1 else a_list[0],
        region_b=b_list if len(b_list) > 1 else b_list[0],
        method=method,
    )


def _compute_tortuosity(geom) -> float:
    """Compute tortuosity of a LineString.

    Args:
        geom: A shapely LineString.

    Returns:
        Tortuosity value (>= 1.0). np.inf for degenerate cases.
    """
    if geom is None or geom.is_empty:
        return np.inf
    length = geom.length
    if length == 0:
        return np.inf
    start, end = geom.coords[0], geom.coords[-1]
    endpoint_dist = np.hypot(end[0] - start[0], end[1] - start[1])
    if endpoint_dist == 0:
        return np.inf
    return length / endpoint_dist


def _graph_method(
    sp,
    graph,
    group_col: str,
    a_list: list[str],
    b_list: list[str],
    min_interface_cells: int,
    coord_type: str,
) -> InterfaceResult:
    """Graph-based interface identification.

    Args:
        sp: spatioloji object.
        graph: PolygonSpatialGraph.
        group_col: Column in cell_meta.
        a_list: Region A labels.
        b_list: Region B labels.
        min_interface_cells: Min cells per segment side.
        coord_type: "global" or "local".

    Returns:
        InterfaceResult.
    """
    cell_index = graph.cell_index
    adj = graph.adjacency
    labels = sp.cell_meta[group_col]

    # Build masks for cells in graph that belong to each region
    graph_labels = labels.reindex(cell_index)
    mask_a = graph_labels.isin(a_list).values
    mask_b = graph_labels.isin(b_list).values

    # Find cross-region edges from COO
    adj_coo = adj.tocoo()
    row, col = adj_coo.row, adj_coo.col
    cross_mask = (mask_a[row] & mask_b[col]) | (mask_b[row] & mask_a[col])

    if not cross_mask.any():
        warnings.warn("No cross-region edges found between the two regions.",
                      UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col, "graph")

    cross_rows = row[cross_mask]
    cross_cols = col[cross_mask]

    # Identify interface cell indices (in graph space)
    interface_a_idx = set(cross_rows[mask_a[cross_rows]]) | set(cross_cols[mask_a[cross_cols]])
    interface_b_idx = set(cross_rows[mask_b[cross_rows]]) | set(cross_cols[mask_b[cross_cols]])

    # Build cell labels
    cell_labels = pd.Series("other", index=sp.cell_index)
    cell_labels[labels.isin(a_list)] = "interior_a"
    cell_labels[labels.isin(b_list)] = "interior_b"

    for idx in interface_a_idx:
        cid = cell_index[idx]
        cell_labels.loc[cid] = "region_a_interface"
    for idx in interface_b_idx:
        cid = cell_index[idx]
        cell_labels.loc[cid] = "region_b_interface"

    # --- Connected components for segment detection ---
    # Build subgraph of cross-region edges only (upper triangle)
    upper = cross_rows < cross_cols
    cr_r, cr_c = cross_rows[upper], cross_cols[upper]
    # All cells involved in cross-region edges
    all_cross_cells = sorted(set(cr_r) | set(cr_c))
    if len(all_cross_cells) == 0:
        return _empty_result(sp, a_list, b_list, group_col, "graph")

    # Reindex to dense
    idx_map = {old: new for new, old in enumerate(all_cross_cells)}
    n_sub = len(all_cross_cells)
    sub_r = np.array([idx_map[r] for r in cr_r])
    sub_c = np.array([idx_map[c] for c in cr_c])
    sub_adj = sparse.csr_matrix(
        (np.ones(len(sub_r)), (sub_r, sub_c)), shape=(n_sub, n_sub)
    )
    sub_adj = sub_adj + sub_adj.T

    n_components, comp_labels = connected_components(sub_adj, directed=False)

    # --- Build contour from shared polygon edges ---
    gdf = sp.to_geopandas(coord_type=coord_type, include_metadata=False)
    geom_dict = {cid: gdf.loc[cid, "geometry"] for cid in gdf.index
                 if cid in set(cell_index[list(interface_a_idx | interface_b_idx)])}

    # For each cross-region pair, compute shared boundary
    shared_lines = []
    pair_component = []  # which component each line belongs to
    seen_pairs = set()
    for r, c in zip(cr_r, cr_c, strict=True):
        pair = (min(r, c), max(r, c))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        cid_r = cell_index[r]
        cid_c = cell_index[c]
        if cid_r not in geom_dict or cid_c not in geom_dict:
            continue

        try:
            shared = geom_dict[cid_r].intersection(geom_dict[cid_c])
        except Exception:
            continue

        if shared.is_empty:
            # Fall back to line between centroids
            cr_geom = geom_dict[cid_r].centroid
            cc_geom = geom_dict[cid_c].centroid
            shared = LineString([cr_geom.coords[0], cc_geom.coords[0]])

        if shared.geom_type == "LineString":
            shared_lines.append(shared)
            pair_component.append(comp_labels[idx_map[r]])
        elif shared.geom_type == "MultiLineString":
            for line in shared.geoms:
                shared_lines.append(line)
                pair_component.append(comp_labels[idx_map[r]])
        elif shared.geom_type == "Point":
            # Point contact — use centroid-to-centroid line
            cr_geom = geom_dict[cid_r].centroid
            cc_geom = geom_dict[cid_c].centroid
            shared_lines.append(LineString([cr_geom.coords[0], cc_geom.coords[0]]))
            pair_component.append(comp_labels[idx_map[r]])

    # --- Build segments GeoDataFrame ---
    seg_rows = []
    for comp_id in range(n_components):
        comp_cell_indices = [all_cross_cells[i] for i in range(n_sub)
                            if comp_labels[i] == comp_id]
        n_ca = sum(1 for i in comp_cell_indices if i in interface_a_idx)
        n_cb = sum(1 for i in comp_cell_indices if i in interface_b_idx)

        if n_ca < min_interface_cells or n_cb < min_interface_cells:
            continue

        # Collect geometry for this component
        comp_lines = [shared_lines[i] for i in range(len(shared_lines))
                      if pair_component[i] == comp_id]
        if not comp_lines:
            continue

        merged = unary_union(comp_lines)
        if merged.geom_type == "Point":
            continue
        if merged.geom_type not in ("LineString", "MultiLineString"):
            continue

        seg_rows.append({
            "segment_id": len(seg_rows),
            "geometry": merged,
            "length": merged.length,
            "tortuosity": _compute_tortuosity(merged) if merged.geom_type == "LineString"
                          else np.mean([_compute_tortuosity(g) for g in merged.geoms]),
            "n_cells_a": n_ca,
            "n_cells_b": n_cb,
        })

    if not seg_rows:
        warnings.warn("All interface segments dropped by min_interface_cells filter.",
                      UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col, "graph")

    segments = gpd.GeoDataFrame(seg_rows, geometry="geometry")
    segments.set_crs(epsg=None, inplace=True) if hasattr(segments, "set_crs") else None

    # Combined contour
    all_geoms = []
    for _, row in segments.iterrows():
        g = row.geometry
        if g.geom_type == "LineString":
            all_geoms.append(g)
        elif g.geom_type == "MultiLineString":
            all_geoms.extend(g.geoms)
    contour = MultiLineString(all_geoms) if all_geoms else None

    summary = {
        "total_length": float(segments["length"].sum()),
        "n_segments": len(segments),
        "mean_tortuosity": float(segments["tortuosity"].replace(np.inf, np.nan).mean()),
        "n_interface_a": int((cell_labels == "region_a_interface").sum()),
        "n_interface_b": int((cell_labels == "region_b_interface").sum()),
    }

    return InterfaceResult(
        cell_labels=cell_labels,
        contour=contour,
        segments=segments,
        summary=summary,
        region_a=a_list if len(a_list) > 1 else a_list[0],
        region_b=b_list if len(b_list) > 1 else b_list[0],
        method="graph",
    )


def identify_interface(
    sp,
    graph=None,
    group_col: str = "cell_type",
    region_a: str | list[str] = "",
    region_b: str | list[str] = "",
    method: Literal["graph", "density"] = "graph",
    min_interface_cells: int = 3,
    bandwidth: float | None = None,
    distance_threshold: float | None = None,
    coord_type: str = "global",
    store: bool = True,
) -> InterfaceResult:
    """Identify interface cells between two spatial regions.

    Finds cells at the boundary between two named cell groups (e.g.,
    tumor vs stroma) and computes the interface contour geometry and
    per-segment metrics.

    Args:
        sp: spatioloji object with polygon data.
        graph: Pre-built ``PolygonSpatialGraph``. Required for
            ``method='graph'``. Optional for ``method='density'``
            (used to auto-estimate ``distance_threshold``).
        group_col: Column in ``cell_meta`` defining cell groups.
        region_a: Label(s) for region A. If a list, all labels are
            treated as a single region.
        region_b: Label(s) for region B.
        method: ``'graph'`` (default) — uses adjacency edges to find
            cross-region contacts. ``'density'`` — uses KDE to find
            the density decision boundary.
        min_interface_cells: Minimum cells on each side of a segment
            for it to be retained.
        bandwidth: KDE bandwidth (density method only). Auto-estimated
            via Scott's rule if ``None``.
        distance_threshold: Max distance from KDE contour to label a
            cell as interface (density method only). Auto-estimated
            from graph if ``None``.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        store: If ``True``, add ``'interface_label'`` to ``cell_meta``.

    Returns:
        InterfaceResult with cell labels, contour geometry, segments,
        and summary metrics.

    Raises:
        ValueError: On invalid inputs.
        ImportError: If ``method='density'`` and scikit-image is not
            installed.

    Example:
        >>> g = build_contact_graph(sp)
        >>> result = identify_interface(sp, g, "cell_type", "Tumor", "Stromal")
        >>> print(result.summary)
    """
    a_list, b_list = _validate_inputs(
        sp, graph, group_col, region_a, region_b, method, distance_threshold
    )

    print(f"\n[Interface] Identifying interface: "
          f"{a_list} vs {b_list} (method={method})")

    if method == "graph":
        result = _graph_method(
            sp, graph, group_col, a_list, b_list, min_interface_cells, coord_type
        )
    else:
        result = _density_method(
            sp, graph, group_col, a_list, b_list, min_interface_cells,
            bandwidth, distance_threshold, coord_type
        )

    if store:
        sp._cell_meta["interface_label"] = result.cell_labels.values
        print("  Stored 'interface_label' in cell_meta")

    n_a = result.summary.get("n_interface_a", 0)
    n_b = result.summary.get("n_interface_b", 0)
    print(f"  {n_a + n_b} interface cells detected "
          f"({n_a} region_a, {n_b} region_b)")
    print(f"  {result.summary['n_segments']} segment(s), "
          f"total length={result.summary['total_length']:.1f}")

    return result


def _density_method(
    sp, graph, group_col, a_list, b_list, min_interface_cells,
    bandwidth, distance_threshold, coord_type,
) -> InterfaceResult:
    """KDE density-based interface identification. Placeholder for Task 5."""
    raise NotImplementedError("Density method not yet implemented")
```

- [ ] **Step 4: Run all interface tests**

Run: `PYTHONIOENCODING=utf-8 pytest tests/unit/test_interface.py -v`
Expected: all TestInterfaceResult and TestPolygonValidation tests PASS, TestPolygonGraphMethod tests PASS (may need to adjust `buffer_distance` in tests if `build_contact_graph` doesn't find cross-region contacts with default params)

- [ ] **Step 5: Debug and fix any test failures**

If `build_contact_graph` with default params finds no cross-region edges (cells are too far apart), use `build_buffer_graph(sp, buffer_distance=50)` instead in tests. Adjust the fixture or test setup accordingly.

- [ ] **Step 6: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/interface.py tests/unit/test_interface.py
git commit -m "feat(spatial/polygon): add graph-based identify_interface"
```

---

### Task 4: Polygon __init__.py exports

**Files:**
- Modify: `src/spatioloji_s/spatial/polygon/__init__.py`

- [ ] **Step 1: Add interface imports and __all__ entries**

Add to imports section:

```python
# Interface
from .interface import identify_interface as identify_interface
from .._interface_types import InterfaceResult as InterfaceResult
```

Add to `__all__`:

```python
    # Interface
    "identify_interface",
    "InterfaceResult",
```

- [ ] **Step 2: Verify import works**

Run: `PYTHONIOENCODING=utf-8 python -c "from spatioloji_s.spatial.polygon import identify_interface, InterfaceResult; print('ok')"`

- [ ] **Step 3: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/__init__.py
git commit -m "feat(spatial/polygon): export identify_interface and InterfaceResult"
```

---

### Task 5: Density method implementation

**Files:**
- Modify: `src/spatioloji_s/spatial/polygon/interface.py`
- Test: `tests/unit/test_interface.py`

- [ ] **Step 1: Write failing tests for density method**

Add to `tests/unit/test_interface.py`:

```python
class TestPolygonDensityMethod:
    """Tests for the density-based interface identification."""

    def test_density_returns_interface_result(self, sp_interface):
        g = build_contact_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    method="density")
        assert isinstance(result, InterfaceResult)
        assert result.method == "density"

    def test_density_without_graph_explicit_threshold(self, sp_interface):
        result = identify_interface(sp_interface, graph=None,
                                    group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    method="density",
                                    distance_threshold=30.0)
        assert isinstance(result, InterfaceResult)

    def test_density_cell_labels_valid(self, sp_interface):
        result = identify_interface(sp_interface, graph=None,
                                    group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    method="density",
                                    distance_threshold=30.0)
        valid = {"region_a_interface", "region_b_interface",
                 "interior_a", "interior_b", "other"}
        assert set(result.cell_labels.unique()).issubset(valid)

    def test_density_contour_is_geometry(self, sp_interface):
        result = identify_interface(sp_interface, graph=None,
                                    group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    method="density",
                                    distance_threshold=30.0)
        if result.contour is not None:
            assert result.contour.geom_type in ("MultiLineString", "LineString")

    def test_density_scikit_image_missing_raises(self, sp_interface, monkeypatch):
        """Should raise ImportError if scikit-image not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "skimage" or name.startswith("skimage."):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="scikit-image"):
            identify_interface(sp_interface, graph=None,
                               group_col="cell_type",
                               region_a="TypeA", region_b="TypeB",
                               method="density", distance_threshold=30.0)
```

- [ ] **Step 2: Implement _density_method**

Replace the placeholder `_density_method` in `polygon/interface.py`:

```python
def _density_method(
    sp, graph, group_col, a_list, b_list, min_interface_cells,
    bandwidth, distance_threshold, coord_type,
) -> InterfaceResult:
    """KDE density-based interface identification.

    Args:
        sp: spatioloji object.
        graph: Optional PolygonSpatialGraph for distance_threshold estimation.
        group_col: Column in cell_meta.
        a_list: Region A labels.
        b_list: Region B labels.
        min_interface_cells: Min cells per segment side.
        bandwidth: KDE bandwidth or None.
        distance_threshold: Max distance from contour or None.
        coord_type: "global" or "local".

    Returns:
        InterfaceResult.
    """
    try:
        from skimage.measure import find_contours
    except ImportError as err:
        raise ImportError(
            "Density method requires scikit-image. "
            "Install with: pip install scikit-image"
        ) from err

    from scipy.stats import gaussian_kde

    labels = sp.cell_meta[group_col]

    # Get coordinates
    if coord_type == "global":
        x_all = np.asarray(sp.spatial.x_global)
        y_all = np.asarray(sp.spatial.y_global)
    else:
        x_all = np.asarray(sp.spatial.x_local)
        y_all = np.asarray(sp.spatial.y_local)

    cell_ids = np.asarray(sp.cell_index)
    mask_a = labels.isin(a_list).values
    mask_b = labels.isin(b_list).values

    # Auto-estimate distance_threshold
    if distance_threshold is None and graph is not None:
        adj_coo = graph.adjacency.tocoo()
        dists = graph.distances.tocoo()
        nonzero = dists.data[dists.data > 0]
        distance_threshold = float(np.median(nonzero)) if len(nonzero) > 0 else 50.0
        print(f"  Auto distance_threshold={distance_threshold:.1f}")

    # Fit KDE for each region
    pts_a = np.vstack([x_all[mask_a], y_all[mask_a]])
    pts_b = np.vstack([x_all[mask_b], y_all[mask_b]])

    bw = bandwidth if bandwidth is not None else "scott"

    try:
        kde_a = gaussian_kde(pts_a, bw_method=bw if isinstance(bw, str) else bw)
        kde_b = gaussian_kde(pts_b, bw_method=bw if isinstance(bw, str) else bw)
    except Exception:
        warnings.warn("KDE fitting failed.", UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col, "density")

    # Evaluate on grid
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    pad = max(x_max - x_min, y_max - y_min) * 0.05
    grid_res = 200
    xi = np.linspace(x_min - pad, x_max + pad, grid_res)
    yi = np.linspace(y_min - pad, y_max + pad, grid_res)
    xx, yy = np.meshgrid(xi, yi)
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])

    za = kde_a(grid_pts).reshape(grid_res, grid_res)
    zb = kde_b(grid_pts).reshape(grid_res, grid_res)
    diff = za - zb

    # Extract zero-contour
    raw_contours = find_contours(diff, level=0.0)

    if not raw_contours:
        warnings.warn("No density boundary found.", UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col, "density")

    # Convert pixel coords back to spatial coords
    contour_lines = []
    for c in raw_contours:
        # c is (n_pts, 2) in (row, col) pixel space
        spatial_x = xi[0] + c[:, 1] * (xi[-1] - xi[0]) / (grid_res - 1)
        spatial_y = yi[0] + c[:, 0] * (yi[-1] - yi[0]) / (grid_res - 1)
        if len(c) >= 2:
            contour_lines.append(LineString(np.column_stack([spatial_x, spatial_y])))

    if not contour_lines:
        return _empty_result(sp, a_list, b_list, group_col, "density")

    # Label cells by distance to nearest contour
    from shapely.geometry import Point

    cell_labels = pd.Series("other", index=sp.cell_index)
    cell_labels[labels.isin(a_list)] = "interior_a"
    cell_labels[labels.isin(b_list)] = "interior_b"

    contour_union = unary_union(contour_lines)

    for i, cid in enumerate(cell_ids):
        if not (mask_a[i] or mask_b[i]):
            continue
        pt = Point(x_all[i], y_all[i])
        dist = contour_union.distance(pt)
        if dist <= distance_threshold:
            if mask_a[i]:
                cell_labels.loc[cid] = "region_a_interface"
            elif mask_b[i]:
                cell_labels.loc[cid] = "region_b_interface"

    # Build segments
    seg_rows = []
    for idx, line in enumerate(contour_lines):
        # Count interface cells near this specific contour line
        n_ca, n_cb = 0, 0
        for i, cid in enumerate(cell_ids):
            if cell_labels.loc[cid] not in ("region_a_interface", "region_b_interface"):
                continue
            pt = Point(x_all[i], y_all[i])
            if line.distance(pt) <= distance_threshold:
                if mask_a[i]:
                    n_ca += 1
                elif mask_b[i]:
                    n_cb += 1

        if n_ca < min_interface_cells or n_cb < min_interface_cells:
            continue

        seg_rows.append({
            "segment_id": len(seg_rows),
            "geometry": line,
            "length": line.length,
            "tortuosity": _compute_tortuosity(line),
            "n_cells_a": n_ca,
            "n_cells_b": n_cb,
        })

    if not seg_rows:
        warnings.warn("All density segments dropped by min_interface_cells.",
                      UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col, "density")

    segments = gpd.GeoDataFrame(seg_rows, geometry="geometry")

    all_geoms = [row.geometry for _, row in segments.iterrows()]
    contour = MultiLineString(all_geoms) if all_geoms else None

    summary = {
        "total_length": float(segments["length"].sum()),
        "n_segments": len(segments),
        "mean_tortuosity": float(segments["tortuosity"].replace(np.inf, np.nan).mean()),
        "n_interface_a": int((cell_labels == "region_a_interface").sum()),
        "n_interface_b": int((cell_labels == "region_b_interface").sum()),
    }

    return InterfaceResult(
        cell_labels=cell_labels, contour=contour, segments=segments,
        summary=summary,
        region_a=a_list if len(a_list) > 1 else a_list[0],
        region_b=b_list if len(b_list) > 1 else b_list[0],
        method="density",
    )
```

- [ ] **Step 3: Run tests**

Run: `PYTHONIOENCODING=utf-8 pytest tests/unit/test_interface.py -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/interface.py tests/unit/test_interface.py
git commit -m "feat(spatial/polygon): add density-based interface method"
```

---

## Chunk 2: Point module, visualization, and exports

### Task 6: Point-based identify_interface

**Files:**
- Create: `src/spatioloji_s/spatial/point/interface.py`
- Test: `tests/unit/test_interface.py`

- [ ] **Step 1: Write failing tests for point interface**

Add to `tests/unit/test_interface.py`:

```python
from spatioloji_s.spatial.point.interface import (
    identify_interface as point_identify_interface,
)
from spatioloji_s.spatial.point.graph import build_knn_graph


class TestPointGraphMethod:
    """Tests for point-based interface identification."""

    def test_returns_interface_result(self, sp_interface):
        g = build_knn_graph(sp_interface, k=10)
        result = point_identify_interface(sp_interface, g, group_col="cell_type",
                                          region_a="TypeA", region_b="TypeB")
        assert isinstance(result, InterfaceResult)

    def test_cell_labels_valid(self, sp_interface):
        g = build_knn_graph(sp_interface, k=10)
        result = point_identify_interface(sp_interface, g, group_col="cell_type",
                                          region_a="TypeA", region_b="TypeB")
        valid = {"region_a_interface", "region_b_interface",
                 "interior_a", "interior_b", "other"}
        assert set(result.cell_labels.unique()).issubset(valid)

    def test_interface_cells_detected(self, sp_interface):
        g = build_knn_graph(sp_interface, k=10)
        result = point_identify_interface(sp_interface, g, group_col="cell_type",
                                          region_a="TypeA", region_b="TypeB")
        assert result.summary["n_interface_a"] > 0 or result.summary["n_interface_b"] > 0

    def test_contour_geometry(self, sp_interface):
        g = build_knn_graph(sp_interface, k=10)
        result = point_identify_interface(sp_interface, g, group_col="cell_type",
                                          region_a="TypeA", region_b="TypeB")
        if result.contour is not None:
            assert result.contour.geom_type in ("MultiLineString", "LineString")

    def test_validation_same_as_polygon(self, sp_interface):
        with pytest.raises(ValueError, match="not found in cell_meta"):
            point_identify_interface(sp_interface, None, group_col="bad",
                                     region_a="TypeA", region_b="TypeB",
                                     method="graph")
```

- [ ] **Step 2: Implement point/interface.py**

```python
# src/spatioloji_s/spatial/point/interface.py
"""Interface cell identification for point-based spatial analysis."""

from __future__ import annotations

import warnings
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

from spatioloji_s.spatial._interface_types import InterfaceResult
# Reuse validation and helpers from polygon module
from spatioloji_s.spatial.polygon.interface import (
    _compute_tortuosity,
    _empty_result,
    _validate_inputs,
)


def _point_graph_method(
    sp, graph, group_col, a_list, b_list, min_interface_cells, coord_type,
) -> InterfaceResult:
    """Graph-based interface for point data (uses midpoints for contour).

    Args:
        sp: spatioloji object.
        graph: PointSpatialGraph.
        group_col: Column in cell_meta.
        a_list: Region A labels.
        b_list: Region B labels.
        min_interface_cells: Min cells per segment side.
        coord_type: "global" or "local".

    Returns:
        InterfaceResult.
    """
    cell_ids_graph = graph.cell_ids
    adj = graph.adjacency
    labels = sp.cell_meta[group_col]

    graph_labels = labels.reindex(cell_ids_graph)
    mask_a = graph_labels.isin(a_list).values
    mask_b = graph_labels.isin(b_list).values

    adj_coo = adj.tocoo()
    row, col = adj_coo.row, adj_coo.col
    cross_mask = (mask_a[row] & mask_b[col]) | (mask_b[row] & mask_a[col])

    if not cross_mask.any():
        warnings.warn("No cross-region edges found.", UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col, "graph")

    cross_rows = row[cross_mask]
    cross_cols = col[cross_mask]

    interface_a_idx = set(cross_rows[mask_a[cross_rows]]) | set(cross_cols[mask_a[cross_cols]])
    interface_b_idx = set(cross_rows[mask_b[cross_rows]]) | set(cross_cols[mask_b[cross_cols]])

    # Cell labels
    cell_labels = pd.Series("other", index=sp.cell_index)
    cell_labels[labels.isin(a_list)] = "interior_a"
    cell_labels[labels.isin(b_list)] = "interior_b"

    for idx in interface_a_idx:
        cell_labels.loc[cell_ids_graph[idx]] = "region_a_interface"
    for idx in interface_b_idx:
        cell_labels.loc[cell_ids_graph[idx]] = "region_b_interface"

    # Get coordinates
    if coord_type == "global":
        x_all = np.asarray(sp.spatial.x_global)
        y_all = np.asarray(sp.spatial.y_global)
    else:
        x_all = np.asarray(sp.spatial.x_local)
        y_all = np.asarray(sp.spatial.y_local)

    # Build coordinate lookup by cell_id
    coord_dict = {}
    for i, cid in enumerate(sp.cell_index):
        coord_dict[cid] = (x_all[i], y_all[i])

    # Connected components
    upper = cross_rows < cross_cols
    cr_r, cr_c = cross_rows[upper], cross_cols[upper]
    all_cross_cells = sorted(set(cr_r) | set(cr_c))

    if len(all_cross_cells) == 0:
        return _empty_result(sp, a_list, b_list, group_col, "graph")

    idx_map = {old: new for new, old in enumerate(all_cross_cells)}
    n_sub = len(all_cross_cells)
    sub_r = np.array([idx_map[r] for r in cr_r])
    sub_c = np.array([idx_map[c] for c in cr_c])
    sub_adj = sparse.csr_matrix(
        (np.ones(len(sub_r)), (sub_r, sub_c)), shape=(n_sub, n_sub)
    )
    sub_adj = sub_adj + sub_adj.T
    n_components, comp_labels = connected_components(sub_adj, directed=False)

    # Build contour from midpoints between cross-region pairs
    midpoints_by_comp = {}
    seen_pairs = set()
    for r, c in zip(cr_r, cr_c, strict=True):
        pair = (min(r, c), max(r, c))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        cid_r = cell_ids_graph[r]
        cid_c = cell_ids_graph[c]
        if cid_r not in coord_dict or cid_c not in coord_dict:
            continue

        mx = (coord_dict[cid_r][0] + coord_dict[cid_c][0]) / 2
        my = (coord_dict[cid_r][1] + coord_dict[cid_c][1]) / 2
        comp_id = comp_labels[idx_map[r]]
        midpoints_by_comp.setdefault(comp_id, []).append((mx, my))

    # Build segments
    seg_rows = []
    for comp_id in range(n_components):
        comp_cell_indices = [all_cross_cells[i] for i in range(n_sub)
                            if comp_labels[i] == comp_id]
        n_ca = sum(1 for i in comp_cell_indices if i in interface_a_idx)
        n_cb = sum(1 for i in comp_cell_indices if i in interface_b_idx)

        if n_ca < min_interface_cells or n_cb < min_interface_cells:
            continue

        pts = midpoints_by_comp.get(comp_id, [])
        if len(pts) < 2:
            continue

        # Order midpoints by shared-cell adjacency walk.
        # Build a graph of midpoints connected by shared cells, then walk it.
        pts_arr = np.array(pts)

        if len(pts_arr) == 2:
            line = LineString(pts_arr)
        else:
            # Build midpoint adjacency: two midpoints are connected if they
            # share at least one cell from the original cross-region edge pair.
            # We already stored (r, c) per midpoint — walk via shared cells.
            # Fallback: use nearest-neighbor ordering if adjacency is sparse.
            from scipy.spatial import KDTree
            tree = KDTree(pts_arr)
            dists_nn, idx_nn = tree.query(pts_arr, k=min(3, len(pts_arr)))
            # Greedy nearest-neighbor walk from the first midpoint
            visited = [0]
            remaining = set(range(1, len(pts_arr)))
            while remaining:
                curr = visited[-1]
                # Find nearest unvisited
                best_dist, best_idx = np.inf, -1
                for j in remaining:
                    d = np.linalg.norm(pts_arr[curr] - pts_arr[j])
                    if d < best_dist:
                        best_dist, best_idx = d, j
                if best_idx >= 0:
                    visited.append(best_idx)
                    remaining.discard(best_idx)
                else:
                    break
            ordered = pts_arr[visited]
            line = LineString(ordered)

        seg_rows.append({
            "segment_id": len(seg_rows),
            "geometry": line,
            "length": line.length,
            "tortuosity": _compute_tortuosity(line),
            "n_cells_a": n_ca,
            "n_cells_b": n_cb,
        })

    if not seg_rows:
        warnings.warn("All segments dropped by min_interface_cells.",
                      UserWarning, stacklevel=3)
        return _empty_result(sp, a_list, b_list, group_col, "graph")

    segments = gpd.GeoDataFrame(seg_rows, geometry="geometry")

    all_geoms = [row.geometry for _, row in segments.iterrows()]
    contour = MultiLineString(all_geoms) if all_geoms else None

    summary = {
        "total_length": float(segments["length"].sum()),
        "n_segments": len(segments),
        "mean_tortuosity": float(segments["tortuosity"].replace(np.inf, np.nan).mean()),
        "n_interface_a": int((cell_labels == "region_a_interface").sum()),
        "n_interface_b": int((cell_labels == "region_b_interface").sum()),
    }

    return InterfaceResult(
        cell_labels=cell_labels, contour=contour, segments=segments,
        summary=summary,
        region_a=a_list if len(a_list) > 1 else a_list[0],
        region_b=b_list if len(b_list) > 1 else b_list[0],
        method="graph",
    )


def _point_density_method(
    sp, graph, group_col, a_list, b_list, min_interface_cells,
    bandwidth, distance_threshold, coord_type,
) -> InterfaceResult:
    """Density method for point data — delegates to polygon implementation.

    Args:
        sp: spatioloji object.
        graph: Optional PointSpatialGraph.
        group_col: Column in cell_meta.
        a_list: Region A labels.
        b_list: Region B labels.
        min_interface_cells: Min cells per segment side.
        bandwidth: KDE bandwidth or None.
        distance_threshold: Max distance from contour or None.
        coord_type: "global" or "local".

    Returns:
        InterfaceResult.
    """
    # The density method is coordinate-based, not polygon-specific.
    # Reuse the polygon implementation directly.
    from spatioloji_s.spatial.polygon.interface import _density_method
    return _density_method(
        sp, graph, group_col, a_list, b_list, min_interface_cells,
        bandwidth, distance_threshold, coord_type,
    )


def identify_interface(
    sp,
    graph=None,
    group_col: str = "cell_type",
    region_a: str | list[str] = "",
    region_b: str | list[str] = "",
    method: Literal["graph", "density"] = "graph",
    min_interface_cells: int = 3,
    bandwidth: float | None = None,
    distance_threshold: float | None = None,
    coord_type: str = "global",
    store: bool = True,
) -> InterfaceResult:
    """Identify interface cells between two spatial regions (point-based).

    Uses cell centroid coordinates and KNN/radius/Delaunay graphs to find
    cross-region contacts. See the polygon version for full parameter docs.

    Args:
        sp: spatioloji object.
        graph: Pre-built ``PointSpatialGraph``. Required for
            ``method='graph'``.
        group_col: Column in ``cell_meta`` defining cell groups.
        region_a: Label(s) for region A.
        region_b: Label(s) for region B.
        method: ``'graph'`` or ``'density'``.
        min_interface_cells: Min cells per segment side.
        bandwidth: KDE bandwidth (density only).
        distance_threshold: Max contour distance (density only).
        coord_type: ``'global'`` or ``'local'``.
        store: If ``True``, add ``'interface_label'`` to ``cell_meta``.

    Returns:
        InterfaceResult.

    Example:
        >>> g = build_knn_graph(sp, k=10)
        >>> result = identify_interface(sp, g, "cell_type", "Tumor", "Stromal")
    """
    a_list, b_list = _validate_inputs(
        sp, graph, group_col, region_a, region_b, method, distance_threshold
    )

    print(f"\n[Interface/Point] Identifying interface: "
          f"{a_list} vs {b_list} (method={method})")

    if method == "graph":
        result = _point_graph_method(
            sp, graph, group_col, a_list, b_list, min_interface_cells, coord_type
        )
    else:
        result = _point_density_method(
            sp, graph, group_col, a_list, b_list, min_interface_cells,
            bandwidth, distance_threshold, coord_type
        )

    if store:
        sp._cell_meta["interface_label"] = result.cell_labels.values
        print("  Stored 'interface_label' in cell_meta")

    n_a = result.summary.get("n_interface_a", 0)
    n_b = result.summary.get("n_interface_b", 0)
    print(f"  {n_a + n_b} interface cells ({n_a} region_a, {n_b} region_b)")
    print(f"  {result.summary['n_segments']} segment(s)")

    return result
```

- [ ] **Step 3: Run tests**

Run: `PYTHONIOENCODING=utf-8 pytest tests/unit/test_interface.py -v`
Expected: all PASS

- [ ] **Step 4: Update point __init__.py**

Add to `src/spatioloji_s/spatial/point/__init__.py`:

```python
# Interface
from .interface import identify_interface as identify_interface
from .._interface_types import InterfaceResult as InterfaceResult
```

And to `__all__`:

```python
    # Interface
    "identify_interface",
    "InterfaceResult",
```

- [ ] **Step 5: Commit**

```bash
git add src/spatioloji_s/spatial/point/interface.py \
        src/spatioloji_s/spatial/point/__init__.py \
        tests/unit/test_interface.py
git commit -m "feat(spatial/point): add point-based identify_interface"
```

---

### Task 7: Visualization — plot_interface_map and plot_interface_metrics

**Files:**
- Modify: `src/spatioloji_s/visualization/polygon_plots.py`
- Modify: `src/spatioloji_s/visualization/point_plots.py`
- Modify: `src/spatioloji_s/visualization/__init__.py`
- Test: `tests/unit/test_interface.py`

- [ ] **Step 1: Write failing tests for visualization**

Add to `tests/unit/test_interface.py`:

```python
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt

from spatioloji_s.visualization.polygon_plots import (
    plot_interface_map,
    plot_interface_metrics,
)


class TestPlotInterfaceMap:
    """Tests for plot_interface_map."""

    def test_returns_figure(self, sp_interface):
        g = build_contact_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        fig = plot_interface_map(sp_interface, result, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_empty_result_no_error(self, sp_interface):
        """Should handle empty InterfaceResult without crashing."""
        empty = InterfaceResult(
            cell_labels=pd.Series("other", index=sp_interface.cell_index),
            contour=None,
            segments=gpd.GeoDataFrame(
                {"segment_id": pd.Series(dtype=int), "length": pd.Series(dtype=float),
                 "tortuosity": pd.Series(dtype=float),
                 "n_cells_a": pd.Series(dtype=int), "n_cells_b": pd.Series(dtype=int)},
                geometry=[]),
            summary={"total_length": 0.0, "n_segments": 0, "mean_tortuosity": 0.0,
                     "n_interface_a": 0, "n_interface_b": 0},
            region_a="TypeA", region_b="TypeB", method="graph",
        )
        fig = plot_interface_map(sp_interface, empty, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_ax(self, sp_interface):
        g = build_contact_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        fig, ax = plt.subplots()
        plot_interface_map(sp_interface, result, ax=ax, show=False)
        plt.close("all")


class TestPlotInterfaceMetrics:
    """Tests for plot_interface_metrics."""

    def test_returns_figure(self, sp_interface):
        g = build_contact_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB")
        fig = plot_interface_metrics(result, metric="length", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_empty_segments(self):
        """Should handle empty segments without crashing."""
        import geopandas as gpd
        empty = InterfaceResult(
            cell_labels=pd.Series(dtype=str),
            contour=None,
            segments=gpd.GeoDataFrame(
                {"segment_id": pd.Series(dtype=int), "length": pd.Series(dtype=float),
                 "tortuosity": pd.Series(dtype=float),
                 "n_cells_a": pd.Series(dtype=int), "n_cells_b": pd.Series(dtype=int)},
                geometry=[]),
            summary={"total_length": 0.0, "n_segments": 0, "mean_tortuosity": 0.0,
                     "n_interface_a": 0, "n_interface_b": 0},
            region_a="A", region_b="B", method="graph",
        )
        fig = plot_interface_metrics(empty, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
```

- [ ] **Step 2: Implement plot_interface_map and plot_interface_metrics**

Append to `src/spatioloji_s/visualization/polygon_plots.py`:

```python
# ══════════════════════════════════════════════════════════════════════════════
# Interface map
# ══════════════════════════════════════════════════════════════════════════════


def plot_interface_map(
    spatioloji_obj,
    interface_result,
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
) -> plt.Figure | None:
    """Polygon map coloured by interface role with contour overlay.

    Args:
        spatioloji_obj: A ``spatioloji`` object with polygon data.
        interface_result: ``InterfaceResult`` from ``identify_interface()``.
        coord_type: ``'global'`` or ``'local'``.
        colors: Dict mapping label → colour. Defaults provide red/blue.
        contour_color: Colour of the interface contour line.
        contour_width: Width of the contour line.
        poly_alpha: Polygon fill opacity.
        show_interior: If ``False``, interior and other cells shown in grey.
        ax: Optional matplotlib Axes to draw into.
        figsize: Figure size (ignored if ``ax`` provided).
        title: Plot title. Auto-generated if ``None``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.polygon.interface.identify_interface(
        ...     sp, g, "cell_type", "Tumor", "Stromal")
        >>> sj.visualization.plot_interface_map(sp, result)
    """
    from pathlib import Path

    default_colors = {
        "region_a_interface": "#e74c3c",
        "region_b_interface": "#3498db",
        "interior_a": "#fadbd8",
        "interior_b": "#d6eaf8",
        "other": "#e8e8e8",
    }
    grey = "#d5d5d5"
    cmap = {**(default_colors), **(colors or {})}

    gdf = spatioloji_obj.to_geopandas(coord_type=coord_type, include_metadata=False)
    labels = interface_result.cell_labels.reindex(gdf.index).fillna("other")

    if show_interior:
        face_colors = [cmap.get(lbl, grey) for lbl in labels]
    else:
        face_colors = [
            cmap.get(lbl, grey) if lbl.endswith("_interface") else grey
            for lbl in labels
        ]

    pc = _build_poly_collection(gdf, face_colors)
    pc.set_alpha(poly_alpha)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.add_collection(pc)

    # Overlay contour
    if interface_result.contour is not None:
        if interface_result.contour.geom_type == "MultiLineString":
            for line in interface_result.contour.geoms:
                xs, ys = line.xy
                ax.plot(xs, ys, color=contour_color, linewidth=contour_width, zorder=3)
        elif interface_result.contour.geom_type == "LineString":
            xs, ys = interface_result.contour.xy
            ax.plot(xs, ys, color=contour_color, linewidth=contour_width, zorder=3)

    ax.autoscale_view()
    ax.set_aspect("equal")

    # Legend
    from matplotlib.patches import Patch
    legend_items = []
    for lbl in ["region_a_interface", "region_b_interface",
                "interior_a", "interior_b", "other"]:
        n = (labels == lbl).sum()
        if n > 0:
            display = lbl.replace("_", " ")
            legend_items.append(Patch(facecolor=cmap.get(lbl, grey),
                                      label=f"{display} ({n})"))
    ax.legend(handles=legend_items, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, frameon=False)

    ra = interface_result.region_a
    rb = interface_result.region_b
    ax.set_title(title or f"Interface: {ra} vs {rb} ({interface_result.method})")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Interface metrics bar chart
# ══════════════════════════════════════════════════════════════════════════════


def plot_interface_metrics(
    interface_result,
    metric: str = "length",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Horizontal bar chart of per-segment interface metrics.

    Args:
        interface_result: ``InterfaceResult`` from ``identify_interface()``.
        metric: Column to plot: ``'length'``, ``'tortuosity'``,
            ``'n_cells_a'``, or ``'n_cells_b'``.
        ax: Optional matplotlib Axes to draw into.
        figsize: Figure size. Auto-computed if ``None``.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> sj.visualization.plot_interface_metrics(result, metric="length")
    """
    segs = interface_result.segments

    if len(segs) == 0:
        figsize = figsize or (6, 3)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.text(0.5, 0.5, "No interface segments found",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title(title or f"Interface segments — {metric}")
        clean_axes(ax)
        return finalize_plot(fig, save_path, dpi, show)

    n = len(segs)
    figsize = figsize or (7, max(3, n * 0.5))
    vals = segs[metric].values
    bar_labels = [f"Segment {i}" for i in range(n)]

    cm_obj = plt.get_cmap("YlOrRd")
    v_min, v_max = float(vals.min()), float(vals.max())
    if v_min == v_max:
        v_min, v_max = v_min - 0.1, v_max + 0.1
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=v_min, vmax=v_max)
    bar_colors = [cm_obj(norm(v)) for v in vals]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.barh(range(n), vals, color=bar_colors, height=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(bar_labels, fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(title or f"Interface segments — {metric}")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)
```

- [ ] **Step 3: Add scatter-based plot_interface_map to point_plots.py**

Append to `src/spatioloji_s/visualization/point_plots.py`:

```python
# ══════════════════════════════════════════════════════════════════════════════
# Interface map (point/scatter mode)
# ══════════════════════════════════════════════════════════════════════════════


def plot_interface_map(
    spatioloji_obj,
    interface_result,
    coord_type: str = "global",
    colors: dict | None = None,
    contour_color: str = "black",
    contour_width: float = 2.0,
    point_size: float = 5.0,
    show_interior: bool = True,
    ax=None,
    figsize: tuple[float, float] = (9, 8),
    title: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    """Scatter plot coloured by interface role with contour overlay.

    Args:
        spatioloji_obj: A ``spatioloji`` object.
        interface_result: ``InterfaceResult`` from ``identify_interface()``.
        coord_type: ``'global'`` or ``'local'``.
        colors: Dict mapping label → colour. Defaults provide red/blue.
        contour_color: Colour of the interface contour line.
        contour_width: Width of the contour line.
        point_size: Scatter dot size.
        show_interior: If ``False``, interior and other cells shown in grey.
        ax: Optional matplotlib Axes to draw into.
        figsize: Figure size (ignored if ``ax`` provided).
        title: Plot title. Auto-generated if ``None``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.point.interface.identify_interface(
        ...     sp, g, "cell_type", "Tumor", "Stromal")
        >>> sj.visualization.point_plots.plot_interface_map(sp, result)
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    default_colors = {
        "region_a_interface": "#e74c3c",
        "region_b_interface": "#3498db",
        "interior_a": "#fadbd8",
        "interior_b": "#d6eaf8",
        "other": "#e8e8e8",
    }
    grey = "#d5d5d5"
    cmap = {**(default_colors), **(colors or {})}

    if coord_type == "global":
        x = spatioloji_obj.spatial.x_global
        y = spatioloji_obj.spatial.y_global
    else:
        x = spatioloji_obj.spatial.x_local
        y = spatioloji_obj.spatial.y_local

    labels = interface_result.cell_labels.reindex(spatioloji_obj.cell_index).fillna("other")

    if show_interior:
        c = [cmap.get(lbl, grey) for lbl in labels]
    else:
        c = [cmap.get(lbl, grey) if lbl.endswith("_interface") else grey for lbl in labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.scatter(x, y, c=c, s=point_size, edgecolors="none", zorder=1)

    # Overlay contour
    if interface_result.contour is not None:
        if interface_result.contour.geom_type == "MultiLineString":
            for line in interface_result.contour.geoms:
                xs, ys = line.xy
                ax.plot(xs, ys, color=contour_color, linewidth=contour_width, zorder=3)
        elif interface_result.contour.geom_type == "LineString":
            xs, ys = interface_result.contour.xy
            ax.plot(xs, ys, color=contour_color, linewidth=contour_width, zorder=3)

    ax.set_aspect("equal")

    # Legend
    legend_items = []
    for lbl in ["region_a_interface", "region_b_interface",
                "interior_a", "interior_b", "other"]:
        n = (labels == lbl).sum()
        if n > 0:
            display = lbl.replace("_", " ")
            legend_items.append(Patch(facecolor=cmap.get(lbl, grey),
                                      label=f"{display} ({n})"))
    ax.legend(handles=legend_items, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, frameon=False)

    ra = interface_result.region_a
    rb = interface_result.region_b
    ax.set_title(title or f"Interface: {ra} vs {rb} ({interface_result.method})")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


def plot_interface_metrics(
    interface_result,
    metric: str = "length",
    ax=None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    """Horizontal bar chart of per-segment interface metrics (point mode).

    Delegates to ``polygon_plots.plot_interface_metrics``.
    """
    from spatioloji_s.visualization.polygon_plots import (
        plot_interface_metrics as _poly_plot_metrics,
    )
    return _poly_plot_metrics(
        interface_result, metric=metric, ax=ax, figsize=figsize,
        title=title, show=show, save_path=save_path, dpi=dpi,
    )
```

- [ ] **Step 4: Update visualization __init__.py**

Add to the `from .polygon_plots import (` block:

```python
    plot_interface_map,
    plot_interface_metrics,
```

Add to `__all__`:

```python
    "plot_interface_map",
    "plot_interface_metrics",
```

- [ ] **Step 4: Run all tests**

Run: `PYTHONIOENCODING=utf-8 pytest tests/unit/test_interface.py -v`
Expected: all PASS

- [ ] **Step 5: Run full test suite**

Run: `PYTHONIOENCODING=utf-8 pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 6: Ruff lint**

Run: `ruff check src/spatioloji_s/spatial/_interface_types.py src/spatioloji_s/spatial/polygon/interface.py src/spatioloji_s/spatial/point/interface.py src/spatioloji_s/visualization/polygon_plots.py tests/unit/test_interface.py --fix`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add src/spatioloji_s/visualization/polygon_plots.py \
        src/spatioloji_s/visualization/point_plots.py \
        src/spatioloji_s/visualization/__init__.py \
        tests/unit/test_interface.py
git commit -m "feat(visualization): add plot_interface_map and plot_interface_metrics"
```

---

### Task 8: Final integration test and cleanup

**Files:**
- Test: `tests/unit/test_interface.py`

- [ ] **Step 1: Add end-to-end integration test**

Add to `tests/unit/test_interface.py`:

```python
class TestIntegration:
    """End-to-end integration tests."""

    def test_polygon_full_pipeline(self, sp_interface):
        """Full pipeline: build graph → identify interface → plot."""
        g = build_contact_graph(sp_interface, buffer_distance=50)
        result = identify_interface(sp_interface, g, group_col="cell_type",
                                    region_a="TypeA", region_b="TypeB",
                                    store=True)
        assert "interface_label" in sp_interface.cell_meta.columns
        assert result.summary["n_segments"] >= 0

        fig = plot_interface_map(sp_interface, result, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

        if result.summary["n_segments"] > 0:
            fig2 = plot_interface_metrics(result, metric="length", show=False)
            assert isinstance(fig2, plt.Figure)
            plt.close("all")

    def test_point_full_pipeline(self, sp_interface):
        """Full pipeline with point-based graph."""
        g = build_knn_graph(sp_interface, k=10)
        result = point_identify_interface(sp_interface, g,
                                          group_col="cell_type",
                                          region_a="TypeA", region_b="TypeB",
                                          store=True)
        assert "interface_label" in sp_interface.cell_meta.columns
        assert isinstance(result, InterfaceResult)

    def test_imports_from_top_level(self):
        """Verify imports work from package top-level paths."""
        from spatioloji_s.spatial.polygon import identify_interface, InterfaceResult
        from spatioloji_s.spatial.point import identify_interface as pi
        from spatioloji_s.visualization import plot_interface_map, plot_interface_metrics
        assert callable(identify_interface)
        assert callable(pi)
        assert callable(plot_interface_map)
        assert callable(plot_interface_metrics)
```

- [ ] **Step 2: Run full test suite**

Run: `PYTHONIOENCODING=utf-8 pytest tests/ -v`
Expected: all PASS

- [ ] **Step 3: Final ruff check on all modified files**

Run: `ruff check src/spatioloji_s/spatial/ src/spatioloji_s/visualization/polygon_plots.py tests/unit/test_interface.py`
Expected: clean

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_interface.py
git commit -m "test: add integration tests for interface analysis"
```
