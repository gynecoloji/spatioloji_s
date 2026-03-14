# Spatial Gradient Analysis & Immune Infiltration Scoring — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two analysis modules that quantify gene expression gradients and immune cell infiltration relative to a spatial interface boundary.

**Architecture:** Two independent modules (`gradient.py`, `infiltration.py`) in `spatial/polygon/`, with thin re-export wrappers in `spatial/point/`. A shared `_distance_utils.py` computes signed distance from cells to the interface contour. Result dataclasses (`GradientResult`, `InfiltrationResult`) follow the existing `InterfaceResult` pattern.

**Tech Stack:** numpy, pandas, scipy (linregress), shapely, matplotlib; optional scikit-learn (NMF/PCA)

**Spec:** `docs/superpowers/specs/2026-03-13-gradient-infiltration-design.md`

---

## Chunk 1: Foundation (Tasks 1–3)

### Task 1: Shared Distance Utility

**Files:**
- Create: `src/spatioloji_s/spatial/_distance_utils.py`
- Create: `tests/unit/test_distance_utils.py`

- [ ] **Step 1: Write failing tests for signed_distance_to_interface**

Create `tests/unit/test_distance_utils.py`:

```python
"""Tests for signed distance to interface utility."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString

from spatioloji_s.spatial._distance_utils import signed_distance_to_interface
from spatioloji_s.spatial._interface_types import InterfaceResult


def _make_interface_result(cell_labels, contour, region_a="TypeA", region_b="TypeB"):
    """Helper to create a minimal InterfaceResult for testing."""
    import geopandas as gpd

    segments = gpd.GeoDataFrame(
        {"segment_id": [0], "geometry": [contour.geoms[0] if contour else None],
         "length": [1.0], "tortuosity": [1.0], "n_cells_a": [1], "n_cells_b": [1]},
    )
    return InterfaceResult(
        cell_labels=cell_labels,
        contour=contour,
        segments=segments,
        summary={"total_length": 1.0, "n_segments": 1,
                 "mean_tortuosity": 1.0, "n_interface_a": 1, "n_interface_b": 1},
        region_a=region_a,
        region_b=region_b,
        method="graph",
    )


class TestSignedDistance:
    """Tests for signed_distance_to_interface."""

    def test_basic_signed_distance(self, sp_interface):
        """Cells on region A side get positive, region B side get negative."""
        from spatioloji_s.spatial.polygon.interface import identify_interface
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        graph = build_buffer_graph(sp_interface, buffer_distance=50)
        iface = identify_interface(
            sp_interface, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
        )
        distances = signed_distance_to_interface(sp_interface, iface, coord_type="global")

        assert isinstance(distances, pd.Series)
        assert len(distances) == len(sp_interface.cell_index)

        # Region A cells should have positive distances
        a_mask = iface.cell_labels.isin(["region_a_interface", "interior_a"])
        assert (distances[a_mask] >= 0).all()

        # Region B cells should have negative distances
        b_mask = iface.cell_labels.isin(["region_b_interface", "interior_b"])
        assert (distances[b_mask] <= 0).all()

    def test_unsigned_distance(self, sp_interface):
        """When unsigned=True, all distances should be non-negative."""
        from spatioloji_s.spatial.polygon.interface import identify_interface
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        graph = build_buffer_graph(sp_interface, buffer_distance=50)
        iface = identify_interface(
            sp_interface, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
        )
        distances = signed_distance_to_interface(
            sp_interface, iface, coord_type="global", unsigned=True,
        )
        assert (distances >= 0).all()

    def test_no_contour_raises(self, sp_interface):
        """Should raise ValueError when contour is None."""
        labels = pd.Series("other", index=sp_interface.cell_index)
        iface = _make_interface_result(labels, contour=None)
        # contour is None — need to handle this
        with pytest.raises(ValueError, match="contour"):
            signed_distance_to_interface(sp_interface, iface)

    def test_returns_series_with_cell_index(self, sp_interface):
        """Result should be indexed by cell ID."""
        # Create a simple vertical line contour at x=500
        contour = MultiLineString([LineString([(500, 0), (500, 1000)])])
        labels = pd.Series(
            ["interior_a"] * 50 + ["interior_b"] * 50,
            index=sp_interface.cell_index,
        )
        iface = _make_interface_result(labels, contour)
        distances = signed_distance_to_interface(sp_interface, iface)
        assert distances.index.equals(sp_interface.cell_index)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_distance_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spatioloji_s.spatial._distance_utils'`

- [ ] **Step 3: Implement signed_distance_to_interface**

Create `src/spatioloji_s/spatial/_distance_utils.py`:

```python
"""Shared distance utilities for interface-based analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Point

from spatioloji_s.spatial._interface_types import InterfaceResult


def signed_distance_to_interface(
    sp,
    interface_result: InterfaceResult,
    coord_type: str = "global",
    unsigned: bool = False,
) -> pd.Series:
    """Compute signed distance from each cell to the interface contour.

    Positive distances indicate cells on the region A side, negative
    distances indicate cells on the region B side.

    Args:
        sp: spatioloji object.
        interface_result: Result from ``identify_interface``.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        unsigned: If True, return absolute distances.

    Returns:
        Series indexed by cell ID with signed (or unsigned) distances.

    Raises:
        ValueError: If ``interface_result.contour`` is None or
            ``coord_type`` is invalid.
    """
    if interface_result.contour is None:
        raise ValueError(
            "InterfaceResult.contour is None — cannot compute distances. "
            "Ensure identify_interface produced a valid contour."
        )

    if coord_type not in ("global", "local"):
        raise ValueError(f"coord_type must be 'global' or 'local', got '{coord_type}'")

    # Get cell coordinates
    if coord_type == "global":
        x = np.asarray(sp.spatial.x_global)
        y = np.asarray(sp.spatial.y_global)
    else:
        x = np.asarray(sp.spatial.x_local)
        y = np.asarray(sp.spatial.y_local)

    contour = interface_result.contour
    labels = interface_result.cell_labels
    cell_ids = sp.cell_index

    # Compute unsigned distances
    raw_distances = np.array([
        Point(xi, yi).distance(contour) for xi, yi in zip(x, y, strict=True)
    ])

    if unsigned:
        return pd.Series(raw_distances, index=cell_ids, name="distance_to_interface")

    # Assign sign based on cell labels
    signs = np.ones(len(cell_ids))
    for i, cid in enumerate(cell_ids):
        label = labels.get(cid, "other")
        if label in ("region_b_interface", "interior_b"):
            signs[i] = -1.0
        # region_a_interface, interior_a, other → positive (default)

    signed = raw_distances * signs
    return pd.Series(signed, index=cell_ids, name="distance_to_interface")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_distance_utils.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/spatioloji_s/spatial/_distance_utils.py tests/unit/test_distance_utils.py
git commit -m "feat: add signed_distance_to_interface utility"
```

---

### Task 2: GradientResult Dataclass

**Files:**
- Create: `src/spatioloji_s/spatial/_gradient_types.py`

- [ ] **Step 1: Create the dataclass file**

Create `src/spatioloji_s/spatial/_gradient_types.py`:

```python
"""Shared data structures for gradient analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class GradientResult:
    """Container for spatial gradient analysis results.

    Attributes:
        distances: Signed distance per cell to the interface contour.
            Positive = region A side, negative = region B side.
        gene_gradients: DataFrame with rows=genes and columns
            ``coef``, ``pvalue``, ``r2``, ``trend``.
        program_gradients: DataFrame with rows=programs and columns
            ``coef``, ``pvalue``, ``r2``, ``trend``.
        program_scores: DataFrame (cells × programs) with per-cell
            mean expression of each gene program.
        bins: Long-form DataFrame with columns ``distance_bin``,
            ``gene``, ``mean_expr``, ``std_expr`` for plotting.
        region_a: Region A label(s) used.
        region_b: Region B label(s) used.
    """

    distances: pd.Series
    gene_gradients: pd.DataFrame
    program_gradients: pd.DataFrame
    program_scores: pd.DataFrame
    bins: pd.DataFrame
    region_a: str | list[str]
    region_b: str | list[str]
```

- [ ] **Step 2: Commit**

```bash
git add src/spatioloji_s/spatial/_gradient_types.py
git commit -m "feat: add GradientResult dataclass"
```

---

### Task 3: InfiltrationResult Dataclass

**Files:**
- Create: `src/spatioloji_s/spatial/_infiltration_types.py`

- [ ] **Step 1: Create the dataclass file**

Create `src/spatioloji_s/spatial/_infiltration_types.py`:

```python
"""Shared data structures for immune infiltration scoring."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class InfiltrationResult:
    """Container for immune infiltration scoring results.

    Attributes:
        distances: Signed distance per cell to the interface contour.
        cell_classifications: Series with values ``"infiltrating"``,
            ``"resident"``, or ``"other"`` per cell.
        per_type_metrics: DataFrame with rows=immune cell types and
            columns ``median_depth``, ``max_depth``, ``density_slope``,
            ``density_pvalue``, ``infiltration_fraction``,
            ``n_infiltrating``, ``n_resident``.
        region_a: Region A label(s) used.
        region_b: Region B label(s) used.
        target_region: Which region immune cells infiltrate into.
    """

    distances: pd.Series
    cell_classifications: pd.Series
    per_type_metrics: pd.DataFrame
    region_a: str | list[str]
    region_b: str | list[str]
    target_region: str
```

- [ ] **Step 2: Commit**

```bash
git add src/spatioloji_s/spatial/_infiltration_types.py
git commit -m "feat: add InfiltrationResult dataclass"
```

---

## Chunk 2: Gradient Analysis (Tasks 4–5)

### Task 4: Test Fixture for Gradient/Infiltration

**Files:**
- Modify: `tests/conftest.py`

The existing `sp_interface` fixture has 100 cells with `TypeA` (x∈[50,500]) and `TypeB` (x∈[550,950]), 10 genes. We need a fixture that also has known expression gradients.

- [ ] **Step 1: Add sp_gradient fixture to conftest.py**

Append to `tests/conftest.py`:

```python
# ===========================================================================
# Fixture 6: gradient/infiltration-specific object
# ===========================================================================


@pytest.fixture
def sp_gradient():
    """200-cell spatioloji with spatial expression gradient + immune cell types.

    Layout:
    - TypeA (cells 0-99): x_global in [50, 490] (tumor-like region)
    - TypeB (cells 100-199): x_global in [510, 950] (stroma-like region)
    - Interface around x=500.
    - 10 cells per side near boundary (x in [460, 540]).

    Expression:
    - gene_0: positively correlated with distance from interface (gradient gene)
    - gene_1: negatively correlated with distance (inverse gradient)
    - gene_2 to gene_9: random noise (no gradient)

    Cell types (in cell_meta 'immune_type' column):
    - TypeA cells: 80 'Tumor', 20 'CD8_T' (infiltrating immune cells)
    - TypeB cells: 60 'Stroma', 30 'CD8_T' (resident), 10 'Macrophage'

    Each cell has a 4x4 square polygon.
    """
    np.random.seed(42)
    n_cells = 200
    n_genes = 10
    n_per_region = 100

    # Spatial layout — two regions with interface at x=500
    x_a = np.concatenate([
        np.random.uniform(50, 460, n_per_region - 10),
        np.random.uniform(460, 490, 10),
    ])
    x_b = np.concatenate([
        np.random.uniform(510, 540, 10),
        np.random.uniform(540, 950, n_per_region - 10),
    ])
    x_global = np.concatenate([x_a, x_b])
    y_global = np.random.uniform(0, 1000, n_cells)

    # Expression with gradient signal
    expression = np.random.poisson(2.0, (n_cells, n_genes)).astype(float)
    # gene_0: expression increases with distance from x=500
    dist_from_interface = np.abs(x_global - 500)
    expression[:, 0] += (dist_from_interface / 100).astype(float)
    # gene_1: expression decreases with distance from x=500
    expression[:, 1] += np.maximum(0, 5 - dist_from_interface / 100).astype(float)

    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # Cell metadata with region and immune type
    region_labels = ["TypeA"] * n_per_region + ["TypeB"] * n_per_region
    # Immune types: CD8_T cells in both regions (20 in A = infiltrating, 30 in B = resident)
    immune_a = ["Tumor"] * 80 + ["CD8_T"] * 20
    immune_b = ["Stroma"] * 60 + ["CD8_T"] * 30 + ["Macrophage"] * 10
    immune_labels = immune_a + immune_b

    cell_meta = pd.DataFrame(
        {
            "cell_type": region_labels,
            "immune_type": immune_labels,
        },
        index=cell_ids,
    )

    spatial = {
        "x_global": x_global,
        "y_global": y_global,
        "x_local": x_global,
        "y_local": y_global,
    }

    # Build 4x4 square polygons
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

- [ ] **Step 2: Verify fixture loads**

Run: `pytest tests/conftest.py --co -q`
Expected: No errors, fixture discovered

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add sp_gradient fixture for gradient/infiltration tests"
```

---

### Task 5: compute_gradient Implementation

**Files:**
- Create: `src/spatioloji_s/spatial/polygon/gradient.py`
- Create: `src/spatioloji_s/spatial/point/gradient.py`
- Create: `tests/unit/test_gradient.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_gradient.py`:

```python
"""Tests for spatial gradient analysis."""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.spatial._gradient_types import GradientResult


class TestComputeGradientBasic:
    """Basic gradient computation tests."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        """Create InterfaceResult from sp_gradient fixture."""
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        return identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )

    def test_returns_gradient_result(self, sp_gradient, interface_result):
        """compute_gradient should return a GradientResult."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(
            sp_gradient, interface_result,
            genes=["gene_0", "gene_1"],
        )
        assert isinstance(result, GradientResult)

    def test_gene_gradients_shape(self, sp_gradient, interface_result):
        """gene_gradients should have one row per gene."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(
            sp_gradient, interface_result,
            genes=["gene_0", "gene_1", "gene_2"],
        )
        assert result.gene_gradients.shape[0] == 3
        assert set(result.gene_gradients.columns) >= {"coef", "pvalue", "r2", "trend"}

    def test_gene_gradients_all_genes(self, sp_gradient, interface_result):
        """genes=None should use all genes."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(sp_gradient, interface_result, genes=None)
        assert result.gene_gradients.shape[0] == 10  # sp_gradient has 10 genes

    def test_bins_dataframe_columns(self, sp_gradient, interface_result):
        """bins should have distance_bin, gene, mean_expr, std_expr."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(
            sp_gradient, interface_result,
            genes=["gene_0"], n_bins=5,
        )
        assert set(result.bins.columns) >= {"distance_bin", "gene", "mean_expr", "std_expr"}
        # 1 gene × 5 bins = up to 5 rows (some bins may be empty)
        assert len(result.bins) <= 5

    def test_distances_series(self, sp_gradient, interface_result):
        """distances should be a Series indexed by cell ID."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(
            sp_gradient, interface_result, genes=["gene_0"],
        )
        assert isinstance(result.distances, pd.Series)
        assert len(result.distances) == len(sp_gradient.cell_index)

    def test_trend_labels(self, sp_gradient, interface_result):
        """trend should be one of the three valid labels."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"])
        valid_trends = {"increasing_toward_a", "increasing_toward_b", "flat"}
        assert set(result.gene_gradients["trend"].unique()).issubset(valid_trends)

    def test_region_labels_propagated(self, sp_gradient, interface_result):
        """region_a/b should be copied from InterfaceResult."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(sp_gradient, interface_result, genes=["gene_0"])
        assert result.region_a == interface_result.region_a
        assert result.region_b == interface_result.region_b


class TestComputeGradientPrograms:
    """Tests for gene program gradient analysis."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        return identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )

    def test_user_programs(self, sp_gradient, interface_result):
        """User-defined programs should produce program_gradients rows."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        programs = {
            "gradient_set": ["gene_0", "gene_1"],
            "noise_set": ["gene_2", "gene_3"],
        }
        result = compute_gradient(
            sp_gradient, interface_result,
            genes=["gene_0"], programs=programs,
        )
        assert result.program_gradients.shape[0] == 2
        assert "gradient_set" in result.program_gradients.index
        assert result.program_scores.shape == (len(sp_gradient.cell_index), 2)

    def test_no_programs(self, sp_gradient, interface_result):
        """No programs → empty program_gradients."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(
            sp_gradient, interface_result, genes=["gene_0"],
        )
        assert result.program_gradients.empty
        assert result.program_scores.empty


class TestComputeGradientAutoPrograms:
    """Tests for auto-discovered gene programs."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        return identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )

    def test_nmf_auto_programs(self, sp_gradient, interface_result):
        """auto_programs='nmf' should discover n_auto_programs programs."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(
            sp_gradient, interface_result,
            genes=["gene_0"],
            auto_programs="nmf", n_auto_programs=3,
        )
        assert result.program_gradients.shape[0] == 3

    def test_pca_auto_programs(self, sp_gradient, interface_result):
        """auto_programs='pca' should discover n_auto_programs programs."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(
            sp_gradient, interface_result,
            genes=["gene_0"],
            auto_programs="pca", n_auto_programs=3,
        )
        assert result.program_gradients.shape[0] == 3

    def test_invalid_auto_programs(self, sp_gradient, interface_result):
        """Invalid auto_programs value should raise ValueError."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        with pytest.raises(ValueError, match="auto_programs"):
            compute_gradient(
                sp_gradient, interface_result,
                auto_programs="invalid",
            )


class TestComputeGradientValidation:
    """Validation and edge case tests."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        return identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )

    def test_invalid_method(self, sp_gradient, interface_result):
        """Invalid method should raise ValueError."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        with pytest.raises(ValueError, match="method"):
            compute_gradient(sp_gradient, interface_result, method="invalid")

    def test_missing_genes(self, sp_gradient, interface_result):
        """Non-existent gene names should raise ValueError."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        with pytest.raises(ValueError, match="not found"):
            compute_gradient(sp_gradient, interface_result, genes=["nonexistent_gene"])

    def test_coord_type_local(self, sp_gradient, interface_result):
        """coord_type='local' should work without error."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(
            sp_gradient, interface_result,
            genes=["gene_0"], coord_type="local",
        )
        assert isinstance(result, GradientResult)

    def test_unsigned_gradient(self, sp_gradient, interface_result):
        """unsigned=True should produce all non-negative distances."""
        from spatioloji_s.spatial.polygon.gradient import compute_gradient

        result = compute_gradient(
            sp_gradient, interface_result,
            genes=["gene_0"], unsigned=True,
        )
        assert (result.distances >= 0).all()


class TestPointGradientReExport:
    """Verify point module re-exports polygon gradient."""

    def test_point_compute_gradient_is_same(self):
        """Point compute_gradient should be the same function as polygon."""
        from spatioloji_s.spatial.point.gradient import compute_gradient as point_cg
        from spatioloji_s.spatial.polygon.gradient import compute_gradient as poly_cg

        assert point_cg is poly_cg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_gradient.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spatioloji_s.spatial.polygon.gradient'`

- [ ] **Step 3: Implement compute_gradient**

Create `src/spatioloji_s/spatial/polygon/gradient.py`:

```python
"""Spatial gradient analysis for polygon-based spatial data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

from spatioloji_s.spatial._distance_utils import signed_distance_to_interface
from spatioloji_s.spatial._gradient_types import GradientResult
from spatioloji_s.spatial._interface_types import InterfaceResult


def _fit_gradient(values: np.ndarray, distances: np.ndarray) -> dict:
    """Fit linear regression of values ~ distances.

    Args:
        values: Expression values (1-D array).
        distances: Signed distances (1-D array, same length).

    Returns:
        Dict with keys coef, pvalue, r2, trend.
    """
    mask = np.isfinite(values) & np.isfinite(distances)
    if mask.sum() < 3:
        return {"coef": np.nan, "pvalue": np.nan, "r2": np.nan, "trend": "flat"}

    result = linregress(distances[mask], values[mask])
    trend = "flat"
    if result.pvalue < 0.05:
        trend = "increasing_toward_a" if result.slope > 0 else "increasing_toward_b"
    return {
        "coef": result.slope,
        "pvalue": result.pvalue,
        "r2": result.rvalue ** 2,
        "trend": trend,
    }


def _bin_expression(
    expr_df: pd.DataFrame,
    distances: pd.Series,
    genes: list[str],
    n_bins: int,
) -> pd.DataFrame:
    """Bin expression by distance for plotting.

    Args:
        expr_df: Expression DataFrame (cells × genes).
        distances: Signed distances indexed by cell ID.
        genes: Gene names to include.
        n_bins: Number of equal-width bins.

    Returns:
        Long-form DataFrame: distance_bin, gene, mean_expr, std_expr.
    """
    bin_edges = np.linspace(distances.min(), distances.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_labels = np.digitize(distances.values, bin_edges[1:-1])  # 0-indexed bins

    rows = []
    for gene in genes:
        gene_vals = expr_df[gene].values
        for b in range(n_bins):
            mask = bin_labels == b
            if mask.sum() == 0:
                continue
            rows.append({
                "distance_bin": bin_centers[b],
                "gene": gene,
                "mean_expr": float(np.mean(gene_vals[mask])),
                "std_expr": float(np.std(gene_vals[mask])),
            })
    return pd.DataFrame(rows)


def _discover_programs_nmf(expr_matrix: np.ndarray, gene_names: list[str], n_programs: int) -> dict[str, list[str]]:
    """Discover gene programs via NMF.

    Args:
        expr_matrix: Dense expression matrix (cells × genes).
        gene_names: Gene names.
        n_programs: Number of programs.

    Returns:
        Dict of {program_name: [top_gene_names]}.
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        raise ImportError(
            "scikit-learn is required for auto_programs='nmf'. "
            "Install with: pip install scikit-learn"
        )

    # Ensure non-negative
    X = np.maximum(expr_matrix, 0)
    n_programs = min(n_programs, min(X.shape))
    model = NMF(n_components=n_programs, random_state=42, max_iter=500)
    model.fit(X)

    programs = {}
    n_top = max(3, len(gene_names) // n_programs)
    for i, component in enumerate(model.components_):
        top_idx = np.argsort(component)[::-1][:n_top]
        programs[f"NMF_{i}"] = [gene_names[j] for j in top_idx]
    return programs


def _discover_programs_pca(expr_matrix: np.ndarray, gene_names: list[str], n_programs: int) -> dict[str, list[str]]:
    """Discover gene programs via PCA loadings.

    Args:
        expr_matrix: Dense expression matrix (cells × genes).
        gene_names: Gene names.
        n_programs: Number of programs.

    Returns:
        Dict of {program_name: [top_gene_names]}.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError(
            "scikit-learn is required for auto_programs='pca'. "
            "Install with: pip install scikit-learn"
        )

    n_programs = min(n_programs, min(expr_matrix.shape))
    model = PCA(n_components=n_programs, random_state=42)
    model.fit(expr_matrix)

    programs = {}
    n_top = max(3, len(gene_names) // n_programs)
    for i, component in enumerate(model.components_):
        top_idx = np.argsort(np.abs(component))[::-1][:n_top]
        programs[f"PC_{i}"] = [gene_names[j] for j in top_idx]
    return programs


def compute_gradient(
    sp,
    interface_result: InterfaceResult,
    genes: list[str] | None = None,
    programs: dict[str, list[str]] | None = None,
    n_bins: int = 20,
    method: str = "ols",
    auto_programs: str | None = None,
    n_auto_programs: int = 5,
    coord_type: str = "global",
    unsigned: bool = False,
) -> GradientResult:
    """Compute spatial expression gradient relative to an interface.

    Fits linear regression of gene expression vs signed distance from
    the interface contour for each gene and/or gene program.

    Args:
        sp: spatioloji object.
        interface_result: Result from ``identify_interface``.
        genes: List of gene names to analyze. ``None`` = all genes.
        programs: Dict of ``{name: [gene_list]}`` for user-defined
            gene modules. ``None`` = skip user programs.
        n_bins: Number of equal-width distance bins for curves.
        method: Regression method. Currently only ``"ols"`` supported.
        auto_programs: ``"nmf"`` or ``"pca"`` to auto-discover gene
            programs. ``None`` = skip auto-discovery.
        n_auto_programs: Number of programs to discover (default 5).
        coord_type: ``'global'`` or ``'local'`` coordinates.
        unsigned: If True, use absolute distances.

    Returns:
        GradientResult with gene/program gradients and binned data.

    Raises:
        ValueError: If genes not found or invalid auto_programs.
    """
    if method != "ols":
        raise ValueError(f"method must be 'ols', got '{method}'")
    if auto_programs is not None and auto_programs not in ("nmf", "pca"):
        raise ValueError(f"auto_programs must be 'nmf', 'pca', or None, got '{auto_programs}'")

    # Compute distances
    distances = signed_distance_to_interface(
        sp, interface_result, coord_type=coord_type, unsigned=unsigned,
    )

    # Get expression
    expr_df = sp.expression.to_dataframe()

    # Resolve gene list
    all_genes = list(expr_df.columns)
    if genes is None:
        genes = all_genes
    else:
        missing = [g for g in genes if g not in all_genes]
        if missing:
            raise ValueError(f"Genes not found in expression matrix: {missing}")

    # Fit gene-level gradients
    gene_rows = []
    dist_arr = distances.values
    for gene in genes:
        vals = expr_df[gene].values
        gene_rows.append({"gene": gene, **_fit_gradient(vals, dist_arr)})
    gene_gradients = pd.DataFrame(gene_rows).set_index("gene")

    # Collect all programs (user-defined + auto-discovered)
    all_programs = dict(programs) if programs else {}

    if auto_programs == "nmf":
        auto = _discover_programs_nmf(
            sp.expression.get_dense(), all_genes, n_auto_programs,
        )
        all_programs.update(auto)
    elif auto_programs == "pca":
        auto = _discover_programs_pca(
            sp.expression.get_dense(), all_genes, n_auto_programs,
        )
        all_programs.update(auto)

    # Fit program-level gradients
    program_rows = []
    program_score_dict = {}
    for pname, pgenes in all_programs.items():
        valid = [g for g in pgenes if g in all_genes]
        if not valid:
            continue
        scores = expr_df[valid].mean(axis=1).values
        program_score_dict[pname] = scores
        program_rows.append({"program": pname, **_fit_gradient(scores, dist_arr)})

    if program_rows:
        program_gradients = pd.DataFrame(program_rows).set_index("program")
        program_scores = pd.DataFrame(program_score_dict, index=expr_df.index)
    else:
        program_gradients = pd.DataFrame(columns=["coef", "pvalue", "r2", "trend"])
        program_scores = pd.DataFrame(index=expr_df.index)

    # Bin expression for plotting (genes + programs)
    bins = _bin_expression(expr_df, distances, genes, n_bins)

    # Also bin program scores so plot_gradient_curve can render them
    if program_score_dict:
        prog_df = pd.DataFrame(program_score_dict, index=expr_df.index)
        prog_bins = _bin_expression(prog_df, distances, list(program_score_dict.keys()), n_bins)
        bins = pd.concat([bins, prog_bins], ignore_index=True)

    return GradientResult(
        distances=distances,
        gene_gradients=gene_gradients,
        program_gradients=program_gradients,
        program_scores=program_scores,
        bins=bins,
        region_a=interface_result.region_a,
        region_b=interface_result.region_b,
    )
```

- [ ] **Step 4: Create point module re-export wrapper**

Create `src/spatioloji_s/spatial/point/gradient.py`:

```python
"""Spatial gradient analysis for point-based spatial data.

Thin wrapper — re-exports ``compute_gradient`` from the polygon module.
Both modes use centroid-based distances, so the logic is identical.
"""

from spatioloji_s.spatial.polygon.gradient import compute_gradient

__all__ = ["compute_gradient"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_gradient.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/gradient.py src/spatioloji_s/spatial/point/gradient.py tests/unit/test_gradient.py
git commit -m "feat: add compute_gradient for spatial expression gradient analysis"
```

---

## Chunk 3: Infiltration Scoring (Task 6)

### Task 6: score_infiltration Implementation

**Files:**
- Create: `src/spatioloji_s/spatial/polygon/infiltration.py`
- Create: `src/spatioloji_s/spatial/point/infiltration.py`
- Create: `tests/unit/test_infiltration.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_infiltration.py`:

```python
"""Tests for immune infiltration scoring."""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.spatial._infiltration_types import InfiltrationResult


class TestScoreInfiltrationBasic:
    """Basic infiltration scoring tests."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        return identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )

    def test_returns_infiltration_result(self, sp_gradient, interface_result):
        """score_infiltration should return an InfiltrationResult."""
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration

        result = score_infiltration(
            sp_gradient, interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
            target_region="TypeA",
        )
        assert isinstance(result, InfiltrationResult)

    def test_cell_classifications(self, sp_gradient, interface_result):
        """Classifications should be infiltrating, resident, or other."""
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration

        result = score_infiltration(
            sp_gradient, interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
            target_region="TypeA",
        )
        valid = {"infiltrating", "resident", "other"}
        assert set(result.cell_classifications.unique()).issubset(valid)
        assert len(result.cell_classifications) == len(sp_gradient.cell_index)

    def test_per_type_metrics_columns(self, sp_gradient, interface_result):
        """per_type_metrics should have expected columns."""
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration

        result = score_infiltration(
            sp_gradient, interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T", "Macrophage"],
            target_region="TypeA",
        )
        expected_cols = {
            "median_depth", "max_depth", "density_slope", "density_pvalue",
            "infiltration_fraction", "n_infiltrating", "n_resident",
        }
        assert expected_cols.issubset(set(result.per_type_metrics.columns))

    def test_per_type_metrics_rows(self, sp_gradient, interface_result):
        """per_type_metrics should have one row per immune type."""
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration

        result = score_infiltration(
            sp_gradient, interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T", "Macrophage"],
            target_region="TypeA",
        )
        assert set(result.per_type_metrics.index) == {"CD8_T", "Macrophage"}

    def test_target_region_auto_detect(self, sp_gradient, interface_result):
        """target_region=None should auto-detect the region with fewer immune cells."""
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration

        result = score_infiltration(
            sp_gradient, interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
        )
        assert result.target_region in ("TypeA", "TypeB")

    def test_infiltration_fraction_range(self, sp_gradient, interface_result):
        """Infiltration fraction should be between 0 and 1."""
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration

        result = score_infiltration(
            sp_gradient, interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
            target_region="TypeA",
        )
        fracs = result.per_type_metrics["infiltration_fraction"]
        assert (fracs >= 0).all() and (fracs <= 1).all()

    def test_distances_series(self, sp_gradient, interface_result):
        """distances should be a Series indexed by cell ID."""
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration

        result = score_infiltration(
            sp_gradient, interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
            target_region="TypeA",
        )
        assert isinstance(result.distances, pd.Series)
        assert len(result.distances) == len(sp_gradient.cell_index)

    def test_region_labels_propagated(self, sp_gradient, interface_result):
        """region_a/b should be copied from InterfaceResult."""
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration

        result = score_infiltration(
            sp_gradient, interface_result,
            immune_col="immune_type",
            immune_types=["CD8_T"],
            target_region="TypeA",
        )
        assert result.region_a == interface_result.region_a
        assert result.region_b == interface_result.region_b


class TestScoreInfiltrationValidation:
    """Input validation tests."""

    @pytest.fixture
    def interface_result(self, sp_gradient):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        return identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )

    def test_invalid_immune_col(self, sp_gradient, interface_result):
        """Invalid immune_col should raise ValueError."""
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration

        with pytest.raises(ValueError, match="not found"):
            score_infiltration(
                sp_gradient, interface_result,
                immune_col="nonexistent",
                immune_types=["CD8_T"],
            )

    def test_invalid_target_region(self, sp_gradient, interface_result):
        """Invalid target_region should raise ValueError."""
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration

        with pytest.raises(ValueError, match="target_region"):
            score_infiltration(
                sp_gradient, interface_result,
                immune_col="immune_type",
                immune_types=["CD8_T"],
                target_region="InvalidRegion",
            )


class TestPointInfiltrationReExport:
    """Verify point module re-exports polygon infiltration."""

    def test_point_score_infiltration_is_same(self):
        """Point score_infiltration should be same function as polygon."""
        from spatioloji_s.spatial.point.infiltration import score_infiltration as point_si
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration as poly_si

        assert point_si is poly_si
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_infiltration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spatioloji_s.spatial.polygon.infiltration'`

- [ ] **Step 3: Implement score_infiltration**

Create `src/spatioloji_s/spatial/polygon/infiltration.py`:

```python
"""Immune infiltration scoring for polygon-based spatial data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

from spatioloji_s.spatial._distance_utils import signed_distance_to_interface
from spatioloji_s.spatial._infiltration_types import InfiltrationResult
from spatioloji_s.spatial._interface_types import InterfaceResult


def score_infiltration(
    sp,
    interface_result: InterfaceResult,
    immune_col: str,
    immune_types: list[str],
    target_region: str | None = None,
    depth_bins: int = 10,
    coord_type: str = "global",
) -> InfiltrationResult:
    """Score immune cell infiltration across a spatial interface.

    Quantifies how deeply immune cells penetrate into a target region,
    computing penetration depth, density gradient, and infiltration
    fraction for each immune cell type.

    Args:
        sp: spatioloji object.
        interface_result: Result from ``identify_interface``.
        immune_col: Column in ``sp.cell_meta`` with cell type labels.
        immune_types: List of cell type labels considered immune.
        target_region: Region immune cells infiltrate into
            (``region_a`` or ``region_b`` label). ``None`` = auto-detect
            as the region with fewer immune cells.
        depth_bins: Number of distance bins for density gradient.
        coord_type: ``'global'`` or ``'local'`` coordinates.

    Returns:
        InfiltrationResult with per-type metrics and cell classifications.

    Raises:
        ValueError: If immune_col not found, target_region invalid.
    """
    # Validate inputs
    if immune_col not in sp.cell_meta.columns:
        raise ValueError(
            f"'{immune_col}' not found in cell_meta. "
            f"Available: {list(sp.cell_meta.columns)}"
        )

    cell_types = sp.cell_meta[immune_col]
    labels = interface_result.cell_labels

    # Normalize region labels for comparison
    region_a = interface_result.region_a
    region_b = interface_result.region_b
    a_list = [region_a] if isinstance(region_a, str) else list(region_a)
    b_list = [region_b] if isinstance(region_b, str) else list(region_b)

    # Determine target region
    if target_region is not None:
        if target_region not in a_list + b_list:
            raise ValueError(
                f"target_region '{target_region}' not in region_a={region_a} "
                f"or region_b={region_b}"
            )
        target_is_a = target_region in a_list
    else:
        # Auto-detect: region with fewer immune cells
        immune_mask = cell_types.isin(immune_types)
        a_mask = labels.isin(["region_a_interface", "interior_a"])
        b_mask = labels.isin(["region_b_interface", "interior_b"])
        n_immune_a = (immune_mask & a_mask).sum()
        n_immune_b = (immune_mask & b_mask).sum()
        target_is_a = n_immune_a <= n_immune_b
        target_region = a_list[0] if target_is_a else b_list[0]

    # Compute signed distances
    distances = signed_distance_to_interface(
        sp, interface_result, coord_type=coord_type,
    )

    # Classify cells
    # Target region cells have positive distance if target=A, negative if target=B
    # "infiltrating" = immune cell in target region
    # "resident" = immune cell NOT in target region
    if target_is_a:
        target_mask = labels.isin(["region_a_interface", "interior_a"])
    else:
        target_mask = labels.isin(["region_b_interface", "interior_b"])

    immune_mask = cell_types.isin(immune_types)
    classifications = pd.Series("other", index=sp.cell_index)
    classifications[immune_mask & target_mask] = "infiltrating"
    classifications[immune_mask & ~target_mask] = "resident"

    # Per-type metrics
    metric_rows = []
    for itype in immune_types:
        type_mask = cell_types == itype
        type_in_target = type_mask & target_mask
        type_not_in_target = type_mask & ~target_mask

        n_infiltrating = int(type_in_target.sum())
        n_resident = int(type_not_in_target.sum())
        n_total = n_infiltrating + n_resident

        # Penetration depth (absolute distance into target region)
        if n_infiltrating > 0:
            depths = distances[type_in_target].abs().values
            median_depth = float(np.median(depths))
            max_depth = float(np.max(depths))
        else:
            median_depth = 0.0
            max_depth = 0.0

        # Infiltration fraction
        infiltration_fraction = n_infiltrating / n_total if n_total > 0 else 0.0

        # Density gradient — bin immune cells by distance, regress count ~ distance
        type_distances = distances[type_mask].values
        if len(type_distances) >= 3:
            bin_edges = np.linspace(
                type_distances.min(), type_distances.max(), depth_bins + 1,
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            counts, _ = np.histogram(type_distances, bins=bin_edges)
            # Fit on all bins (including empty ones with count=0)
            if len(bin_centers) >= 3:
                result = linregress(bin_centers, counts.astype(float))
                density_slope = result.slope
                density_pvalue = result.pvalue
            else:
                density_slope = np.nan
                density_pvalue = np.nan
        else:
            density_slope = np.nan
            density_pvalue = np.nan

        metric_rows.append({
            "immune_type": itype,
            "median_depth": median_depth,
            "max_depth": max_depth,
            "density_slope": density_slope,
            "density_pvalue": density_pvalue,
            "infiltration_fraction": infiltration_fraction,
            "n_infiltrating": n_infiltrating,
            "n_resident": n_resident,
        })

    per_type_metrics = pd.DataFrame(metric_rows).set_index("immune_type")

    return InfiltrationResult(
        distances=distances,
        cell_classifications=classifications,
        per_type_metrics=per_type_metrics,
        region_a=region_a,
        region_b=region_b,
        target_region=target_region,
    )
```

- [ ] **Step 4: Create point module re-export wrapper**

Create `src/spatioloji_s/spatial/point/infiltration.py`:

```python
"""Immune infiltration scoring for point-based spatial data.

Thin wrapper — re-exports ``score_infiltration`` from the polygon module.
Both modes use centroid-based distances, so the logic is identical.
"""

from spatioloji_s.spatial.polygon.infiltration import score_infiltration

__all__ = ["score_infiltration"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_infiltration.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/infiltration.py src/spatioloji_s/spatial/point/infiltration.py tests/unit/test_infiltration.py
git commit -m "feat: add score_infiltration for immune cell infiltration analysis"
```

---

## Chunk 4: Visualization & Wiring (Tasks 7–8)

### Task 7: Visualization Functions

**Files:**
- Modify: `src/spatioloji_s/visualization/polygon_plots.py`
- Modify: `src/spatioloji_s/visualization/point_plots.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_gradient.py`:

```python
class TestPlotGradientCurve:
    """Tests for plot_gradient_curve."""

    @pytest.fixture
    def gradient_result(self, sp_gradient):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        from spatioloji_s.spatial.polygon.interface import identify_interface

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        iface = identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )
        return compute_gradient(
            sp_gradient, iface, genes=["gene_0", "gene_1"],
        )

    def test_returns_figure(self, gradient_result):
        import matplotlib
        from spatioloji_s.visualization.polygon_plots import plot_gradient_curve

        fig = plot_gradient_curve(gradient_result, genes=["gene_0", "gene_1"])
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_programs_plot(self, sp_gradient):
        import matplotlib
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        from spatioloji_s.spatial.polygon.interface import identify_interface
        from spatioloji_s.visualization.polygon_plots import plot_gradient_curve

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        iface = identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )
        result = compute_gradient(
            sp_gradient, iface, genes=["gene_0"],
            programs={"test_prog": ["gene_0", "gene_1"]},
        )
        fig = plot_gradient_curve(result, programs=["test_prog"])
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotSpatialDistance:
    """Tests for plot_spatial_distance."""

    def test_returns_figure(self, sp_gradient):
        import matplotlib
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        from spatioloji_s.spatial.polygon.interface import identify_interface
        from spatioloji_s.visualization.polygon_plots import plot_spatial_distance

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        iface = identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )
        result = compute_gradient(sp_gradient, iface, genes=["gene_0"])
        fig = plot_spatial_distance(sp_gradient, result.distances)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_contour_overlay(self, sp_gradient):
        """plot_spatial_distance should render contour when interface_result provided."""
        import matplotlib
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        from spatioloji_s.spatial.polygon.interface import identify_interface
        from spatioloji_s.visualization.polygon_plots import plot_spatial_distance

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        iface = identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )
        result = compute_gradient(sp_gradient, iface, genes=["gene_0"])
        fig = plot_spatial_distance(sp_gradient, result.distances, interface_result=iface)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)
```

Add to `tests/unit/test_infiltration.py`:

```python
class TestPlotInfiltrationSummary:
    """Tests for plot_infiltration_summary."""

    def test_returns_figure(self, sp_gradient):
        import matplotlib
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration
        from spatioloji_s.spatial.polygon.interface import identify_interface
        from spatioloji_s.visualization.polygon_plots import plot_infiltration_summary

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        iface = identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )
        result = score_infiltration(
            sp_gradient, iface,
            immune_col="immune_type",
            immune_types=["CD8_T", "Macrophage"],
            target_region="TypeA",
        )
        fig = plot_infiltration_summary(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_gradient.py::TestPlotGradientCurve tests/unit/test_gradient.py::TestPlotSpatialDistance tests/unit/test_infiltration.py::TestPlotInfiltrationSummary -v`
Expected: FAIL with `ImportError: cannot import name 'plot_gradient_curve'`

- [ ] **Step 3: Implement plot_gradient_curve, plot_spatial_distance, plot_infiltration_summary**

Append to `src/spatioloji_s/visualization/polygon_plots.py`.

First, add these imports near the top of the file (after existing imports):

```python
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from spatioloji_s.spatial._gradient_types import GradientResult
from spatioloji_s.spatial._infiltration_types import InfiltrationResult
from spatioloji_s.spatial._interface_types import InterfaceResult
```

Then append the functions:

```python
# ---------------------------------------------------------------------------
# Gradient & infiltration plots
# ---------------------------------------------------------------------------


def plot_gradient_curve(
    gradient_result,
    genes: list[str] | None = None,
    programs: list[str] | None = None,
    n_cols: int = 3,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Expression vs distance-from-interface curve plot.

    Args:
        gradient_result: GradientResult from ``compute_gradient``.
        genes: Genes to plot (from gene_gradients). None = all.
        programs: Programs to plot (from program_gradients). None = skip.
        n_cols: Number of columns in subplot grid.
        figsize: Figure size. Auto-calculated if None.
        save_path: Save figure to path if provided.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    bins = gradient_result.bins
    items = []

    # Collect genes
    if genes is None and programs is None:
        genes = list(gradient_result.gene_gradients.index)
    if genes:
        for g in genes:
            if g in gradient_result.gene_gradients.index:
                items.append(("gene", g))
    if programs:
        for p in programs:
            if p in gradient_result.program_gradients.index:
                items.append(("program", p))

    if not items:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No items to plot", ha="center", va="center", transform=ax.transAxes)
        return fig

    n_items = len(items)
    n_rows = int(np.ceil(n_items / n_cols))
    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx, (item_type, name) in enumerate(items):
        ax = axes[idx // n_cols, idx % n_cols]

        if item_type == "gene":
            subset = bins[bins["gene"] == name]
            grad_row = gradient_result.gene_gradients.loc[name]
        else:
            # For programs, compute bins from program_scores
            subset = bins[bins["gene"] == name] if name in bins["gene"].values else None
            grad_row = gradient_result.program_gradients.loc[name]

        if subset is not None and not subset.empty:
            x = subset["distance_bin"].values
            y = subset["mean_expr"].values
            std = subset["std_expr"].values
            ax.plot(x, y, "-o", markersize=3, color="steelblue")
            ax.fill_between(x, y - std, y + std, alpha=0.2, color="steelblue")

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        coef = grad_row.get("coef", np.nan)
        r2 = grad_row.get("r2", np.nan)
        pval = grad_row.get("pvalue", np.nan)
        ax.set_title(name, fontsize=10)
        ax.annotate(
            f"slope={coef:.3f}\nR²={r2:.3f}\np={pval:.2e}",
            xy=(0.02, 0.98), xycoords="axes fraction",
            va="top", fontsize=7, family="monospace",
        )
        ax.set_xlabel("Distance from interface")
        ax.set_ylabel("Expression")

    # Hide empty subplots
    for idx in range(n_items, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(Path(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_spatial_distance(
    sp,
    distances: pd.Series,
    interface_result: InterfaceResult | None = None,
    coord_type: str = "global",
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Spatial map colored by signed distance from interface.

    Args:
        sp: spatioloji object.
        distances: Series of signed distances indexed by cell ID.
        interface_result: Optional InterfaceResult to overlay contour.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        cmap: Diverging colormap name.
        figsize: Figure size.
        save_path: Save figure to path if provided.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    if figsize is None:
        figsize = (10, 8)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get coordinates
    if coord_type == "global":
        x = np.asarray(sp.spatial.x_global)
        y = np.asarray(sp.spatial.y_global)
    else:
        x = np.asarray(sp.spatial.x_local)
        y = np.asarray(sp.spatial.y_local)

    # Align distances to cell order
    d_vals = distances.reindex(sp.cell_index).values

    # Center colormap at 0
    vmax = max(abs(np.nanmin(d_vals)), abs(np.nanmax(d_vals)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    sc = ax.scatter(x, y, c=d_vals, cmap=cmap, norm=norm, s=8, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Signed distance from interface")

    # Overlay contour
    if interface_result is not None and interface_result.contour is not None:
        for geom in interface_result.contour.geoms:
            coords = np.array(geom.coords)
            ax.plot(coords[:, 0], coords[:, 1], "k-", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Signed Distance from Interface")
    ax.set_aspect("equal")
    fig.tight_layout()

    if save_path:
        fig.savefig(Path(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_infiltration_summary(
    infiltration_result: InfiltrationResult,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Bar chart summarizing infiltration metrics per immune type.

    Three side-by-side panels: penetration depth, density slope,
    infiltration fraction.

    Args:
        infiltration_result: InfiltrationResult from ``score_infiltration``.
        figsize: Figure size.
        save_path: Save figure to path if provided.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = infiltration_result.per_type_metrics
    if figsize is None:
        figsize = (12, 4)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    types = metrics.index.tolist()
    y_pos = np.arange(len(types))

    # Panel 1: Median penetration depth
    ax = axes[0]
    ax.barh(y_pos, metrics["median_depth"].values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)
    ax.set_xlabel("Median depth")
    ax.set_title("Penetration Depth")

    # Panel 2: Density slope
    ax = axes[1]
    vals = metrics["density_slope"].values
    colors = ["salmon" if v < 0 else "steelblue" for v in vals]
    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)
    ax.set_xlabel("Density slope")
    ax.set_title("Density Gradient")

    # Panel 3: Infiltration fraction
    ax = axes[2]
    ax.barh(y_pos, metrics["infiltration_fraction"].values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)
    ax.set_xlabel("Fraction")
    ax.set_title("Infiltration Fraction")
    ax.set_xlim(0, 1)

    fig.suptitle(f"Immune Infiltration into {infiltration_result.target_region}", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(Path(save_path), dpi=150, bbox_inches="tight")
    return fig
```

Also append to `src/spatioloji_s/visualization/point_plots.py` (delegate to polygon_plots):

```python
# ---------------------------------------------------------------------------
# Gradient & infiltration plots (delegates to polygon_plots)
# ---------------------------------------------------------------------------

from spatioloji_s.visualization.polygon_plots import (
    plot_gradient_curve,
    plot_infiltration_summary,
    plot_spatial_distance,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_gradient.py::TestPlotGradientCurve tests/unit/test_gradient.py::TestPlotSpatialDistance tests/unit/test_infiltration.py::TestPlotInfiltrationSummary -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatioloji_s/visualization/polygon_plots.py src/spatioloji_s/visualization/point_plots.py tests/unit/test_gradient.py tests/unit/test_infiltration.py
git commit -m "feat: add gradient curve, spatial distance, and infiltration summary plots"
```

---

### Task 8: Module Exports & Integration

**Files:**
- Modify: `src/spatioloji_s/spatial/polygon/__init__.py`
- Modify: `src/spatioloji_s/spatial/point/__init__.py`
- Modify: `src/spatioloji_s/visualization/__init__.py`

- [ ] **Step 1: Update polygon/__init__.py**

Add imports and __all__ entries:

```python
# After the Interface section, add:

# Gradient
from .gradient import compute_gradient
from .._gradient_types import GradientResult

# Infiltration
from .infiltration import score_infiltration
from .._infiltration_types import InfiltrationResult
```

Add to `__all__`:

```python
    # Gradient
    "compute_gradient",
    "GradientResult",
    # Infiltration
    "score_infiltration",
    "InfiltrationResult",
```

- [ ] **Step 2: Update point/__init__.py**

Add imports and __all__ entries:

```python
# After the Interface section, add:

# Gradient
from .gradient import compute_gradient
from .._gradient_types import GradientResult

# Infiltration
from .infiltration import score_infiltration
from .._infiltration_types import InfiltrationResult
```

Add to `__all__`:

```python
    # Gradient
    "compute_gradient",
    "GradientResult",
    # Infiltration
    "score_infiltration",
    "InfiltrationResult",
```

- [ ] **Step 3: Update visualization/__init__.py**

Add to polygon_plots imports:

```python
    plot_gradient_curve,
    plot_infiltration_summary,
    plot_spatial_distance,
```

Add to `__all__`:

```python
    "plot_gradient_curve",
    "plot_spatial_distance",
    "plot_infiltration_summary",
```

- [ ] **Step 4: Write integration test**

Add to `tests/unit/test_gradient.py`:

```python
class TestIntegration:
    """End-to-end integration tests."""

    def test_full_gradient_workflow(self, sp_gradient):
        """Full workflow: interface → gradient → plot."""
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface
        from spatioloji_s.spatial.polygon.gradient import compute_gradient
        from spatioloji_s.visualization.polygon_plots import plot_gradient_curve, plot_spatial_distance
        import matplotlib.pyplot as plt

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        iface = identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )
        result = compute_gradient(
            sp_gradient, iface,
            genes=["gene_0", "gene_1"],
            programs={"test": ["gene_0", "gene_1"]},
        )

        assert isinstance(result, GradientResult)
        assert result.gene_gradients.shape[0] == 2
        assert result.program_gradients.shape[0] == 1

        fig1 = plot_gradient_curve(result, genes=["gene_0"])
        fig2 = plot_spatial_distance(sp_gradient, result.distances, iface)
        plt.close(fig1)
        plt.close(fig2)
```

Add to `tests/unit/test_infiltration.py`:

```python
class TestIntegration:
    """End-to-end integration tests."""

    def test_full_infiltration_workflow(self, sp_gradient):
        """Full workflow: interface → infiltration → plot."""
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.interface import identify_interface
        from spatioloji_s.spatial.polygon.infiltration import score_infiltration
        from spatioloji_s.visualization.polygon_plots import plot_infiltration_summary
        import matplotlib.pyplot as plt

        graph = build_buffer_graph(sp_gradient, buffer_distance=50)
        iface = identify_interface(
            sp_gradient, graph, group_col="cell_type",
            region_a="TypeA", region_b="TypeB", method="graph",
            min_interface_cells=1,
        )
        result = score_infiltration(
            sp_gradient, iface,
            immune_col="immune_type",
            immune_types=["CD8_T", "Macrophage"],
            target_region="TypeA",
        )

        assert isinstance(result, InfiltrationResult)
        assert "CD8_T" in result.per_type_metrics.index

        fig = plot_infiltration_summary(result)
        plt.close(fig)
```

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/__init__.py src/spatioloji_s/spatial/point/__init__.py src/spatioloji_s/visualization/__init__.py tests/unit/test_gradient.py tests/unit/test_infiltration.py
git commit -m "feat: wire gradient and infiltration modules into package exports"
```
