# Hierarchical Spatial Motif Discovery — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-scale spatial motif discovery that identifies local cellular neighborhood motifs, mesoscale tissue assemblies, and matches them to known biological structures.

**Architecture:** Three-stage pipeline — (1) cluster cells by neighborhood composition into motif classes, (2) find spatially contiguous motif instances and cluster their arrangements into assemblies, (3) match discovered patterns against known structure signatures. All stages operate on existing spatial graphs (polygon or point) and share result dataclasses in `_motif_types.py`.

**Tech Stack:** numpy, pandas, scipy (sparse, csgraph), scikit-learn (MiniBatchKMeans, NearestNeighbors, calinski_harabasz_score); optional leidenalg/igraph for Leiden clustering

**Spec:** `docs/superpowers/specs/2026-03-14-spatial-motifs-design.md`

---

## Chunk 1: Foundation (Tasks 1–2)

### Task 1: Dataclasses & Graph Helpers

**Files:**
- Create: `src/spatioloji_s/spatial/_motif_types.py`

- [ ] **Step 1: Write the dataclasses and helpers**

Create `src/spatioloji_s/spatial/_motif_types.py`:

```python
"""Shared data structures and helpers for spatial motif analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse


@dataclass
class MotifCatalog:
    """Container for local motif discovery results.

    Attributes:
        labels: Series mapping cell_id → motif_id (int).
        signatures: DataFrame (motif_id × cell_type) with mean composition.
        counts: Series mapping motif_id → number of cells.
        group_col: Cell-type column name used for composition.
        feature_matrix: Sparse feature matrix used for clustering.
            Retained only if ``keep_features=True``.
        params: Parameters used for discovery.
    """

    labels: pd.Series
    signatures: pd.DataFrame
    counts: pd.Series
    group_col: str
    feature_matrix: scipy.sparse.csr_matrix | None
    params: dict


@dataclass
class AssemblyCatalog:
    """Container for mesoscale assembly detection results.

    Attributes:
        labels: Series mapping cell_id → assembly_id (int, -1 = unassigned).
        composition: DataFrame (assembly_id × motif_id) with mean motif proportions.
        instances: DataFrame with one row per motif instance.
            Columns: instance_id, assembly_id, motif_id, n_cells, centroid_x, centroid_y.
        adjacency_pattern: Long-form DataFrame of motif-pair adjacency frequencies
            per assembly type. Columns: assembly_id, motif_a, motif_b, frequency.
        params: Parameters used for detection.
    """

    labels: pd.Series
    composition: pd.DataFrame
    instances: pd.DataFrame
    adjacency_pattern: pd.DataFrame
    params: dict


@dataclass
class StructureMatches:
    """Container for known structure matching results.

    Attributes:
        matches: DataFrame with columns structure_name, target_type
            ("motif"/"assembly"), target_id, similarity, n_cells,
            centroid_x, centroid_y.
        per_cell: Series mapping cell_id → matched structure name or "unmatched".
        signatures_used: Dict of signatures that were queried.
    """

    matches: pd.DataFrame
    per_cell: pd.Series
    signatures_used: dict


@dataclass
class MotifResult:
    """Top-level container for the full motif pipeline.

    Attributes:
        motif_catalog: Local motif discovery results.
        assembly_catalog: Mesoscale assembly results (None if skipped).
        structure_matches: Known structure matches (None if skipped).
        params: Pipeline parameters.
    """

    motif_catalog: MotifCatalog
    assembly_catalog: AssemblyCatalog | None
    structure_matches: StructureMatches | None
    params: dict


def _get_cell_ids(graph) -> pd.Index:
    """Return cell IDs from either PolygonSpatialGraph or PointSpatialGraph."""
    if hasattr(graph, "cell_index"):
        return graph.cell_index
    return graph.cell_ids


def _get_sparse_adjacency(graph) -> scipy.sparse.csr_matrix:
    """Return sparse adjacency matrix from either graph type."""
    return graph.adjacency
```

- [ ] **Step 2: Lint**

Run: `ruff check src/spatioloji_s/spatial/_motif_types.py --fix`
Expected: All checks passed

- [ ] **Step 3: Commit**

```bash
git add src/spatioloji_s/spatial/_motif_types.py
git commit -m "feat: add motif analysis dataclasses and graph helpers"
```

---

### Task 2: Test Fixture

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/unit/test_motifs.py` (initial skeleton)

- [ ] **Step 1: Add sp_motif fixture to conftest.py**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def sp_motif():
    """500-cell spatioloji with structured tissue layout for motif tests.

    Layout:
    - Center (100 cells, x=400-600, y=400-600): Dense Tumor core (80% Tumor, 20% Fibroblast)
    - Inner ring (100 cells, x=250-750, y=250-750 minus center): Macrophage + Fibroblast stroma
    - Left lobe (100 cells, x=50-250, y=350-650): T_cell + B_cell aggregate (TLS-like)
    - Right lobe (100 cells, x=750-950, y=350-650): Scattered T_cell + Macrophage
    - Periphery (100 cells, x=0-1000, y=0-1000 outer): Sparse Fibroblast + Tumor
    """
    np.random.seed(42)
    n_cells = 500
    n_genes = 10
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # -- Coordinates --
    # Center cluster (Tumor core)
    cx = np.random.uniform(400, 600, 100)
    cy = np.random.uniform(400, 600, 100)

    # Inner ring (stroma)
    rx = np.random.uniform(250, 750, 100)
    ry = np.random.uniform(250, 750, 100)
    # Push away from center
    for i in range(100):
        while 400 <= rx[i] <= 600 and 400 <= ry[i] <= 600:
            rx[i] = np.random.uniform(250, 750)
            ry[i] = np.random.uniform(250, 750)

    # Left lobe (TLS-like)
    lx = np.random.uniform(50, 250, 100)
    ly = np.random.uniform(350, 650, 100)

    # Right lobe (immune infiltrate)
    rlx = np.random.uniform(750, 950, 100)
    rly = np.random.uniform(350, 650, 100)

    # Periphery
    px = np.random.uniform(0, 1000, 100)
    py = np.random.uniform(0, 1000, 100)

    x_global = np.concatenate([cx, rx, lx, rlx, px])
    y_global = np.concatenate([cy, ry, ly, rly, py])

    # -- Cell types --
    cell_types = []
    # Center: 80 Tumor, 20 Fibroblast
    cell_types += ["Tumor"] * 80 + ["Fibroblast"] * 20
    # Inner ring: 60 Macrophage, 40 Fibroblast
    cell_types += ["Macrophage"] * 60 + ["Fibroblast"] * 40
    # Left lobe: 50 T_cell, 40 B_cell, 10 Macrophage
    cell_types += ["T_cell"] * 50 + ["B_cell"] * 40 + ["Macrophage"] * 10
    # Right lobe: 60 T_cell, 30 Macrophage, 10 Fibroblast
    cell_types += ["T_cell"] * 60 + ["Macrophage"] * 30 + ["Fibroblast"] * 10
    # Periphery: 50 Fibroblast, 30 Tumor, 20 T_cell
    cell_types += ["Fibroblast"] * 50 + ["Tumor"] * 30 + ["T_cell"] * 20

    cell_meta = pd.DataFrame({"cell_type": cell_types}, index=cell_ids)

    # -- Expression --
    expression = np.random.poisson(2.0, (n_cells, n_genes)).astype(float)

    # -- Spatial data --
    spatial = pd.DataFrame(
        {
            "x_global": x_global,
            "y_global": y_global,
            "x_local": x_global,
            "y_local": y_global,
            "fov": 1,
        },
        index=cell_ids,
    )

    # -- Polygons: 4x4 squares --
    polygons = {}
    for i, cid in enumerate(cell_ids):
        bx, by = x_global[i], y_global[i]
        polygons[cid] = np.array([
            [bx - 2, by - 2],
            [bx + 2, by - 2],
            [bx + 2, by + 2],
            [bx - 2, by + 2],
            [bx - 2, by - 2],
        ])

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        spatial_coords=spatial,
        polygons=polygons,
    )
```

- [ ] **Step 2: Create test skeleton**

Create `tests/unit/test_motifs.py`:

```python
"""Tests for hierarchical spatial motif discovery."""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.spatial._motif_types import (
    AssemblyCatalog,
    MotifCatalog,
    MotifResult,
    StructureMatches,
)


class TestMotifTypes:
    """Verify dataclass imports and basic construction."""

    def test_motif_catalog_fields(self):
        mc = MotifCatalog(
            labels=pd.Series([0, 1], index=["a", "b"]),
            signatures=pd.DataFrame({"T": [0.5, 0.5]}, index=[0, 1]),
            counts=pd.Series([1, 1], index=[0, 1]),
            group_col="cell_type",
            feature_matrix=None,
            params={},
        )
        assert mc.group_col == "cell_type"
        assert len(mc.labels) == 2

    def test_assembly_catalog_fields(self):
        ac = AssemblyCatalog(
            labels=pd.Series([0, -1], index=["a", "b"]),
            composition=pd.DataFrame(),
            instances=pd.DataFrame(),
            adjacency_pattern=pd.DataFrame(),
            params={},
        )
        assert (ac.labels == [0, -1]).all()

    def test_structure_matches_fields(self):
        sm = StructureMatches(
            matches=pd.DataFrame(),
            per_cell=pd.Series(dtype=str),
            signatures_used={},
        )
        assert sm.matches.empty

    def test_motif_result_fields(self):
        mc = MotifCatalog(
            labels=pd.Series(dtype=int),
            signatures=pd.DataFrame(),
            counts=pd.Series(dtype=int),
            group_col="ct",
            feature_matrix=None,
            params={},
        )
        mr = MotifResult(
            motif_catalog=mc,
            assembly_catalog=None,
            structure_matches=None,
            params={},
        )
        assert mr.assembly_catalog is None
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_motifs.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 4: Verify sp_motif fixture**

Run: `pytest tests/unit/test_motifs.py -v -k "test_motif"` (just to exercise conftest import)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatioloji_s/spatial/_motif_types.py tests/conftest.py tests/unit/test_motifs.py
git commit -m "feat: add motif dataclasses, sp_motif fixture, and test skeleton"
```

---

## Chunk 2: Local Motif Discovery (Task 3)

### Task 3: discover_motifs Implementation

**Files:**
- Create: `src/spatioloji_s/spatial/polygon/motifs.py`
- Create: `src/spatioloji_s/spatial/point/motifs.py`
- Modify: `tests/unit/test_motifs.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_motifs.py`:

```python
class TestDiscoverMotifsKMeans:
    """Tests for discover_motifs with KMeans."""

    @pytest.fixture
    def graph(self, sp_motif):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        return build_buffer_graph(sp_motif, buffer_distance=30)

    def test_returns_motif_catalog(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        result = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert isinstance(result, MotifCatalog)

    def test_labels_cover_all_cells(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        result = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert len(result.labels) == len(sp_motif.cell_index)
        assert result.labels.index.equals(sp_motif.cell_index)

    def test_n_motifs_respected(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        result = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=4)
        assert result.signatures.shape[0] == 4
        assert len(result.counts) == 4

    def test_signatures_sum_to_one(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        result = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        # Each row sums to ~1 (composition fractions, L2-normalized then averaged back)
        row_sums = result.signatures.sum(axis=1)
        assert (row_sums > 0).all()

    def test_group_col_stored(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        result = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert result.group_col == "cell_type"

    def test_feature_matrix_none_by_default(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        result = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert result.feature_matrix is None

    def test_keep_features(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        result = discover_motifs(
            sp_motif, graph, group_col="cell_type", n_motifs=5, keep_features=True,
        )
        assert result.feature_matrix is not None
        assert result.feature_matrix.shape[0] == len(sp_motif.cell_index)

    def test_store_writes_to_cell_meta(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5, store=True)
        assert "motif_label" in sp_motif.cell_meta.columns

    def test_auto_n_motifs(self, sp_motif, graph):
        """n_motifs=None should auto-select via Calinski-Harabasz."""
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        result = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=None)
        assert isinstance(result, MotifCatalog)
        assert result.signatures.shape[0] >= 2


class TestDiscoverMotifsLeiden:
    """Tests for discover_motifs with Leiden."""

    @pytest.fixture
    def graph(self, sp_motif):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        return build_buffer_graph(sp_motif, buffer_distance=30)

    def test_leiden_returns_motif_catalog(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        result = discover_motifs(
            sp_motif, graph, group_col="cell_type", method="leiden", resolution=1.0,
        )
        assert isinstance(result, MotifCatalog)
        assert len(result.labels) == len(sp_motif.cell_index)


class TestDiscoverMotifsValidation:
    """Validation and edge case tests."""

    @pytest.fixture
    def graph(self, sp_motif):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        return build_buffer_graph(sp_motif, buffer_distance=30)

    def test_invalid_method(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        with pytest.raises(ValueError, match="method"):
            discover_motifs(sp_motif, graph, group_col="cell_type", method="invalid")

    def test_invalid_group_col(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        with pytest.raises(ValueError, match="not found"):
            discover_motifs(sp_motif, graph, group_col="nonexistent")

    def test_morphology_with_point_graph_raises(self, sp_motif):
        from spatioloji_s.spatial.point.graph import build_knn_graph
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        pg = build_knn_graph(sp_motif, k=10)
        with pytest.raises(ValueError, match="polygon graph"):
            discover_motifs(
                sp_motif, pg, group_col="cell_type", n_motifs=5, include_morphology=True,
            )

    def test_density_with_point_graph_raises(self, sp_motif):
        from spatioloji_s.spatial.point.graph import build_knn_graph
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        pg = build_knn_graph(sp_motif, k=10)
        with pytest.raises(ValueError, match="polygon graph"):
            discover_motifs(
                sp_motif, pg, group_col="cell_type", n_motifs=5, include_density=True,
            )

    def test_single_motif(self, sp_motif):
        """n_motifs=1 should assign all cells to motif 0."""
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        result = discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=1)
        assert result.labels.nunique() == 1


class TestPointMotifReExport:
    """Verify point module re-exports polygon motifs."""

    def test_discover_motifs_is_same(self):
        from spatioloji_s.spatial.point.motifs import discover_motifs as point_dm
        from spatioloji_s.spatial.polygon.motifs import discover_motifs as poly_dm
        assert point_dm is poly_dm
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_motifs.py::TestDiscoverMotifsKMeans::test_returns_motif_catalog -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spatioloji_s.spatial.polygon.motifs'`

- [ ] **Step 3: Implement discover_motifs**

Create `src/spatioloji_s/spatial/polygon/motifs.py`:

```python
"""Hierarchical spatial motif discovery for polygon-based spatial data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from spatioloji_s.spatial._motif_types import (
    AssemblyCatalog,
    MotifCatalog,
    MotifResult,
    StructureMatches,
    _get_cell_ids,
    _get_sparse_adjacency,
)


def _build_composition_features(
    sp,
    graph,
    group_col: str,
    k_hops: int,
) -> tuple[sp_sparse.csr_matrix, list[str]]:
    """Build sparse neighborhood composition matrix.

    Args:
        sp: spatioloji object.
        graph: Spatial graph (polygon or point).
        group_col: Cell-type column name.
        k_hops: Neighborhood radius in hops.

    Returns:
        Tuple of (sparse CSR feature matrix, list of cell-type names).
    """
    cell_ids = _get_cell_ids(graph)
    adj = _get_sparse_adjacency(graph)
    labels = sp.cell_meta[group_col]
    types = sorted(labels.unique())
    type_to_idx = {t: i for i, t in enumerate(types)}
    n_cells = len(cell_ids)
    n_types = len(types)

    # Expand adjacency for k-hops
    if k_hops > 1:
        expanded = adj.copy()
        power = adj.copy()
        for _ in range(k_hops - 1):
            power = power @ adj
            expanded = expanded + power
        # Binarize
        expanded.data[:] = 1.0
        adj = expanded

    # Build composition matrix
    # For each cell, count types of neighbors
    # Reindex labels to match graph cell ordering
    aligned_labels = labels.reindex(cell_ids)
    label_indices = np.array([type_to_idx[aligned_labels.iloc[i]] for i in range(n_cells)])
    # One-hot encode cell types
    one_hot = sp_sparse.csr_matrix(
        (np.ones(n_cells), (np.arange(n_cells), label_indices)),
        shape=(n_cells, n_types),
    )
    # Composition = adjacency @ one_hot → (n_cells × n_types)
    comp = adj @ one_hot

    # Normalize to fractions
    row_sums = np.asarray(comp.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    diag_inv = sp_sparse.diags(1.0 / row_sums)
    comp = diag_inv @ comp

    return comp, types


def _auto_select_n_motifs(
    features: np.ndarray,
    k_range: range,
    random_state: int,
    n_jobs: int,
) -> int:
    """Auto-select n_motifs via Calinski-Harabasz score on subsample.

    Args:
        features: Dense feature matrix (subsample).
        k_range: Range of k values to test.
        random_state: Random seed.
        n_jobs: Parallelism.

    Returns:
        Best k value.
    """
    best_k = k_range[0]
    best_score = -1.0
    for k in k_range:
        km = MiniBatchKMeans(
            n_clusters=k, random_state=random_state, batch_size=min(10000, features.shape[0]),
        )
        cluster_labels = km.fit_predict(features)
        if len(set(cluster_labels)) < 2:
            continue
        score = calinski_harabasz_score(features, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _is_point_graph(graph) -> bool:
    """Check if graph is a PointSpatialGraph (has cell_ids, not cell_index)."""
    return hasattr(graph, "cell_ids") and not hasattr(graph, "cell_index")


def discover_motifs(
    sp,
    graph,
    group_col: str,
    method: str = "kmeans",
    n_motifs: int | None = None,
    resolution: float = 1.0,
    k_hops: int = 1,
    include_morphology: bool = False,
    include_density: bool = False,
    keep_features: bool = False,
    random_state: int = 42,
    n_jobs: int = 1,
    store: bool = True,
) -> MotifCatalog:
    """Discover local spatial motifs by clustering neighborhood composition.

    Each cell is characterized by the cell-type composition of its
    k-hop neighborhood, then clustered to find recurring local motif
    patterns. Motifs capture microenvironment context, not cell identity.

    Args:
        sp: spatioloji object.
        graph: PolygonSpatialGraph or PointSpatialGraph.
        group_col: Column in ``sp.cell_meta`` with cell type labels.
        method: Clustering method — ``"kmeans"`` (default) or ``"leiden"``.
        n_motifs: Number of motif classes for KMeans. ``None`` = auto-select
            via Calinski-Harabasz score. Ignored for Leiden.
        resolution: Leiden resolution parameter (higher = more motifs).
        k_hops: Neighborhood radius in graph hops.
        include_morphology: Add mean morphology stats of neighbors.
            Requires polygon graph.
        include_density: Add local cell density feature.
            Requires polygon graph.
        keep_features: Retain sparse feature matrix in output.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel threads.
        store: Write ``"motif_label"`` column to ``sp.cell_meta``.

    Returns:
        MotifCatalog with motif labels, signatures, and counts.

    Raises:
        ValueError: If method invalid, group_col not found, or
            morphology/density requested with point graph.
    """
    if method not in ("kmeans", "leiden"):
        raise ValueError(f"method must be 'kmeans' or 'leiden', got '{method}'")
    if group_col not in sp.cell_meta.columns:
        raise ValueError(
            f"'{group_col}' not found in cell_meta. "
            f"Available: {list(sp.cell_meta.columns)}"
        )
    if (include_morphology or include_density) and _is_point_graph(graph):
        raise ValueError(
            "Morphology/density features require a polygon graph. "
            "Use a PolygonSpatialGraph or set include_morphology=False, include_density=False."
        )

    cell_ids = _get_cell_ids(graph)
    n_cells = len(cell_ids)

    # Build feature matrix
    comp_sparse, type_names = _build_composition_features(sp, graph, group_col, k_hops)

    # Optionally add morphology features
    feature_blocks = [comp_sparse]
    if include_morphology:
        morph_cols = [c for c in sp.cell_meta.columns if c.startswith("morph_")]
        if morph_cols:
            morph_vals = sp.cell_meta[morph_cols].values.astype(float)
            # Mean of neighbor morphology
            adj = _get_sparse_adjacency(graph)
            row_sums = np.asarray(adj.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            diag_inv = sp_sparse.diags(1.0 / row_sums)
            neighbor_morph = (diag_inv @ adj) @ morph_vals
            feature_blocks.append(sp_sparse.csr_matrix(neighbor_morph))

    if include_density:
        if "density" in sp.cell_meta.columns:
            dens = sp.cell_meta["density"].values.reshape(-1, 1).astype(float)
        else:
            # Compute simple density: 1 / cell area if polygons available
            areas = sp.cell_meta.get("morph_area", pd.Series(np.ones(n_cells), index=cell_ids))
            dens = (1.0 / np.maximum(areas.values, 1e-6)).reshape(-1, 1)
        feature_blocks.append(sp_sparse.csr_matrix(dens))

    # Stack features
    if len(feature_blocks) > 1:
        features_sparse = sp_sparse.hstack(feature_blocks, format="csr")
    else:
        features_sparse = comp_sparse

    # L2-normalize rows
    features_sparse = normalize(features_sparse, norm="l2", axis=1)

    # Cluster
    if method == "kmeans":
        # Need dense for MiniBatchKMeans
        features_dense = features_sparse.toarray()

        if n_motifs is None:
            # Auto-select via Calinski-Harabasz on subsample
            max_sample = min(50000, n_cells)
            rng = np.random.RandomState(random_state)
            if n_cells > max_sample:
                idx = rng.choice(n_cells, max_sample, replace=False)
                subsample = features_dense[idx]
            else:
                subsample = features_dense
            k_max = min(26, n_cells)
            k_min = min(2, k_max)
            n_motifs = _auto_select_n_motifs(
                subsample, range(k_min, k_max), random_state, n_jobs,
            )

        km = MiniBatchKMeans(
            n_clusters=n_motifs,
            random_state=random_state,
            batch_size=min(10000, n_cells),
            n_init=3,
        )
        cluster_labels = km.fit_predict(features_dense)

    else:  # leiden
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError(
                "Leiden clustering requires leidenalg and igraph. "
                "Install with: pip install spatioloji_s[clustering]"
            ) from None

        # Build KNN graph in feature space
        k_nn = min(15, n_cells - 1)
        nn = NearestNeighbors(n_neighbors=k_nn, algorithm="ball_tree", n_jobs=n_jobs)
        nn.fit(features_sparse)
        knn_graph = nn.kneighbors_graph(mode="connectivity")

        # Symmetrize
        knn_sym = knn_graph + knn_graph.T
        knn_sym.data[:] = 1.0

        # Convert to igraph
        sources, targets = knn_sym.nonzero()
        mask = sources < targets  # upper triangle to avoid duplicates
        edges = list(zip(sources[mask].tolist(), targets[mask].tolist()))
        g = ig.Graph(n=n_cells, edges=edges, directed=False)

        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution,
            seed=random_state,
        )
        cluster_labels = np.array(partition.membership)

    # Build output
    labels_series = pd.Series(cluster_labels, index=cell_ids, name="motif_label")

    # Compute signatures: mean composition per motif
    comp_dense = comp_sparse.toarray()
    comp_df = pd.DataFrame(comp_dense, index=cell_ids, columns=type_names)
    signatures = comp_df.groupby(cluster_labels).mean()
    signatures.index.name = "motif_id"

    counts = labels_series.value_counts().sort_index()
    counts.index.name = "motif_id"

    if store:
        sp.cell_meta["motif_label"] = labels_series

    return MotifCatalog(
        labels=labels_series,
        signatures=signatures,
        counts=counts,
        group_col=group_col,
        feature_matrix=features_sparse if keep_features else None,
        params={
            "method": method,
            "n_motifs": int(labels_series.nunique()),
            "k_hops": k_hops,
            "resolution": resolution,
            "include_morphology": include_morphology,
            "include_density": include_density,
            "random_state": random_state,
        },
    )
```

- [ ] **Step 4: Create point module re-export**

Create `src/spatioloji_s/spatial/point/motifs.py`:

```python
"""Spatial motif discovery for point-based spatial data.

Thin wrapper — re-exports from the polygon module.
Both modes use the same graph adjacency interface.
"""

from spatioloji_s.spatial.polygon.motifs import (
    detect_assemblies,
    discover_motifs,
    match_known_structures,
    run_motif_pipeline,
)

__all__ = ["discover_motifs", "detect_assemblies", "match_known_structures", "run_motif_pipeline"]
```

Note: `detect_assemblies`, `match_known_structures`, and `run_motif_pipeline` do not exist yet — the import will fail until Task 4–5 implement them. To avoid import errors during Task 3, temporarily export only `discover_motifs`:

```python
"""Spatial motif discovery for point-based spatial data.

Thin wrapper — re-exports from the polygon module.
Both modes use the same graph adjacency interface.
"""

from spatioloji_s.spatial.polygon.motifs import discover_motifs

__all__ = ["discover_motifs"]
```

Update to full exports in Task 6 (after all functions exist).

- [ ] **Step 5: Lint and run tests**

Run: `ruff check src/spatioloji_s/spatial/polygon/motifs.py src/spatioloji_s/spatial/point/motifs.py --fix`
Run: `pytest tests/unit/test_motifs.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/motifs.py src/spatioloji_s/spatial/point/motifs.py tests/unit/test_motifs.py
git commit -m "feat: add discover_motifs for local spatial motif discovery"
```

---

## Chunk 3: Mesoscale Assembly Detection (Task 4)

### Task 4: detect_assemblies Implementation

**Files:**
- Modify: `src/spatioloji_s/spatial/polygon/motifs.py`
- Modify: `tests/unit/test_motifs.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_motifs.py`:

```python
class TestDetectAssemblies:
    """Tests for detect_assemblies."""

    @pytest.fixture
    def graph(self, sp_motif):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        return build_buffer_graph(sp_motif, buffer_distance=30)

    @pytest.fixture
    def motif_catalog(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        return discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)

    def test_returns_assembly_catalog(self, sp_motif, graph, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import detect_assemblies
        result = detect_assemblies(sp_motif, graph, motif_catalog)
        assert isinstance(result, AssemblyCatalog)

    def test_labels_cover_all_cells(self, sp_motif, graph, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import detect_assemblies
        result = detect_assemblies(sp_motif, graph, motif_catalog)
        assert len(result.labels) == len(sp_motif.cell_index)

    def test_instances_columns(self, sp_motif, graph, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import detect_assemblies
        result = detect_assemblies(sp_motif, graph, motif_catalog)
        expected = {"instance_id", "assembly_id", "motif_id", "n_cells", "centroid_x", "centroid_y"}
        assert expected.issubset(set(result.instances.columns))

    def test_composition_shape(self, sp_motif, graph, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import detect_assemblies
        result = detect_assemblies(sp_motif, graph, motif_catalog)
        n_assemblies = result.labels[result.labels >= 0].nunique()
        if n_assemblies > 0:
            assert result.composition.shape[0] == n_assemblies

    def test_adjacency_pattern_columns(self, sp_motif, graph, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import detect_assemblies
        result = detect_assemblies(sp_motif, graph, motif_catalog)
        if not result.adjacency_pattern.empty:
            expected = {"assembly_id", "motif_a", "motif_b", "frequency"}
            assert expected.issubset(set(result.adjacency_pattern.columns))

    def test_min_assembly_cells_filter(self, sp_motif, graph, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import detect_assemblies
        result = detect_assemblies(
            sp_motif, graph, motif_catalog, min_assembly_cells=9999,
        )
        # All should be unassigned
        assert (result.labels == -1).all()

    def test_store_writes_to_cell_meta(self, sp_motif, graph, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import detect_assemblies
        detect_assemblies(sp_motif, graph, motif_catalog, store=True)
        assert "assembly_label" in sp_motif.cell_meta.columns

    def test_kmeans_method(self, sp_motif, graph, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import detect_assemblies
        result = detect_assemblies(
            sp_motif, graph, motif_catalog, method="kmeans", n_assemblies=3,
        )
        assert isinstance(result, AssemblyCatalog)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_motifs.py::TestDetectAssemblies::test_returns_assembly_catalog -v`
Expected: FAIL with `ImportError: cannot import name 'detect_assemblies'`

- [ ] **Step 3: Implement detect_assemblies**

Append to `src/spatioloji_s/spatial/polygon/motifs.py`:

```python
def detect_assemblies(
    sp,
    graph,
    motif_catalog: MotifCatalog,
    method: str = "leiden",
    resolution: float = 0.5,
    n_assemblies: int | None = None,
    min_assembly_cells: int = 10,
    random_state: int = 42,
    store: bool = True,
) -> AssemblyCatalog:
    """Detect mesoscale tissue assemblies from spatial motif patterns.

    Groups spatially contiguous cells with the same motif into instances,
    builds a region graph over instances, and clusters to find recurring
    multi-motif tissue structures.

    Args:
        sp: spatioloji object.
        graph: PolygonSpatialGraph or PointSpatialGraph.
        motif_catalog: Output from ``discover_motifs``.
        method: ``"leiden"`` (default) or ``"kmeans"``.
        resolution: Leiden resolution parameter.
        n_assemblies: Number of assemblies for KMeans. Ignored for Leiden.
        min_assembly_cells: Minimum cells for an assembly to be retained.
        random_state: Random seed.
        store: Write ``"assembly_label"`` to ``sp.cell_meta``.

    Returns:
        AssemblyCatalog with assembly labels, composition, instances,
        and adjacency patterns.
    """
    if method not in ("kmeans", "leiden"):
        raise ValueError(f"method must be 'kmeans' or 'leiden', got '{method}'")

    cell_ids = _get_cell_ids(graph)
    adj = _get_sparse_adjacency(graph)
    motif_labels = motif_catalog.labels.reindex(cell_ids).values
    n_cells = len(cell_ids)
    n_motifs = int(motif_labels.max()) + 1

    # --- Step 1: Connected components per motif ---
    instance_ids = np.full(n_cells, -1, dtype=int)
    instance_counter = 0
    instance_motif = []   # motif_id for each instance
    instance_cells = []   # list of cell indices for each instance

    for m in range(n_motifs):
        mask = motif_labels == m
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0]
        sub_adj = adj[np.ix_(indices, indices)]
        n_comp, comp_labels = connected_components(sub_adj, directed=False)
        for c in range(n_comp):
            cells_in_comp = indices[comp_labels == c]
            instance_ids[cells_in_comp] = instance_counter
            instance_motif.append(m)
            instance_cells.append(cells_in_comp)
            instance_counter += 1

    n_instances = instance_counter
    if n_instances == 0:
        empty_labels = pd.Series(-1, index=cell_ids, name="assembly_label")
        if store:
            sp.cell_meta["assembly_label"] = empty_labels
        return AssemblyCatalog(
            labels=empty_labels,
            composition=pd.DataFrame(),
            instances=pd.DataFrame(),
            adjacency_pattern=pd.DataFrame(),
            params={"method": method, "min_assembly_cells": min_assembly_cells},
        )

    instance_motif = np.array(instance_motif)

    # --- Step 2: Compute instance centroids ---
    # Align coordinates to graph cell ordering
    all_x = np.asarray(sp.spatial.x_global)
    all_y = np.asarray(sp.spatial.y_global)
    pos_indices = sp.cell_index.get_indexer(cell_ids)
    x_coords = all_x[pos_indices]
    y_coords = all_y[pos_indices]

    inst_centroid_x = np.zeros(n_instances)
    inst_centroid_y = np.zeros(n_instances)
    inst_n_cells = np.zeros(n_instances, dtype=int)
    for inst_id in range(n_instances):
        cells = instance_cells[inst_id]
        inst_centroid_x[inst_id] = x_coords[cells].mean()
        inst_centroid_y[inst_id] = y_coords[cells].mean()
        inst_n_cells[inst_id] = len(cells)

    # --- Step 3: Build region graph ---
    # Edge between instances if any of their cells are neighbors
    region_rows = []
    region_cols = []
    for inst_a in range(n_instances):
        cells_a = instance_cells[inst_a]
        # Find all neighbors of cells in instance A
        neighbor_indices = set()
        for ci in cells_a:
            neighbor_indices.update(adj[ci].nonzero()[1].tolist())
        # Which instances do those neighbors belong to?
        for ni in neighbor_indices:
            inst_b = instance_ids[ni]
            if inst_b != inst_a and inst_b >= 0:
                region_rows.append(inst_a)
                region_cols.append(inst_b)

    if region_rows:
        region_adj = sp_sparse.csr_matrix(
            (np.ones(len(region_rows)), (region_rows, region_cols)),
            shape=(n_instances, n_instances),
        )
        region_adj = (region_adj + region_adj.T)
        region_adj.data[:] = 1.0
    else:
        region_adj = sp_sparse.csr_matrix((n_instances, n_instances))

    # --- Step 4: Featurize instances ---
    # One-hot motif + log size + neighbor motif composition
    motif_onehot = np.zeros((n_instances, n_motifs))
    for i in range(n_instances):
        motif_onehot[i, instance_motif[i]] = 1.0

    log_size = np.log1p(inst_n_cells).reshape(-1, 1)
    log_size = log_size / (log_size.max() + 1e-8)  # normalize

    # Neighbor motif composition
    neighbor_comp = np.zeros((n_instances, n_motifs))
    for i in range(n_instances):
        neighbors = region_adj[i].nonzero()[1]
        if len(neighbors) > 0:
            for ni in neighbors:
                neighbor_comp[i, instance_motif[ni]] += 1
            neighbor_comp[i] /= neighbor_comp[i].sum()

    region_features = np.hstack([motif_onehot, log_size, neighbor_comp])

    # --- Step 5: Cluster region graph ---
    if method == "kmeans":
        if n_assemblies is None:
            n_assemblies = min(max(2, n_instances // 10), 20)
        n_assemblies = min(n_assemblies, n_instances)
        km = MiniBatchKMeans(
            n_clusters=n_assemblies, random_state=random_state,
            batch_size=min(1000, n_instances), n_init=3,
        )
        assembly_labels = km.fit_predict(region_features)
    else:  # leiden
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError(
                "Leiden clustering requires leidenalg and igraph. "
                "Install with: pip install spatioloji_s[clustering]"
            ) from None

        if region_adj.nnz > 0:
            sources, targets = region_adj.nonzero()
            mask = sources < targets
            edges = list(zip(sources[mask].tolist(), targets[mask].tolist()))
            g = ig.Graph(n=n_instances, edges=edges, directed=False)
            partition = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution, seed=random_state,
            )
            assembly_labels = np.array(partition.membership)
        else:
            assembly_labels = np.arange(n_instances)

    # --- Step 6: Filter small assemblies ---
    assembly_cell_counts = {}
    for inst_id in range(n_instances):
        a_id = assembly_labels[inst_id]
        assembly_cell_counts[a_id] = assembly_cell_counts.get(a_id, 0) + inst_n_cells[inst_id]

    for a_id, count in assembly_cell_counts.items():
        if count < min_assembly_cells:
            assembly_labels[assembly_labels == a_id] = -1

    # Remap to contiguous IDs (keep -1)
    unique_labels = sorted(set(assembly_labels) - {-1})
    remap = {old: new for new, old in enumerate(unique_labels)}
    remap[-1] = -1
    assembly_labels = np.array([remap[a] for a in assembly_labels])

    # --- Step 7: Propagate to cells ---
    cell_assembly = np.full(n_cells, -1, dtype=int)
    for inst_id in range(n_instances):
        cells = instance_cells[inst_id]
        cell_assembly[cells] = assembly_labels[inst_id]

    labels_series = pd.Series(cell_assembly, index=cell_ids, name="assembly_label")

    # --- Step 8: Build instances DataFrame ---
    instances_df = pd.DataFrame({
        "instance_id": np.arange(n_instances),
        "assembly_id": assembly_labels,
        "motif_id": instance_motif,
        "n_cells": inst_n_cells,
        "centroid_x": inst_centroid_x,
        "centroid_y": inst_centroid_y,
    })

    # --- Step 9: Composition (assembly × motif proportions) ---
    valid_assemblies = sorted(set(assembly_labels) - {-1})
    if valid_assemblies:
        comp_rows = []
        for a_id in valid_assemblies:
            inst_mask = assembly_labels == a_id
            total_cells = inst_n_cells[inst_mask].sum()
            motif_fracs = np.zeros(n_motifs)
            for inst_id in np.where(inst_mask)[0]:
                motif_fracs[instance_motif[inst_id]] += inst_n_cells[inst_id]
            motif_fracs /= total_cells
            comp_rows.append(motif_fracs)
        composition = pd.DataFrame(
            comp_rows, index=valid_assemblies,
            columns=[f"motif_{i}" for i in range(n_motifs)],
        )
        composition.index.name = "assembly_id"
    else:
        composition = pd.DataFrame()

    # --- Step 10: Adjacency pattern ---
    adj_rows = []
    for a_id in valid_assemblies:
        inst_mask = assembly_labels == a_id
        inst_indices = np.where(inst_mask)[0]
        pair_counts = {}
        for inst_id in inst_indices:
            m_a = instance_motif[inst_id]
            neighbors = region_adj[inst_id].nonzero()[1]
            for ni in neighbors:
                if inst_mask[ni]:
                    m_b = instance_motif[ni]
                    key = (min(m_a, m_b), max(m_a, m_b))
                    pair_counts[key] = pair_counts.get(key, 0) + 1
        total = sum(pair_counts.values()) or 1
        for (m_a, m_b), count in pair_counts.items():
            adj_rows.append({
                "assembly_id": a_id,
                "motif_a": m_a,
                "motif_b": m_b,
                "frequency": count / total,
            })
    adjacency_pattern = pd.DataFrame(adj_rows)

    if store:
        sp.cell_meta["assembly_label"] = labels_series

    return AssemblyCatalog(
        labels=labels_series,
        composition=composition,
        instances=instances_df,
        adjacency_pattern=adjacency_pattern,
        params={
            "method": method,
            "resolution": resolution,
            "n_assemblies": len(valid_assemblies),
            "min_assembly_cells": min_assembly_cells,
            "random_state": random_state,
        },
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_motifs.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/motifs.py tests/unit/test_motifs.py
git commit -m "feat: add detect_assemblies for mesoscale tissue structure detection"
```

---

## Chunk 4: Known Structure Matching & Pipeline (Tasks 5–6)

### Task 5: match_known_structures Implementation

**Files:**
- Modify: `src/spatioloji_s/spatial/polygon/motifs.py`
- Modify: `tests/unit/test_motifs.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_motifs.py`:

```python
class TestMatchKnownStructures:
    """Tests for match_known_structures."""

    @pytest.fixture
    def graph(self, sp_motif):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        return build_buffer_graph(sp_motif, buffer_distance=30)

    @pytest.fixture
    def motif_catalog(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import discover_motifs
        return discover_motifs(sp_motif, graph, group_col="cell_type", n_motifs=5)

    def test_returns_structure_matches(self, sp_motif, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import match_known_structures
        sigs = {"tumor_core": {"Tumor": 0.6}}
        result = match_known_structures(sp_motif, motif_catalog, signatures=sigs)
        assert isinstance(result, StructureMatches)

    def test_matches_columns(self, sp_motif, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import match_known_structures
        sigs = {"tumor_core": {"Tumor": 0.6}}
        result = match_known_structures(sp_motif, motif_catalog, signatures=sigs)
        expected = {"structure_name", "target_type", "target_id", "similarity", "n_cells", "centroid_x", "centroid_y"}
        assert expected.issubset(set(result.matches.columns))

    def test_per_cell_covers_all(self, sp_motif, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import match_known_structures
        sigs = {"tumor_core": {"Tumor": 0.6}}
        result = match_known_structures(sp_motif, motif_catalog, signatures=sigs)
        assert len(result.per_cell) == len(sp_motif.cell_index)

    def test_no_matches_below_threshold(self, sp_motif, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import match_known_structures
        sigs = {"impossible": {"Nonexistent_type": 0.99}}
        result = match_known_structures(sp_motif, motif_catalog, signatures=sigs, threshold=0.99)
        assert result.matches.empty
        assert (result.per_cell == "unmatched").all()

    def test_absence_filter(self, sp_motif, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import match_known_structures
        # Require Tumor absent — should not match motifs with Tumor
        sigs = {"no_tumor": {"T_cell": 0.3, "Tumor": 0.0}}
        result = match_known_structures(sp_motif, motif_catalog, signatures=sigs)
        # Any match should have low Tumor fraction
        for _, row in result.matches.iterrows():
            if row["target_type"] == "motif":
                motif_sig = motif_catalog.signatures.loc[row["target_id"]]
                if "Tumor" in motif_sig.index:
                    assert motif_sig["Tumor"] <= 0.05

    def test_builtin_tme(self, sp_motif, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import match_known_structures
        result = match_known_structures(sp_motif, motif_catalog, builtin="TME")
        assert isinstance(result, StructureMatches)
        assert "TLS" in result.signatures_used

    def test_invalid_builtin(self, sp_motif, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import match_known_structures
        with pytest.raises(ValueError, match="builtin"):
            match_known_structures(sp_motif, motif_catalog, builtin="nonexistent")

    def test_with_assembly_catalog(self, sp_motif, graph, motif_catalog):
        from spatioloji_s.spatial.polygon.motifs import detect_assemblies, match_known_structures
        assembly_cat = detect_assemblies(sp_motif, graph, motif_catalog)
        sigs = {"tumor_core": {"Tumor": 0.6}}
        result = match_known_structures(
            sp_motif, motif_catalog, assembly_catalog=assembly_cat, signatures=sigs,
        )
        assert isinstance(result, StructureMatches)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_motifs.py::TestMatchKnownStructures::test_returns_structure_matches -v`
Expected: FAIL with `ImportError: cannot import name 'match_known_structures'`

- [ ] **Step 3: Implement match_known_structures**

Append to `src/spatioloji_s/spatial/polygon/motifs.py`:

```python
# Built-in structure signature presets
# Note: signatures use generic cell-type names. Users must ensure their
# group_col values match (e.g., "B_cell" not "B cells"). Unrecognized
# types are silently ignored, so presets work across different datasets.
_BUILTIN_SIGNATURES = {
    "TME": {
        "TLS": {"B_cell": 0.25, "T_cell": 0.15, "DC": 0.05},
        "immune_aggregate": {"CD8_T": 0.2, "CD4_T": 0.15, "Macrophage": 0.15},
        "tumor_bud": {"Tumor": 0.8},
        "perivascular_niche": {"Endothelial": 0.2, "Pericyte": 0.15},
        "immune_desert": {"Tumor": 0.9, "T_cell": 0.0},
    },
}


def match_known_structures(
    sp,
    motif_catalog: MotifCatalog,
    assembly_catalog: AssemblyCatalog | None = None,
    signatures: dict[str, dict[str, float]] | None = None,
    builtin: str | None = None,
    threshold: float = 0.5,
    absence_threshold: float = 0.05,
    coord_type: str = "global",
) -> StructureMatches:
    """Match discovered motifs/assemblies against known structure signatures.

    Args:
        sp: spatioloji object.
        motif_catalog: Output from ``discover_motifs``.
        assembly_catalog: Output from ``detect_assemblies``. Optional.
        signatures: User-defined signatures as ``{name: {cell_type: fraction}}``.
            Zero values mean "must be absent."
        builtin: Load built-in presets. Currently supports ``"TME"``.
        threshold: Minimum cosine similarity for a match.
        absence_threshold: Maximum fraction for "must be absent" types.
        coord_type: ``'global'`` or ``'local'`` for centroid computation.

    Returns:
        StructureMatches with matches, per-cell labels, and signatures used.

    Raises:
        ValueError: If builtin name not recognized.
    """
    # Collect signatures
    all_sigs: dict[str, dict[str, float]] = {}
    if builtin is not None:
        if builtin not in _BUILTIN_SIGNATURES:
            available = list(_BUILTIN_SIGNATURES.keys())
            raise ValueError(f"Unknown builtin '{builtin}'. Available: {available}")
        all_sigs.update(_BUILTIN_SIGNATURES[builtin])
    if signatures is not None:
        all_sigs.update(signatures)

    if not all_sigs:
        return StructureMatches(
            matches=pd.DataFrame(
                columns=["structure_name", "target_type", "target_id",
                         "similarity", "n_cells", "centroid_x", "centroid_y"],
            ),
            per_cell=pd.Series("unmatched", index=sp.cell_index),
            signatures_used=all_sigs,
        )

    motif_sigs = motif_catalog.signatures
    type_cols = list(motif_sigs.columns)

    match_rows = []

    for struct_name, struct_sig in all_sigs.items():
        # Build reference vector aligned to motif signature columns
        ref_vec = np.zeros(len(type_cols))
        absence_types = set()
        for ctype, val in struct_sig.items():
            if ctype not in type_cols:
                warnings.warn(
                    f"Cell type '{ctype}' in signature '{struct_name}' not found in "
                    f"group_col values. Ignoring.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            idx = type_cols.index(ctype)
            if val == 0.0:
                absence_types.add(ctype)
            else:
                ref_vec[idx] = val

        ref_norm = np.linalg.norm(ref_vec)
        if ref_norm == 0:
            continue
        ref_vec = ref_vec / ref_norm

        # Match against motifs
        for motif_id in motif_sigs.index:
            motif_vec = motif_sigs.loc[motif_id].values.astype(float)

            # Absence filter
            skip = False
            for atype in absence_types:
                if atype in type_cols:
                    aidx = type_cols.index(atype)
                    if motif_vec[aidx] > absence_threshold:
                        skip = True
                        break
            if skip:
                continue

            motif_norm = np.linalg.norm(motif_vec)
            if motif_norm == 0:
                continue
            sim = float(np.dot(ref_vec, motif_vec / motif_norm))

            if sim >= threshold:
                # Compute centroid for this motif
                cells_mask = motif_catalog.labels == motif_id
                cell_idx = motif_catalog.labels[cells_mask].index
                pos_idx = sp.cell_index.get_indexer(cell_idx)
                pos_idx = pos_idx[pos_idx >= 0]
                if coord_type == "global":
                    cx = float(np.mean(np.asarray(sp.spatial.x_global)[pos_idx]))
                    cy = float(np.mean(np.asarray(sp.spatial.y_global)[pos_idx]))
                else:
                    cx = float(np.mean(np.asarray(sp.spatial.x_local)[pos_idx]))
                    cy = float(np.mean(np.asarray(sp.spatial.y_local)[pos_idx]))

                match_rows.append({
                    "structure_name": struct_name,
                    "target_type": "motif",
                    "target_id": motif_id,
                    "similarity": sim,
                    "n_cells": int(cells_mask.sum()),
                    "centroid_x": cx,
                    "centroid_y": cy,
                })

        # Match against assemblies if provided
        if assembly_catalog is not None and not assembly_catalog.composition.empty:
            for a_id in assembly_catalog.composition.index:
                # Weighted average of motif signatures
                weights = assembly_catalog.composition.loc[a_id].values
                motif_ids = [int(c.split("_")[1]) for c in assembly_catalog.composition.columns]
                a_vec = np.zeros(len(type_cols))
                for m_id, w in zip(motif_ids, weights, strict=False):
                    if m_id in motif_sigs.index:
                        a_vec += w * motif_sigs.loc[m_id].values.astype(float)

                # Absence filter
                skip = False
                for atype in absence_types:
                    if atype in type_cols:
                        aidx = type_cols.index(atype)
                        if a_vec[aidx] > absence_threshold:
                            skip = True
                            break
                if skip:
                    continue

                a_norm = np.linalg.norm(a_vec)
                if a_norm == 0:
                    continue
                sim = float(np.dot(ref_vec, a_vec / a_norm))

                if sim >= threshold:
                    a_cells = assembly_catalog.labels == a_id
                    cell_idx = assembly_catalog.labels[a_cells].index
                    if coord_type == "global":
                        cx = float(np.mean(sp.spatial.x_global.reindex(cell_idx)))
                        cy = float(np.mean(sp.spatial.y_global.reindex(cell_idx)))
                    else:
                        cx = float(np.mean(sp.spatial.x_local.reindex(cell_idx)))
                        cy = float(np.mean(sp.spatial.y_local.reindex(cell_idx)))

                    match_rows.append({
                        "structure_name": struct_name,
                        "target_type": "assembly",
                        "target_id": a_id,
                        "similarity": sim,
                        "n_cells": int(a_cells.sum()),
                        "centroid_x": cx,
                        "centroid_y": cy,
                    })

    matches_df = pd.DataFrame(match_rows)
    if not matches_df.empty:
        matches_df = matches_df.sort_values("similarity", ascending=False).reset_index(drop=True)

    # Build per-cell labels — best matching structure for each cell
    per_cell = pd.Series("unmatched", index=sp.cell_index)
    if not matches_df.empty:
        # Priority: highest similarity first
        for _, row in matches_df.iterrows():
            if row["target_type"] == "motif":
                cells_mask = motif_catalog.labels == row["target_id"]
            else:
                cells_mask = assembly_catalog.labels == row["target_id"]
            # Only overwrite "unmatched" cells (first match wins = highest similarity)
            unmatched = per_cell == "unmatched"
            per_cell[cells_mask & unmatched] = row["structure_name"]

    return StructureMatches(
        matches=matches_df,
        per_cell=per_cell,
        signatures_used=all_sigs,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_motifs.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/motifs.py tests/unit/test_motifs.py
git commit -m "feat: add match_known_structures with builtin TME presets"
```

---

### Task 6: run_motif_pipeline & Final Re-exports

**Files:**
- Modify: `src/spatioloji_s/spatial/polygon/motifs.py`
- Modify: `src/spatioloji_s/spatial/point/motifs.py`
- Modify: `tests/unit/test_motifs.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_motifs.py`:

```python
class TestRunMotifPipeline:
    """Tests for run_motif_pipeline convenience wrapper."""

    @pytest.fixture
    def graph(self, sp_motif):
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        return build_buffer_graph(sp_motif, buffer_distance=30)

    def test_returns_motif_result(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import run_motif_pipeline
        result = run_motif_pipeline(sp_motif, graph, group_col="cell_type", n_motifs=5)
        assert isinstance(result, MotifResult)
        assert isinstance(result.motif_catalog, MotifCatalog)
        assert isinstance(result.assembly_catalog, AssemblyCatalog)

    def test_skip_assemblies(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import run_motif_pipeline
        result = run_motif_pipeline(
            sp_motif, graph, group_col="cell_type", n_motifs=5,
            detect_assemblies_flag=False,
        )
        assert result.assembly_catalog is None

    def test_with_matching(self, sp_motif, graph):
        from spatioloji_s.spatial.polygon.motifs import run_motif_pipeline
        result = run_motif_pipeline(
            sp_motif, graph, group_col="cell_type", n_motifs=5,
            match_builtin="TME",
        )
        assert result.structure_matches is not None

    def test_full_pipeline_point_graph(self, sp_motif):
        from spatioloji_s.spatial.point.graph import build_knn_graph
        from spatioloji_s.spatial.point.motifs import run_motif_pipeline
        pg = build_knn_graph(sp_motif, k=10)
        result = run_motif_pipeline(sp_motif, pg, group_col="cell_type", n_motifs=5)
        assert isinstance(result, MotifResult)


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, sp_motif):
        """Full: graph → motifs → assemblies → match → verify."""
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.motifs import run_motif_pipeline

        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        result = run_motif_pipeline(
            sp_motif, graph, group_col="cell_type", n_motifs=5,
            match_signatures={"tumor_core": {"Tumor": 0.6}},
        )
        assert isinstance(result, MotifResult)
        assert result.motif_catalog.labels.nunique() == 5
        assert "motif_label" in sp_motif.cell_meta.columns
        assert "assembly_label" in sp_motif.cell_meta.columns
```

- [ ] **Step 2: Implement run_motif_pipeline**

Append to `src/spatioloji_s/spatial/polygon/motifs.py`:

```python
def run_motif_pipeline(
    sp,
    graph,
    group_col: str,
    method: str = "kmeans",
    n_motifs: int | None = None,
    resolution: float = 1.0,
    k_hops: int = 1,
    include_morphology: bool = False,
    include_density: bool = False,
    detect_assemblies_flag: bool = True,
    assembly_method: str = "leiden",
    assembly_resolution: float = 0.5,
    n_assemblies: int | None = None,
    min_assembly_cells: int = 10,
    match_signatures: dict | None = None,
    match_builtin: str | None = None,
    match_threshold: float = 0.5,
    coord_type: str = "global",
    random_state: int = 42,
    n_jobs: int = 1,
    store: bool = True,
) -> MotifResult:
    """Run the full motif discovery pipeline.

    Convenience wrapper that calls ``discover_motifs``,
    ``detect_assemblies``, and ``match_known_structures`` in sequence.

    Args:
        sp: spatioloji object.
        graph: PolygonSpatialGraph or PointSpatialGraph.
        group_col: Cell-type column in ``sp.cell_meta``.
        method: Clustering method for motif discovery.
        n_motifs: Number of motifs (KMeans). None = auto.
        resolution: Leiden resolution for motif discovery.
        k_hops: Neighborhood radius.
        include_morphology: Include morphology features (polygon only).
        include_density: Include density features (polygon only).
        detect_assemblies_flag: Whether to run assembly detection.
        assembly_method: Clustering method for assemblies.
        assembly_resolution: Leiden resolution for assemblies.
        n_assemblies: Number of assemblies (KMeans).
        min_assembly_cells: Minimum cells per assembly.
        match_signatures: User-defined structure signatures.
        match_builtin: Built-in signature presets.
        match_threshold: Minimum similarity for matching.
        random_state: Random seed.
        n_jobs: Parallelism.
        store: Write labels to ``sp.cell_meta``.

    Returns:
        MotifResult with motif catalog, assembly catalog, and matches.
    """
    motif_cat = discover_motifs(
        sp, graph, group_col,
        method=method, n_motifs=n_motifs, resolution=resolution,
        k_hops=k_hops, include_morphology=include_morphology,
        include_density=include_density, random_state=random_state,
        n_jobs=n_jobs, store=store,
    )

    assembly_cat = None
    if detect_assemblies_flag:
        assembly_cat = detect_assemblies(
            sp, graph, motif_cat,
            method=assembly_method, resolution=assembly_resolution,
            n_assemblies=n_assemblies, min_assembly_cells=min_assembly_cells,
            random_state=random_state, store=store,
        )

    matches = None
    if match_signatures or match_builtin:
        matches = match_known_structures(
            sp, motif_cat,
            assembly_catalog=assembly_cat,
            signatures=match_signatures,
            builtin=match_builtin,
            threshold=match_threshold,
            coord_type=coord_type,
        )

    return MotifResult(
        motif_catalog=motif_cat,
        assembly_catalog=assembly_cat,
        structure_matches=matches,
        params={
            "method": method,
            "n_motifs": motif_cat.params.get("n_motifs"),
            "detect_assemblies": detect_assemblies_flag,
            "match_builtin": match_builtin,
        },
    )
```

- [ ] **Step 3: Update point re-export**

Replace `src/spatioloji_s/spatial/point/motifs.py` with the full exports:

```python
"""Spatial motif discovery for point-based spatial data.

Thin wrapper — re-exports from the polygon module.
Both modes use the same graph adjacency interface.
"""

from spatioloji_s.spatial.polygon.motifs import (
    detect_assemblies,
    discover_motifs,
    match_known_structures,
    run_motif_pipeline,
)

__all__ = ["discover_motifs", "detect_assemblies", "match_known_structures", "run_motif_pipeline"]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_motifs.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/motifs.py src/spatioloji_s/spatial/point/motifs.py tests/unit/test_motifs.py
git commit -m "feat: add run_motif_pipeline and update point re-exports"
```

---

## Chunk 5: Visualization & Module Wiring (Tasks 7–8)

### Task 7: Visualization Functions

**Files:**
- Modify: `src/spatioloji_s/visualization/polygon_plots.py`
- Modify: `src/spatioloji_s/visualization/point_plots.py`
- Modify: `tests/unit/test_motifs.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_motifs.py`:

```python
class TestPlotMotifMap:
    """Tests for plot_motif_map."""

    def test_returns_figure(self, sp_motif):
        import matplotlib
        import matplotlib.pyplot as plt
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.motifs import run_motif_pipeline
        from spatioloji_s.visualization.polygon_plots import plot_motif_map

        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        result = run_motif_pipeline(sp_motif, graph, group_col="cell_type", n_motifs=5)
        fig = plot_motif_map(sp_motif, result, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestPlotMotifComposition:
    """Tests for plot_motif_composition."""

    def test_returns_figure(self, sp_motif):
        import matplotlib
        import matplotlib.pyplot as plt
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.motifs import run_motif_pipeline
        from spatioloji_s.visualization.polygon_plots import plot_motif_composition

        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        result = run_motif_pipeline(sp_motif, graph, group_col="cell_type", n_motifs=5)
        fig = plot_motif_composition(result, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestPlotAssemblyMap:
    """Tests for plot_assembly_map."""

    def test_returns_figure(self, sp_motif):
        import matplotlib
        import matplotlib.pyplot as plt
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.motifs import run_motif_pipeline
        from spatioloji_s.visualization.polygon_plots import plot_assembly_map

        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        result = run_motif_pipeline(sp_motif, graph, group_col="cell_type", n_motifs=5)
        fig = plot_assembly_map(sp_motif, result, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestPlotStructureMatches:
    """Tests for plot_structure_matches."""

    def test_returns_figure(self, sp_motif):
        import matplotlib
        import matplotlib.pyplot as plt
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph
        from spatioloji_s.spatial.polygon.motifs import run_motif_pipeline
        from spatioloji_s.visualization.polygon_plots import plot_structure_matches

        graph = build_buffer_graph(sp_motif, buffer_distance=30)
        result = run_motif_pipeline(
            sp_motif, graph, group_col="cell_type", n_motifs=5,
            match_signatures={"tumor_core": {"Tumor": 0.6}},
        )
        fig = plot_structure_matches(sp_motif, result, "tumor_core", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
```

- [ ] **Step 2: Implement plot functions**

Append to `src/spatioloji_s/visualization/polygon_plots.py` (after the infiltration plots section):

```python
# ══════════════════════════════════════════════════════════════════════════════
# Spatial motif plots
# ══════════════════════════════════════════════════════════════════════════════


def plot_motif_map(
    sp,
    motif_result,
    coord_type: str = "global",
    point_size: float = 8,
    palette: str | None = "tab20",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Spatial map colored by motif label.

    Args:
        sp: spatioloji object.
        motif_result: MotifResult from ``run_motif_pipeline``.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        point_size: Scatter point size.
        palette: Categorical colormap name.
        figsize: Figure size.
        show: Whether to display the figure.
        save_path: Save figure to path if provided.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure.
    """
    import numpy as np

    if figsize is None:
        figsize = (10, 8)

    fig, ax = plt.subplots(figsize=figsize)
    labels = motif_result.motif_catalog.labels.reindex(sp.cell_index)

    if coord_type == "global":
        x = np.asarray(sp.spatial.x_global)
        y = np.asarray(sp.spatial.y_global)
    else:
        x = np.asarray(sp.spatial.x_local)
        y = np.asarray(sp.spatial.y_local)

    colors, _, handles = categorical_colors(labels.astype(str), palette)
    ax.scatter(x, y, s=point_size, c=colors, edgecolors="none")
    ax.legend(
        handles=handles, title="Motif", bbox_to_anchor=(1.01, 1),
        loc="upper left", fontsize=7, title_fontsize=8, frameon=False,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Spatial Motif Map")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


def plot_motif_composition(
    motif_result,
    figsize: tuple[float, float] | None = None,
    palette: str | None = "tab20",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Stacked bar chart of cell-type composition per motif.

    Args:
        motif_result: MotifResult from ``run_motif_pipeline``.
        figsize: Figure size.
        palette: Colormap for cell types.
        show: Whether to display the figure.
        save_path: Save figure to path if provided.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure.
    """
    import seaborn as sns

    sigs = motif_result.motif_catalog.signatures
    n_motifs = sigs.shape[0]

    if figsize is None:
        figsize = (8, max(3, n_motifs * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    pal = sns.color_palette(palette or "tab20", sigs.shape[1])

    y_pos = range(n_motifs)
    left = np.zeros(n_motifs)
    for j, col in enumerate(sigs.columns):
        vals = sigs[col].values
        ax.barh(y_pos, vals, left=left, color=pal[j % len(pal)], label=col, height=0.6)
        left += vals

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"Motif {i}" for i in sigs.index], fontsize=8)
    ax.set_xlabel("Cell-type fraction")
    ax.set_title("Motif Composition")
    ax.legend(
        title="Cell type", bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=7, title_fontsize=8, frameon=False,
    )
    ax.set_xlim(0, 1)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


def plot_assembly_map(
    sp,
    motif_result,
    coord_type: str = "global",
    point_size: float = 8,
    palette: str | None = "Set2",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Spatial map colored by assembly label.

    Args:
        sp: spatioloji object.
        motif_result: MotifResult with assembly_catalog.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        point_size: Scatter point size.
        palette: Categorical colormap name.
        figsize: Figure size.
        show: Whether to display the figure.
        save_path: Save figure to path if provided.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If ``motif_result.assembly_catalog`` is None.
    """
    import numpy as np

    if motif_result.assembly_catalog is None:
        raise ValueError("No assembly catalog — run detect_assemblies first.")

    if figsize is None:
        figsize = (10, 8)

    fig, ax = plt.subplots(figsize=figsize)
    labels = motif_result.assembly_catalog.labels.reindex(sp.cell_index)

    if coord_type == "global":
        x = np.asarray(sp.spatial.x_global)
        y = np.asarray(sp.spatial.y_global)
    else:
        x = np.asarray(sp.spatial.x_local)
        y = np.asarray(sp.spatial.y_local)

    # Unassigned in grey
    display_labels = labels.astype(str).replace("-1", "unassigned")
    colors, _, handles = categorical_colors(display_labels, palette)
    ax.scatter(x, y, s=point_size, c=colors, edgecolors="none")
    ax.legend(
        handles=handles, title="Assembly", bbox_to_anchor=(1.01, 1),
        loc="upper left", fontsize=7, title_fontsize=8, frameon=False,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Tissue Assembly Map")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


def plot_structure_matches(
    sp,
    motif_result,
    structure_name: str,
    coord_type: str = "global",
    point_size: float = 8,
    highlight_color: str = "#d62728",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Highlight cells matching a known structure on the spatial map.

    Args:
        sp: spatioloji object.
        motif_result: MotifResult with structure_matches.
        structure_name: Name of the structure to highlight.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        point_size: Scatter point size.
        highlight_color: Color for matched cells.
        figsize: Figure size.
        show: Whether to display the figure.
        save_path: Save figure to path if provided.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If ``motif_result.structure_matches`` is None.
    """
    import numpy as np
    from matplotlib.patches import Patch

    if motif_result.structure_matches is None:
        raise ValueError("No structure matches — run match_known_structures first.")

    if figsize is None:
        figsize = (10, 8)

    fig, ax = plt.subplots(figsize=figsize)

    if coord_type == "global":
        x = np.asarray(sp.spatial.x_global)
        y = np.asarray(sp.spatial.y_global)
    else:
        x = np.asarray(sp.spatial.x_local)
        y = np.asarray(sp.spatial.y_local)

    per_cell = motif_result.structure_matches.per_cell.reindex(sp.cell_index)
    matched = per_cell == structure_name
    n_matched = matched.sum()

    # Background cells
    ax.scatter(x[~matched], y[~matched], s=point_size * 0.5, c="#dddddd", edgecolors="none", alpha=0.5)
    # Matched cells
    ax.scatter(x[matched], y[matched], s=point_size, c=highlight_color, edgecolors="none")

    handles = [
        Patch(facecolor=highlight_color, label=f"{structure_name} ({n_matched})"),
        Patch(facecolor="#dddddd", label=f"other ({(~matched).sum()})"),
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, frameon=False)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Structure: {structure_name}")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)
```

Also add to `__all__` in `polygon_plots.py`:

```python
    "plot_motif_map",
    "plot_motif_composition",
    "plot_assembly_map",
    "plot_structure_matches",
```

And append re-exports to `src/spatioloji_s/visualization/point_plots.py` (after the existing gradient/infiltration imports):

```python
from spatioloji_s.visualization.polygon_plots import (  # noqa: E402
    plot_assembly_map,
    plot_motif_composition,
    plot_motif_map,
    plot_structure_matches,
)
```

Also add to `__all__` in `point_plots.py`:

```python
    "plot_motif_map",
    "plot_motif_composition",
    "plot_assembly_map",
    "plot_structure_matches",
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_motifs.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/spatioloji_s/visualization/polygon_plots.py src/spatioloji_s/visualization/point_plots.py tests/unit/test_motifs.py
git commit -m "feat(visualization): add motif map, composition, assembly, and structure match plots"
```

---

### Task 8: Module Exports

**Files:**
- Modify: `src/spatioloji_s/spatial/polygon/__init__.py`
- Modify: `src/spatioloji_s/spatial/point/__init__.py`
- Modify: `src/spatioloji_s/visualization/__init__.py`

- [ ] **Step 1: Update polygon/__init__.py**

Add imports after the Infiltration section:

```python
# Motifs
from .motifs import detect_assemblies, discover_motifs, match_known_structures, run_motif_pipeline
from .._motif_types import AssemblyCatalog, MotifCatalog, MotifResult, StructureMatches
```

Add to `__all__`:

```python
    # Motifs
    "discover_motifs",
    "detect_assemblies",
    "match_known_structures",
    "run_motif_pipeline",
    "MotifCatalog",
    "AssemblyCatalog",
    "MotifResult",
    "StructureMatches",
```

- [ ] **Step 2: Update point/__init__.py**

Same imports from point module:

```python
# Motifs
from .motifs import detect_assemblies, discover_motifs, match_known_structures, run_motif_pipeline
from .._motif_types import AssemblyCatalog, MotifCatalog, MotifResult, StructureMatches
```

Same `__all__` entries.

- [ ] **Step 3: Update visualization/__init__.py**

Add to polygon_plots imports:

```python
    plot_assembly_map,
    plot_motif_composition,
    plot_motif_map,
    plot_structure_matches,
```

Add to `__all__`:

```python
    "plot_motif_map",
    "plot_motif_composition",
    "plot_assembly_map",
    "plot_structure_matches",
```

- [ ] **Step 4: Lint and run full test suite**

Run: `ruff check src/ tests/ --fix`
Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/spatioloji_s/spatial/polygon/__init__.py src/spatioloji_s/spatial/point/__init__.py src/spatioloji_s/visualization/__init__.py
git commit -m "feat: wire motif modules into package exports"
```
