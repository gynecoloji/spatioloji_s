# Hierarchical Spatial Motif Discovery & Tissue Structure Detection — Design Spec

**Date:** 2026-03-14
**Status:** Approved
**Module:** `spatial/polygon/motifs.py` (with thin point re-export)

---

## 1. Goal

Add a multi-scale spatial motif discovery framework that identifies:
1. **Local motifs** — recurring cellular neighborhood patterns (3-10 cells)
2. **Mesoscale assemblies** — higher-order tissue structures composed of motifs (10-100+ cells)
3. **Named structure matches** — map discovered patterns to known biological structures (TLS, tumor buds, immune aggregates, etc.)

**Target users:** Tumor microenvironment researchers and clinical/translational scientists who need to move beyond pairwise spatial statistics to detect and quantify complex tissue architecture.

**Differentiator:** No existing Python spatial-omics package (Squidpy, BANKSY, CellCharter) offers integrated multi-scale motif discovery with a built-in known-structure query layer.

---

## 2. Architecture

### 2.1 Pipeline Overview

```
Stage 1: discover_motifs()
    cells → neighborhood feature vectors → cluster → MotifCatalog

Stage 2: detect_assemblies()
    motif instances → region graph → cluster → AssemblyCatalog

Query: match_known_structures()
    motif/assembly signatures × known structure signatures → StructureMatches

Convenience: run_motif_pipeline()
    runs all three stages with sensible defaults
```

### 2.2 File Layout

```
src/spatioloji_s/spatial/
├── _motif_types.py              # MotifCatalog, AssemblyCatalog, StructureMatches, MotifResult
├── polygon/motifs.py            # discover_motifs, detect_assemblies, match_known_structures, run_motif_pipeline
└── point/motifs.py              # thin re-export

src/spatioloji_s/visualization/
├── polygon_plots.py             # append: plot_motif_map, plot_motif_composition, plot_assembly_map, plot_structure_matches
└── point_plots.py               # re-export delegates

tests/unit/
└── test_motifs.py
```

### 2.3 Dependencies

**Required (already in package):** numpy, pandas, scipy, scikit-learn, shapely

**Optional (already optional dep):** leidenalg, igraph — only if user selects `method="leiden"`

**No new dependencies.**

### 2.4 Graph Type Handling

Both `PolygonSpatialGraph` and `PointSpatialGraph` are accepted. They differ in cell ID accessors (`.cell_index` vs `.cell_ids`) and adjacency storage. All functions use a private helper to normalize access:

```python
def _get_cell_ids(graph) -> np.ndarray:
    """Return cell IDs from either graph type."""
    if hasattr(graph, "cell_index"):
        return np.asarray(graph.cell_index)
    return np.asarray(graph.cell_ids)

def _get_sparse_adjacency(graph) -> scipy.sparse.csr_matrix:
    """Return sparse adjacency from either graph type."""
    # Both graph types store adjacency as sparse or can be converted
    return graph.adjacency  # or build from graph.neighbors dict
```

The `graph` parameter is typed as `PolygonSpatialGraph | PointSpatialGraph` in all public signatures. The point re-export is a plain `from ... import` (same function, no wrapping needed) because the implementation handles both graph types internally.

---

## 3. Stage 1 — Local Motif Discovery

### 3.1 Function Signature

```python
def discover_motifs(
    sp,
    graph: PolygonSpatialGraph | PointSpatialGraph,
    group_col: str,
    method: str = "kmeans",           # "kmeans" or "leiden"
    n_motifs: int | None = None,      # required for kmeans, ignored for leiden
    resolution: float = 1.0,          # leiden only
    k_hops: int = 1,                  # neighborhood radius
    include_morphology: bool = False, # add mean morphology of neighbors (polygon graphs only)
    include_density: bool = False,    # add local cell density (polygon graphs only)
    keep_features: bool = False,      # retain feature_matrix in output
    random_state: int = 42,           # reproducibility
    n_jobs: int = 1,                  # parallelism for KMeans/KNN
    store: bool = True,               # write "motif_label" to sp.cell_meta
) -> MotifCatalog:
```

### 3.2 Feature Vector Construction

For each cell, build a feature vector from its k-hop neighborhood:

| Feature block | Dimensions | Source | Availability |
|---------------|-----------|--------|-------------|
| Cell-type composition | n_types | Proportion of each cell type in k-hop neighborhood | All graphs |
| Morphology stats (optional) | 4 | Mean area, perimeter, circularity, contour_entropy of neighbors | Polygon graphs only |
| Local density (optional) | 1 | From `cell_density_map` (uses polygon area) | Polygon graphs only |

**Polygon-only features:** If `include_morphology=True` or `include_density=True` is passed with a `PointSpatialGraph`, raise `ValueError` with message: "Morphology/density features require a polygon graph. Use a PolygonSpatialGraph or set include_morphology=False."

**Implementation:**
- Build sparse neighborhood composition matrix (CSR) directly from graph adjacency — avoid dense n×n_types matrix for memory
- For k_hops > 1: sparse matrix power `A^k` (capped at nonzero) to expand neighborhoods
- L2-normalize rows before clustering

### 3.3 Clustering

**KMeans (default, scalable):**
- `MiniBatchKMeans` from scikit-learn — O(n × n_iter × batch_size)
- `n_motifs` required; if None, auto-select via Calinski-Harabasz score on subsample (max 50k cells, testing k=5..25). Calinski-Harabasz is O(n) unlike silhouette O(n²), making it practical for large subsamples.

**Leiden (opt-in, more flexible):**
- Build approximate KNN graph (k=15) over feature vectors using `sklearn.neighbors.NearestNeighbors(algorithm='ball_tree')`
- Convert to igraph and run `leidenalg.find_partition` directly (not via scanpy — consistent with existing `identify_niches` in point module)
- Guard with `try: import leidenalg, igraph except ImportError: raise ImportError("Install with: pip install spatioloji_s[clustering]")`
- `resolution` controls granularity

### 3.4 Output — MotifCatalog

```python
@dataclass
class MotifCatalog:
    labels: pd.Series              # cell_id → motif_id (int)
    signatures: pd.DataFrame       # motif_id × cell_type — mean composition
    counts: pd.Series              # motif_id → n_cells
    group_col: str                 # the cell-type column used (needed for matching)
    feature_matrix: scipy.sparse.csr_matrix | None  # retained only if keep_features=True
    params: dict                   # parameters used (method, k_hops, n_motifs, etc.)
```

### 3.5 Key Design Decision

Motifs are defined by **neighborhood composition**, not by the cell's own type. A tumor cell in an immune-rich neighborhood and a tumor cell in a tumor-only neighborhood belong to different motifs. This captures the microenvironment context.

---

## 4. Stage 2 — Mesoscale Assembly Detection

### 4.1 Function Signature

```python
def detect_assemblies(
    sp,
    graph: PolygonSpatialGraph | PointSpatialGraph,
    motif_catalog: MotifCatalog,
    method: str = "leiden",           # "leiden" or "kmeans"
    resolution: float = 0.5,          # leiden only
    n_assemblies: int | None = None,  # kmeans only
    min_assembly_cells: int = 10,     # filter small regions
    random_state: int = 42,
    store: bool = True,               # write "assembly_label" to sp.cell_meta
) -> AssemblyCatalog:
```

### 4.2 Algorithm

1. **Region building:** `scipy.sparse.csgraph.connected_components` on the subgraph of cells with the same motif label. Each connected component = one motif instance. O(n) with sparse adjacency.

2. **Region graph construction:** Nodes = motif instances. Edge between two instances if any of their cells are neighbors in the original graph. Each node featurized by:
   - One-hot motif identity
   - Size (log cell count, normalized)
   - Composition of adjacent instances (what fraction of neighbors are each motif type)

3. **Assembly clustering:** Leiden (default) or KMeans on the region graph. The region graph is much smaller than the cell graph (typically 100-10,000× fewer nodes), so this is fast.

4. **Propagation:** Each cell inherits the assembly label of its motif instance.

5. **Filtering:** Assemblies with fewer than `min_assembly_cells` total cells across all their instances are labeled "unassigned" (assembly_id = -1).

### 4.3 Output — AssemblyCatalog

```python
@dataclass
class AssemblyCatalog:
    labels: pd.Series              # cell_id → assembly_id (int, -1 = unassigned)
    composition: pd.DataFrame      # assembly_id × motif_id — mean motif proportions
    instances: pd.DataFrame        # see column spec below
    adjacency_pattern: pd.DataFrame  # see column spec below
    params: dict
```

**`instances` DataFrame columns:**

| Column | Type | Description |
|--------|------|-------------|
| `instance_id` | int | Unique ID for this motif instance |
| `assembly_id` | int | Assembly it belongs to (-1 = unassigned) |
| `motif_id` | int | Which motif class this instance is |
| `n_cells` | int | Number of cells in this instance |
| `centroid_x` | float | Mean x-coordinate of member cells |
| `centroid_y` | float | Mean y-coordinate of member cells |

Note: No `area` column — centroids are computed as mean of cell coordinates (works for both point and polygon graphs). Area can be derived from polygon geometries downstream if needed.

**`adjacency_pattern` DataFrame columns:**

A single long-form DataFrame (not a dict of DataFrames) with columns:

| Column | Type | Description |
|--------|------|-------------|
| `assembly_id` | int | Assembly type |
| `motif_a` | int | Source motif type |
| `motif_b` | int | Adjacent motif type |
| `frequency` | float | Normalized frequency of this motif-pair adjacency within this assembly type |

---

## 5. Query Layer — Known Structure Matching

### 5.1 Function Signature

```python
def match_known_structures(
    sp,
    motif_catalog: MotifCatalog,
    assembly_catalog: AssemblyCatalog | None = None,
    signatures: dict[str, dict[str, float]] | None = None,
    builtin: str | None = None,       # "TME" for tumor microenvironment presets
    threshold: float = 0.5,           # minimum cosine similarity
    absence_threshold: float = 0.05,  # max fraction for "must be absent" types
    coord_type: str = "global",
) -> StructureMatches:
```

The `sp` parameter is needed to compute centroids for the `matches` DataFrame output and to access `cell_index` for the `per_cell` Series.

### 5.2 Signature Format

User-defined:
```python
signatures = {
    "TLS": {"B_cell": 0.25, "T_cell": 0.15, "DC": 0.05},
    "immune_exclusion": {"Tumor": 0.8, "T_cell": 0.0},
}
```

- Positive values = minimum expected fraction (used to build the reference vector for cosine similarity)
- Zero = must be absent (hard filter: reject if that type exceeds `absence_threshold` in the motif)

Built-in `"TME"` presets (initial curated set):
- `TLS` — B_cell + T_cell + DC rich
- `immune_aggregate` — CD8_T + CD4_T + Macrophage cluster
- `tumor_bud` — small isolated Tumor cluster (<20 cells)
- `perivascular_niche` — Endothelial + Pericyte + immune mix
- `immune_desert` — Tumor >90%, immune <5%

### 5.3 Matching Logic

1. For each motif signature (row of `motif_catalog.signatures`), compute cosine similarity against each query signature
2. Apply hard "must be absent" filter using `absence_threshold`
3. If `assembly_catalog` provided, also match assembly compositions (weighted average of constituent motif signatures, weighted by motif instance cell counts)
4. Compute centroids: for each matched motif/assembly, mean of cell coordinates (retrieved from `sp.spatial`)
5. Return matches above `threshold`, ranked by similarity

### 5.4 Output — StructureMatches

```python
@dataclass
class StructureMatches:
    matches: pd.DataFrame          # structure_name, target_type ("motif"/"assembly"), target_id, similarity, n_cells, centroid_x, centroid_y
    per_cell: pd.Series            # cell_id → matched structure name or "unmatched"
    signatures_used: dict          # the signatures that were queried
```

---

## 6. Convenience Wrapper

```python
def run_motif_pipeline(
    sp,
    graph: PolygonSpatialGraph | PointSpatialGraph,
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
    random_state: int = 42,
    n_jobs: int = 1,
    store: bool = True,
) -> MotifResult:
```

### 6.1 Top-level Result

```python
@dataclass
class MotifResult:
    motif_catalog: MotifCatalog
    assembly_catalog: AssemblyCatalog | None
    structure_matches: StructureMatches | None
    params: dict
```

---

## 7. Scalability

### 7.1 Design Targets

| Dataset size | Pipeline time | Memory overhead |
|-------------|--------------|-----------------|
| 1M cells | <2 minutes | <4 GB |
| 5M cells | <10 minutes | <16 GB |

### 7.2 Scalability Strategies

| Component | Strategy |
|-----------|----------|
| Feature vector construction | Sparse CSR matrix for neighborhood composition; chunked processing (100k cells/chunk) for >500k cells |
| KMeans clustering | `MiniBatchKMeans` (batch_size=10000) — O(n) per iteration |
| KMeans auto-selection | Calinski-Harabasz score (O(n)) on subsample, testing k=5..25 |
| Leiden clustering (opt-in) | Approximate KNN graph via `NearestNeighbors(algorithm='ball_tree')` — O(n·k·log n), then `leidenalg` directly (not scanpy) |
| Connected components | `scipy.sparse.csgraph.connected_components` on sparse adjacency — O(n) |
| Region graph | Depends on motif instance count (100-10,000× fewer than cells) — trivially fast |
| Region featurization | Vectorized pandas groupby, no Python per-instance loops |
| Cosine similarity | Vectorized scipy/sklearn — O(n_motifs × n_signatures) |
| Parallelism | `n_jobs` parameter on MiniBatchKMeans and NearestNeighbors |

### 7.3 Memory Management

- Neighborhood composition stored as `scipy.sparse.csr_matrix`, not dense DataFrame
- Feature matrix discarded after clustering by default (`keep_features=False` in `discover_motifs`)
- Graph adjacency reused from existing spatial graph (no duplication)

---

## 8. Visualization

Four plot functions appended to `polygon_plots.py`, following existing conventions (`show`, `save_path`, `dpi`, `figsize`, return `plt.Figure`). All accept the top-level `MotifResult` for convenience, extracting the needed sub-catalog internally.

### 8.1 `plot_motif_map(sp, motif_result: MotifResult, coord_type, ...)`
Spatial scatter/polygon map colored by motif label. Categorical colormap.

### 8.2 `plot_motif_composition(motif_result: MotifResult, ...)`
Stacked horizontal bar chart: x-axis = cell-type fraction, y-axis = motif classes. Shows what each motif "is made of."

### 8.3 `plot_assembly_map(sp, motif_result: MotifResult, coord_type, show_motif_boundaries, ...)`
Spatial map colored by assembly label. Optional thin outlines showing motif boundaries within each assembly. Uses polygon fills if available, scatter otherwise. Requires `motif_result.assembly_catalog` to be non-None.

### 8.4 `plot_structure_matches(sp, motif_result: MotifResult, structure_name, coord_type, ...)`
Highlights cells matching a specific known structure on the spatial map. Matched cells in color, all others in light grey. Annotates each matched region with its similarity score. Requires `motif_result.structure_matches` to be non-None.

---

## 9. Error Handling

| Condition | Behavior |
|-----------|----------|
| `n_motifs=None` with `method="kmeans"` | Auto-select via Calinski-Harabasz on subsample (max 50k cells, k=5..25) |
| `include_morphology=True` with PointSpatialGraph | `ValueError`: "Morphology features require a polygon graph" |
| `include_density=True` with PointSpatialGraph | `ValueError`: "Density features require a polygon graph" |
| No contiguous regions found | `detect_assemblies` returns AssemblyCatalog with all cells "unassigned" (-1) |
| `builtin` name not recognized | `ValueError` with list of available presets |
| Cell type in signature not in `group_col` values | `UserWarning` — that type is ignored in matching |
| `method="leiden"` without leidenalg installed | `ImportError` with install instructions |
| All motifs below match threshold | `StructureMatches.matches` is empty DataFrame, `per_cell` all "unmatched" |
| `assembly_catalog` is None but `plot_assembly_map` called | `ValueError`: "No assembly catalog — run detect_assemblies first" |

---

## 10. Testing Strategy

### 10.1 Test Fixture — `sp_motif`

Create a synthetic spatioloji object with **5 cell types** (Tumor, T_cell, B_cell, Macrophage, Fibroblast) and **500 cells** arranged in a structured layout:

- **Center cluster (100 cells):** Dense Tumor core (>80% Tumor, some Fibroblast)
- **Inner ring (100 cells):** Mixed Macrophage + Fibroblast stroma surrounding tumor
- **Left lobe (100 cells):** T_cell + B_cell aggregate (resembling TLS-like structure)
- **Right lobe (100 cells):** Scattered T_cell + Macrophage (immune infiltrate)
- **Periphery (100 cells):** Sparse Fibroblast + Tumor mix

This layout should produce at least 3 distinct motifs and 2 recognizable assemblies.

### 10.2 Test Coverage

- **Unit tests** for each function: `discover_motifs`, `detect_assemblies`, `match_known_structures`, `run_motif_pipeline`
- **Dataclass validation:** verify all output fields have correct types, shapes, and index alignment
- **Edge cases:** single motif (all cells same neighborhood), all cells same type, empty graph, min_assembly_cells larger than all regions
- **Integration test:** full `run_motif_pipeline` end-to-end with `sp_motif` fixture
- **Re-export test:** verify `point.motifs.discover_motifs is polygon.motifs.discover_motifs`
- **Visualization tests:** each plot function returns `plt.Figure`, no crash on empty results
- **Scalability test (manual, not CI):** synthetic 100k cell dataset, verify <30s runtime
