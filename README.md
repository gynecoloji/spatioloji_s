# spatioloji_s

[![PyPI version](https://img.shields.io/pypi/v/spatioloji-s.svg)](https://pypi.org/project/spatioloji-s/)
[![Documentation Status](https://readthedocs.org/projects/spatioloji_s/badge/?version=latest)](https://spatioloji_s.readthedocs.io/en/latest/?version=latest)

**spatioloji_s** is a Python package for spatial transcriptomics analysis, purpose-built for image-based single-cell RNA sequencing data (CosMx, MERFISH, Xenium). It provides an integrated workflow — from raw data loading through quality control, processing, spatial analysis, and polygon-native cell-cell communication — all within a consistent, polygon-aware data structure.

* PyPI: https://pypi.org/project/spatioloji_s/
* GitHub: https://github.com/gynecoloji/spatioloji_s
* Documentation: https://spatioloji_s.readthedocs.io
* License: MIT

---

## Key Features

- **Custom data structure** — A `spatioloji` object that unifies expression matrices, cell metadata, spatial coordinates, cell polygons, and FOV images under a single master cell index, ensuring automatic alignment across all components.
- **Efficient memory handling** — Automatic sparse/dense matrix switching (`ExpressionMatrix`), and lazy-loading of FOV images with LRU caching (`ImageHandler`).
- **Quality control** — Comprehensive QC metrics for cells, genes, and FOVs with diagnostic plots. Built-in NegProbe-aware gene filtering for CosMx/MERFISH.
- **Processing pipeline** — Normalization, feature selection, dimensionality reduction (PCA, UMAP, t-SNE), clustering (Leiden, K-Means, hierarchical), batch correction (ComBat, Harmony, scVI), and imputation (MAGIC, ALRA, KNN, DCA, scVI).
- **Spatial analysis** — Two complementary modes: centroid-based (fast, large datasets) and polygon-based (topologically accurate). Includes neighborhood enrichment, spatial autocorrelation, Ripley's K/L/G, and pattern analysis.
- **Interface & gradient analysis** — Detect tissue interfaces between cell type regions, compute gene expression gradients across boundaries, and score immune cell infiltration depth and density.
- **Spatial motif discovery** — Multi-scale tissue architecture analysis: discover recurring local cellular motifs, detect mesoscale assemblies (TLS, tumor buds, immune aggregates), and match against known structure signatures. Scalable to millions of cells.
- **Polygon morphology** — Cell shape metrics including area, circularity, elongation, solidity, convexity, compactness, and Shannon entropy of boundary curvature distribution. Proximity contact mode for buffer graph interactions.
- **Cell-cell communication (CCC)** — A 3-layer polygon-native CCC framework: spatial discovery (Bivariate Moran's I), cell-pair scoring (Polygon OT + Message Passing), and pattern detection (Contrastive Scoring + NMF). No other CCC tool uses polygon contact geometry.
- **Visualization** — 40+ static and interactive spatial plots supporting scatter (dot), polygon (cell boundary), and analysis-specific rendering (gradient curves, motif maps, infiltration summaries).

---

## Installation

```bash
pip install spatioloji-s
```

Requires Python >= 3.12.

For optional extras:
```bash
pip install "spatioloji-s[clustering]"   # Leiden clustering
pip install "spatioloji-s[reduction]"    # UMAP
pip install "spatioloji-s[batch]"        # Harmony, ComBat
pip install "spatioloji-s[anndata]"      # AnnData/scanpy interop
pip install "spatioloji-s[all]"          # Everything above
```

MAGIC imputation requires a separate conda environment:
```bash
conda create -n spatioloji_magic python=3.12
pip install magic-impute spatioloji-s
```

---

## Quick Start

```python
import spatioloji_s as sj

# Load from files
sp = sj.spatioloji.from_files(
    polygons_path      = "polygons.csv",
    cell_meta_path     = "cell_metadata.csv",
    expression_path    = "expression.npz",
    fov_positions_path = "fov_positions.csv",
    images_folder      = "images/"
)

# Or load from a saved object
sp = sj.spatioloji.from_pickle("my_data.pkl")

# Quick summary
sj.data.utils.quick_summary(sp)
```

---

## Module Overview

```
spatioloji_s/
├── data/               # Core data structure, QC, and utilities
│   ├── core.py             # spatioloji class (master cell index)
│   ├── qc.py               # QC filtering, NegProbe-aware metrics
│   ├── expression.py       # ExpressionMatrix (auto sparse/dense)
│   ├── images.py           # ImageHandler (lazy load + LRU cache)
│   └── config.py / utils.py
│
├── processing/         # Single-cell processing pipeline
│   ├── normalization.py        # total, log, scale, Pearson residuals
│   ├── feature_selection.py    # highly variable genes
│   ├── dimension_reduction.py  # PCA, UMAP, t-SNE, diffusion map
│   ├── clustering.py           # Leiden, KMeans, hierarchical, spatial
│   ├── batch_correction.py     # Harmony, ComBat, CCA, rPCA, scVI
│   └── imputation.py           # MAGIC, ALRA, KNN-smooth, DCA, scVI
│
├── spatial/            # Spatial analysis (two complementary modes)
│   ├── point/              # Centroid-based (fast, large datasets)
│   │   ├── graph.py            # KNN / radius / Delaunay graphs
│   │   ├── neighborhoods.py    # Cell-type neighborhood enrichment
│   │   ├── statistics.py       # Nearest-neighbor distances, proximity
│   │   ├── ripley.py           # Ripley's K/L/G functions
│   │   └── patterns.py         # Moran's I, Getis-Ord, co-occurrence
│   ├── polygon/            # Polygon-based (accurate topology)
│   │   ├── graph.py            # Contact / buffer / KNN graphs
│   │   ├── boundaries.py       # Contact/free-boundary fractions + proximity mode
│   │   ├── morphology.py       # Shape metrics + contour entropy
│   │   ├── neighborhoods.py    # Contact-aware neighborhoods + niches
│   │   ├── statistics.py       # Permutation tests, association tests
│   │   ├── patterns.py         # Density, hotspots, autocorrelation
│   │   ├── interface.py        # Tissue interface detection
│   │   ├── gradient.py         # Expression gradients across interfaces
│   │   ├── infiltration.py     # Immune infiltration scoring
│   │   └── motifs.py           # Spatial motif discovery + assemblies
│   └── _*.py               # Shared types (InterfaceResult, GradientResult, etc.)
│
├── ccc/                # Cell-Cell Communication (3-layer framework)
│   ├── database.py         # LR pair loading (CellChatDB, builtin, custom)
│   ├── layer1.py           # Discovery — Bivariate Moran's I + Spatial Lag
│   ├── layer2.py           # Cell-pair scoring — Polygon OT + Message Passing
│   ├── layer3.py           # Patterns — Contrastive Scoring + NMF
│   └── run.py              # CCCConfig, run_ccc, run_ccc_multifov
│
└── visualization/      # Plotting (40+ functions)
    ├── basic_plots.py      # UMAP, PCA, heatmap, violin, dotplot
    ├── plots.py            # Spatial maps (dot and polygon rendering)
    ├── point_plots.py      # Point-based analysis plots
    ├── polygon_plots.py    # Polygon analysis, gradient, motif plots
    └── interactive_plots.py # Plotly-based interactive visualization
```

---

## Usage Pattern

```python
import spatioloji_s as sj

# --- QC ---
qc = sj.data.qc.spatioloji_qc(sp)
qc.filter_cells()
qc.filter_genes(method='percentile')
qc.run_all(output_dir="my_qc_output/")

# --- Processing ---
sj.processing.normalization.normalize_total(sp)
sj.processing.normalization.log_transform(sp)
sj.processing.feature_selection.highly_variable_genes(sp)
sj.processing.dimension_reduction.pca(sp)
sj.processing.dimension_reduction.umap(sp)
sj.processing.clustering.leiden(sp)

# --- Spatial analysis ---
graph = sj.spatial.polygon.build_buffer_graph(sp, buffer_distance=15)  # 15 μm for Xenium
sj.spatial.polygon.neighborhood_enrichment(sp, graph, 'cell_type')
sj.spatial.polygon.compute_morphology(sp, store=True)

# --- Interface & gradient analysis ---
from spatioloji_s.spatial.polygon import identify_interface, compute_gradient, score_infiltration

iface = identify_interface(sp, graph, group_col='cell_type',
                           region_a='Tumor', region_b='Stroma')
gradient = compute_gradient(sp, iface, genes=['MKI67', 'VIM', 'CDH1'])
infiltration = score_infiltration(sp, iface, immune_col='cell_type',
                                  immune_types=['CD8_T', 'Macrophage'],
                                  target_region='Tumor')

# --- Spatial motif discovery ---
from spatioloji_s.spatial.polygon import run_motif_pipeline

motifs = run_motif_pipeline(
    sp, graph, group_col='cell_type', n_motifs=8,
    match_builtin='TME',  # auto-match TLS, tumor buds, immune desert, etc.
)
# motifs.motif_catalog     → local neighborhood motifs
# motifs.assembly_catalog  → mesoscale tissue structures
# motifs.structure_matches → TLS, tumor bud matches with locations

# --- Cell-Cell Communication ---
from spatioloji_s.ccc import CCCConfig, run_ccc, summarize_ccc

config = CCCConfig(
    cell_type_col = 'cell_type',
    layer         = 'log_normalized',
    db_source     = 'cellchatdb',
    db_csv_path   = 'CellChatDB.csv',
    K             = 5,
)
ccc_results = run_ccc(sp_fov, config)
summarize_ccc(ccc_results)

# --- Visualization ---
from spatioloji_s.visualization import (
    plot_gradient_curve, plot_spatial_distance,
    plot_motif_map, plot_assembly_map, plot_structure_matches,
    plot_infiltration_summary,
)
plot_motif_map(sp, motifs, show=True)
plot_gradient_curve(gradient, genes=['MKI67', 'VIM'])
plot_structure_matches(sp, motifs, 'TLS')
```

---

## Data Structure

The `spatioloji` object stores all data aligned to a **master cell index**:

| Component | Type | Description |
|---|---|---|
| `sp.expression` | `ExpressionMatrix` | Sparse/dense gene × cell matrix (auto-switched) |
| `sp.cell_meta` | `pd.DataFrame` | Per-cell metadata, QC metrics, cluster labels |
| `sp.gene_meta` | `pd.DataFrame` | Per-gene metadata (NegProbe flags, HVG status) |
| `sp.spatial` | `SpatialData` | Global and local x/y coordinates per cell |
| `sp.polygons` | `GeoDataFrame` | Cell boundary polygons (Shapely) |
| `sp.images` | `ImageHandler` | Lazy-loaded FOV images with LRU cache |
| `sp.fov_positions` | `pd.DataFrame` | FOV global offsets for stitching |
| `sp.embeddings` | `dict` | PCA, UMAP, t-SNE, diffusion map coordinates |
| `sp.layers` | `dict` | Named expression layers (raw, normalized, scaled) |

---

## Cell-Cell Communication: 3-Layer Framework

The `ccc` module implements a polygon-native CCC framework that uses actual cell geometry — not centroid distances — at every step.

### Why polygon geometry matters

Existing CCC tools (CellChat, COMMOT, SpatialDM) use centroid distance as the spatial proxy. For juxtacrine signals, what physically matters is the **fraction of shared membrane**, not distance. For ECM signals, it is the **morphological complexity and membrane exposure** of the receiver cell. spatioloji_s computes these directly from polygon boundaries.

### Layer 1 — Discovery

Identifies which LR pairs show significant spatial coupling between cell types.

- **Bivariate Moran's I**: tests whether high-ligand senders are non-randomly co-localized with high-receptor receivers, using polygon-geometry weight matrices
- **Spatial lag regression**: estimates effect size (ρ) with confounders (library size, cell density, morphology), FDR-corrected
- **Output**: ranked `(lr_pair, sender_type → receiver_type)` combinations with `layer1_score = |I_bivar| × |ρ|`

### Layer 2 — Cell-Pair Scoring

Scores every contacting cell pair for each significant LR pair.

- **Polygon OT**: entropy-regularized optimal transport with geometry-specific cost matrices (contact fraction for juxtacrine; membrane exposure × distance for secreted; contour entropy × free boundary for ECM)
- **Message passing**: `m[i,j] = √(L_i × R_j) × geo_weight[i,j]`
- **Combined score**: geometric mean of normalized OT and MP scores per cell
- **Hub detection**: top-percentile sender and receiver cells per cell type

### Layer 3 — Pattern Detection

- **Contrastive scoring**: two permutation null models (shuffle expression vs shuffle geometry) classify each LR pair as `SYNERGISTIC`, `EXPRESSION_DRIVEN`, `GEOMETRY_DRIVEN`, or `WEAK`
- **NMF communication programs**: decomposes all LR pair scores into K recurring spatial programs, with polygon Laplacian regularization for spatial coherence
- **Output**: per-cell sender/receiver loadings (A, B matrices) and per-LR-pair loadings (H matrix)

### CCC vs other tools

| Feature | CellChat | NicheNet | CellPhoneDB | SpatialDM | COMMOT | **spatioloji_s** |
|---|---|---|---|---|---|---|
| Single-cell resolution | Partial | No | No | Partial | Yes | Yes |
| Spatial statistics (Moran's I) | No | No | No | Yes | No | Yes (Layer 1) |
| Optimal transport scoring | No | No | No | No | Yes (centroid) | Yes (polygon) |
| Polygon contact geometry | No | No | No | No | No | Yes |
| Membrane complexity (contour entropy) | No | No | No | No | No | Yes |
| Expression vs geometry decomposition | No | Partial | No | No | No | Yes (Layer 3) |
| NMF communication programs | No | No | No | No | No | Yes (Layer 3) |
| Multi-subunit LR complexes | Yes | Partial | Yes | Yes | Partial | Yes |
| Multi-FOV / multi-sample | Yes | Yes | Yes | Partial | No | Yes |

---

## Spatial Motif Discovery

The `spatial.polygon.motifs` module provides hierarchical tissue architecture analysis — identifying recurring cellular patterns at multiple scales.

### Stage 1 — Local Motif Discovery

Each cell is characterized by the cell-type composition of its k-hop neighborhood, then clustered to find recurring local motifs. Motifs capture **microenvironment context**, not cell identity — a tumor cell in an immune-rich neighborhood and a tumor cell in a tumor-only neighborhood are different motifs.

- **Methods**: MiniBatchKMeans (default, scalable) or Leiden (flexible)
- **Auto-selection**: Calinski-Harabasz score when `n_motifs=None`
- **Optional features**: neighbor morphology stats, local cell density (polygon graphs only)

### Stage 2 — Mesoscale Assembly Detection

Groups spatially contiguous cells with the same motif into instances, builds a region graph over instances, and clusters to find recurring multi-motif tissue structures (e.g., a B-cell follicle adjacent to a T-cell zone forming a TLS-like assembly).

### Stage 3 — Known Structure Matching

Matches discovered motifs and assemblies against known biological structure signatures using cosine similarity. Includes built-in TME presets:

- **TLS** (tertiary lymphoid structures)
- **Immune aggregate** (T-cell + macrophage clusters)
- **Tumor bud** (isolated tumor clusters)
- **Perivascular niche** (endothelial + pericyte)
- **Immune desert** (tumor-dominant, immune-excluded)

Users can define custom signatures:

```python
signatures = {
    "my_niche": {"B_cell": 0.3, "T_cell": 0.2, "Tumor": 0.0},  # 0.0 = must be absent
}
```

---

## How spatioloji Compares (Spatial Tools)

| Feature | **spatioloji** | Squidpy | Giotto | SpatialData (scverse) |
|---|---|---|---|---|
| **Primary language** | Python | Python | Python / R | Python |
| **Data structure** | Custom (`spatioloji`) | AnnData | GiottoObject | SpatialData |
| **AnnData dependency** | None (optional) | Required | Required | Required |
| **Image-based ST focus** | First-class | Partial | Partial | Yes |
| **FOV image handling** | Lazy load + LRU cache | No | No | Yes |
| **Master index consistency** | Auto-enforced | Manual | Manual | Manual |
| **Auto sparse/dense matrix** | Yes | No | No | No |
| **Cell polygon analysis** | Full polygon module | Limited | Limited | Partial |
| **Contact-based neighborhoods** | Polygon graph | No | No | No |
| **Proximity contact mode** | Auto buffer-graph support | No | No | No |
| **Polygon shape metrics** | 8 metrics incl. contour entropy | No | No | No |
| **Interface detection** | Graph-based + density-based | No | No | No |
| **Expression gradients** | OLS + gene programs (NMF/PCA) | No | No | No |
| **Immune infiltration scoring** | Depth + density + fraction | No | No | No |
| **Spatial motif discovery** | Multi-scale (local + mesoscale) | No | No | No |
| **Known structure matching** | Built-in TME presets | No | No | No |
| **3-layer CCC framework** | Yes | No | No | No |
| **Ripley's K/L/G** | Yes | Yes | Partial | No |
| **Batch correction** | ComBat + Harmony + scVI | Via scanpy | Limited | Via scanpy |
| **Imputation** | MAGIC + ALRA + KNN + DCA | No | No | No |
| **NegProbe-aware QC** | Built-in | No | No | No |
| **Interactive visualization** | Yes (Plotly) | Yes | Yes | Yes |
| **Polygon visualization** | Yes | Partial | Partial | Partial |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

> Package created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) template.

---

## TODO

### In Progress
- [ ] Visualization methods for CCC results (Layer 1 Moran maps, Layer 2 hub plots, Layer 3 program plots)
- [ ] Interactive spatial plots for CCC and morphology

### Planned
- [ ] Xenium native file format loader
- [ ] AI-supported integrated analysis for histology and gene expression
- [ ] Multi-FOV spatial alignment and cross-sample comparison

### Done
- [x] Core `spatioloji` data structure with master cell index
- [x] QC pipeline with NegProbe-aware gene filtering
- [x] Batch correction (ComBat, Harmony, scVI, CCA, rPCA)
- [x] Imputation (MAGIC, ALRA, KNN-smooth, DCA, scVI)
- [x] Polygon morphology (8 shape metrics including contour entropy)
- [x] 3-layer CCC framework (Moran + Polygon OT + Contrastive NMF)
- [x] Multi-FOV CCC with cross-FOV secreted/ECM graph support
- [x] Tissue interface detection (graph-based + density-based)
- [x] Expression gradient analysis across interfaces (OLS + NMF/PCA programs)
- [x] Immune infiltration scoring (depth, density gradient, fraction)
- [x] Hierarchical spatial motif discovery with known-structure matching
- [x] Proximity contact mode for buffer-graph interactions
- [x] 40+ visualization functions (static + interactive)
