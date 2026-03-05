# spatioloji_s

[![PyPI version](https://img.shields.io/pypi/v/spatioloji-s.svg)](https://pypi.org/project/spatioloji-s/)
[![Documentation Status](https://readthedocs.org/projects/spatioloji_s/badge/?version=latest)](https://spatioloji_s.readthedocs.io/en/latest/?version=latest)

**spatioloji_s** is a Python package for spatial transcriptomics analysis, purpose-built for image-based single-cell RNA sequencing data (e.g., CosMx, MERFISH, Xenium). It provides an integrated, computationally efficient workflow — from raw data loading through quality control, processing, and spatial analysis — all within a consistent, easy-to-use data structure.

image_based spatial RNA sequencing

* PyPI package: https://pypi.org/project/spatioloji_s/
* Free software: MIT License
* Documentation: https://spatioloji_s.readthedocs.io.


## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
> 📦 GitHub: [gynecoloji/spatioloji_s](https://github.com/gynecoloji/spatioloji_s)

---

## Key Features

- **Custom data structure** — A `spatioloji` object that unifies expression matrices, cell metadata, spatial coordinates, cell polygons, and FOV images under a single master cell index, ensuring automatic alignment across all components.
- **Efficient memory handling** — Automatic sparse/dense matrix switching (`ExpressionMatrix`), and lazy-loading of FOV images with LRU caching (`ImageHandler`).
- **Quality control** — Comprehensive QC metrics for cells, genes, and FOVs with diagnostic plots.
- **Processing pipeline** — Normalization, feature selection, dimensionality reduction (PCA, UMAP, t-SNE), clustering (Leiden, Louvain, K-Means), batch correction (ComBat, Harmony, scVI), and imputation (MAGIC, scVI, ALRA, DCA, KNN).
- **Spatial analysis** — Point-based and polygon-based spatial analysis including neighborhood enrichment, spatial statistics, Ripley's functions, and pattern analysis.
- **Visualization** — Static and interactive spatial plots supporting both scatter (dot) and polygon (cell boundary) rendering, with flexible color customization.

---

## Installation

```bash
pip install spatioloji-s
```

For optional dependencies (e.g., MAGIC imputation):
```bash
conda create -n spatioloji_magic python=3.10
pip install magic-impute
```

---

## Quick Start

```python
import spatioloji_s as sj

# Load from files
sp = sj.spatioloji.from_files(
    polygons_path    = "polygons.csv",
    cell_meta_path   = "cell_metadata.csv",
    expression_path  = "expression.npz",
    fov_positions_path = "fov_positions.csv",
    images_folder    = "images/"
)

# Or load from a saved object
sp = sj.spatioloji.from_pickle("my_data.pkl")

# Quick summary
sj.data.utils.quick_summary(sp)
```

---

## Module Overview

```
spatioloji/
├── data/               # Core data structure, QC, and utilities
│   ├── spatioloji      # Main object class
│   ├── spatioloji_qc   # Quality control pipeline
│   └── ExpressionMatrix / ImageHandler / SpatialData
│
├── processing/         # Analysis pipeline
│   ├── normalization       # Library size, log-normalization
│   ├── feature_selection   # Highly variable genes
│   ├── dimension_reduction # PCA, UMAP, t-SNE
│   ├── clustering          # Leiden, Louvain, K-Means
│   ├── batch_correction    # ComBat, Harmony
│   └── imputation          # MAGIC
│
├── spatial/            # Spatial analysis
│   ├── point/              # Centroid-based analysis
│   │   ├── graph           # Spatial neighbor graphs (kNN, radius)
│   │   ├── neighborhoods   # Cell-type neighborhood enrichment
│   │   ├── statistics      # Spatial autocorrelation (Moran's I)
│   │   ├── ripley          # Ripley's K/L/G functions
│   │   └── patterns        # Spatial pattern detection
│   └── polygon/            # Cell boundary-based analysis
│       ├── graph           # Contact graph from polygon intersections
│       ├── boundaries      # Boundary detection
│       ├── morphology      # Cell shape metrics
│       ├── neighborhoods   # Contact-based neighborhoods
│       ├── statistics      # Polygon spatial statistics
│       └── patterns        # Polygon pattern analysis
│
└── visualization/      # Plotting
    ├── basic_plots     # QC and expression plots
    ├── spatial_plots   # FOV scatter and polygon plots
    └── interactive_plots  # Interactive (Plotly-based) views
```

---

## Usage Pattern

```python
import spatioloji_s as sj

# --- QC ---
qc = sj.spatioloji_qc(sp)
qc.filter_cells()
qc.filter_genes(method='percentile')
qc.run_all(output_dir="my_qc_output/")

# --- Processing ---
sj.processing.normalize(sp)
sj.processing.select_hvg(sp)
sj.processing.pca(sp)
sj.processing.umap(sp)
sj.processing.leiden(sp)

# --- Spatial ---
sj.spatial.point.graph.build_knn_graph(sp, k=10)
sj.spatial.point.neighborhoods.neighborhood_enrichment(sp)

# --- Visualization ---
sj.visualization.spatial_plots.plot_fov(sp, fov=1, color_by='leiden')
sj.visualization.spatial_plots.plot_polygons(sp, fov=1, color_by='cell_type')
```

---

## Data Structure

The `spatioloji` object stores all data aligned to a **master cell index**:

| Component | Description |
|-----------|-------------|
| `sp.expression` | `ExpressionMatrix` — sparse/dense gene × cell matrix |
| `sp.cell_meta` | `pd.DataFrame` — per-cell metadata and annotations |
| `sp.gene_meta` | `pd.DataFrame` — per-gene metadata (incl. NegProbe flags) |
| `sp.spatial` | `SpatialData` — global x/y coordinates per cell |
| `sp.polygons` | `GeoDataFrame` — cell boundary polygons |
| `sp.images` | `ImageHandler` — lazy-loaded FOV images with LRU cache |
| `sp.fov_positions` | `pd.DataFrame` — FOV global offsets |
| `sp.embeddings` | `dict` — PCA, UMAP, t-SNE coordinates |

---

## How spatioloji Compares

The table below benchmarks spatioloji against the most widely-used image-based spatial transcriptomics packages.

| Feature | **spatioloji** | Squidpy | Giotto | SpatialData (scverse) |
|---|---|---|---|---|
| **Primary language** | Python | Python | Python / R | Python |
| **Data structure** | Custom (`spatioloji`) | AnnData | GiottoObject | SpatialData |
| **AnnData dependency** | ✅ None (optional import) | ❌ Required | ❌ Required | ❌ Required |
| **Image-based ST focus** | ✅ First-class | ⚠️ Partial | ⚠️ Partial | ✅ Yes |
| **FOV image handling** | ✅ Lazy load + LRU cache | ❌ No | ❌ No | ✅ Yes |
| **Master index consistency** | ✅ Auto-enforced | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |
| **Auto sparse/dense matrix** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Cell polygon analysis** | ✅ Full polygon module | ⚠️ Limited | ⚠️ Limited | ✅ Partial |
| **Contact-based neighborhoods** | ✅ Polygon graph | ❌ No | ❌ No | ❌ No |
| **Ripley's K/L/G** | ✅ Yes | ✅ Yes | ⚠️ Partial | ❌ No |
| **Batch correction** | ✅ ComBat + Harmony | ⚠️ Via scanpy | ⚠️ Limited | ⚠️ Via scanpy |
| **Imputation (MAGIC)** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **NegProbe-aware QC** | ✅ Built-in | ❌ No | ❌ No | ❌ No |
| **Interactive visualization** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Polygon visualization** | ✅ Yes | ⚠️ Partial | ⚠️ Partial | ✅ Partial |

> ✅ Fully supported · ⚠️ Partial/indirect support · ❌ Not supported

### Key Design Advantages

**1. No AnnData lock-in.** AnnData is a general-purpose format not designed around the multi-FOV, multi-image structure of image-based ST data. spatioloji's custom object natively represents FOV images, global/local coordinates, and polygon boundaries without workarounds.

**2. Master index as single source of truth.** All data components (expression, metadata, coordinates, polygons) are automatically aligned and validated against one master cell index. Misalignment bugs — a common pain point in AnnData-based workflows — are caught immediately at load time.

**3. Auto sparse/dense switching.** `ExpressionMatrix` automatically selects sparse or dense representation based on sparsity, with no user configuration needed. This directly reduces memory footprint for high-gene-count platforms like CosMx (1000+ genes).

**4. Polygon-native spatial analysis.** Unlike centroid-only approaches, spatioloji builds spatial graphs from actual cell boundary contacts (via GeoDataFrame + polygon intersection), enabling physically meaningful neighborhood and interaction analysis.

**5. Platform-aware QC.** Built-in handling of negative control probes (`NegProbe`) for gene filtering — a CosMx/MERFISH-specific requirement that generic tools ignore.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## TODO

### In Progress
- [ ] Add exclusive supportive visualization methods for spatial analysis
- [ ] Add interactive plot for simple visualization

### Planned
- [ ] Add AI-supported integrated analysis for histology and gene expression
- [ ] Support Xenium native file format loader

### Done ✅
- [x] Core spatioloji data structure
- [x] QC pipeline with diagnostic plots
- [x] Batch correction (ComBat, Harmony)
- [x] Add spatially variable gene detection
