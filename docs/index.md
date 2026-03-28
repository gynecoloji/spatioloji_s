# spatioloji_s

**Image-based spatial transcriptomics analysis for CosMx, MERFISH, and Xenium.**

spatioloji_s provides an integrated workflow from raw data loading through quality control, processing, spatial analysis, and polygon-native cell-cell communication — all within a consistent, polygon-aware data structure.

```{image} https://img.shields.io/pypi/v/spatioloji-s.svg
:target: https://pypi.org/project/spatioloji-s/
```

---

## Modules at a glance

| Module | Description |
|--------|-------------|
| {doc}`Data <api/data>` | `spatioloji` object with master cell index, auto sparse/dense expression, lazy image loading |
| {doc}`Processing <api/processing>` | Normalization, HVG, PCA/UMAP/tSNE, Leiden/KMeans, batch correction, imputation |
| {doc}`Spatial <api/spatial>` | Dual point/polygon analysis: neighborhoods, Ripley's K/L, Moran's I, morphology, interface detection |
| {doc}`Motifs <api/motifs>` | Multi-scale tissue architecture: local motifs, mesoscale assemblies, TLS/tumor bud matching |
| {doc}`CCC <api/ccc>` | 3-layer polygon-native CCC: Bivariate Moran's I, Polygon OT, Contrastive NMF |
| {doc}`Visualization <api/visualization>` | 40+ static and interactive plots for embeddings, spatial maps, analysis results |

---

## Quick start

```python
import spatioloji_s as sj

# Load data
sp = sj.spatioloji.from_files(
    polygons_path="polygons.csv",
    cell_meta_path="metadata.csv",
    expression_path="expression.npz",
)

# Process
sj.processing.normalization.normalize_total(sp)
sj.processing.normalization.log_transform(sp)
sj.processing.dimension_reduction.pca(sp)
sj.processing.dimension_reduction.umap(sp)
sj.processing.clustering.leiden(sp)

# Spatial analysis
graph = sj.spatial.polygon.build_buffer_graph(sp, buffer_distance=15)
sj.spatial.polygon.neighborhood_enrichment(sp, graph, "cell_type")

# Motif discovery
from spatioloji_s.spatial.polygon import run_motif_pipeline
motifs = run_motif_pipeline(sp, graph, group_col="cell_type", match_builtin="TME")
```

See {doc}`tutorials/quickstart` for a complete walkthrough.

---

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
tutorials/quickstart
tutorials/spatial_analysis
tutorials/interface_gradient
tutorials/motif_discovery
tutorials/ccc_analysis
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/data
api/processing
api/spatial
api/motifs
api/ccc
api/visualization
```

```{toctree}
:maxdepth: 1
:caption: About

changelog
contributing
```
