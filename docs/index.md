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
| {doc}`Processing <api/processing>` | Normalization, HVG, PCA/UMAP/tSNE, Leiden/KMeans, batch correction, imputation, DEG, pathway scoring |
| {doc}`Spatial <api/spatial>` | Dual point/polygon analysis: neighborhoods, Ripley's K/L, Moran's I, morphology, interface detection |
| {doc}`Motifs <api/motifs>` | Multi-scale tissue architecture: local motifs, mesoscale assemblies, TLS/tumor bud matching |
| {doc}`CCC <api/ccc>` | Polygon-native cell-cell communication: edge scoring, significance testing, zone comparison |
| {doc}`Visualization <api/visualization>` | 40+ static and interactive plots for embeddings, spatial maps, analysis results |

---

## Quick start

```python
import spatioloji_s as sj

# Load data
sp = sj.spatioloji.from_xenium("path/to/xenium_bundle/")

# Process
sj.processing.normalization.normalize_total(sp)
sj.processing.normalization.log_transform(sp)
sj.processing.dimension_reduction.pca(sp)
sj.processing.dimension_reduction.umap(sp)
sj.processing.clustering.leiden_clustering(sp)

# Spatial analysis
from spatioloji_s.spatial.polygon import build_buffer_graph, neighborhood_enrichment
graph = build_buffer_graph(sp, buffer_distance=15)  # 15 μm for Xenium
neighborhood_enrichment(sp, graph, "cell_type")

# Motif discovery
from spatioloji_s.spatial.polygon import run_motif_pipeline
motifs = run_motif_pipeline(sp, graph, group_col="cell_type", match_builtin="TME")

# Cell-cell communication
from spatioloji_s.ccc import CCCConfig, run_ccc
config = CCCConfig(cell_type_col="cell_type", layer="log_normalized")
results = run_ccc(sp, graph, lr_pairs, "cell_type", config)
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
tutorials/deg_pathway
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/index
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
