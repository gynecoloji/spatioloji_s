# Quick Start

This guide walks through a typical spatioloji_s workflow: loading data, QC, processing, and basic spatial analysis.

## Loading data

### From Xenium bundle

```python
import spatioloji_s as sj

sp = sj.spatioloji.from_xenium("path/to/xenium_bundle/")
```

### From files (CosMx/MERFISH)

```python
sp = sj.spatioloji.from_files(
    expression_file="expression.npz",
    spatial_file="spatial.csv",
    polygon_file="polygons.csv",
)
```

### From AnnData

```python
sp = sj.spatioloji.from_anndata(adata, spatial_key="spatial")
```

### From saved object

```python
sp = sj.spatioloji.from_pickle("my_analysis.pkl")
```

## Quality control

```python
# CosMx/MERFISH QC
qc = sj.data.qc.QCConfig(sp)
qc.qc_cell_metrics(plot=True)
qc.qc_negative_probes(plot=True)
qc.filter_cells()
qc.filter_genes(method="percentile")
sp = qc.apply_filters()

# Xenium QC
qc = sj.data.qc.XeniumQCConfig(sp)
qc.run_qc_pipeline(plot=True)
sp = qc.apply_filters()

sj.data.utils.quick_summary(sp)
```

## Processing

```python
# Normalize and log-transform
sj.processing.normalization.normalize_total(sp)
sj.processing.normalization.log_transform(sp)

# Feature selection
sj.processing.feature_selection.highly_variable_genes(sp, n_top_genes=2000)

# Dimensionality reduction
sj.processing.dimension_reduction.pca(sp, n_components=50)
sj.processing.dimension_reduction.umap(sp, n_neighbors=15, min_dist=0.1)

# Clustering
sj.processing.clustering.leiden_clustering(sp, resolution=1.0)
```

## Visualization

```python
from spatioloji_s.visualization import plot_umap, plot_global_polygon

# UMAP colored by cluster
plot_umap(sp, color_by="leiden")

# Spatial map
plot_global_polygon(sp, color_by="cell_type")
```

## Spatial analysis

```python
from spatioloji_s.spatial.polygon import build_buffer_graph, neighborhood_enrichment

# Build spatial graph
# Xenium: 15 μm (coords in μm)
# CosMx: ~80 px (coords in pixels, 0.18 μm/px)
graph = build_buffer_graph(sp, buffer_distance=15)

# Neighborhood enrichment
enrichment = neighborhood_enrichment(sp, graph, "cell_type")
```

## Saving

```python
sp.to_pickle("my_analysis.pkl")

# Export for other tools
adata = sp.to_anndata()  # → AnnData for scanpy
```

## Next steps

- {doc}`spatial_analysis` — Graphs, neighborhoods, patterns, Ripley's statistics
- {doc}`interface_gradient` — Interface detection and gradient analysis
- {doc}`motif_discovery` — Tissue architecture discovery
- {doc}`ccc_analysis` — Cell-cell communication
- {doc}`deg_pathway` — Differential expression and pathway scoring
