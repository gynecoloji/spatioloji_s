# Quick Start

This guide walks through a typical spatioloji_s workflow: loading data, QC, processing, and basic spatial analysis.

## Loading data

### From files (CosMx/MERFISH)

```python
import spatioloji_s as sj

sp = sj.spatioloji.from_files(
    polygons_path="polygons.csv",
    cell_meta_path="cell_metadata.csv",
    expression_path="expression.npz",
    fov_positions_path="fov_positions.csv",
    images_folder="images/",
)
```

### From Xenium bundle

```python
sp = sj.spatioloji.from_xenium("path/to/xenium_bundle/")
```

### From AnnData

```python
sp = sj.spatioloji.from_anndata(adata, spatial_key="spatial")
```

## Quality control

```python
qc = sj.data.qc.QCConfig(sp)
qc.qc_cell_metrics(plot=True)
qc.qc_negative_probes(plot=True)
qc.filter_cells()
qc.filter_genes(method="percentile")
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

# Build spatial graph (15 μm buffer for Xenium, ~80 px for CosMx)
graph = build_buffer_graph(sp, buffer_distance=15)

# Neighborhood enrichment
enrichment = neighborhood_enrichment(sp, graph, "cell_type")
```

## Saving

```python
sp.to_pickle("my_analysis.pkl")

# Or export components
sp.save_components("output_dir/")
```

## Next steps

- {doc}`spatial_analysis` — Deeper spatial analysis
- {doc}`interface_gradient` — Interface detection and gradient analysis
- {doc}`motif_discovery` — Tissue architecture discovery
- {doc}`ccc_analysis` — Cell-cell communication
