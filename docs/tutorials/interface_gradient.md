# Interface & Gradient Analysis

Detect tissue interfaces between cell-type regions, compute expression gradients across boundaries, and score immune cell infiltration.

## Interface detection

```python
from spatioloji_s.spatial.polygon import build_buffer_graph, identify_interface

graph = build_buffer_graph(sp, buffer_distance=15)

iface = identify_interface(
    sp, graph,
    group_col="cell_type",
    region_a="Tumor",
    region_b="Stroma",
)

# Results
iface.cell_labels   # per-cell: "region_a_interface", "interior_a", etc.
iface.contour       # MultiLineString geometry of the interface
iface.segments      # GeoDataFrame with per-segment metrics
iface.summary       # dict with total_length, n_segments, etc.
```

## Expression gradients

Fit gene expression as a function of signed distance from the interface.

```python
from spatioloji_s.spatial.polygon import compute_gradient

gradient = compute_gradient(
    sp, iface,
    genes=["MKI67", "VIM", "CDH1"],          # individual genes
    programs={"EMT": ["VIM", "CDH2", "FN1"]}, # gene programs
    auto_programs="nmf",                       # auto-discover programs
    n_auto_programs=5,
)

# Per-gene results
gradient.gene_gradients
#          coef     pvalue    r2      trend
# MKI67   0.0023   0.001    0.15   increasing_toward_a
# VIM    -0.0015   0.012    0.08   increasing_toward_b

# Binned expression for plotting
gradient.bins  # distance_bin, gene, mean_expr, std_expr

# Signed distances (positive = region A, negative = region B)
gradient.distances
```

## Immune infiltration scoring

Quantify how deeply immune cells penetrate into a target region.

```python
from spatioloji_s.spatial.polygon import score_infiltration

infil = score_infiltration(
    sp, iface,
    immune_col="cell_type",
    immune_types=["CD8_T", "Macrophage"],
    target_region="Tumor",  # None = auto-detect
)

# Per-type metrics
infil.per_type_metrics
#             median_depth  max_depth  density_slope  infiltration_fraction
# CD8_T       45.2          120.0      -0.023         0.35
# Macrophage  28.1          85.0       -0.015         0.42

# Per-cell classification
infil.cell_classifications  # "infiltrating", "resident", or "other"
```

## Visualization

```python
from spatioloji_s.visualization import (
    plot_interface_polygon_map, plot_gradient_curve,
    plot_spatial_distance, plot_infiltration_summary,
)

# Interface map
plot_interface_polygon_map(sp, iface)

# Expression gradient curves
plot_gradient_curve(gradient, genes=["MKI67", "VIM"])

# Distance heatmap
plot_spatial_distance(sp, gradient.distances, interface_result=iface)

# Infiltration summary
plot_infiltration_summary(infil)
```
