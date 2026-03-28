# Spatial Analysis — `spatioloji_s.spatial`

Two complementary analysis modes that share a common interface:

- **Point-based** (`spatial.point`) — Uses cell centroids. Fast, suitable for large datasets.
- **Polygon-based** (`spatial.polygon`) — Uses actual cell boundary polygons. Topologically accurate.

## Graph construction

All spatial analysis starts with building a spatial graph.

```python
from spatioloji_s.spatial.point import build_knn_graph, build_radius_graph, build_delaunay_graph
from spatioloji_s.spatial.polygon import build_contact_graph, build_buffer_graph

# Point graphs
knn_graph = build_knn_graph(sp, k=10)
radius_graph = build_radius_graph(sp, radius=50)
delaunay_graph = build_delaunay_graph(sp)

# Polygon graphs
contact_graph = build_contact_graph(sp)           # physically touching cells
buffer_graph = build_buffer_graph(sp, buffer_distance=15)  # 15 μm proximity
```

### Point graph

```{eval-rst}
.. autoclass:: spatioloji_s.spatial.point.graph.PointSpatialGraph
   :members:

.. autofunction:: spatioloji_s.spatial.point.graph.build_knn_graph

.. autofunction:: spatioloji_s.spatial.point.graph.build_radius_graph

.. autofunction:: spatioloji_s.spatial.point.graph.build_delaunay_graph
```

### Polygon graph

```{eval-rst}
.. autoclass:: spatioloji_s.spatial.polygon.graph.PolygonSpatialGraph
   :members:

.. autofunction:: spatioloji_s.spatial.polygon.graph.build_contact_graph

.. autofunction:: spatioloji_s.spatial.polygon.graph.build_buffer_graph

.. autofunction:: spatioloji_s.spatial.polygon.graph.build_knn_graph
```

## Neighborhoods

```{eval-rst}
.. automodule:: spatioloji_s.spatial.point.neighborhoods
   :members:

.. automodule:: spatioloji_s.spatial.polygon.neighborhoods
   :members:
```

## Spatial patterns

Point-based: Moran's I, Getis-Ord Gi*, spatially variable genes, co-occurrence.

```{eval-rst}
.. automodule:: spatioloji_s.spatial.point.patterns
   :members:
```

Polygon-based: cell density, hotspots, spatial autocorrelation, colocalization.

```{eval-rst}
.. automodule:: spatioloji_s.spatial.polygon.patterns
   :members:
```

## Ripley's statistics

K, L, cross-K, cross-L functions with simulation envelopes.

```{eval-rst}
.. autoclass:: spatioloji_s.spatial.point.ripley.RipleyResult
   :members:

.. automodule:: spatioloji_s.spatial.point.ripley
   :members:
   :exclude-members: RipleyResult
```

## Cell morphology

Eight shape metrics per cell, including Shannon entropy of boundary curvature.

```python
from spatioloji_s.spatial.polygon import compute_morphology, classify_morphology

compute_morphology(sp, store=True)
# Adds: morph_area, morph_perimeter, morph_circularity, morph_elongation,
#        morph_solidity, morph_compactness, morph_convexity, morph_contour_entropy
```

```{eval-rst}
.. automodule:: spatioloji_s.spatial.polygon.morphology
   :members:
```

## Boundaries & contact

Contact length, contact fraction, free boundary fraction. Automatically uses **proximity mode** when given a buffer graph — buffered neighbor polygons for intersection so that non-touching cells within buffer distance get meaningful contact metrics.

```python
from spatioloji_s.spatial.polygon.boundaries import contact_length, contact_fraction, free_boundary_fraction

buffer_graph = build_buffer_graph(sp, buffer_distance=15)
edges = contact_fraction(sp, buffer_graph)  # auto proximity mode
free = free_boundary_fraction(sp, buffer_graph)
```

```{eval-rst}
.. automodule:: spatioloji_s.spatial.polygon.boundaries
   :members:
```

## Interface detection

Detect boundaries between cell-type regions.

```python
from spatioloji_s.spatial.polygon import identify_interface

iface = identify_interface(sp, graph, group_col="cell_type",
                           region_a="Tumor", region_b="Stroma")
# iface.cell_labels  — per-cell interface/interior labels
# iface.contour      — MultiLineString boundary geometry
# iface.segments     — GeoDataFrame of interface segments
```

```{eval-rst}
.. autoclass:: spatioloji_s.spatial._interface_types.InterfaceResult
   :members:

.. automodule:: spatioloji_s.spatial.polygon.interface
   :members:
```

## Expression gradients

Fit gene expression vs signed distance from interface.

```python
from spatioloji_s.spatial.polygon import compute_gradient

gradient = compute_gradient(sp, iface, genes=["MKI67", "VIM"],
                            programs={"EMT": ["VIM", "CDH2", "SNAI1"]})
# gradient.gene_gradients    — coef, pvalue, r2, trend per gene
# gradient.program_gradients — same for gene programs
# gradient.bins              — binned expression for plotting
```

```{eval-rst}
.. autoclass:: spatioloji_s.spatial._gradient_types.GradientResult
   :members:

.. automodule:: spatioloji_s.spatial.polygon.gradient
   :members:
```

## Immune infiltration

Score immune cell penetration across an interface.

```python
from spatioloji_s.spatial.polygon import score_infiltration

infil = score_infiltration(sp, iface, immune_col="cell_type",
                           immune_types=["CD8_T", "Macrophage"],
                           target_region="Tumor")
# infil.per_type_metrics — depth, density slope, infiltration fraction
# infil.cell_classifications — "infiltrating", "resident", or "other"
```

```{eval-rst}
.. autoclass:: spatioloji_s.spatial._infiltration_types.InfiltrationResult
   :members:

.. automodule:: spatioloji_s.spatial.polygon.infiltration
   :members:
```

## Statistical tests

```{eval-rst}
.. automodule:: spatioloji_s.spatial.point.statistics
   :members:

.. automodule:: spatioloji_s.spatial.polygon.statistics
   :members:
```
