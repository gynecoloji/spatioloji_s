# Spatial Analysis

spatioloji_s provides two complementary spatial analysis modes:

- **Point-based** — Uses cell centroids. Fast for large datasets.
- **Polygon-based** — Uses cell boundary polygons. Topologically accurate.

Both share a common pattern: build a graph, then run analyses on it.

## Building spatial graphs

### Point graphs

```python
from spatioloji_s.spatial.point import build_knn_graph, build_radius_graph, build_delaunay_graph

# K-nearest neighbors (most common)
knn = build_knn_graph(sp, k=10)

# Distance threshold
radius = build_radius_graph(sp, radius=50)

# Delaunay triangulation
delaunay = build_delaunay_graph(sp)
```

### Polygon graphs

```python
from spatioloji_s.spatial.polygon import build_contact_graph, build_buffer_graph

# Direct physical contact
contact = build_contact_graph(sp)

# Proximity within buffer distance (recommended for paracrine signaling)
# Xenium: buffer_distance in μm (coords are μm)
# CosMx: buffer_distance in pixels (~0.18 μm/px, so 15 μm ≈ 83 px)
buffer = build_buffer_graph(sp, buffer_distance=15)
```

## Neighborhood analysis

```python
from spatioloji_s.spatial.polygon import (
    neighborhood_composition, neighborhood_enrichment,
    niche_identification, boundary_cells,
)

# Cell-type composition of each cell's neighborhood
comp = neighborhood_composition(sp, graph, "cell_type")

# Enrichment/depletion of cell type co-localization
enrichment = neighborhood_enrichment(sp, graph, "cell_type")

# Identify spatial niches
niches = niche_identification(sp, graph, "cell_type", n_niches=5)

# Find cells at region boundaries
border_cells = boundary_cells(sp, graph, "cell_type")
```

## Cell morphology

```python
from spatioloji_s.spatial.polygon import compute_morphology, classify_morphology

# Compute 8 shape metrics per cell
compute_morphology(sp, store=True)
# Adds to sp.cell_meta: morph_area, morph_perimeter, morph_circularity,
#   morph_elongation, morph_solidity, morph_compactness, morph_convexity,
#   morph_contour_entropy

# Classify cells by morphology
classes = classify_morphology(sp, n_classes=3)
```

## Contact and boundary analysis

```python
from spatioloji_s.spatial.polygon.boundaries import (
    contact_length, contact_fraction, free_boundary_fraction,
)

# With a buffer graph, auto-detects proximity mode
edges = contact_fraction(sp, buffer_graph)    # facing boundary fractions
free = free_boundary_fraction(sp, buffer_graph)  # exposed boundary fractions
```

## Spatial patterns

```python
from spatioloji_s.spatial.point import morans_i, getis_ord_gi, spatially_variable_genes

# Global spatial autocorrelation
mi = morans_i(sp, graph, gene_name="MKI67")

# Local hot/cold spots
gi = getis_ord_gi(sp, graph, gene_name="MKI67")

# Screen for spatially variable genes
svg = spatially_variable_genes(sp, graph)
```

## Ripley's statistics

```python
from spatioloji_s.spatial.point import ripleys_l, simulation_envelope, cross_l

# L function with CSR envelope
L = ripleys_l(sp, "cell_type", category="Tumor")
envelope = simulation_envelope(sp, "cell_type", "Tumor", n_simulations=99)

# Cross-type L function
cross = cross_l(sp, "cell_type", "Tumor", "CD8_T")
```

## Statistical tests

```python
from spatioloji_s.spatial.polygon import contact_permutation_test, morphology_association_test

# Test if cell types contact more/less than expected
perm = contact_permutation_test(sp, graph, "cell_type", n_permutations=999)

# Test morphology-phenotype association
assoc = morphology_association_test(sp, "morph_circularity", "cell_type")
```
