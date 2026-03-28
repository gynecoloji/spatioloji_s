# Spatial Motif Discovery

Discover recurring tissue architecture at multiple scales — from local cellular neighborhoods to mesoscale tissue structures like tertiary lymphoid structures (TLS) and tumor buds.

## Full pipeline

```python
from spatioloji_s.spatial.polygon import build_buffer_graph, run_motif_pipeline

graph = build_buffer_graph(sp, buffer_distance=15)
result = run_motif_pipeline(
    sp, graph,
    group_col="cell_type",
    n_motifs=8,                 # or None for auto-selection
    match_builtin="TME",        # match against known structures
    match_signatures={          # add custom signatures
        "my_niche": {"B_cell": 0.3, "T_cell": 0.2},
    },
)

# Access results
result.motif_catalog        # local motifs
result.assembly_catalog     # mesoscale assemblies
result.structure_matches    # known structure matches
```

## Step-by-step

### Stage 1: Local motifs

```python
from spatioloji_s.spatial.polygon.motifs import discover_motifs

motif_cat = discover_motifs(
    sp, graph,
    group_col="cell_type",
    method="kmeans",    # or "leiden"
    n_motifs=8,         # None = auto-select via Calinski-Harabasz
    k_hops=1,           # neighborhood radius
    keep_features=True, # retain feature matrix
    store=True,         # add "motif_label" to sp.cell_meta
)

# What each motif "looks like"
print(motif_cat.signatures)
#          Tumor  Stroma  CD8_T  Macrophage  B_cell
# 0         0.82    0.10   0.02        0.04    0.02   <- tumor core
# 1         0.15    0.45   0.20        0.15    0.05   <- tumor-stroma interface
# 2         0.05    0.10   0.35        0.10    0.40   <- TLS-like
```

### Stage 2: Mesoscale assemblies

```python
from spatioloji_s.spatial.polygon.motifs import detect_assemblies

assembly_cat = detect_assemblies(
    sp, graph, motif_cat,
    method="leiden",
    min_assembly_cells=10,
)

# Motif composition of each assembly
print(assembly_cat.composition)

# Individual instances with locations
print(assembly_cat.instances[["assembly_id", "motif_id", "n_cells", "centroid_x", "centroid_y"]])
```

### Stage 3: Known structure matching

```python
from spatioloji_s.spatial.polygon.motifs import match_known_structures

# Built-in TME presets
matches = match_known_structures(sp, motif_cat, builtin="TME")

# Custom signatures (0.0 = must be absent)
matches = match_known_structures(sp, motif_cat, signatures={
    "TLS": {"B_cell": 0.25, "T_cell": 0.15, "DC": 0.05},
    "immune_exclusion": {"Tumor": 0.85, "T_cell": 0.0},
})

# Where are the matches?
print(matches.matches[["structure_name", "similarity", "n_cells", "centroid_x", "centroid_y"]])

# Per-cell labels
print(matches.per_cell.value_counts())
```

## Visualization

```python
from spatioloji_s.visualization import (
    plot_motif_map, plot_motif_composition,
    plot_assembly_map, plot_structure_matches,
)

# Spatial motif map
plot_motif_map(sp, result)

# What each motif is made of
plot_motif_composition(result)

# Assembly map
plot_assembly_map(sp, result)

# Highlight specific structures
plot_structure_matches(sp, result, "TLS")
```

## Tips

- **Buffer distance matters**: Use ~15 μm for Xenium (μm coords), ~80 px for CosMx (pixel coords)
- **n_motifs**: Start with `None` (auto-select), then refine. Too few = merged patterns, too many = noise
- **k_hops**: `1` captures immediate neighbors (default). `2` captures broader context but blurs local patterns
- **Leiden vs KMeans**: Leiden finds the number of motifs automatically; KMeans requires `n_motifs` but is faster and more reproducible
- **Scalability**: Tested to 1M+ cells. Uses MiniBatchKMeans and sparse matrices internally
