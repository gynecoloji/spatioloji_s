# Spatial Motif Discovery — `spatioloji_s.spatial.polygon.motifs`

Multi-scale tissue architecture analysis: discover recurring local cellular motifs, detect mesoscale assemblies, and match against known biological structures.

## Overview

The motif pipeline has three stages:

1. **Local motif discovery** — Cluster cells by neighborhood composition to find recurring local patterns
2. **Mesoscale assembly detection** — Group contiguous motif instances and cluster their arrangements
3. **Known structure matching** — Match discoveries against biological structure signatures (TLS, tumor buds, etc.)

```python
from spatioloji_s.spatial.polygon import run_motif_pipeline, build_buffer_graph

graph = build_buffer_graph(sp, buffer_distance=15)
result = run_motif_pipeline(
    sp, graph, group_col="cell_type", n_motifs=8,
    match_builtin="TME",  # auto-match known structures
)

# Results
result.motif_catalog       # local motifs
result.assembly_catalog    # mesoscale assemblies
result.structure_matches   # TLS, tumor bud matches
```

## Stage 1: discover_motifs

Each cell is characterized by its k-hop neighborhood composition, then clustered. Motifs capture **microenvironment context** — a tumor cell in an immune-rich neighborhood is a different motif than one in a tumor-only neighborhood.

```python
from spatioloji_s.spatial.polygon.motifs import discover_motifs

motif_cat = discover_motifs(
    sp, graph, group_col="cell_type",
    method="kmeans",   # or "leiden"
    n_motifs=8,        # None = auto-select via Calinski-Harabasz
    k_hops=1,          # neighborhood radius
)
```

```{eval-rst}
.. autofunction:: spatioloji_s.spatial.polygon.motifs.discover_motifs
```

## Stage 2: detect_assemblies

Groups contiguous cells with the same motif label into instances, builds a region graph, and clusters to find recurring multi-motif tissue structures.

```python
from spatioloji_s.spatial.polygon.motifs import detect_assemblies

assembly_cat = detect_assemblies(
    sp, graph, motif_cat,
    method="leiden",
    min_assembly_cells=10,
)
```

```{eval-rst}
.. autofunction:: spatioloji_s.spatial.polygon.motifs.detect_assemblies
```

## Stage 3: match_known_structures

Matches discovered motifs/assemblies against known biological structure signatures using cosine similarity.

### Built-in TME presets

| Structure | Key cell types |
|-----------|---------------|
| TLS | B_cell + T_cell + DC |
| immune_aggregate | CD8_T + CD4_T + Macrophage |
| tumor_bud | Tumor (dominant) |
| perivascular_niche | Endothelial + Pericyte |
| immune_desert | Tumor >90%, T_cell absent |

### Custom signatures

```python
signatures = {
    "my_niche": {"B_cell": 0.3, "T_cell": 0.2, "Tumor": 0.0},  # 0.0 = must be absent
}
matches = match_known_structures(sp, motif_cat, signatures=signatures)
```

```{eval-rst}
.. autofunction:: spatioloji_s.spatial.polygon.motifs.match_known_structures
```

## Convenience wrapper

```{eval-rst}
.. autofunction:: spatioloji_s.spatial.polygon.motifs.run_motif_pipeline
```

## Result types

```{eval-rst}
.. autoclass:: spatioloji_s.spatial._motif_types.MotifCatalog
   :members:

.. autoclass:: spatioloji_s.spatial._motif_types.AssemblyCatalog
   :members:

.. autoclass:: spatioloji_s.spatial._motif_types.StructureMatches
   :members:

.. autoclass:: spatioloji_s.spatial._motif_types.MotifResult
   :members:
```
