# Cell-Cell Communication — `spatioloji_s.ccc`

Polygon-native cell-cell communication analysis that uses actual cell boundary geometry for scoring ligand-receptor interactions.

## Overview

The CCC module has four components:

| Component | Module | Purpose |
|-----------|--------|---------|
| **Database** | `ccc.database` | Load and filter ligand-receptor pair databases |
| **Scoring** | `ccc.scoring` | Score cell-pair interactions for each LR pair |
| **Zones** | `ccc.zones` | Compare communication across spatial zones and morphological contexts |
| **Pipeline** | `ccc.run` | End-to-end orchestration with `CCCConfig` and `CCCResult` |

## Quick start

```python
from spatioloji_s.ccc import CCCConfig, CCCResult, run_ccc
from spatioloji_s.ccc.database import load_lr_database, filter_to_expressed
from spatioloji_s.spatial.polygon import build_buffer_graph

# Load LR database
lr_pairs = load_lr_database(source="cellchatdb", csv_path="CellChatDB.csv")
lr_pairs = filter_to_expressed(lr_pairs, sp, min_pct=0.1)

# Build spatial graph
graph = build_buffer_graph(sp, buffer_distance=15)

# Configure and run
config = CCCConfig(cell_type_col="cell_type", layer="log_normalized")
results = run_ccc(sp, graph, lr_pairs, "cell_type", config)
```

## LR database

Load ligand-receptor pair databases from CellChatDB, custom CSV, or built-in sources. Supports multi-subunit complexes and signaling type classification (juxtacrine, secreted, ECM).

```python
from spatioloji_s.ccc.database import (
    LRPair, load_lr_database, load_from_cellchatdb_csv,
    filter_to_expressed, lr_pairs_to_dataframe,
)

# Load from CellChatDB CSV
lr_pairs = load_from_cellchatdb_csv("CellChatDB.csv")

# Filter to expressed in dataset
lr_pairs = filter_to_expressed(lr_pairs, sp, min_pct=0.1)
print(f"{len(lr_pairs)} expressed LR pairs")

# Convert to table
df = lr_pairs_to_dataframe(lr_pairs)
```

```{eval-rst}
.. autoclass:: spatioloji_s.ccc.database.LRPair
   :members:

.. autofunction:: spatioloji_s.ccc.database.load_lr_database

.. autofunction:: spatioloji_s.ccc.database.load_from_cellchatdb_csv

.. autofunction:: spatioloji_s.ccc.database.filter_to_expressed

.. autofunction:: spatioloji_s.ccc.database.lr_pairs_to_dataframe
```

## Edge scoring

Score cell-pair interactions and test significance via permutation.

```python
from spatioloji_s.ccc.scoring import score_edges, aggregate_scores, test_significance

# Score all edges
edge_scores = score_edges(sp, graph, lr_pairs, interaction_type="expression_product")

# Aggregate per cell type pair
summary = aggregate_scores(edge_scores, aggregation_method="mean")

# Permutation test for significance
sig_results = test_significance(sp, lr_pairs, n_permutations=100)
```

```{eval-rst}
.. automodule:: spatioloji_s.ccc.scoring
   :members:
```

## Spatial zone comparison

Compare communication patterns across spatial zones, along interface gradients, and between morphological contexts.

```python
from spatioloji_s.ccc.zones import compare_zones, communication_gradient, compare_morphology

# Compare CCC across spatial zones (e.g., tumor core vs margin)
zone_comparison = compare_zones(sp, zone_col="spatial_zone")

# Communication gradient across an interface
from spatioloji_s.spatial.polygon import identify_interface
iface = identify_interface(sp, graph, group_col="cell_type",
                           region_a="Tumor", region_b="Stroma")
gradient = communication_gradient(sp, iface, zone_col="cell_type")

# Compare CCC by cell morphology
morph_comparison = compare_morphology(sp, morphology_col="morph_circularity")
```

```{eval-rst}
.. automodule:: spatioloji_s.ccc.zones
   :members:
```

## Pipeline orchestration

```{eval-rst}
.. autoclass:: spatioloji_s.ccc.run.CCCConfig
   :members:

.. autoclass:: spatioloji_s.ccc.run.CCCResult
   :members:

.. autofunction:: spatioloji_s.ccc.run.run_ccc
```

## Why polygon geometry matters

| Signal type | Centroid-based proxy | Polygon-based proxy (spatioloji_s) |
|---|---|---|
| Juxtacrine | Distance < threshold | Shared membrane fraction |
| Secreted | 1/distance | Membrane exposure x distance |
| ECM | Fixed radius | Contour entropy x free boundary |
