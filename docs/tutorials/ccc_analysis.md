# Cell-Cell Communication

spatioloji_s provides polygon-native cell-cell communication analysis. Unlike other tools, it uses actual cell boundary geometry for scoring interactions.

## Quick start

```python
from spatioloji_s.ccc import CCCConfig, run_ccc
from spatioloji_s.ccc.database import load_lr_database, filter_to_expressed
from spatioloji_s.spatial.polygon import build_buffer_graph

# 1. Load LR database
lr_pairs = load_lr_database(source="cellchatdb", csv_path="CellChatDB.csv")
lr_pairs = filter_to_expressed(lr_pairs, sp, min_pct=0.1)
print(f"{len(lr_pairs)} expressed LR pairs")

# 2. Build spatial graph
graph = build_buffer_graph(sp, buffer_distance=15)

# 3. Run CCC pipeline
config = CCCConfig(cell_type_col="cell_type", layer="log_normalized")
results = run_ccc(sp, graph, lr_pairs, "cell_type", config)
```

## LR database

```python
from spatioloji_s.ccc.database import (
    load_lr_database, load_from_cellchatdb_csv,
    filter_to_expressed, lr_pairs_to_dataframe,
)

# Load and inspect
lr_pairs = load_from_cellchatdb_csv("CellChatDB.csv")
df = lr_pairs_to_dataframe(lr_pairs)
print(df.head())

# Filter to expressed pairs
lr_pairs = filter_to_expressed(lr_pairs, sp, min_pct=0.1)
```

## Edge scoring

Score individual cell-cell interactions for each LR pair.

```python
from spatioloji_s.ccc.scoring import score_edges, aggregate_scores, test_significance

# Score all edges
edge_scores = score_edges(sp, graph, lr_pairs, interaction_type="expression_product")

# Aggregate by cell type pair
summary = aggregate_scores(edge_scores, aggregation_method="mean")

# Permutation test
sig = test_significance(sp, lr_pairs, n_permutations=100)
```

## Zone comparison

Compare communication patterns across spatial regions.

```python
from spatioloji_s.ccc.zones import compare_zones, communication_gradient, compare_morphology

# Compare CCC across spatial zones
zone_comp = compare_zones(sp, zone_col="spatial_zone")

# Communication gradient across tissue interface
from spatioloji_s.spatial.polygon import identify_interface
iface = identify_interface(sp, group_col="cell_type",
                           region_a="Tumor", region_b="Stroma")
grad = communication_gradient(sp, iface, zone_col="cell_type")

# Stratify by morphology
morph_comp = compare_morphology(sp, morphology_col="morph_circularity")
```

## Multi-sample analysis

```python
# Run across multiple samples
sp_dict = {"sample_1": sp1, "sample_2": sp2, "sample_3": sp3}
for name, sp_obj in sp_dict.items():
    graph = build_buffer_graph(sp_obj, buffer_distance=15)
    results[name] = run_ccc(sp_obj, graph, lr_pairs, "cell_type", config)
```
