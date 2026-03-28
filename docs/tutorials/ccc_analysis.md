# Cell-Cell Communication

spatioloji_s provides a 3-layer polygon-native CCC framework. Unlike other tools, it uses actual cell boundary geometry for scoring — not centroid distances.

## Quick start

```python
from spatioloji_s.ccc import CCCConfig, run_ccc, summarize_ccc

config = CCCConfig(
    cell_type_col="cell_type",
    layer="log_normalized",
    db_source="cellchatdb",
    db_csv_path="CellChatDB.csv",
    K=5,  # NMF programs (None = auto)
)

results = run_ccc(sp, config)
summarize_ccc(results)
```

## LR database

```python
from spatioloji_s.ccc.database import load_lr_database, filter_to_expressed

# Load from CellChatDB
lr_pairs = load_lr_database(source="cellchatdb", csv_path="CellChatDB.csv")

# Filter to expressed pairs
lr_pairs = filter_to_expressed(lr_pairs, sp, min_pct=0.1)
print(f"{len(lr_pairs)} expressed LR pairs")
```

## Layer 1 — Discovery

Identifies which LR pairs show significant spatial coupling using Bivariate Moran's I and spatial lag regression.

```python
# Results
sig_pairs = results["significant_pairs"]  # ranked LR pairs
```

## Layer 2 — Cell-Pair Scoring

Scores every contacting cell pair using polygon OT and message passing.

```python
scores = results["scores"]       # per-cell-pair LR activity
hubs = results.get("hubs", {})   # hub sender/receiver cells
```

## Layer 3 — Pattern Detection

Classifies LR pairs by driver mechanism and decomposes into spatial communication programs.

```python
programs = results["programs"]   # NMF factors
drivers = results.get("driver_classification", {})  # expression vs geometry driven
```

## Multi-FOV analysis

```python
from spatioloji_s.ccc import run_ccc_multifov

# Run across all FOVs
sp_dict = {f"fov_{i}": sp.subset_by_fovs([f"fov_{i}"]) for i in range(1, 5)}
multi_results = run_ccc_multifov(sp_dict, config)
```

## Why polygon geometry matters

| Signal type | Centroid-based proxy | Polygon-based proxy (spatioloji_s) |
|---|---|---|
| Juxtacrine | Distance < threshold | Shared membrane fraction |
| Secreted | 1/distance | Membrane exposure x distance |
| ECM | Fixed radius | Contour entropy x free boundary |
