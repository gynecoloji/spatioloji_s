# Cell-Cell Communication — `spatioloji_s.ccc`

A 3-layer polygon-native CCC framework that uses actual cell geometry — not centroid distances — at every step. No other CCC tool uses polygon contact geometry for scoring.

## Why polygon geometry matters

Existing CCC tools (CellChat, COMMOT, SpatialDM) use centroid distance as the spatial proxy. For juxtacrine signals, what physically matters is the **fraction of shared membrane**, not distance. For ECM signals, it is the **morphological complexity and membrane exposure** of the receiver cell. spatioloji_s computes these directly from polygon boundaries.

## Quick start

```python
from spatioloji_s.ccc import CCCConfig, run_ccc, summarize_ccc

config = CCCConfig(
    cell_type_col="cell_type",
    layer="log_normalized",
    db_source="cellchatdb",
    db_csv_path="CellChatDB.csv",
    K=5,
)
results = run_ccc(sp, config)
summarize_ccc(results)
```

## Layer 1 — Discovery

Identifies which LR pairs show significant spatial coupling between cell types.

- **Bivariate Moran's I**: tests spatial co-localization of ligand senders and receptor receivers using polygon-geometry weight matrices
- **Spatial lag regression**: estimates effect size with confounders (library size, cell density, morphology), FDR-corrected
- **Output**: ranked `(lr_pair, sender → receiver)` combinations

## Layer 2 — Cell-Pair Scoring

Scores every contacting cell pair for each significant LR pair.

- **Polygon OT**: entropy-regularized optimal transport with geometry-specific cost matrices
- **Message passing**: `m[i,j] = sqrt(L_i * R_j) * geo_weight[i,j]`
- **Hub detection**: top-percentile sender and receiver cells per type

## Layer 3 — Pattern Detection

- **Contrastive scoring**: classifies each LR pair as expression-driven, geometry-driven, or synergistic
- **NMF communication programs**: decomposes LR pair scores into K spatial programs with polygon Laplacian regularization

## LR database

```{eval-rst}
.. automodule:: spatioloji_s.ccc.database
   :members:
```

## Configuration & orchestration

```{eval-rst}
.. automodule:: spatioloji_s.ccc.run
   :members:
```
