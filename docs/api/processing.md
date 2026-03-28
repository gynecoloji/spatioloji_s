# Processing — `spatioloji_s.processing`

Single-cell processing pipeline for normalization, feature selection, dimensionality reduction, clustering, batch correction, imputation, differential expression, and pathway activity scoring.

## Normalization

```{eval-rst}
.. automodule:: spatioloji_s.processing.normalization
   :members:
```

## Feature selection

```{eval-rst}
.. automodule:: spatioloji_s.processing.feature_selection
   :members:
```

## Dimensionality reduction

Supported methods: PCA, UMAP, t-SNE, diffusion map. GPU acceleration available for large datasets.

```python
import spatioloji_s as sj

sj.processing.dimension_reduction.pca(sp, n_components=50)
sj.processing.dimension_reduction.umap(sp, n_neighbors=15, min_dist=0.1)

# Parallel UMAP for large datasets (non-reproducible)
sj.processing.dimension_reduction.umap(sp, n_jobs=-1, random_state=None)
```

```{eval-rst}
.. automodule:: spatioloji_s.processing.dimension_reduction
   :members:
```

## Clustering

Supported methods: Leiden, KMeans, hierarchical, spatial, spatially-constrained.

```python
sj.processing.clustering.leiden_clustering(sp, resolution=1.0)
sj.processing.clustering.spatial_clustering(sp, coord_type="global", resolution=1.0)

# Find optimal resolution
sj.processing.clustering.leiden_resolution_sweep(sp, resolution_range=[0.5, 1.0, 1.5, 2.0])
```

```{eval-rst}
.. automodule:: spatioloji_s.processing.clustering
   :members:
```

## Batch correction

Supported methods: ComBat, Harmony, scVI, CCA, rPCA, regress_out.

```{eval-rst}
.. automodule:: spatioloji_s.processing.batch_correction
   :members:
```

## Imputation

Supported methods: MAGIC, ALRA, KNN-smooth, scVI.

```{eval-rst}
.. automodule:: spatioloji_s.processing.imputation
   :members:
```

## Differential expression

Five statistical methods for identifying differentially expressed genes between cell groups.

```python
from spatioloji_s.processing.DEG import run_deg, deg_wilcoxon

# Quick DEG analysis
results = run_deg(sp, group_col="cell_type", method="wilcoxon")

# Specific comparison
results = deg_wilcoxon(sp, group_col="cell_type", groupA="Tumor", groupB="Stroma")
```

| Method | Function | Description |
|--------|----------|-------------|
| Wilcoxon rank-sum | `deg_wilcoxon` | Non-parametric, robust default |
| t-test | `deg_ttest` | Parametric, assumes normality |
| MAST | `deg_mast` | Hurdle model for scRNA-seq |
| NB-GLM | `deg_nb_glm` | Negative binomial GLM |
| DESeq2 | `deg_deseq2` | Variance-stabilized (requires pydeseq2) |

```{eval-rst}
.. automodule:: spatioloji_s.processing.DEG
   :members:
```

## Pathway activity scoring

Score gene set activities per cell using decoupler.

```python
from spatioloji_s.processing.decoupler import load_gene_sets, score_gene_sets

gene_sets = load_gene_sets(resource="GO:BP")
score_gene_sets(sp, gene_sets, method="mean", store_key="pathway_scores")
```

```{eval-rst}
.. automodule:: spatioloji_s.processing.decoupler
   :members:
```
