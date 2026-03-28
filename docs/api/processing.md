# Processing — `spatioloji_s.processing`

Single-cell processing pipeline for normalization, feature selection, dimensionality reduction, clustering, batch correction, and imputation.

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

Supported methods: PCA, UMAP, t-SNE, diffusion map.

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

Supported methods: MAGIC, ALRA, KNN-smooth, DCA, scVI.

```{eval-rst}
.. automodule:: spatioloji_s.processing.imputation
   :members:
```
