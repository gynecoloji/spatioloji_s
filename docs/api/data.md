# Data — `spatioloji_s.data`

The data module provides the core `spatioloji` object and supporting infrastructure for loading, storing, and managing spatial transcriptomics data.

## Core object

The `spatioloji` class is the central data structure. All analysis functions take it as their first argument.

| Component | Type | Description |
|---|---|---|
| `sp.expression` | `ExpressionMatrix` | Sparse/dense gene x cell matrix (auto-switched) |
| `sp.cell_meta` | `pd.DataFrame` | Per-cell metadata, QC metrics, cluster labels |
| `sp.gene_meta` | `pd.DataFrame` | Per-gene metadata (NegProbe flags, HVG status) |
| `sp.spatial` | `SpatialData` | Global and local x/y coordinates per cell |
| `sp.polygons` | `pd.DataFrame` | Cell boundary polygon vertices |
| `sp.images` | `ImageHandler` | Lazy-loaded FOV images with LRU cache |
| `sp.layers` | `dict` | Named expression layers (raw, normalized, scaled) |
| `sp.embeddings` | `dict` | PCA, UMAP, tSNE coordinates |

### Creating a spatioloji object

```python
import spatioloji_s as sj

# From files
sp = sj.spatioloji.from_files(
    expression_file="expression.npz",
    spatial_file="spatial.csv",
    polygon_file="polygons.csv",
)

# From Xenium bundle
sp = sj.spatioloji.from_xenium("path/to/xenium_bundle/")

# From AnnData
sp = sj.spatioloji.from_anndata(adata)

# From saved object
sp = sj.spatioloji.from_pickle("my_data.pkl")
```

### Subsetting

```python
sp_fov1 = sp.subset_by_fovs(["fov_1"])
sp_tumor = sp.subset_by_cells(tumor_cell_ids)
sp_hvg = sp.subset_by_genes(hvg_names)
```

```{eval-rst}
.. autoclass:: spatioloji_s.data.core.spatioloji
   :members:
   :undoc-members:
   :show-inheritance:
```

## Configuration

```{eval-rst}
.. autoclass:: spatioloji_s.data.config.SpatiolojiConfig
   :members:

.. autoclass:: spatioloji_s.data.config.SpatialData
   :members:
```

## Expression matrix

```{eval-rst}
.. autoclass:: spatioloji_s.data.expression.ExpressionMatrix
   :members:
```

## Image handling

```{eval-rst}
.. autoclass:: spatioloji_s.data.images.ImageHandler
   :members:
```

## Quality control

```{eval-rst}
.. autoclass:: spatioloji_s.data.qc.QCConfig
   :members:

.. autoclass:: spatioloji_s.data.qc.XeniumQCConfig
   :members:
```

## Utilities

```{eval-rst}
.. automodule:: spatioloji_s.data.utils
   :members:
```
