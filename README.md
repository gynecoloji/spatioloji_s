# spatioloji_s

[![Tests](https://github.com/gynecoloji/spatioloji_s/actions/workflows/test.yml/badge.svg)](https://github.com/gynecoloji/spatioloji_s/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/spatioloji-s.svg)](https://pypi.org/project/spatioloji-s/)
[![Release](https://img.shields.io/github/v/release/gynecoloji/spatioloji_s)](https://github.com/gynecoloji/spatioloji_s/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/spatioloji-s)](https://pypi.org/project/spatioloji-s/)
[![DOI](https://zenodo.org/badge/1123073729.svg)](https://doi.org/10.5281/zenodo.21753918)


**spatioloji_s** is a Python package for image-based spatial transcriptomics analysis, purpose-built for CosMx, MERFISH, and Xenium. It provides an integrated workflow from raw data loading through quality control, processing, spatial analysis, and polygon-native cell-cell communication.

- **PyPI**: [pypi.org/project/spatioloji-s](https://pypi.org/project/spatioloji-s/)
- **GitHub**: [github.com/gynecoloji/spatioloji_s](https://github.com/gynecoloji/spatioloji_s)
- **License**: MIT

---

## Key Features

- **Custom data structure** --- A `spatioloji` object that unifies expression matrices, cell metadata, spatial coordinates, cell polygons, and FOV images under a single master cell index.
- **Two spatial modes** --- Point-based (centroid, fast) and polygon-based (boundary geometry, accurate) analysis with shared interface, gradient, infiltration, and motif APIs.
- **Polygon-native CCC** --- Cell-cell communication using actual membrane geometry, not centroid distance. Edge-level scoring with exponential distance decay, analytical and permutation significance testing, interface zone stratification, and morphology-aware analysis.
- **80+ visualization functions** --- Embedding plots, spatial maps (dot and polygon), neighborhood enrichment, Ripley's K/L, morphology, CCC heatmaps/networks/gradients, motif maps, and more.
- **Comprehensive processing** --- Normalization, HVG selection (6 methods), PCA/UMAP/tSNE/diffusion maps, Leiden/KMeans/hierarchical clustering, DEG (5 methods), batch correction (ComBat, Harmony, scVI), imputation, and gene set scoring.

---

## Installation

```bash
pip install spatioloji-s
```

Requires Python >= 3.12.

Optional extras:

```bash
pip install "spatioloji-s[clustering]"   # Leiden (leidenalg + igraph)
pip install "spatioloji-s[reduction]"    # UMAP
pip install "spatioloji-s[batch]"        # Harmony, ComBat
pip install "spatioloji-s[deg]"          # DESeq2, statsmodels
pip install "spatioloji-s[anndata]"      # AnnData/scanpy interop
pip install "spatioloji-s[annotation]"   # Cell type annotation (CellTypist)
pip install "spatioloji-s[decoupler]"    # Pathway scoring
pip install "spatioloji-s[ripley]"       # Ripley's K/L
pip install "spatioloji-s[all]"          # All of the above
```

`[all]` covers every pip-installable analysis feature. The deep-learning
imputation backends are excluded on purpose — scVI and MAGIC are meant to run in
a dedicated environment (pass `conda_env=` to `scvi_impute` / `magic_impute`),
and pulling PyTorch into a default install costs several GB. Install them
in-process with `pip install "spatioloji-s[imputation]"` if you prefer.

Not sure what you have? `spatioloji_s info` lists every optional feature and
whether it is available.

---

## Quick Start

```python
import spatioloji_s as sj

# 1. Load data
sp = sj.spatioloji(
    expression=expression_df,
    cell_meta=cell_meta_df,
    spatial_coords=spatial_df,
    polygons=polygon_dict,
)

# 2. Process
sj.processing.normalize_total(sp)
sj.processing.log_transform(sp)
sj.processing.pca(sp)
sj.processing.umap(sp)
sj.processing.leiden_clustering(sp)

# 3. Spatial analysis
from spatioloji_s.spatial.polygon import build_buffer_graph, neighborhood_enrichment
graph = build_buffer_graph(sp, buffer_distance=15)
neighborhood_enrichment(sp, graph, "cell_type")

# 4. Cell-cell communication
from spatioloji_s.ccc import CCCConfig, run_ccc
config = CCCConfig(group_col="cell_type", layer="log_normalized")
result = run_ccc(sp, config)

# 5. Visualize
sj.visualization.plot_umap(sp, color_by="cell_type")
sj.visualization.plot_ccc_heatmap(result)
sj.visualization.plot_ccc_network(result)
```

---

## Module Overview

| Module | Functions | Description |
|--------|-----------|-------------|
| `sj.data` | 12 | Core `spatioloji` object, QC, image handling, export |
| `sj.processing` | 44 | Normalization, HVG, dim reduction, clustering, DEG, batch correction, imputation, gene sets |
| `sj.spatial.point` | 30 | Centroid-based: KNN/radius/Delaunay graphs, neighborhoods, Moran's I, Getis-Ord, Ripley's K/L, motifs |
| `sj.spatial.polygon` | 35 | Polygon-based: contact graphs, morphology, boundaries, neighborhoods, interface, gradient, infiltration, motifs |
| `sj.ccc` | 14 | LR database, edge scoring, significance testing, zone/gradient/morphology stratification |
| `sj.visualization` | 74 | Embedding, spatial maps, point/polygon analysis, CCC plots |

---

## Cell-Cell Communication

spatioloji_s uses polygon geometry for biologically accurate CCC inference:

**Scoring formula**: `score(i,j) = sqrt(L_i x R_j) x w_ij`

| Signal type | Weight `w_ij` | Graph |
|-------------|---------------|-------|
| Juxtacrine | `contact_frac_a x contact_frac_b` | Polygon contact graph |
| Secreted | `exp(-distance / sigma)` | Radius graph (default 200 um) |
| ECM | `exp(-distance / sigma)` | Radius graph |

> **Sizing the radius:** expected neighbors per cell ≈ π·r²·density, so edge
> count (memory and run time) grows quadratically with the radius. The 200 µm
> default suits sparse tissue — at ~6,000 cells/mm² (a typical Xenium tumor
> section) it already means ~740 neighbors per cell; use ~100 µm there and
> less for denser tissue (e.g. 75 µm for a reactive lymph node at
> ~15,000 cells/mm²).

Features:
- **50 built-in LR pairs** + CellChatDB (3,234 interactions) support
- **Analytical z-score** (fast) or **permutation** significance testing with BH-FDR
- **Interface zone comparison** --- interface vs. interior enrichment
- **Communication gradient** --- OLS regression of score vs. signed distance
- **Morphology stratification** --- CCC by sender cell shape (round/elongated)

| Feature | CellChat | COMMOT | SpatialDM | spatioloji_s |
|---------|----------|--------|-----------|--------------|
| Single-cell resolution | Partial | Yes | Partial | Yes |
| Polygon contact geometry | No | No | No | **Yes** |
| Spatial edge scoring | No | OT (centroid) | Moran's I | sqrt(LR) x w |
| Interface zone analysis | No | No | No | **Yes** |
| Communication gradient | No | No | No | **Yes** |
| Morphology stratification | No | No | No | **Yes** |
| Significance testing | Permutation | No | Permutation | Both |

---

## Spatial Analysis Highlights

### Interface & Gradient

```python
from spatioloji_s.spatial.polygon import identify_interface, compute_gradient

iface = identify_interface(sp, group_col="cell_type", region_a="Tumor", region_b="Stroma")
grad = compute_gradient(sp, iface, genes=["TGFB1", "VIM"])
```

### Morphology

```python
from spatioloji_s.spatial.polygon import compute_morphology, classify_morphology

compute_morphology(sp, store=True)  # area, circularity, elongation, solidity, ...
classify_morphology(sp)             # round / intermediate / elongated
```

### Spatial Motifs

```python
from spatioloji_s.spatial.polygon import run_motif_pipeline

motifs = run_motif_pipeline(sp, graph, group_col="cell_type", match_builtin="TME")
# Built-in: TLS, tumor buds, immune aggregate, perivascular niche, immune desert
```

---

## Data Structure

The `spatioloji` object stores all data aligned to a **master cell index**:

| Component | Type | Description |
|-----------|------|-------------|
| `sp.expression` | `ExpressionMatrix` | Sparse/dense gene x cell matrix (auto-switched) |
| `sp.cell_meta` | `pd.DataFrame` | Per-cell metadata, QC metrics, cluster labels |
| `sp.gene_meta` | `pd.DataFrame` | Per-gene metadata (NegProbe flags, HVG status) |
| `sp.spatial` | `SpatialData` | Global and local x/y coordinates |
| `sp.polygons` | `GeoDataFrame` | Cell boundary polygons (Shapely) |
| `sp.images` | `ImageHandler` | Lazy-loaded FOV images with LRU cache |
| `sp.embeddings` | `dict` | PCA, UMAP, tSNE, diffusion map coordinates |
| `sp.layers` | `dict` | Named expression layers (raw, normalized, scaled) |

---

## Examples

Example notebooks are being reworked alongside the documentation rebuild. The
workflows they cover:

- **Basic Workflow** --- Loading, QC, normalization, clustering, visualization
- **Spatial Analysis** --- Graphs, neighborhoods, Moran's I, interface detection
- **CCC Analysis** --- LR scoring, significance, zone comparison, morphology
- **Interface Detection** --- Tumor-stroma boundary, gradient analysis
- **Pathway Scoring** --- Gene set analysis via decoupler

---

## Citation

If you use spatioloji_s in your research, please cite it. Use the **"Cite this
repository"** button on the GitHub repository page (generated from
[`CITATION.cff`](CITATION.cff)), or cite the archived release on Zenodo via the
DOI badge above --- the concept DOI always resolves to the latest version.

---

## License

MIT License --- see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/gynecoloji/spatioloji_s.git
cd spatioloji_s
pip install -e ".[test]"
pytest tests/ -v
ruff check src/ tests/ --fix
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org);
versioning, [`CHANGELOG.md`](CHANGELOG.md), and releases are automated with
release-please. See [RELEASING.md](RELEASING.md).
