# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

New entries are generated automatically by
[release-please](https://github.com/googleapis/release-please) from
[Conventional Commit](https://www.conventionalcommits.org) messages — do not edit
released sections by hand.

## [0.3.0](https://github.com/gynecoloji/spatioloji_s/releases/tag/v0.3.0) (2026-03-01)

### Added

* **Spatial motif discovery** — multi-scale tissue architecture analysis with `discover_motifs`, `detect_assemblies`, and `match_known_structures`, plus built-in TME presets for TLS, tumor buds, and immune desert detection.
* **Interface detection** — graph-based and density-based tissue interface identification between cell-type regions.
* **Expression gradients** — OLS regression of gene expression against signed distance from an interface, with NMF/PCA gene program discovery.
* **Immune infiltration scoring** — penetration depth, density gradient, and infiltration fraction per immune cell type.
* **Proximity contact mode** — buffer graphs use buffered neighbor polygons for contact metrics, making `contact_length`/`contact_fraction` meaningful for non-touching cells.
* **Differential expression** — five statistical methods (Wilcoxon, t-test, MAST, NB-GLM, DESeq2).
* **Pathway activity scoring** — gene set scoring via decoupler integration.
* **GPU-accelerated dimensionality reduction** — cross-environment PCA/UMAP with GPU support for large datasets.
* **UMAP parallelism** — `n_jobs` parameter for parallel UMAP execution.
* **Weighted KNN graphs** — inverse-distance weighted spatial graphs for both point and polygon modes.
* **40+ visualization functions** — gradient curves, motif maps, assembly maps, structure match highlights, and infiltration summaries.

### Changed

* **CCC module rewritten** — the 3-layer architecture (`layer1`/`layer2`/`layer3`) was replaced with a modular scoring + zones design exposing a new `CCCConfig`/`CCCResult` API.
* **Spatial shared modules** — interface, gradient, and infiltration logic moved to shared `spatial/_*.py` modules, accessible from both the point and polygon subpackages.
* **KNN graphs default to directed** — asymmetric neighbor relationships are now preserved by default.

## [0.2.0](https://github.com/gynecoloji/spatioloji_s/releases/tag/v0.2.0) (2026-03-01)

### Added

* 3-layer CCC framework (bivariate Moran's I, polygon optimal transport, contrastive NMF).
* Polygon morphology metrics (8 shape descriptors including contour entropy).
* Multi-FOV CCC with cross-FOV graph support.

## [0.1.0](https://github.com/gynecoloji/spatioloji_s/releases/tag/v0.1.0) (2026-03-01)

### Added

* First release on PyPI.
* Core `spatioloji` data structure.
* QC pipeline with NegProbe-aware filtering.
* Processing pipeline (normalization, clustering, batch correction, imputation).
* Point-based and polygon-based spatial analysis.
* Static and interactive visualization.
