# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

New entries are generated automatically by
[release-please](https://github.com/googleapis/release-please) from
[Conventional Commit](https://www.conventionalcommits.org) messages — do not edit
released sections by hand.

## [0.4.4](https://github.com/gynecoloji/spatioloji_s/compare/v0.4.3...v0.4.4) (2026-08-03)


### Fixed

* update publish.yml and release-please.yml to turn publish into automatic move ([9b875f4](https://github.com/gynecoloji/spatioloji_s/commit/9b875f43a81eb573acd7307dedeb34547a4e7c2e))

## [0.4.3](https://github.com/gynecoloji/spatioloji_s/compare/v0.4.2...v0.4.3) (2026-08-03)


### Fixed

* **clustering:** build the neighbor graph the way sc.pp.neighbors does ([50670de](https://github.com/gynecoloji/spatioloji_s/commit/50670def0932cff748e09ff1718e0575d5cb8786))
* **clustering:** do not reproduce umap's pre-0.5.11 bandwidth bug ([271a90c](https://github.com/gynecoloji/spatioloji_s/commit/271a90c3dab348e3d89206e7f87ab0497a57659a))
* **deg:** rank wilcoxon markers by signed z-score, not two-sided padj ([5f17d89](https://github.com/gynecoloji/spatioloji_s/commit/5f17d894a9dda016972b1d7915e37b8155f85e82))
* **hvg:** implement Seurat v3 VST and accumulate moments in float64 ([251b7fe](https://github.com/gynecoloji/spatioloji_s/commit/251b7fe5e1391bbc18bb8f548d44ea7416737cdb))
* **pca:** use arpack solver to match scanpy ([a789a9f](https://github.com/gynecoloji/spatioloji_s/commit/a789a9f521c0149b65383909b382ab1c6a1410ef))
* scanpy numerical concordance ([ccb9e48](https://github.com/gynecoloji/spatioloji_s/commit/ccb9e488e19144cf68deb171656406003517d1ab))

## [0.4.2](https://github.com/gynecoloji/spatioloji_s/compare/v0.4.1...v0.4.2) (2026-08-02)


### Documentation

* add Zenodo DOI badge and sync Python badge to PyPI metadata ([442e82a](https://github.com/gynecoloji/spatioloji_s/commit/442e82a678a385be565019e194d3baa2d503ffe5))

## [0.4.1](https://github.com/gynecoloji/spatioloji_s/compare/v0.4.0...v0.4.1) (2026-08-02)


### Documentation

* **changelog:** drop reverted C++ entries from the 0.4.0 notes ([8b3f5bd](https://github.com/gynecoloji/spatioloji_s/commit/8b3f5bda2f72adf6fd3c7deadda07dc9e5a3bf0f))

## [0.4.0](https://github.com/gynecoloji/spatioloji_s/compare/v0.3.0...v0.4.0) (2026-08-02)


### Added

* add 3 layer analysis for ccc ([ca45a7e](https://github.com/gynecoloji/spatioloji_s/commit/ca45a7e1690ce84e5709b45eec7ebcb892999593))
* add batch_correction, imputation methods, debug dimension_reduction methods and add ipynb tutorials for imputation and batch_correction ([d09a66b](https://github.com/gynecoloji/spatioloji_s/commit/d09a66bae57b659910c977ae9024f4afcb806862))
* add compute_gradient for spatial expression gradient analysis ([d143166](https://github.com/gynecoloji/spatioloji_s/commit/d143166e385a09cdef0fe94ddce9b05f32be86c9))
* add detect_assemblies for mesoscale tissue structure detection ([ccf6023](https://github.com/gynecoloji/spatioloji_s/commit/ccf602386f5baf9e5ede0f1fa8b10de871a89260))
* add discover_motifs for local spatial motif discovery ([664731b](https://github.com/gynecoloji/spatioloji_s/commit/664731b6d7e3cb385b56bea1e583f62898ba4de2))
* add GradientResult dataclass ([760569d](https://github.com/gynecoloji/spatioloji_s/commit/760569d5938164c6dc4bc13c977bf3025513d2a1))
* add InfiltrationResult dataclass ([0e8c486](https://github.com/gynecoloji/spatioloji_s/commit/0e8c486aec5496b60b1f1234f834ae04ed27dea5))
* add match_known_structures with builtin TME presets ([1d9814b](https://github.com/gynecoloji/spatioloji_s/commit/1d9814b20844e0789f02711c71af06da573bdc7d))
* add motif analysis dataclasses and graph helpers ([8392f7a](https://github.com/gynecoloji/spatioloji_s/commit/8392f7a623561462ee48d92b7080c5eebcf35753))
* add run_motif_pipeline and update point re-exports ([36f2454](https://github.com/gynecoloji/spatioloji_s/commit/36f24545200d5d5c72135cf2c6894ca2bc628711))
* add score_infiltration for immune cell infiltration analysis ([7e40196](https://github.com/gynecoloji/spatioloji_s/commit/7e40196855e1ac1a9246d1ed8406a1d9106a9995))
* add signed_distance_to_interface utility ([4e5b179](https://github.com/gynecoloji/spatioloji_s/commit/4e5b1794249aef93b5c61c609b0690a2c0bf5abf))
* **ccc/spatial:** replace solidity with contour_entropy×free_boundary for ECM weighting ([3a57bed](https://github.com/gynecoloji/spatioloji_s/commit/3a57bed8972aca7c2386d74163c18bcef626c896))
* **ccc:** add scoring.py with score_edges, aggregate_scores, test_significance ([54f6525](https://github.com/gynecoloji/spatioloji_s/commit/54f6525a2d272c2eb1dec41a1b9e0f4b44bacce1))
* **ccc:** add zones.py with compare_zones, communication_gradient, compare_morphology ([650ba8c](https://github.com/gynecoloji/spatioloji_s/commit/650ba8c66f4851f3e1ec914412796c91bfaeac77))
* **ccc:** implement 3-layer CCC framework with ruff formatting cleanup ([93e75f5](https://github.com/gynecoloji/spatioloji_s/commit/93e75f5e803446a82397dfb2a6e5ea1b05616d07))
* **ccc:** rewrite run.py with CCCConfig, CCCResult, run_ccc ([13fb9f6](https://github.com/gynecoloji/spatioloji_s/commit/13fb9f65821301f06bafba78882a06a916e62f8c))
* **deg:** add convenience wrappers, __init__.py integration, lint fixes ([57f5382](https://github.com/gynecoloji/spatioloji_s/commit/57f53821482c07391716d4c5dcd702eb79cff9c0))
* **deg:** add DEG module with 5 statistical methods ([6022276](https://github.com/gynecoloji/spatioloji_s/commit/60222765e3ccd5a875b4322108d78632e4403040))
* **deg:** add DEG.py skeleton and [deg] optional deps ([5837bb5](https://github.com/gynecoloji/spatioloji_s/commit/5837bb5996428a2fbfaae530774029a51f8fec4a))
* **deg:** implement _aggregate_pseudobulk and _deseq2_backend ([62a3b6b](https://github.com/gynecoloji/spatioloji_s/commit/62a3b6b2e2dfaa8d240007b33f092fe0dbdd2cf8))
* **deg:** implement _build_cell_mask, _apply_correction, _build_result_df ([0a7f5d7](https://github.com/gynecoloji/spatioloji_s/commit/0a7f5d75ffe77a88ef87329e8a857792829ffd0a))
* **deg:** implement _mast_backend and _nb_glm_backend ([5179489](https://github.com/gynecoloji/spatioloji_s/commit/5179489987964adb5db9964d8c07adb0def81b18))
* **deg:** implement _wilcoxon_backend and _ttest_backend ([bb2bfd9](https://github.com/gynecoloji/spatioloji_s/commit/bb2bfd9916f4263028276836b8ed5619cca428f2))
* **deg:** implement run_deg core pipeline (wilcoxon + ttest paths) ([9083c56](https://github.com/gynecoloji/spatioloji_s/commit/9083c56d1818453492a2646e8fb715fedb5e575e))
* optimize leiden_resolution_sweep function ([f84a99a](https://github.com/gynecoloji/spatioloji_s/commit/f84a99aa000ebd5e56f93d116ea5b7cd44f88b5b))
* optimize speed of morphology-related calculation and add morphology-genes analysis ([022c840](https://github.com/gynecoloji/spatioloji_s/commit/022c840cf22efad348a6ec3e6d5db078a57cd4f6))
* **processing:** add CellTypist-based automated cell-type annotation ([0a6c219](https://github.com/gynecoloji/spatioloji_s/commit/0a6c21956c65c155ac50379f7cd557e865f9d02b))
* **processing:** add cross-env dimension reduction and GPU support ([bc3b54c](https://github.com/gynecoloji/spatioloji_s/commit/bc3b54c956da78d35010e52a663bedc2de48dead))
* **processing:** add decoupler module for pathway activity scoring ([6862aae](https://github.com/gynecoloji/spatioloji_s/commit/6862aaeb37db95310d97272905ef35710f17e390))
* **processing:** add n_jobs parameter to UMAP for parallel execution ([5ff6b11](https://github.com/gynecoloji/spatioloji_s/commit/5ff6b1143426392b45609d7fc62584970216e277))
* **spatial/point:** add point-based identify_interface ([3450912](https://github.com/gynecoloji/spatioloji_s/commit/34509127c3df92e03131f877c8f4410f65767464))
* **spatial/polygon:** add density-based interface method ([8bc584c](https://github.com/gynecoloji/spatioloji_s/commit/8bc584c35b2c2e19f0c1b2c3702ffec8c7fa0282))
* **spatial/polygon:** add graph-based identify_interface ([9b9b884](https://github.com/gynecoloji/spatioloji_s/commit/9b9b88406b18055653f6630e31305a8a18b30f02))
* **spatial/polygon:** add proximity contact mode for buffer graphs ([929b059](https://github.com/gynecoloji/spatioloji_s/commit/929b059413f76d5128bd83fdf3080c347a763588))
* **spatial/polygon:** export identify_interface and InterfaceResult ([9a90689](https://github.com/gynecoloji/spatioloji_s/commit/9a90689de9be1dc7fa915278253f7eb9cdef7bc5))
* **spatial:** add InterfaceResult dataclass ([c6858d0](https://github.com/gynecoloji/spatioloji_s/commit/c6858d004fab7d4cf1227bbd4e58d2e090a08acd))
* **spatial:** add weighted KNN graphs, morphology features, and gene support for polygon patterns ([ecf5a51](https://github.com/gynecoloji/spatioloji_s/commit/ecf5a5133453d2f617cd97165a57121fe0a00304))
* **visualization:** add CCC plot functions ([e64e7b9](https://github.com/gynecoloji/spatioloji_s/commit/e64e7b9fc86eef477a89105b18aea557dd464618))
* **visualization:** add gradient curve, spatial distance, and infiltration summary plots ([f0f4107](https://github.com/gynecoloji/spatioloji_s/commit/f0f4107fe7026a99a6470bec86ad0867e6441582))
* **visualization:** add motif map, composition, assembly, and structure match plots ([5111075](https://github.com/gynecoloji/spatioloji_s/commit/5111075852ef30ff07d6651055a258a70b79f55a))
* **visualization:** add new plot functions and fix finalize_plot ([5bf2629](https://github.com/gynecoloji/spatioloji_s/commit/5bf2629affba4618a242ba49cd19320265a5da53))
* **visualization:** add plot_interface_map and plot_interface_metrics ([a97a32d](https://github.com/gynecoloji/spatioloji_s/commit/a97a32d346a5961b640a4ffb5b42d67e90cd5cc7))
* wire gradient and infiltration modules into package exports ([34ef1e7](https://github.com/gynecoloji/spatioloji_s/commit/34ef1e7551729447d18f4924847d8329e77cd3c4))
* wire motif modules into package exports ([5ce848b](https://github.com/gynecoloji/spatioloji_s/commit/5ce848be275d37e9d738c7c4b4dba5e765fdad6a))


### Fixed

* **cli:** replace broken stub with version and info commands ([9b1546a](https://github.com/gynecoloji/spatioloji_s/commit/9b1546a797495577d7d74d1c85b9495d8ee565cb))
* **docs:** add numba to autodoc_mock_imports for RTD build ([93f2ac6](https://github.com/gynecoloji/spatioloji_s/commit/93f2ac63e6b8ec3bc0d7598e65a66505379fb424))
* **docs:** remove duplicate .readthedocs.yml that pointed to wrong conf.py path ([eb5c277](https://github.com/gynecoloji/spatioloji_s/commit/eb5c27705eb5760ab43813641f24b5017e1f69ce))
* make package importable without optional dependencies ([bb12eca](https://github.com/gynecoloji/spatioloji_s/commit/bb12eca599bf8b04d4b8404b5bbdc2ee48d2df2a))
* remove unused variable in plot_structure_matches ([c0113f8](https://github.com/gynecoloji/spatioloji_s/commit/c0113f85b682d436dceb53051a2a1910e562803b))
* reset interface labels when all segments dropped by min_interface_cells ([e8ce226](https://github.com/gynecoloji/spatioloji_s/commit/e8ce226210a8cdda2d8faf2fab785675642564d0))
* **spatial/point:** fix MemoryError in build_radius_graph for large datasets ([69c32ec](https://github.com/gynecoloji/spatioloji_s/commit/69c32ec55efb62e020c05747e1868015027e9892))
* **spatial/polygon:** handle directed graphs in enrichment and colocalization ([4880bfa](https://github.com/gynecoloji/spatioloji_s/commit/4880bfa5986db54f6a83ec85ce4a89375498492e))
* **visualization:** don't close caller's figure when ax is provided ([26708d6](https://github.com/gynecoloji/spatioloji_s/commit/26708d678d5a7cc5f4a775aa8cdb2f19644e902b))
* **visualization:** fix KeyError in plot_ccc_gradient nlargest call ([80f7332](https://github.com/gynecoloji/spatioloji_s/commit/80f733291ff168d1e32fb9836237553b51cde86d))
* **visualization:** return the figure from plot helpers ([de5d9ac](https://github.com/gynecoloji/spatioloji_s/commit/de5d9acb1dcc2bd998c3c4275c023c134ce7610a))


### Changed

* **ccc:** delete layer1, layer2, layer3 — replaced by scoring + zones ([d3245e2](https://github.com/gynecoloji/spatioloji_s/commit/d3245e285f8247310536795ee8b478666cbe70c9))
* **spatial:** default KNN graphs to directed (asymmetric) ([fe877e8](https://github.com/gynecoloji/spatioloji_s/commit/fe877e8b6f8ffbf981047875eca59da7d3fb5c76))
* **spatial:** unify interface, infiltration, and gradient into shared modules ([2cdee1c](https://github.com/gynecoloji/spatioloji_s/commit/2cdee1c5ada18b848a0796c573a2bfc9dec75665))
* **visualization:** disambiguate interface plot names for point vs polygon ([19ffa19](https://github.com/gynecoloji/spatioloji_s/commit/19ffa1912c265981587f969497bb469cf154681d))


### Documentation

* add comprehensive Sphinx documentation site ([b78eaa3](https://github.com/gynecoloji/spatioloji_s/commit/b78eaa35ffaf53e8f159981cead7973d38e8765d))
* add DEG implementation plan (post-review) ([a913803](https://github.com/gynecoloji/spatioloji_s/commit/a9138035532873f9659b5f2cdc67242783c30275))
* add DEG.py design spec ([f3bf789](https://github.com/gynecoloji/spatioloji_s/commit/f3bf7898a7e3e6ded0e6bffc84d75fa5cfbeb323))
* add example Jupyter notebooks to documentation ([95258f8](https://github.com/gynecoloji/spatioloji_s/commit/95258f849e469370679ee492231832aeec2bdd1e))
* add example Jupyter notebooks with title fixes ([870225a](https://github.com/gynecoloji/spatioloji_s/commit/870225ad6f2fceacbb75c9d06e857d6e548d7e5b))
* add interface cells analysis design spec ([99f8a1c](https://github.com/gynecoloji/spatioloji_s/commit/99f8a1c0267c43333dde705ef210da01dec83a6e))
* add interface cells implementation plan ([67058a4](https://github.com/gynecoloji/spatioloji_s/commit/67058a471ff88edc5055428066e92e7696290822))
* add proximity contact mode design spec ([ec7728a](https://github.com/gynecoloji/spatioloji_s/commit/ec7728a11a9badd190083f76e5b4a5664684ea33))
* add spatial motifs design spec ([c1aa024](https://github.com/gynecoloji/spatioloji_s/commit/c1aa024894b05b4c8a73d85d38a1378ffb83ff1a))
* add spatial motifs implementation plan ([2f69288](https://github.com/gynecoloji/spatioloji_s/commit/2f6928879e5c0f46375a43c670d0f1de2c568c8f))
* regenerate documentation for refactored package ([90cd377](https://github.com/gynecoloji/spatioloji_s/commit/90cd37746f5f5526c0b84d4a613f6546522323b7))
* remove Sphinx docs and Read the Docs integration ([cfe94ec](https://github.com/gynecoloji/spatioloji_s/commit/cfe94ec1547ee1f48de0f99261db2b73950b2b1e))
* revise DEG spec with reviewer feedback ([a566400](https://github.com/gynecoloji/spatioloji_s/commit/a56640033db347479b0dbf8703d59c2b314686a0))
* **spatial/polygon:** add docstrings to morphology numba functions ([404ea07](https://github.com/gynecoloji/spatioloji_s/commit/404ea07ee72306db79698d3525b3680ea385d17a))
* update README with CCC 3-layer framework, shape metrics, and comparison tables ([72d0d97](https://github.com/gynecoloji/spatioloji_s/commit/72d0d97086ef2f7d058cc63a1fd32a303b3c4d4b))

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
