# Installation

## Requirements

- Python >= 3.12
- Core dependencies are installed automatically: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, geopandas, shapely, networkx, opencv-python

## From PyPI (recommended)

```bash
pip install spatioloji-s
```

## Optional extras

spatioloji_s uses optional dependencies for specialized functionality:

```bash
# Leiden clustering (leidenalg + igraph)
pip install "spatioloji-s[clustering]"

# UMAP dimensionality reduction
pip install "spatioloji-s[reduction]"

# Batch correction (Harmony, ComBat)
pip install "spatioloji-s[batch]"

# AnnData/scanpy interoperability
pip install "spatioloji-s[anndata]"

# Everything
pip install "spatioloji-s[all]"
```

## MAGIC imputation

MAGIC requires a separate conda environment due to dependency conflicts:

```bash
conda create -n spatioloji_magic python=3.12
conda activate spatioloji_magic
pip install magic-impute spatioloji-s
```

## Development installation

```bash
git clone https://github.com/gynecoloji/spatioloji_s.git
cd spatioloji_s
pip install -e ".[test]"
```

## Verifying installation

```python
import spatioloji_s as sj
print(sj.__version__)
```
