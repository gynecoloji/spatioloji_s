Installation
============

Requirements
------------

- Python >= 3.12
- Core dependencies are installed automatically: numpy, pandas, scipy,
  scikit-learn, matplotlib, seaborn, geopandas, shapely, networkx, opencv-python

From PyPI (recommended)
-----------------------

.. code-block:: bash

   pip install spatioloji-s

Optional extras
---------------

spatioloji_s uses optional dependencies for specialized functionality:

.. code-block:: bash

   # Leiden clustering (leidenalg + igraph)
   pip install "spatioloji-s[clustering]"

   # UMAP dimensionality reduction
   pip install "spatioloji-s[reduction]"

   # Batch correction (Harmony, ComBat)
   pip install "spatioloji-s[batch]"

   # Differential expression (statsmodels, pydeseq2)
   pip install "spatioloji-s[deg]"

   # AnnData/scanpy interoperability
   pip install "spatioloji-s[anndata]"

   # Ripley's K/L statistics
   pip install "spatioloji-s[ripley]"

   # Pathway scoring (decoupler)
   pip install "spatioloji-s[decoupler]"

   # Everything
   pip install "spatioloji-s[all]"

Development installation
------------------------

.. code-block:: bash

   git clone https://github.com/gynecoloji/spatioloji_s.git
   cd spatioloji_s
   pip install -e ".[test]"

Verifying installation
----------------------

.. code-block:: python

   import spatioloji_s as sj
   print(sj.__version__)
