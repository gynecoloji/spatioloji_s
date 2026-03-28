# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "spatioloji_s"
copyright = "2024-2026, Ji Wang"
author = "Ji Wang"
release = "0.3.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "superpowers", ".ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "logo_only": False,
}
html_title = "spatioloji_s"
html_short_title = "spatioloji_s"

# -- Extension configuration -------------------------------------------------

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autosummary settings
autosummary_generate = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_mock_imports = [
    "anndata",
    "scanpy",
    "leidenalg",
    "igraph",
    "umap",
    "harmonypy",
    "combat",
    "magic",
    "scvi",
    "plotly",
    "cv2",
    "decoupler",
    "pydeseq2",
    "pynndescent",
    "pointpats",
    "DCA",
    "cuml",
    "cupy",
]

# MyST settings (Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}
