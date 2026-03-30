# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata

# -- Project information -----------------------------------------------------
project = "spatioloji_s"
author = "Ji Wang"
copyright = "2024-2026, Ji Wang"
release = importlib.metadata.version("spatioloji_s")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Napoleon settings (Google-style docstrings) -----------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True

# -- Autodoc settings --------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- Autosummary settings ----------------------------------------------------
autosummary_generate = True

# -- Mock imports (not available at doc build time) --------------------------
autodoc_mock_imports = [
    "cv2",
    "geopandas",
    "shapely",
    "networkx",
    "natsort",
    "tqdm",
    "joblib",
    "leidenalg",
    "igraph",
    "umap",
    "harmonypy",
    "combat",
    "magic",
    "scvi",
    "scanpy",
    "anndata",
    "plotly",
    "decoupler",
    "pointpats",
    "pynndescent",
    "statsmodels",
    "pydeseq2",
    "DCA",
]

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

# -- nbsphinx settings -------------------------------------------------------
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# -- HTML output --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
    "logo_only": False,
}
html_static_path = ["_static"]
html_title = f"spatioloji_s {release}"
html_show_sourcelink = False

# -- Copy button settings ----------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
