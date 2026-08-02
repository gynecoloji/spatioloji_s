"""
decoupler.py - Gene set activity inference via decoupler-py

Wraps decoupler-py methods (AUCell, ULM, MLM, WMEAN) to score gene set
activity per cell.  Supports pre-defined collections (PROGENy pathways,
DoRothEA / CollecTRI transcription factors, MSigDB, KEGG, etc.) and
user-supplied custom gene sets.

All results are stored in ``sp._embeddings`` under a configurable key
and optionally added to ``sp.cell_meta`` for easy access.

Requires: ``pip install decoupler`` (or ``pip install spatioloji_s[decoupler]``)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Column name aliases used by different decoupler resources
_SOURCE_ALIASES = {"geneset", "gene_set", "term", "pathway", "tf", "collection_name", "gs_name"}
_TARGET_ALIASES = {"genesymbol", "gene_symbol", "gene", "target_gene", "gene_id", "gs_gene"}


def _normalize_net_columns(net: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard (source, target, weight) format.

    Different decoupler resources return varying column names
    (e.g., 'geneset'/'genesymbol' for MSigDB vs 'source'/'target'
    for PROGENy).  This normalizes them so downstream code works
    uniformly.
    """
    col_map = {}

    # Already standard
    if "source" in net.columns and "target" in net.columns:
        return net

    # Find source column
    for alias in _SOURCE_ALIASES:
        matches = [c for c in net.columns if c.lower() == alias]
        if matches:
            col_map[matches[0]] = "source"
            break

    # Find target column
    for alias in _TARGET_ALIASES:
        matches = [c for c in net.columns if c.lower() == alias]
        if matches:
            col_map[matches[0]] = "target"
            break

    if col_map:
        net = net.rename(columns=col_map)

    # Ensure source and target exist
    if "source" not in net.columns or "target" not in net.columns:
        print(f"  Warning: could not identify source/target columns. Found: {list(net.columns)}")

    return net


# ---------------------------------------------------------------------------
# Public helpers — gene set loading
# ---------------------------------------------------------------------------


def load_gene_sets(
    collection: str = "progeny",
    organism: str = "human",
    top: int | None = None,
    levels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load a pre-defined gene set collection via decoupler-py.

    Parameters
    ----------
    collection : str
        Which collection to load.  Supported shortcuts:

        - ``'progeny'`` — 14 signaling pathways (PROGENy).
        - ``'dorothea'`` — transcription factor → target interactions.
        - ``'collectri'`` — comprehensive TF network (12 resources).
        - ``'msigdb'`` — Molecular Signatures Database (broad).
        - ``'kegg'`` — KEGG pathway gene sets.
        - ``'reactome'`` — Reactome pathway gene sets.
        - Any name accepted by ``decoupler.get_resource()``.
    organism : str
        ``'human'`` or ``'mouse'``, by default ``'human'``.
    top : int or None
        For PROGENy: limit to top N genes per pathway.
    levels : list of str or None
        For DoRothEA: confidence levels to include, e.g.
        ``['A', 'B', 'C']``.  Default ``['A', 'B']``.

    Returns
    -------
    pd.DataFrame
        Long-format network with columns ``source``, ``target``,
        and optionally ``weight``.

    Examples
    --------
    >>> net = load_gene_sets('progeny')
    >>> net = load_gene_sets('dorothea', levels=['A', 'B', 'C'])
    >>> net = load_gene_sets('msigdb', organism='mouse')
    """
    dc = _import_decoupler()
    _op = _get_dc_ops(dc)

    collection_lower = collection.lower()

    if collection_lower == "progeny":
        kwargs = {"organism": organism}
        if top is not None:
            kwargs["top"] = top
        net = _op["progeny"](**kwargs)
        print(f"  Loaded PROGENy: {net['source'].nunique()} pathways, {len(net)} interactions")

    elif collection_lower == "dorothea":
        if levels is None:
            levels = ["A", "B"]
        net = _op["dorothea"](organism=organism, levels=levels)
        print(
            f"  Loaded DoRothEA (levels={levels}): "
            f"{net['source'].nunique()} TFs, {len(net)} interactions"
        )

    elif collection_lower == "collectri":
        net = _op["collectri"](organism=organism)
        print(f"  Loaded CollecTRI: {net['source'].nunique()} TFs, {len(net)} interactions")

    else:
        # Generic resource via decoupler
        net = _op["resource"](collection, organism=organism)
        # Decoupler resources may use different column names depending on
        # the collection.  Normalize to the standard (source, target, weight)
        # format so downstream code works uniformly.
        net = _normalize_net_columns(net)
        print(f"  Loaded '{collection}': {net['source'].nunique()} sets, {len(net)} interactions")

    return net


def make_gene_set_net(
    gene_sets: dict[str, list[str]],
    weights: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """
    Convert a custom gene set dictionary to decoupler network format.

    Parameters
    ----------
    gene_sets : dict
        ``{set_name: [gene1, gene2, ...]}`` mapping.
    weights : dict or None
        ``{set_name: [w1, w2, ...]}`` matching gene_sets.
        If None, all weights are set to 1.0.

    Returns
    -------
    pd.DataFrame
        Long-format network with columns ``source``, ``target``, ``weight``.

    Examples
    --------
    >>> net = make_gene_set_net({
    ...     'EMT': ['VIM', 'CDH2', 'SNAI1', 'ZEB1'],
    ...     'Proliferation': ['MKI67', 'TOP2A', 'PCNA'],
    ... })
    """
    rows = []
    for name, genes in gene_sets.items():
        ws = weights[name] if weights is not None else [1.0] * len(genes)
        if len(ws) != len(genes):
            raise ValueError(f"Weight length mismatch for '{name}': {len(ws)} weights vs {len(genes)} genes")
        for gene, w in zip(genes, ws, strict=True):
            rows.append({"source": name, "target": gene, "weight": w})

    net = pd.DataFrame(rows, columns=["source", "target", "weight"])
    print(f"  Custom gene sets: {net['source'].nunique()} sets, {len(net)} gene-set pairs")
    return net


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------


def score_gene_sets(
    sp: spatioloji,
    net: pd.DataFrame | dict[str, list[str]],
    method: str = "aucell",
    layer: str | None = "log_normalized",
    min_n: int = 5,
    n_permutations: int = 1000,
    seed: int = 42,
    store: bool = True,
    output_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Score gene set activity per cell using decoupler-py.

    Parameters
    ----------
    sp : spatioloji
        spatioloji object with expression data.
    net : pd.DataFrame or dict
        Gene set network.  Can be:

        - A decoupler-format DataFrame (columns: ``source``, ``target``,
          optionally ``weight``) from :func:`load_gene_sets`.
        - A plain dict ``{set_name: [gene1, gene2, ...]}`` — auto-converted
          via :func:`make_gene_set_net`.
    method : str
        Scoring method:

        - ``'aucell'`` — rank-based, no weights needed, no p-values.
          Best for unweighted gene sets (MSigDB, custom lists).
        - ``'ulm'`` — univariate linear model, returns t-values + p-values.
          Good for weighted networks (DoRothEA, PROGENy).
        - ``'mlm'`` — multivariate linear model, all regulators together.
          Most accurate for overlapping networks, slowest.
        - ``'wmean'`` — weighted mean with permutation p-values.
          Robust, fast, good default for weighted networks.
        - ``'consensus'`` — runs ulm + mlm + wmean and returns consensus.
    layer : str or None
        Expression layer to use, by default ``'log_normalized'``.
        None uses raw expression.
    min_n : int
        Minimum number of target genes per source to keep, by default 5.
    n_permutations : int
        Number of permutations for wmean p-values, by default 1000.
    seed : int
        Random seed, by default 42.
    store : bool
        If True, store results in ``sp._embeddings[output_key]`` and
        add activity scores to ``sp.cell_meta``.
    output_key : str or None
        Key for storing in embeddings.  If None, auto-generates from
        method name (e.g. ``'aucell_estimate'``).

    Returns
    -------
    dict with keys:
        - ``'estimate'`` : pd.DataFrame (n_cells × n_sources) — activity scores
        - ``'pvals'`` : pd.DataFrame or None — p-values (not available for AUCell)

    Raises
    ------
    ImportError
        If ``decoupler`` is not installed.
    ValueError
        If no genes overlap between expression and network.

    Examples
    --------
    >>> # AUCell with PROGENy pathways
    >>> net = sj.processing.load_gene_sets('progeny')
    >>> result = sj.processing.score_gene_sets(sp, net, method='aucell')
    >>>
    >>> # Custom gene sets
    >>> result = sj.processing.score_gene_sets(sp, {
    ...     'EMT': ['VIM', 'CDH2', 'SNAI1'],
    ...     'Immune': ['CD3D', 'CD8A', 'GZMB'],
    ... })
    >>>
    >>> # ULM with DoRothEA TF activities
    >>> net = sj.processing.load_gene_sets('dorothea', levels=['A', 'B'])
    >>> result = sj.processing.score_gene_sets(sp, net, method='ulm')
    >>>
    >>> # Consensus scoring
    >>> result = sj.processing.score_gene_sets(sp, net, method='consensus')
    """
    valid_methods = {"aucell", "ulm", "mlm", "wmean", "consensus"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {sorted(valid_methods)}")

    dc = _import_decoupler()

    # Auto-convert dict to network DataFrame
    if isinstance(net, dict):
        net = make_gene_set_net(net)

    # Validate network columns
    if "source" not in net.columns or "target" not in net.columns:
        raise ValueError("Network must have 'source' and 'target' columns")

    # Get expression matrix — keep sparse when possible (avoid .toarray())
    print(f"\n[Decoupler] Scoring gene set activity (method={method})")
    if layer is None:
        X = sp.expression.get_dense()
    else:
        X = sp.get_layer(layer)
        # Keep sparse — decoupler handles it internally

    gene_names = list(sp.gene_index)
    cell_ids = list(sp.cell_index)
    n_cells = len(cell_ids)

    # Subset to only genes in the network (avoids building huge DataFrame)
    net_genes = set(net["target"].unique())
    expr_genes = set(gene_names)
    overlap = sorted(net_genes & expr_genes)
    n_sources = net.loc[net["target"].isin(overlap), "source"].nunique()

    if len(overlap) == 0:
        raise ValueError(
            f"No genes overlap between expression ({len(expr_genes)} genes) "
            f"and network ({len(net_genes)} targets). Check gene name format."
        )

    print(f"  Gene overlap: {len(overlap)}/{len(net_genes)} network targets found in expression")
    print(f"  Active sources (>= {min_n} targets): {n_sources}")
    print(f"  Subsetting expression to {len(overlap)} overlapping genes (saves memory)")

    # Build a slim expression DataFrame with only overlapping genes
    gene_idx = [gene_names.index(g) for g in overlap]
    X_sub = X[:, gene_idx]
    if sparse.issparse(X_sub):
        X_sub = X_sub.toarray()
    expr_df = pd.DataFrame(X_sub, index=cell_ids, columns=overlap)

    # Also filter network to overlapping genes
    net = net[net["target"].isin(overlap)].copy()

    # Ensure weight column exists for methods that need it
    if method in ("ulm", "mlm", "wmean", "consensus") and "weight" not in net.columns:
        net["weight"] = 1.0

    # Resolve decoupler API — v2 uses dc.mt.*, v1 uses dc.run_*
    _run = _get_dc_runners(dc)
    is_v2 = hasattr(dc, "mt")

    # v1 uses min_n, v2 uses tmin
    tmin_kw = {"tmin": min_n} if is_v2 else {"min_n": min_n}

    # Run scoring
    estimate = None
    pvals = None

    if method == "aucell":
        if is_v2:
            estimate, _ = _run["aucell"](expr_df, net, verbose=False, **tmin_kw)
        else:
            estimate, _ = _run["aucell"](expr_df, net, seed=seed, verbose=False, **tmin_kw)

    elif method == "ulm":
        estimate, pvals = _run["ulm"](expr_df, net, verbose=False, **tmin_kw)

    elif method == "mlm":
        estimate, pvals = _run["mlm"](expr_df, net, verbose=False, **tmin_kw)

    elif method == "wmean":
        estimate, pvals = _run["wmean"](
            expr_df, net, times=n_permutations, seed=seed, verbose=False, **tmin_kw,
        )

    elif method == "consensus":
        if is_v2:
            results = _run["decouple"](
                expr_df, net, methods=["mlm", "ulm", "waggr"], cons=True, verbose=False, **tmin_kw,
            )
        else:
            results = _run["decouple"](
                expr_df, net, methods=["mlm", "ulm", "wmean"], consensus=True, verbose=False, **tmin_kw,
            )
        estimate = results.get("consensus_estimate")
        pvals = results.get("consensus_pvals")

    # Free the sliced expression DataFrame
    del expr_df

    # Align index to sp.cell_index
    estimate = estimate.reindex(sp.cell_index)
    if pvals is not None:
        pvals = pvals.reindex(sp.cell_index)

    n_sources_out = estimate.shape[1]
    print(f"  ✓ Scored {n_sources_out} gene sets across {n_cells} cells")

    # Store results
    if store:
        if not hasattr(sp, "_embeddings"):
            sp._embeddings = {}

        key = output_key or f"{method}_estimate"
        sp._embeddings[key] = estimate.values
        sp._embeddings[f"{key}_names"] = np.array(estimate.columns.tolist())

        if pvals is not None:
            sp._embeddings[f"{method}_pvals"] = pvals.values

        # Batch-assign to cell_meta (faster than column-by-column)
        prefixed = estimate.rename(columns=lambda c: f"{method}_{c}")
        for col in prefixed.columns:
            sp._cell_meta[col] = prefixed[col].values

        print(f"  ✓ Stored in embeddings['{key}'] and cell_meta['{method}_*']")

    return {"estimate": estimate, "pvals": pvals}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _import_decoupler():
    """Lazy import with actionable error."""
    try:
        import decoupler
    except ImportError as err:
        raise ImportError(
            "decoupler is required for gene set activity scoring. "
            "Install with: pip install spatioloji_s[decoupler]"
        ) from err
    return decoupler


def _get_dc_ops(dc):
    """Resolve decoupler resource loaders across v1 and v2 API.

    v1: ``dc.get_progeny``, ``dc.get_dorothea``, etc.
    v2: ``dc.op.progeny``, ``dc.op.dorothea``, etc.
    """
    if hasattr(dc, "op"):
        op = dc.op
        return {
            "progeny": op.progeny,
            "dorothea": op.dorothea,
            "collectri": op.collectri,
            "resource": op.resource if hasattr(op, "resource") else dc.get_resource,
        }
    return {
        "progeny": dc.get_progeny,
        "dorothea": dc.get_dorothea,
        "collectri": dc.get_collectri,
        "resource": dc.get_resource,
    }


def _get_dc_runners(dc):
    """Resolve decoupler method functions across v1 and v2 API.

    v1 (<=1.9.x): ``dc.run_aucell``, ``dc.run_ulm``, etc.
    v2 (>=2.0):   ``dc.mt.aucell``, ``dc.mt.ulm``, etc.

    Returns
    -------
    dict
        ``{method_name: callable}`` mapping.
    """
    # Try v2 API first (dc.mt.*)
    if hasattr(dc, "mt"):
        mt = dc.mt
        # v2 renamed wmean → waggr
        wmean_fn = mt.waggr if hasattr(mt, "waggr") else getattr(mt, "wmean", None)
        decouple_fn = mt.decouple if hasattr(mt, "decouple") else dc.decouple
        return {
            "aucell": mt.aucell,
            "ulm": mt.ulm,
            "mlm": mt.mlm,
            "wmean": wmean_fn,
            "decouple": decouple_fn,
        }

    # Fall back to v1 API (dc.run_*)
    return {
        "aucell": dc.run_aucell,
        "ulm": dc.run_ulm,
        "mlm": dc.run_mlm,
        "wmean": dc.run_wmean,
        "decouple": dc.decouple,
    }
