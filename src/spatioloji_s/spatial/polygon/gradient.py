"""Spatial gradient analysis for polygon-based spatial data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

from spatioloji_s.spatial._distance_utils import signed_distance_to_interface
from spatioloji_s.spatial._gradient_types import GradientResult
from spatioloji_s.spatial._interface_types import InterfaceResult


def _fit_gradient(values: np.ndarray, distances: np.ndarray) -> dict:
    """Fit linear regression of values ~ distances.

    Args:
        values: Expression values (1-D array).
        distances: Signed distances (1-D array, same length).

    Returns:
        Dict with keys coef, pvalue, r2, trend.
    """
    mask = np.isfinite(values) & np.isfinite(distances)
    if mask.sum() < 3:
        return {"coef": np.nan, "pvalue": np.nan, "r2": np.nan, "trend": "flat"}

    result = linregress(distances[mask], values[mask])
    trend = "flat"
    if result.pvalue < 0.05:
        trend = "increasing_toward_a" if result.slope > 0 else "increasing_toward_b"
    return {
        "coef": result.slope,
        "pvalue": result.pvalue,
        "r2": result.rvalue ** 2,
        "trend": trend,
    }


def _bin_expression(
    expr_df: pd.DataFrame,
    distances: pd.Series,
    genes: list[str],
    n_bins: int,
) -> pd.DataFrame:
    """Bin expression by distance for plotting.

    Args:
        expr_df: Expression DataFrame (cells x genes).
        distances: Signed distances indexed by cell ID.
        genes: Gene names to include.
        n_bins: Number of equal-width bins.

    Returns:
        Long-form DataFrame: distance_bin, gene, mean_expr, std_expr.
    """
    bin_edges = np.linspace(distances.min(), distances.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_labels = np.digitize(distances.values, bin_edges[1:-1])  # 0-indexed bins

    rows = []
    for gene in genes:
        gene_vals = expr_df[gene].values
        for b in range(n_bins):
            mask = bin_labels == b
            if mask.sum() == 0:
                continue
            rows.append({
                "distance_bin": bin_centers[b],
                "gene": gene,
                "mean_expr": float(np.mean(gene_vals[mask])),
                "std_expr": float(np.std(gene_vals[mask])),
            })
    return pd.DataFrame(rows)


def _discover_programs_nmf(expr_matrix: np.ndarray, gene_names: list[str], n_programs: int) -> dict[str, list[str]]:
    """Discover gene programs via NMF.

    Args:
        expr_matrix: Dense expression matrix (cells x genes).
        gene_names: Gene names.
        n_programs: Number of programs.

    Returns:
        Dict of {program_name: [top_gene_names]}.
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required for auto_programs='nmf'. "
            "Install with: pip install scikit-learn"
        ) from err

    X = np.maximum(expr_matrix, 0)
    n_programs = min(n_programs, min(X.shape))
    model = NMF(n_components=n_programs, random_state=42, max_iter=500)
    model.fit(X)

    programs = {}
    n_top = max(3, len(gene_names) // n_programs)
    for i, component in enumerate(model.components_):
        top_idx = np.argsort(component)[::-1][:n_top]
        programs[f"NMF_{i}"] = [gene_names[j] for j in top_idx]
    return programs


def _discover_programs_pca(expr_matrix: np.ndarray, gene_names: list[str], n_programs: int) -> dict[str, list[str]]:
    """Discover gene programs via PCA loadings.

    Args:
        expr_matrix: Dense expression matrix (cells x genes).
        gene_names: Gene names.
        n_programs: Number of programs.

    Returns:
        Dict of {program_name: [top_gene_names]}.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required for auto_programs='pca'. "
            "Install with: pip install scikit-learn"
        ) from err

    n_programs = min(n_programs, min(expr_matrix.shape))
    model = PCA(n_components=n_programs, random_state=42)
    model.fit(expr_matrix)

    programs = {}
    n_top = max(3, len(gene_names) // n_programs)
    for i, component in enumerate(model.components_):
        top_idx = np.argsort(np.abs(component))[::-1][:n_top]
        programs[f"PC_{i}"] = [gene_names[j] for j in top_idx]
    return programs


def compute_gradient(
    sp,
    interface_result: InterfaceResult,
    genes: list[str] | None = None,
    programs: dict[str, list[str]] | None = None,
    n_bins: int = 20,
    method: str = "ols",
    auto_programs: str | None = None,
    n_auto_programs: int = 5,
    coord_type: str = "global",
    unsigned: bool = False,
) -> GradientResult:
    """Compute spatial expression gradient relative to an interface.

    Fits linear regression of gene expression vs signed distance from
    the interface contour for each gene and/or gene program.

    Args:
        sp: spatioloji object.
        interface_result: Result from ``identify_interface``.
        genes: List of gene names to analyze. ``None`` = all genes.
        programs: Dict of ``{name: [gene_list]}`` for user-defined
            gene modules. ``None`` = skip user programs.
        n_bins: Number of equal-width distance bins for curves.
        method: Regression method. Currently only ``"ols"`` supported.
        auto_programs: ``"nmf"`` or ``"pca"`` to auto-discover gene
            programs. ``None`` = skip auto-discovery.
        n_auto_programs: Number of programs to discover (default 5).
        coord_type: ``'global'`` or ``'local'`` coordinates.
        unsigned: If True, use absolute distances.

    Returns:
        GradientResult with gene/program gradients and binned data.

    Raises:
        ValueError: If genes not found or invalid auto_programs.
    """
    if method != "ols":
        raise ValueError(f"method must be 'ols', got '{method}'")
    if auto_programs is not None and auto_programs not in ("nmf", "pca"):
        raise ValueError(f"auto_programs must be 'nmf', 'pca', or None, got '{auto_programs}'")

    distances = signed_distance_to_interface(
        sp, interface_result, coord_type=coord_type, unsigned=unsigned,
    )

    expr_df = sp.expression.to_dataframe()

    all_genes = list(expr_df.columns)
    if genes is None:
        genes = all_genes
    else:
        missing = [g for g in genes if g not in all_genes]
        if missing:
            raise ValueError(f"Genes not found in expression matrix: {missing}")

    gene_rows = []
    dist_arr = distances.values
    for gene in genes:
        vals = expr_df[gene].values
        gene_rows.append({"gene": gene, **_fit_gradient(vals, dist_arr)})
    gene_gradients = pd.DataFrame(gene_rows).set_index("gene")

    all_programs = dict(programs) if programs else {}

    if auto_programs == "nmf":
        auto = _discover_programs_nmf(
            sp.expression.get_dense(), all_genes, n_auto_programs,
        )
        all_programs.update(auto)
    elif auto_programs == "pca":
        auto = _discover_programs_pca(
            sp.expression.get_dense(), all_genes, n_auto_programs,
        )
        all_programs.update(auto)

    program_rows = []
    program_score_dict = {}
    for pname, pgenes in all_programs.items():
        valid = [g for g in pgenes if g in all_genes]
        if not valid:
            continue
        scores = expr_df[valid].mean(axis=1).values
        program_score_dict[pname] = scores
        program_rows.append({"program": pname, **_fit_gradient(scores, dist_arr)})

    if program_rows:
        program_gradients = pd.DataFrame(program_rows).set_index("program")
        program_scores = pd.DataFrame(program_score_dict, index=expr_df.index)
    else:
        program_gradients = pd.DataFrame(columns=["coef", "pvalue", "r2", "trend"])
        program_scores = pd.DataFrame(index=expr_df.index)

    bins = _bin_expression(expr_df, distances, genes, n_bins)

    if program_score_dict:
        prog_df = pd.DataFrame(program_score_dict, index=expr_df.index)
        prog_bins = _bin_expression(prog_df, distances, list(program_score_dict.keys()), n_bins)
        bins = pd.concat([bins, prog_bins], ignore_index=True)

    return GradientResult(
        distances=distances,
        gene_gradients=gene_gradients,
        program_gradients=program_gradients,
        program_scores=program_scores,
        bins=bins,
        region_a=interface_result.region_a,
        region_b=interface_result.region_b,
    )
