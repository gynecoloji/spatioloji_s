"""Immune infiltration scoring for polygon-based spatial data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

from spatioloji_s.spatial._distance_utils import signed_distance_to_interface
from spatioloji_s.spatial._infiltration_types import InfiltrationResult
from spatioloji_s.spatial._interface_types import InterfaceResult


def score_infiltration(
    sp,
    interface_result: InterfaceResult,
    immune_col: str,
    immune_types: list[str],
    target_region: str | None = None,
    depth_bins: int = 10,
    coord_type: str = "global",
) -> InfiltrationResult:
    """Score immune cell infiltration across a spatial interface.

    Quantifies how deeply immune cells penetrate into a target region,
    computing penetration depth, density gradient, and infiltration
    fraction for each immune cell type.

    Args:
        sp: spatioloji object.
        interface_result: Result from ``identify_interface``.
        immune_col: Column in ``sp.cell_meta`` with cell type labels.
        immune_types: List of cell type labels considered immune.
        target_region: Region immune cells infiltrate into
            (``region_a`` or ``region_b`` label). ``None`` = auto-detect
            as the region with fewer immune cells.
        depth_bins: Number of distance bins for density gradient.
        coord_type: ``'global'`` or ``'local'`` coordinates.

    Returns:
        InfiltrationResult with per-type metrics and cell classifications.

    Raises:
        ValueError: If immune_col not found, target_region invalid.

    Example:
        >>> from spatioloji_s.spatial.polygon.infiltration import score_infiltration
        >>> result = score_infiltration(
        ...     sp, interface_result,
        ...     immune_col="immune_type", immune_types=["CD8_T"],
        ...     target_region="TypeA",
        ... )
    """
    if immune_col not in sp.cell_meta.columns:
        raise ValueError(f"'{immune_col}' not found in cell_meta. Available: {list(sp.cell_meta.columns)}")

    cell_types = sp.cell_meta[immune_col]
    labels = interface_result.cell_labels

    region_a = interface_result.region_a
    region_b = interface_result.region_b
    a_list = [region_a] if isinstance(region_a, str) else list(region_a)
    b_list = [region_b] if isinstance(region_b, str) else list(region_b)

    if target_region is not None:
        if target_region not in a_list + b_list:
            raise ValueError(f"target_region '{target_region}' not in region_a={region_a} or region_b={region_b}")
        target_is_a = target_region in a_list
    else:
        immune_mask = cell_types.isin(immune_types)
        a_mask = labels.isin(["region_a_interface", "interior_a"])
        b_mask = labels.isin(["region_b_interface", "interior_b"])
        n_immune_a = (immune_mask & a_mask).sum()
        n_immune_b = (immune_mask & b_mask).sum()
        target_is_a = n_immune_a <= n_immune_b
        target_region = a_list[0] if target_is_a else b_list[0]

    distances = signed_distance_to_interface(
        sp,
        interface_result,
        coord_type=coord_type,
    )

    if target_is_a:
        target_mask = labels.isin(["region_a_interface", "interior_a"])
    else:
        target_mask = labels.isin(["region_b_interface", "interior_b"])

    immune_mask = cell_types.isin(immune_types)
    classifications = pd.Series("other", index=sp.cell_index)
    classifications[immune_mask & target_mask] = "infiltrating"
    classifications[immune_mask & ~target_mask] = "resident"

    metric_rows = []
    for itype in immune_types:
        type_mask = cell_types == itype
        type_in_target = type_mask & target_mask
        type_not_in_target = type_mask & ~target_mask

        n_infiltrating = int(type_in_target.sum())
        n_resident = int(type_not_in_target.sum())
        n_total = n_infiltrating + n_resident

        if n_infiltrating > 0:
            depths = distances[type_in_target].abs().values
            median_depth = float(np.median(depths))
            max_depth = float(np.max(depths))
        else:
            median_depth = 0.0
            max_depth = 0.0

        infiltration_fraction = n_infiltrating / n_total if n_total > 0 else 0.0

        type_distances = distances[type_mask].values
        if len(type_distances) >= 3:
            bin_edges = np.linspace(
                type_distances.min(),
                type_distances.max(),
                depth_bins + 1,
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            counts, _ = np.histogram(type_distances, bins=bin_edges)
            if len(bin_centers) >= 3:
                result = linregress(bin_centers, counts.astype(float))
                density_slope = result.slope
                density_pvalue = result.pvalue
            else:
                density_slope = np.nan
                density_pvalue = np.nan
        else:
            density_slope = np.nan
            density_pvalue = np.nan

        metric_rows.append(
            {
                "immune_type": itype,
                "median_depth": median_depth,
                "max_depth": max_depth,
                "density_slope": density_slope,
                "density_pvalue": density_pvalue,
                "infiltration_fraction": infiltration_fraction,
                "n_infiltrating": n_infiltrating,
                "n_resident": n_resident,
            }
        )

    per_type_metrics = pd.DataFrame(metric_rows).set_index("immune_type")

    return InfiltrationResult(
        distances=distances,
        cell_classifications=classifications,
        per_type_metrics=per_type_metrics,
        region_a=region_a,
        region_b=region_b,
        target_region=target_region,
    )
