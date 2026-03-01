"""
morphology.py - Cell shape features from polygon geometry

Extracts biologically meaningful shape descriptors per cell.
All metrics are computed from polygon geometry and stored
back into sp.cell_meta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji


import numpy as np
import pandas as pd

from .graph import _get_gdf


def compute_morphology(
    sp: spatioloji, coord_type: str = "global", metrics: list[str] | None = None, store: bool = True
) -> pd.DataFrame:
    """
    Compute cell shape metrics from polygon geometry.

    Metrics computed:
    - area: polygon area
    - perimeter: polygon perimeter length
    - circularity: 4π × area / perimeter² (1.0 = perfect circle)
    - elongation: major_axis / minor_axis of fitted ellipse (1.0 = round)
    - solidity: area / convex_hull_area (1.0 = no concavities)
    - compactness: sqrt(4 × area / π) / major_axis
    - convexity: convex_hull_perimeter / perimeter

    Parameters
    ----------
    sp : spatioloji
        spatioloji object with polygon data
    coord_type : str
        'global' or 'local' coordinates
    metrics : list of str, optional
        Specific metrics to compute. If None, compute all.
    store : bool
        If True, store results in sp.cell_meta with 'morph_' prefix

    Returns
    -------
    pd.DataFrame
        Shape metrics per cell, indexed by cell_index

    Examples
    --------
    >>> from spatioloji_s.spatial.polygon.morphology import compute_morphology
    >>> morph = compute_morphology(sp)
    >>> print(morph[['area', 'circularity', 'elongation']].describe())
    >>>
    >>> # Access via cell_meta after store=True
    >>> sp.cell_meta['morph_circularity']
    """

    all_metrics = ["area", "perimeter", "circularity", "elongation", "solidity", "compactness", "convexity"]

    if metrics is not None:
        invalid = set(metrics) - set(all_metrics)
        if invalid:
            raise ValueError(f"Unknown metrics: {invalid}. " f"Available: {all_metrics}")
        compute = metrics
    else:
        compute = all_metrics

    print(f"\n[Morphology] Computing shape metrics ({len(compute)} metrics)...")

    gdf = _get_gdf(sp, coord_type=coord_type)

    # Initialize result arrays
    n_cells = len(gdf)
    results = {m: np.full(n_cells, np.nan) for m in compute}

    for i, (_cell_id, row) in enumerate(gdf.iterrows()):
        geom = row.geometry

        # Skip invalid/missing
        if geom is None or geom.is_empty or not geom.is_valid:
            continue

        poly_area = geom.area
        poly_perimeter = geom.length

        if "area" in compute:
            results["area"][i] = poly_area

        if "perimeter" in compute:
            results["perimeter"][i] = poly_perimeter

        if "circularity" in compute:
            if poly_perimeter > 0:
                results["circularity"][i] = 4.0 * np.pi * poly_area / (poly_perimeter**2)

        # Elongation and compactness need minimum rotated rectangle
        if any(m in compute for m in ["elongation", "compactness"]):
            try:
                mrr = geom.minimum_rotated_rectangle
                coords = np.array(mrr.exterior.coords)
                edge_lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
                major = edge_lengths.max()
                minor = edge_lengths.min()

                if "elongation" in compute and minor > 0:
                    results["elongation"][i] = major / minor

                if "compactness" in compute and major > 0:
                    results["compactness"][i] = np.sqrt(4.0 * poly_area / np.pi) / major
            except Exception:
                pass

        if "solidity" in compute:
            convex = geom.convex_hull
            convex_area = convex.area
            if convex_area > 0:
                results["solidity"][i] = poly_area / convex_area

        if "convexity" in compute:
            convex = geom.convex_hull
            convex_perimeter = convex.length
            if poly_perimeter > 0:
                results["convexity"][i] = convex_perimeter / poly_perimeter

    # Build DataFrame aligned to cell_index
    morph_df = pd.DataFrame(results, index=gdf.index)
    morph_df = morph_df.reindex(sp.cell_index)

    # Store in cell_meta
    if store:
        for col in morph_df.columns:
            sp.cell_meta[f"morph_{col}"] = morph_df[col].values
        print("  ✓ Stored in cell_meta with 'morph_' prefix")

    n_valid = morph_df.notna().all(axis=1).sum()
    print(f"  ✓ Computed {len(compute)} metrics for {n_valid}/{len(sp.cell_index)} cells")

    return morph_df


def classify_morphology(
    sp: spatioloji,
    method: str = "threshold",
    n_classes: int = 3,
    circularity_thresh: float = 0.7,
    elongation_thresh: float = 2.0,
    solidity_thresh: float = 0.85,
    store: bool = True,
) -> pd.Series:
    """
    Classify cells by shape into morphological categories.

    Methods:
    - 'threshold': Rule-based using circularity, elongation, solidity.
        Categories: 'round', 'elongated', 'irregular'
    - 'kmeans': Data-driven clustering on all morphology metrics.
        Categories: integer labels (0, 1, ..., n_classes-1)

    Parameters
    ----------
    sp : spatioloji
        spatioloji object. Must have morph_ columns in cell_meta
        (run compute_morphology first).
    method : str
        'threshold' or 'kmeans'
    n_classes : int
        Number of clusters for 'kmeans' method
    circularity_thresh : float
        Cells above this are considered 'round' (threshold method)
    elongation_thresh : float
        Cells above this are considered 'elongated' (threshold method)
    solidity_thresh : float
        Cells below this are considered 'irregular' (threshold method)
    store : bool
        If True, store result in sp.cell_meta['morph_class']

    Returns
    -------
    pd.Series
        Morphological class per cell

    Examples
    --------
    >>> compute_morphology(sp)
    >>> classes = classify_morphology(sp, method='threshold')
    >>> sp.cell_meta['morph_class'].value_counts()
    round        1523
    elongated     892
    irregular     341
    """
    print(f"\n[Morphology] Classifying cells (method='{method}')...")

    # Check that morphology has been computed
    morph_cols = [c for c in sp.cell_meta.columns if c.startswith("morph_")]
    if not morph_cols:
        raise ValueError("No morphology metrics found in cell_meta. " "Run compute_morphology(sp) first.")

    labels = pd.Series(index=sp.cell_index, dtype=object)

    if method == "threshold":
        # Validate required columns exist
        required = ["morph_circularity", "morph_elongation", "morph_solidity"]
        missing = [c for c in required if c not in sp.cell_meta.columns]
        if missing:
            raise ValueError(
                f"Missing metrics for threshold classification: {missing}. "
                "Run compute_morphology(sp) with these metrics."
            )

        circ = sp.cell_meta["morph_circularity"]
        elong = sp.cell_meta["morph_elongation"]
        solid = sp.cell_meta["morph_solidity"]

        # Classification rules (order matters: first match wins)
        # 1. Low solidity → irregular (concavities dominate)
        # 2. High elongation → elongated
        # 3. High circularity → round
        # 4. Fallback → irregular
        labels[:] = "irregular"
        labels[circ >= circularity_thresh] = "round"
        labels[elong >= elongation_thresh] = "elongated"
        labels[solid < solidity_thresh] = "irregular"

        # NaN metrics → unclassified
        has_nan = circ.isna() | elong.isna() | solid.isna()
        labels[has_nan] = np.nan

    elif method == "kmeans":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Gather all morph metrics
        morph_data = sp.cell_meta[morph_cols].copy()

        # Only classify cells with complete metrics
        valid_mask = morph_data.notna().all(axis=1)
        valid_data = morph_data[valid_mask]

        if len(valid_data) < n_classes:
            raise ValueError(
                f"Only {len(valid_data)} cells with complete metrics, "
                f"need at least {n_classes} for {n_classes} clusters."
            )

        # Scale and cluster
        scaler = StandardScaler()
        scaled = scaler.fit_transform(valid_data)

        km = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(scaled)

        labels[valid_mask] = cluster_labels.astype(str)
        labels[~valid_mask] = np.nan

        print(f"  → KMeans inertia: {km.inertia_:.1f}")

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'threshold' or 'kmeans'.")

    # Store
    if store:
        sp.cell_meta["morph_class"] = labels.values
        print("  ✓ Stored in cell_meta['morph_class']")

    # Report
    counts = labels.value_counts(dropna=False)
    print("  ✓ Classification results:")
    for cls, n in counts.items():
        label = cls if pd.notna(cls) else "unclassified"
        print(f"    {label}: {n:,} cells")

    return labels


def morphology_by_group(
    sp: spatioloji, group_col: str, metrics: list[str] | None = None, test: str = "kruskal"
) -> dict[str, pd.DataFrame]:
    """
    Compare morphology metrics across cell groups.

    For each metric, computes per-group summary statistics and
    runs a statistical test for differences between groups.

    Parameters
    ----------
    sp : spatioloji
        spatioloji object with morph_ columns in cell_meta
    group_col : str
        Column in cell_meta defining groups (e.g., 'cell_type')
    metrics : list of str, optional
        Morphology metrics to compare (without 'morph_' prefix).
        If None, uses all available morph_ columns.
    test : str
        Statistical test: 'kruskal' (Kruskal-Wallis) or 'anova' (one-way ANOVA)

    Returns
    -------
    dict with two keys:
        'summary' : pd.DataFrame
            Per-group mean, median, std for each metric (MultiIndex columns)
        'tests' : pd.DataFrame
            Test statistic and p-value for each metric

    Examples
    --------
    >>> compute_morphology(sp)
    >>> results = morphology_by_group(sp, group_col='cell_type')
    >>> print(results['tests'])
    >>>
    >>> # Summary stats per cell type
    >>> print(results['summary']['circularity'])
    """
    from scipy import stats as scipy_stats

    print(f"\n[Morphology] Comparing metrics by '{group_col}'...")

    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"'{group_col}' not found in cell_meta")

    # Find available morphology columns
    all_morph_cols = [c for c in sp.cell_meta.columns if c.startswith("morph_") and c != "morph_class"]

    if not all_morph_cols:
        raise ValueError("No morphology metrics found. Run compute_morphology(sp) first.")

    # Select metrics
    if metrics is not None:
        morph_cols = [f"morph_{m}" for m in metrics]
        missing = [c for c in morph_cols if c not in sp.cell_meta.columns]
        if missing:
            raise ValueError(f"Metrics not found in cell_meta: {missing}")
    else:
        morph_cols = all_morph_cols

    # Clean metric names (without prefix) for display
    metric_names = [c.replace("morph_", "") for c in morph_cols]

    groups = sp.cell_meta[group_col].dropna()
    unique_groups = groups.unique()
    n_groups = len(unique_groups)

    print(f"  → {n_groups} groups, {len(morph_cols)} metrics")

    # Per-group summary
    summary_records = []
    for col, name in zip(morph_cols, metric_names, strict=True):
        data = sp.cell_meta[[group_col, col]].dropna()
        grouped = data.groupby(group_col)[col]

        for grp_name, grp_data in grouped:
            summary_records.append(
                {
                    "metric": name,
                    "group": grp_name,
                    "n": len(grp_data),
                    "mean": grp_data.mean(),
                    "median": grp_data.median(),
                    "std": grp_data.std(),
                    "q25": grp_data.quantile(0.25),
                    "q75": grp_data.quantile(0.75),
                }
            )

    summary_df = pd.DataFrame(summary_records)

    # Statistical tests
    test_records = []
    for col, name in zip(morph_cols, metric_names, strict=True):
        data = sp.cell_meta[[group_col, col]].dropna()

        # Split into groups
        group_arrays = [grp_data[col].values for _, grp_data in data.groupby(group_col)]

        # Need at least 2 groups with data
        group_arrays = [g for g in group_arrays if len(g) > 0]

        if len(group_arrays) < 2:
            test_records.append(
                {"metric": name, "test": test, "statistic": np.nan, "p_value": np.nan, "significant": False}
            )
            continue

        if test == "kruskal":
            stat, pval = scipy_stats.kruskal(*group_arrays)
        elif test == "anova":
            stat, pval = scipy_stats.f_oneway(*group_arrays)
        else:
            raise ValueError(f"Unknown test: '{test}'. Use 'kruskal' or 'anova'.")

        test_records.append(
            {"metric": name, "test": test, "statistic": stat, "p_value": pval, "significant": pval < 0.05}
        )

    tests_df = pd.DataFrame(test_records)

    # Report
    print(f"\n  Statistical tests ({test}):")
    for _, row in tests_df.iterrows():
        sig = (
            "***"
            if row["p_value"] < 0.001
            else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "ns"))
        )
        print(f"    {row['metric']:>15s}: p={row['p_value']:.2e} {sig}")

    return {"summary": summary_df, "tests": tests_df}
