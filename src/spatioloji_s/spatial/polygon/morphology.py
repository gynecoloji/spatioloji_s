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

from concurrent.futures import ProcessPoolExecutor
import numba as nb
import numpy as np
import os
import pandas as pd
from scipy import stats
from shapely.geometry import Polygon
from statsmodels.stats.multitest import multipletests


from .graph import _get_gdf


# ============================================================
# Geometry Extraction
# ============================================================

def extract_polygon_arrays(gdf):
    """
    Convert GeoDataFrame polygons into flattened coordinate arrays.

    Returns
    -------
    all_x : np.ndarray
    all_y : np.ndarray
    offsets : np.ndarray
        offsets[i] : start index of polygon i
    """

    xs = []
    ys = []
    offsets = [0]

    for geom in gdf.geometry:

        if geom is None or geom.is_empty:
            offsets.append(offsets[-1])
            continue

        coords = np.asarray(geom.exterior.coords)

        xs.extend(coords[:, 0])
        ys.extend(coords[:, 1])

        offsets.append(len(xs))

    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(offsets, dtype=np.int64),
    )


import numpy as np
import pandas as pd
import numba as nb
from shapely.geometry import Polygon

# ============================================================
# Numba Kernels
# ============================================================

@nb.njit(parallel=True, fastmath=True)
def polygon_area_perimeter(all_x, all_y, offsets):
    n_cells = len(offsets) - 1
    areas = np.zeros(n_cells)
    perims = np.zeros(n_cells)
    for i in nb.prange(n_cells):
        start, end = offsets[i], offsets[i + 1]
        if end - start < 3:
            areas[i] = np.nan
            perims[i] = np.nan
            continue
        area, perim = 0.0, 0.0
        for j in range(start, end):
            j_next = start if j + 1 == end else j + 1
            x1, y1 = all_x[j], all_y[j]
            x2, y2 = all_x[j_next], all_y[j_next]
            area += x1 * y2 - x2 * y1
            dx, dy = x2 - x1, y2 - y1
            perim += (dx * dx + dy * dy) ** 0.5
        areas[i] = abs(area) * 0.5
        perims[i] = perim
    return areas, perims

@nb.njit(parallel=True, fastmath=True)
def compute_circularity(areas, perims):
    n = len(areas)
    circ = np.full(n, np.nan)
    for i in nb.prange(n):
        p = perims[i]
        if p > 0:
            circ[i] = 4.0 * np.pi * areas[i] / (p * p)
    return circ

@nb.njit(fastmath=True)
def polygon_entropy(x, y):
    n = len(x)
    if n < 3:
        return np.nan

    angles = np.zeros(n)
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        v1x, v1y = x[i] - x[i_prev], y[i] - y[i_prev]
        v2x, v2y = x[i_next] - x[i], y[i_next] - y[i]
        dot = v1x * v2x + v1y * v2y
        n1 = np.sqrt(v1x**2 + v1y**2)
        n2 = np.sqrt(v2x**2 + v2y**2)
        if n1 > 0 and n2 > 0:
            cosang = dot / (n1 * n2)
            if cosang > 1.0:
                cosang = 1.0
            elif cosang < -1.0:
                cosang = -1.0
            angles[i] = np.arccos(cosang)

    hist = np.histogram(angles, bins=16)[0]
    total = hist.sum()
    if total == 0:
        return 0.0

    # Numba-friendly entropy computation
    entropy = 0.0
    for h in hist:
        if h > 0:
            p = h / total
            entropy -= p * np.log2(p)

    return entropy

@nb.njit(parallel=True)
def batch_entropy(all_x, all_y, offsets):
    n_cells = len(offsets) - 1
    out = np.full(n_cells, np.nan)
    for i in nb.prange(n_cells):
        start, end = offsets[i], offsets[i + 1]
        if end - start < 3:
            continue
        x, y = all_x[start:end], all_y[start:end]
        out[i] = polygon_entropy(x, y)
    return out

# ============================================================
# Main API with Shapely Elongation
# ============================================================

def compute_morphology(
    sp,
    coord_type="global",
    metrics=None,
    store=True,
):
    """
    High-performance morphology computation using Numba + Shapely.

    Metrics supported:
        - area
        - perimeter
        - circularity
        - elongation (minimum rotated rectangle)
        - solidity (convex hull)
        - compactness (sqrt(4*area/pi)/major_axis)
        - convexity (convex hull / perimeter)
        - contour_entropy
    """

    all_metrics = [
        "area", "perimeter", "circularity",
        "elongation", "solidity", "compactness",
        "convexity", "contour_entropy",
    ]

    if metrics is None:
        metrics = all_metrics

    gdf = _get_gdf(sp, coord_type=coord_type)

    print("\n[Morphology] Extracting polygon coordinates...")
    all_x, all_y, offsets = extract_polygon_arrays(gdf)

    results = {}

    # --- Area + Perimeter + Circularity ---
    if any(m in metrics for m in ["area", "perimeter", "circularity"]):
        print("[Morphology] Computing area and perimeter...")
        areas, perims = polygon_area_perimeter(all_x, all_y, offsets)
        if "area" in metrics:
            results["area"] = areas
        if "perimeter" in metrics:
            results["perimeter"] = perims
        if "circularity" in metrics:
            results["circularity"] = compute_circularity(areas, perims)

    # --- Elongation + Solidity + Compactness + Convexity ---
    if any(m in metrics for m in ["elongation", "solidity", "compactness", "convexity"]):
        print("[Morphology] Computing elongation, solidity, convexity, compactness (shapely)...")
        elong, sol, comp, conv = [], [], [], []

        for poly, area, perim in zip(gdf.geometry, results.get("area", [None]*len(gdf)), results.get("perimeter", [None]*len(gdf))):
            if poly is None or poly.is_empty or not poly.is_valid:
                elong.append(np.nan)
                sol.append(np.nan)
                comp.append(np.nan)
                conv.append(np.nan)
                continue

            # Elongation via minimum rotated rectangle
            try:
                mrr = poly.minimum_rotated_rectangle
                coords = np.array(mrr.exterior.coords)
                edge_lengths = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
                major = edge_lengths.max()
                minor = edge_lengths.min()
                elong.append(major / minor if minor > 0 else np.nan)
            except Exception:
                elong.append(np.nan)

            # Solidity & Convexity
            convex = poly.convex_hull
            convex_area = convex.area
            convex_perim = convex.length
            sol.append(area / convex_area if convex_area > 0 else np.nan)
            conv.append(convex_perim / perim if perim > 0 else np.nan)

            # Compactness
            comp.append(np.sqrt(4*area/np.pi)/major if major > 0 else np.nan)

        if "elongation" in metrics:
            results["elongation"] = np.array(elong)
        if "solidity" in metrics:
            results["solidity"] = np.array(sol)
        if "convexity" in metrics:
            results["convexity"] = np.array(conv)
        if "compactness" in metrics:
            results["compactness"] = np.array(comp)

    # --- Contour Entropy ---
    if "contour_entropy" in metrics:
        print("[Morphology] Computing contour entropy...")
        results["contour_entropy"] = batch_entropy(all_x, all_y, offsets)

    # --- Build DataFrame ---
    morph_df = pd.DataFrame(results, index=gdf.index).reindex(sp.cell_index)

    if store:
        for col in morph_df.columns:
            sp.cell_meta[f"morph_{col}"] = morph_df[col].values
        print("✓ Stored in sp.cell_meta")

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
        raise ValueError("No morphology metrics found in cell_meta. Run compute_morphology(sp) first.")

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
    sp: spatioloji,
    group_col: str,
    metrics: list[str] | None = None,
    test: str = "kruskal",
) -> dict[str, pd.DataFrame]:

    from scipy import stats as scipy_stats

    print(f"\n[Morphology] Comparing metrics by '{group_col}'...")

    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"{group_col} not found in cell_meta")

    # --------------------------------------------------
    # Find morphology columns
    # --------------------------------------------------

    all_morph_cols = [
        c for c in sp.cell_meta.columns
        if c.startswith("morph_") and c != "morph_class"
    ]

    if not all_morph_cols:
        raise ValueError("No morphology metrics found. Run compute_morphology() first.")

    if metrics is None:
        morph_cols = all_morph_cols
    else:
        morph_cols = [f"morph_{m}" for m in metrics]

    metric_names = [c.replace("morph_", "") for c in morph_cols]

    # --------------------------------------------------
    # Filter dataframe once
    # --------------------------------------------------

    df = sp.cell_meta[[group_col] + morph_cols].dropna()

    if df.empty:
        raise ValueError("No valid rows after dropping NaNs")

    groups = df[group_col]

    unique_groups = groups.unique()

    print(f"  → {len(unique_groups)} groups, {len(morph_cols)} metrics")

    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------

    summary = (
        df.groupby(group_col)[morph_cols]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )

    # rename columns
    summary.columns = pd.MultiIndex.from_tuples(
        [(c.replace("morph_", ""), stat) for c, stat in summary.columns]
    )

    # --------------------------------------------------
    # Statistical tests
    # --------------------------------------------------

    grouped = df.groupby(group_col)

    test_records = []

    for col, name in zip(morph_cols, metric_names):

        group_arrays = [
            g[col].values
            for _, g in grouped
            if len(g[col].values) > 0
        ]

        if len(group_arrays) < 2:

            test_records.append(
                {
                    "metric": name,
                    "test": test,
                    "statistic": np.nan,
                    "p_value": np.nan,
                }
            )
            continue

        if test == "kruskal":

            stat, pval = scipy_stats.kruskal(*group_arrays)

        elif test == "anova":

            stat, pval = scipy_stats.f_oneway(*group_arrays)

        else:

            raise ValueError("test must be 'kruskal' or 'anova'")

        test_records.append(
            {
                "metric": name,
                "test": test,
                "statistic": stat,
                "p_value": pval,
            }
        )

    tests_df = pd.DataFrame(test_records)

    # --------------------------------------------------
    # Significance labels
    # --------------------------------------------------

    def p_to_star(p):

        if pd.isna(p):
            return "NA"
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    tests_df["significance"] = tests_df["p_value"].apply(p_to_star)

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print(f"\n  Statistical tests ({test})")

    for _, row in tests_df.iterrows():

        print(
            f"    {row['metric']:>15s} "
            f"p={row['p_value']:.2e} "
            f"{row['significance']}"
        )

    return {
        "summary": summary,
        "tests": tests_df,
    }


def _correlation_worker(gene_chunk, morph_mat_processed, n_cells, gene_names_chunk, method):
    """
    Worker function handles ranking (if Spearman) and correlation.
    """
    # 1. Ranking (The heavy lifter for Spearman)
    if method == "spearman":
        # Process each gene column in the chunk
        gene_chunk = stats.rankdata(gene_chunk, axis=0).astype(np.float32)
    else:
        gene_chunk = gene_chunk.astype(np.float32)

    # 2. Standardize Gene Chunk
    gene_chunk -= np.mean(gene_chunk, axis=0)
    g_std = np.std(gene_chunk, axis=0, ddof=1)
    g_std[g_std == 0] = 1.0  # Avoid division by zero for constant genes
    gene_chunk /= g_std

    # 3. Correlation (Dot Product)
    corr_chunk = np.dot(morph_mat_processed.T, gene_chunk) / (n_cells - 1)

    # 4. P-value calculation
    df_val = n_cells - 2
    corr_clipped = np.clip(corr_chunk, -0.999999, 0.999999)
    t_stat = corr_clipped * np.sqrt(df_val / (1 - corr_clipped**2))
    p_chunk = 2 * stats.t.sf(np.abs(t_stat), df_val)

    return corr_chunk, p_chunk, gene_names_chunk

def morphology_gene_correlation(
    sp,
    morph_metrics=None,
    genes=None,
    layer=None,
    method="spearman",
    n_workers=None,
    chunk_size=1000  # Smaller chunks often better for parallel overhead
):
    if n_workers is None:
        n_workers = os.cpu_count()

    # --- Pre-processing Morphology (Main Process) ---
    morph_cols = [c for c in sp.cell_meta.columns if c.startswith("morph_")]
    if morph_metrics:
        morph_cols = [f"morph_{m}" for m in morph_metrics]
    
    valid_cells = sp.cell_meta[morph_cols].notna().all(axis=1).values
    morph_mat = sp.cell_meta.loc[valid_cells, morph_cols].values.astype(np.float32)
    n_cells = morph_mat.shape[0]

    # Process morphology once so workers don't repeat this
    if method == "spearman":
        morph_mat = stats.rankdata(morph_mat, axis=0).astype(np.float32)
    
    morph_mat -= np.mean(morph_mat, axis=0)
    m_std = np.std(morph_mat, axis=0, ddof=1)
    m_std[m_std == 0] = 1.0
    morph_mat /= m_std

    # --- Prepare Gene Data ---
    X = sp.layers[layer] if layer else sp.X
    gene_names = np.array(sp.gene_index)
    
    if genes is not None:
        gene_mask = np.isin(gene_names, genes)
        X = X[:, gene_mask]
        gene_names = gene_names[gene_mask]

    # Slice cells (keep sparse if possible until chunking)
    X = X[valid_cells, :]

    # --- Parallel Dispatch ---
    n_genes = X.shape[1]
    tasks = []
    
    print(f"[Morphology] Parallel {method} correlation: {n_genes} genes, {n_cells} cells")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(0, n_genes, chunk_size):
            # Densify only the chunk to keep memory usage low
            g_chunk = X[:, i : i + chunk_size]
            if hasattr(g_chunk, "toarray"):
                g_chunk = g_chunk.toarray()
                
            n_chunk = gene_names[i : i + chunk_size]
            
            futures.append(
                executor.submit(_correlation_worker, g_chunk, morph_mat, n_cells, n_chunk, method)
            )

        all_corrs, all_ps, all_genes = [], [], []
        for f in futures:
            c, p, g = f.result()
            all_corrs.append(c)
            all_ps.append(p)
            all_genes.append(g)

    # --- Fast Recombine ---
    full_corr = np.hstack(all_corrs)
    full_p = np.hstack(all_ps)
    full_gene_names = np.concatenate(all_genes)
    morph_names = [m.replace("morph_", "") for m in morph_cols]

    res_df = pd.DataFrame({
        "metric": np.repeat(morph_names, full_corr.shape[1]),
        "gene": np.tile(full_gene_names, full_corr.shape[0]),
        "correlation": full_corr.ravel(),
        "p_value": full_p.ravel(),
        "n_cells": n_cells
    })

    res_df["p_adj"] = multipletests(res_df["p_value"], method="fdr_bh")[1]
    res_df["significant"] = res_df["p_adj"] < 0.05
    
    return res_df.sort_values("p_value")

