"""
zones.py - Interface-aware CCC and morphology stratification.

Provides three analytical functions that stratify edge-level CCC scores
(from score_edges) by:

  - Spatial zone relative to a tissue interface (compare_zones)
  - Signed distance gradient across the interface (communication_gradient)
  - Sender cell morphology category (compare_morphology)

Typical usage
-------------
>>> from spatioloji_s.ccc.zones import compare_zones, communication_gradient, compare_morphology
>>> from spatioloji_s.ccc.scoring import score_edges
>>> from spatioloji_s.spatial._interface import identify_interface
>>>
>>> iface = identify_interface(sp, group_col="cell_type", region_a="Tumor", region_b="Stroma")
>>> edges = score_edges(sp, pairs, graph_diffusible=graph)
>>> zone_df = compare_zones(edges, sp, iface)
>>> grad_df = communication_gradient(edges, sp, iface)
>>> morph_df = compare_morphology(edges, sp, morphology_col="morph_class")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    from spatioloji_s.spatial._interface_types import InterfaceResult

# ── Public API ────────────────────────────────────────────────────────────────


def compare_zones(
    edge_df: pd.DataFrame,
    sp: spatioloji,
    interface_result: InterfaceResult,
    group_col: str = "cell_type",
    coord_type: str = "global",
) -> pd.DataFrame:
    """Compare CCC scores across spatial zones relative to a tissue interface.

    Each cell is classified into a zone based on its interface label:
    ``'interface'``, ``'interior_a'``, ``'interior_b'``, or ``'other'``.
    Edges are then grouped by zone (an edge belongs to a zone if either
    the sender or receiver is in that zone) and aggregated by
    (lr_name, sender_type, receiver_type).

    Args:
        edge_df: Output of :func:`~spatioloji_s.ccc.scoring.score_edges`.
            Must contain columns: sender, receiver, lr_name, sender_type,
            receiver_type, score.
        sp: spatioloji object.
        interface_result: Result from
            :func:`~spatioloji_s.spatial._interface.identify_interface`.
        group_col: Column in ``sp.cell_meta`` for cell type labels.
            Currently unused in computation but validated for consistency.
        coord_type: ``'global'`` or ``'local'`` coordinates passed to
            :func:`~spatioloji_s.spatial._distance_utils.signed_distance_to_interface`.

    Returns:
        DataFrame with columns: lr_name, sender_type, receiver_type, zone,
        mean_score, sum_score, n_edges, fold_change.

        ``fold_change`` is zone_mean_score / overall_mean_score for each
        (lr_name, sender_type, receiver_type) group.  Returns NaN when
        overall_mean_score is zero.

    Raises:
        ValueError: If ``edge_df`` is missing required columns.

    Example:
        >>> iface = identify_interface(sp, group_col="cell_type",
        ...     region_a="Tumor", region_b="Stroma")
        >>> edges = score_edges(sp, pairs, graph_diffusible=graph)
        >>> zone_df = compare_zones(edges, sp, iface)
    """
    _validate_edge_df(edge_df)

    if edge_df.empty:
        return pd.DataFrame(
            columns=["lr_name", "sender_type", "receiver_type", "zone", "mean_score", "sum_score", "n_edges",
                     "fold_change"]
        )

    # ── Zone classification ───────────────────────────────────────────────
    cell_labels = interface_result.cell_labels  # Series indexed by cell_id

    label_to_zone = {
        "region_a_interface": "interface",
        "region_b_interface": "interface",
        "interior_a": "interior_a",
        "interior_b": "interior_b",
    }

    cell_zones: pd.Series = cell_labels.map(lambda lbl: label_to_zone.get(lbl, "other"))

    # ── Tag each edge with sender/receiver zone ───────────────────────────
    df = edge_df.copy()
    df["sender_zone"] = df["sender"].map(cell_zones).fillna("other")
    df["receiver_zone"] = df["receiver"].map(cell_zones).fillna("other")

    # ── Overall mean score per (lr_name, sender_type, receiver_type) ─────
    overall_mean = (
        df.groupby(["lr_name", "sender_type", "receiver_type"])["score"]
        .mean()
        .rename("overall_mean")
    )

    # ── Aggregate per zone ────────────────────────────────────────────────
    zone_records: list[dict] = []
    for zone in ("interface", "interior_a", "interior_b"):
        zone_mask = (df["sender_zone"] == zone) | (df["receiver_zone"] == zone)
        sub = df[zone_mask]
        if sub.empty:
            continue

        grouped = sub.groupby(["lr_name", "sender_type", "receiver_type"])["score"]
        zone_agg = pd.DataFrame(
            {
                "mean_score": grouped.mean(),
                "sum_score": grouped.sum(),
                "n_edges": grouped.count(),
            }
        ).reset_index()
        zone_agg["zone"] = zone
        zone_records.append(zone_agg)

    if not zone_records:
        return pd.DataFrame(
            columns=["lr_name", "sender_type", "receiver_type", "zone", "mean_score", "sum_score", "n_edges",
                     "fold_change"]
        )

    result = pd.concat(zone_records, ignore_index=True)

    # ── Fold change ───────────────────────────────────────────────────────
    result = result.join(overall_mean, on=["lr_name", "sender_type", "receiver_type"])
    result["fold_change"] = result["mean_score"] / result["overall_mean"].replace(0, np.nan)
    result = result.drop(columns=["overall_mean"])

    return result[["lr_name", "sender_type", "receiver_type", "zone", "mean_score", "sum_score", "n_edges",
                   "fold_change"]].reset_index(drop=True)


def communication_gradient(
    edge_df: pd.DataFrame,
    sp: spatioloji,
    interface_result: InterfaceResult,
    group_col: str = "cell_type",
    n_bins: int = 20,
    coord_type: str = "global",
) -> pd.DataFrame:
    """Fit a linear gradient of CCC scores across the interface axis.

    For each (lr_name, sender_type, receiver_type) group, edges are
    positioned along the signed-distance axis and binned.  A linear
    regression is fit to the mean score per bin to detect gradients
    toward or away from the interface.

    Args:
        edge_df: Output of :func:`~spatioloji_s.ccc.scoring.score_edges`.
        sp: spatioloji object.
        interface_result: Result from
            :func:`~spatioloji_s.spatial._interface.identify_interface`.
        group_col: Column in ``sp.cell_meta`` for cell type labels.
        n_bins: Number of equal-width bins along the signed-distance axis.
        coord_type: ``'global'`` or ``'local'``.

    Returns:
        DataFrame with columns: lr_name, sender_type, receiver_type,
        slope, pvalue, r2, trend.

        ``trend`` is one of:

        - ``'increasing_toward_a'``: slope > 0, p < 0.05
        - ``'increasing_toward_b'``: slope < 0, p < 0.05
        - ``'flat'``: p >= 0.05

    Raises:
        ValueError: If ``edge_df`` is missing required columns or
            ``interface_result.contour`` is None.

    Example:
        >>> grad_df = communication_gradient(edges, sp, iface, n_bins=10)
        >>> sig = grad_df[grad_df["trend"] != "flat"]
    """
    from scipy.stats import linregress

    from spatioloji_s.spatial._distance_utils import signed_distance_to_interface

    _validate_edge_df(edge_df)

    if edge_df.empty:
        return pd.DataFrame(columns=["lr_name", "sender_type", "receiver_type", "slope", "pvalue", "r2", "trend"])

    # ── Signed distances ──────────────────────────────────────────────────
    distances = signed_distance_to_interface(sp, interface_result, coord_type=coord_type)

    # ── Edge position = midpoint of sender and receiver distances ─────────
    df = edge_df.copy()
    sender_dist = df["sender"].map(distances)
    receiver_dist = df["receiver"].map(distances)
    df["edge_distance"] = (sender_dist + receiver_dist) / 2.0

    # Drop edges where distance could not be computed
    df = df.dropna(subset=["edge_distance"])

    if df.empty:
        return pd.DataFrame(columns=["lr_name", "sender_type", "receiver_type", "slope", "pvalue", "r2", "trend"])

    # ── Bin edges and regress ─────────────────────────────────────────────
    dist_min = df["edge_distance"].min()
    dist_max = df["edge_distance"].max()

    if dist_min == dist_max:
        # All edges at same distance — flat by definition
        groups = df.groupby(["lr_name", "sender_type", "receiver_type"])
        records = []
        for (lr, st, rt), _ in groups:
            records.append(
                {"lr_name": lr, "sender_type": st, "receiver_type": rt,
                 "slope": 0.0, "pvalue": 1.0, "r2": 0.0, "trend": "flat"}
            )
        return pd.DataFrame(records)

    bin_edges = np.linspace(dist_min, dist_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    df["bin"] = pd.cut(df["edge_distance"], bins=bin_edges, labels=False, include_lowest=True)

    records: list[dict] = []
    groups = df.groupby(["lr_name", "sender_type", "receiver_type"])

    for (lr, st, rt), grp in groups:
        bin_means = grp.groupby("bin")["score"].mean()

        # Align to all bin centers; skip if fewer than 2 data points
        bin_idx = bin_means.index.astype(int)
        x = bin_centers[bin_idx]
        y = bin_means.values

        if len(x) < 2:
            records.append(
                {"lr_name": lr, "sender_type": st, "receiver_type": rt,
                 "slope": 0.0, "pvalue": 1.0, "r2": 0.0, "trend": "flat"}
            )
            continue

        reg = linregress(x, y)
        slope = float(reg.slope)
        pvalue = float(reg.pvalue)
        r2 = float(reg.rvalue ** 2)

        if slope > 0 and pvalue < 0.05:
            trend = "increasing_toward_a"
        elif slope < 0 and pvalue < 0.05:
            trend = "increasing_toward_b"
        else:
            trend = "flat"

        records.append(
            {"lr_name": lr, "sender_type": st, "receiver_type": rt,
             "slope": slope, "pvalue": pvalue, "r2": r2, "trend": trend}
        )

    return pd.DataFrame(records)[["lr_name", "sender_type", "receiver_type", "slope", "pvalue", "r2", "trend"]]


def compare_morphology(
    edge_df: pd.DataFrame,
    sp: spatioloji,
    morphology_col: str,
    group_col: str = "cell_type",
) -> pd.DataFrame:
    """Stratify CCC scores by sender cell morphology.

    Tags each edge with the sender cell's morphology category and
    aggregates scores by (lr_name, sender_type, receiver_type, morphology_group).

    Args:
        edge_df: Output of :func:`~spatioloji_s.ccc.scoring.score_edges`.
        sp: spatioloji object.
        morphology_col: Column in ``sp.cell_meta`` containing morphology
            category labels (e.g. ``'morph_class'`` with ``'round'``/
            ``'elongated'``).
        group_col: Column in ``sp.cell_meta`` for cell type labels.
            Currently unused in computation but kept for API consistency.

    Returns:
        DataFrame with columns: lr_name, sender_type, receiver_type,
        morphology_group, mean_score, sum_score, n_edges, fold_change.

        ``fold_change`` is group_mean_score / overall_mean_score for each
        (lr_name, sender_type, receiver_type) group.  Returns NaN when
        overall_mean_score is zero.

    Raises:
        ValueError: If ``morphology_col`` is not found in ``sp.cell_meta``
            or ``edge_df`` is missing required columns.

    Example:
        >>> morph_df = compare_morphology(edges, sp, morphology_col="morph_class")
        >>> elongated = morph_df[morph_df["morphology_group"] == "elongated"]
    """
    _validate_edge_df(edge_df)

    if morphology_col not in sp.cell_meta.columns:
        raise ValueError(f"'{morphology_col}' not found in cell_meta. Available: {list(sp.cell_meta.columns)}")

    if edge_df.empty:
        return pd.DataFrame(
            columns=["lr_name", "sender_type", "receiver_type", "morphology_group", "mean_score", "sum_score",
                     "n_edges", "fold_change"]
        )

    morph_labels: pd.Series = sp.cell_meta[morphology_col]

    # ── Tag sender morphology ─────────────────────────────────────────────
    df = edge_df.copy()
    df["sender_morphology"] = df["sender"].map(morph_labels)

    # ── Overall mean score per (lr_name, sender_type, receiver_type) ─────
    overall_mean = (
        df.groupby(["lr_name", "sender_type", "receiver_type"])["score"]
        .mean()
        .rename("overall_mean")
    )

    # ── Aggregate by morphology group ─────────────────────────────────────
    grouped = df.dropna(subset=["sender_morphology"]).groupby(
        ["lr_name", "sender_type", "receiver_type", "sender_morphology"]
    )["score"]

    result = pd.DataFrame(
        {
            "mean_score": grouped.mean(),
            "sum_score": grouped.sum(),
            "n_edges": grouped.count(),
        }
    ).reset_index()

    result = result.rename(columns={"sender_morphology": "morphology_group"})

    # ── Fold change ───────────────────────────────────────────────────────
    result = result.join(overall_mean, on=["lr_name", "sender_type", "receiver_type"])
    result["fold_change"] = result["mean_score"] / result["overall_mean"].replace(0, np.nan)
    result = result.drop(columns=["overall_mean"])

    return result[
        ["lr_name", "sender_type", "receiver_type", "morphology_group", "mean_score", "sum_score", "n_edges",
         "fold_change"]
    ].reset_index(drop=True)


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate_edge_df(edge_df: pd.DataFrame) -> None:
    """Validate that edge_df has the required columns.

    Args:
        edge_df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"sender", "receiver", "lr_name", "sender_type", "receiver_type", "score"}
    missing = required - set(edge_df.columns)
    if missing:
        raise ValueError(f"edge_df is missing required columns: {sorted(missing)}")
