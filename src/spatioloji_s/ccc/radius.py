"""radius.py - Per-cell-type contact-radius optimization for juxtacrine CCC.

Juxtacrine signalling needs a *contact* graph. The accurate way to build one is
from cell boundary polygons (:func:`~spatioloji_s.spatial.polygon.graph.build_buffer_graph`),
but polygons are expensive and are not always available. The cheap stand-in is a
centroid radius graph — and a single radius cannot serve a tissue whose cell
types differ in size: 17 um hepatocytes need a much larger radius than the 8 um
lymphocytes beside them.

This module fits the radius **per cell type** against the polygon contact graph:

1. Build (or accept) the polygon contact graph as ground truth.
2. Sweep candidate radii, comparing edge sets at *edge* resolution.
3. Report, for each cell type, the radius maximising Jaccard agreement, keeping
   the two error directions (missed real contacts vs invented ones) separate.
4. Hand the resulting ``{cell_type: radius}`` map to
   :func:`build_typed_radius_graph`, whose union rule connects two cells when
   their centroid distance is within the **larger** of their two radii.

Typical usage
-------------
>>> from spatioloji_s.ccc.radius import optimize_contact_radius, build_typed_radius_graph
>>> opt = optimize_contact_radius(sp_roi, group_col="cell_type")
>>> opt.radius_map
{'Tumor': 14.0, 'T_NK': 8.0, 'Fibroblast': 12.0}
>>> graph = build_typed_radius_graph(sp_big, opt.radius_map, group_col="cell_type")

or straight through the pipeline::

>>> config = CCCConfig(group_col="cell_type", juxtacrine_radius_map=opt.radius_map)
>>> result = run_ccc(sp_big, config)

Interpreting the fit
--------------------
Fitting needs the polygon graph the radius graph is meant to replace, so the
two workflows this supports are (a) fit on one ROI or sample that has polygons
and reuse the map where polygons are missing or too slow, and (b) diagnose how
badly centroid geometry misrepresents contact in *your* tissue.

Treat the reported Jaccard as that diagnosis. Benchmarking across four Xenium
cohorts found the best achievable agreement to be 0.67-0.80 even when the
radius was fitted against the ground truth, with the optimum spanning 10-24 um
across tissues — a centroid radius approximates a contact graph, it does not
reproduce one. :func:`optimize_contact_radius` warns when the best fit is weak.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    from spatioloji_s.spatial.point.graph import PointSpatialGraph

# Candidate radii (um). Spans the cell-diameter range seen across imaging-based
# platforms, from small lymphocytes to large parenchymal cells.
DEFAULT_RADII: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 18.0, 24.0, 32.0)

# Segmentation rarely draws adjacent cells as exactly touching; a small buffer
# is what makes the truth graph a cell packing (mean degree ~6) rather than a
# mostly-isolated one.
DEFAULT_TRUTH_BUFFER: float = 0.25

# Below this many ground-truth contacts, a per-type optimum is noise.
DEFAULT_MIN_EDGES: int = 200

# Jaccard below this means the radius approximation is poor for this tissue.
WEAK_FIT_JACCARD: float = 0.5


@dataclass
class RadiusOptimizationResult:
    """Result of :func:`optimize_contact_radius`.

    Attributes:
        radius_map: ``{cell_type: best_radius}`` — pass straight to
            :func:`build_typed_radius_graph` or ``CCCConfig.juxtacrine_radius_map``.
        per_type: One row per cell type: ``cell_type``, ``best_radius``,
            ``jaccard``, ``n_truth_edges``, ``missed_rate``, ``false_rate``.
        per_pair: One row per unordered cell-type pair (``"A|B"``): ``pair``,
            ``best_radius``, ``jaccard``, ``n_truth_edges``. Diagnostic — shows
            how far apart different pairs' optima sit.
        sweep: Full grid. Global metrics per radius plus one row per
            (radius, cell_type).
        global_best: Best single radius for the whole tissue.
        global_jaccard: Jaccard achieved at ``global_best``.
        truth_params: How the ground-truth graph was built.
    """

    radius_map: dict[str, float]
    per_type: pd.DataFrame
    per_pair: pd.DataFrame
    sweep: pd.DataFrame
    global_best: float
    global_jaccard: float
    truth_params: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        spread = f"{min(self.radius_map.values()):g}-{max(self.radius_map.values()):g}" if self.radius_map else "n/a"
        return (
            f"RadiusOptimizationResult(n_types={len(self.radius_map)}, "
            f"per_type_radius={spread} um, global_best={self.global_best:g} um, "
            f"global_jaccard={self.global_jaccard:.3f})"
        )


# ── Edge-set helpers ─────────────────────────────────────────────────────────


def _edge_keys(adjacency: sparse.spmatrix, n_cells: int) -> np.ndarray:
    """Undirected edge set as sorted unique int64 keys (``i < j``)."""
    coo = adjacency.tocoo()
    i = coo.row.astype(np.int64)
    j = coo.col.astype(np.int64)
    keep = i != j
    i, j = i[keep], j[keep]
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    return np.unique(lo * np.int64(n_cells) + hi)


def _compare(truth: np.ndarray, test: np.ndarray) -> dict:
    """Edge-level agreement with the two error directions kept apart."""
    inter = np.intersect1d(truth, test, assume_unique=True).size
    union = truth.size + test.size - inter
    return {
        "n_truth_edges": int(truth.size),
        "n_test_edges": int(test.size),
        "n_shared": int(inter),
        "jaccard": inter / union if union else np.nan,
        "missed_rate": (truth.size - inter) / truth.size if truth.size else np.nan,
        "false_rate": (test.size - inter) / test.size if test.size else np.nan,
    }


def _pair_labels(keys: np.ndarray, types: np.ndarray, n_cells: int) -> np.ndarray:
    """Unordered ``"A|B"`` type label for each edge key."""
    lo = types[(keys // n_cells).astype(np.int64)]
    hi = types[(keys % n_cells).astype(np.int64)]
    first = np.where(lo < hi, lo, hi)
    second = np.where(lo < hi, hi, lo)
    return np.char.add(np.char.add(first.astype(str), "|"), second.astype(str))


def _endpoint_types(keys: np.ndarray, types: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Both endpoint types for each edge key."""
    return types[(keys // n_cells).astype(np.int64)], types[(keys % n_cells).astype(np.int64)]


# ── Public API ───────────────────────────────────────────────────────────────


def optimize_contact_radius(
    sp: spatioloji,
    group_col: str = "cell_type",
    radii: tuple[float, ...] | list[float] = DEFAULT_RADII,
    truth_graph: object | None = None,
    truth_buffer: float = DEFAULT_TRUTH_BUFFER,
    coord_type: str = "global",
    min_type_edges: int = DEFAULT_MIN_EDGES,
    min_pair_edges: int = DEFAULT_MIN_EDGES,
    verbose: bool = True,
) -> RadiusOptimizationResult:
    """Fit the centroid radius that best reproduces polygon contact, per cell type.

    Args:
        sp: spatioloji object with polygons (unless ``truth_graph`` is given)
            and ``group_col`` in ``cell_meta``.
        group_col: Column in ``sp.cell_meta`` holding cell type labels.
        radii: Candidate radii in coordinate units (um for Xenium/MERSCOPE).
        truth_graph: Pre-built polygon contact graph to score against. When
            ``None``, one is built with
            :func:`~spatioloji_s.spatial.polygon.graph.build_buffer_graph`
            at ``truth_buffer``.
        truth_buffer: Buffer distance for the ground-truth contact graph.
            Default 0.25 — segmentation rarely draws adjacent cells as exactly
            touching, and a zero buffer yields a mostly-isolated graph.
        coord_type: ``'global'`` or ``'local'``.
        min_type_edges: Cell types with fewer ground-truth contacts than this
            are reported but excluded from ``radius_map`` (their optimum is noise).
        min_pair_edges: Same threshold for the per-pair diagnostic table.
        verbose: Print the sweep and the fitted map.

    Returns:
        :class:`RadiusOptimizationResult`.

    Raises:
        ValueError: If ``group_col`` is missing, ``radii`` is empty, or no
            ground-truth contacts exist.

    Example:
        >>> opt = optimize_contact_radius(sp, group_col="cell_type")
        >>> opt.radius_map
        {'Tumor': 14.0, 'T_NK': 8.0}
    """
    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"group_col '{group_col}' not found in cell_meta")
    radii = tuple(float(r) for r in radii)
    if not radii:
        raise ValueError("radii must not be empty")

    from spatioloji_s.spatial.point.graph import build_radius_graph

    n = sp.n_cells
    types = sp.cell_meta[group_col].astype(str).to_numpy()

    if truth_graph is None:
        from spatioloji_s.spatial.polygon.graph import build_buffer_graph

        if sp.polygons is None:
            raise ValueError(
                "No polygons on this object: pass truth_graph= a polygon contact "
                "graph, or load boundaries first. A radius cannot be fitted "
                "without the contact graph it is approximating."
            )
        if verbose:
            print(f"[radius] Building ground-truth contact graph (buffer={truth_buffer})")
        truth_graph = build_buffer_graph(sp, buffer_distance=truth_buffer, coord_type=coord_type)

    truth = _edge_keys(truth_graph.adjacency, n)
    if truth.size == 0:
        raise ValueError(
            "Ground-truth contact graph has no edges — check truth_buffer "
            "(segmentation rarely draws cells as exactly touching)."
        )
    truth_pair = _pair_labels(truth, types, n)
    truth_ta, truth_tb = _endpoint_types(truth, types, n)

    global_rows: list[dict] = []
    type_rows: list[dict] = []
    pair_rows: list[dict] = []

    for r in radii:
        test = _edge_keys(build_radius_graph(sp, radius=r, coord_type=coord_type).adjacency, n)
        metrics = _compare(truth, test)
        metrics.update(radius=r, scope="global", key="__all__")
        global_rows.append(metrics)

        found = np.isin(truth, test, assume_unique=True)
        test_pair = _pair_labels(test, types, n)
        test_ta, test_tb = _endpoint_types(test, types, n)

        # ── per cell type: every edge with this type at either endpoint ──
        n_test_by_type = pd.Series(np.concatenate([test_ta, test_tb])).value_counts()
        truth_by_type = pd.DataFrame(
            {"t": np.concatenate([truth_ta, truth_tb]), "found": np.concatenate([found, found])}
        )
        for t, grp in truth_by_type.groupby("t"):
            n_truth = int(len(grp))
            n_found = int(grp["found"].sum())
            n_test = int(n_test_by_type.get(t, 0))
            union = n_truth + n_test - n_found
            type_rows.append(
                {
                    "radius": r,
                    "scope": "cell_type",
                    "key": str(t),
                    "n_truth_edges": n_truth,
                    "n_test_edges": n_test,
                    "n_shared": n_found,
                    "jaccard": n_found / union if union else np.nan,
                    "missed_rate": (n_truth - n_found) / n_truth if n_truth else np.nan,
                    "false_rate": (n_test - n_found) / n_test if n_test else np.nan,
                }
            )

        # ── per unordered cell-type pair (diagnostic) ──
        n_test_by_pair = pd.Series(test_pair).value_counts()
        tp = pd.DataFrame({"pair": truth_pair, "found": found})
        agg = tp.groupby("pair").agg(n_truth=("found", "size"), n_found=("found", "sum")).reset_index()
        agg["n_test"] = agg["pair"].map(n_test_by_pair).fillna(0).astype(int)
        union = agg.n_truth + agg.n_test - agg.n_found
        agg["jaccard"] = np.where(union > 0, agg.n_found / union, np.nan)
        for row in agg.itertuples(index=False):
            pair_rows.append(
                {
                    "radius": r,
                    "scope": "pair",
                    "key": row.pair,
                    "n_truth_edges": int(row.n_truth),
                    "n_test_edges": int(row.n_test),
                    "n_shared": int(row.n_found),
                    "jaccard": row.jaccard,
                }
            )

    sweep = pd.DataFrame(global_rows + type_rows + pair_rows)

    def _argmax_by(frame: pd.DataFrame, min_edges: int) -> pd.DataFrame:
        eligible = frame[frame["n_truth_edges"] >= min_edges]
        if eligible.empty:  # keep something reportable rather than nothing
            eligible = frame
        best = eligible.loc[eligible.groupby("key")["jaccard"].idxmax()].copy()
        return best.rename(columns={"radius": "best_radius"}).sort_values("key")

    g = pd.DataFrame(global_rows)
    global_best_row = g.loc[g["jaccard"].idxmax()]
    global_best = float(global_best_row["radius"])
    global_jaccard = float(global_best_row["jaccard"])

    per_type = _argmax_by(pd.DataFrame(type_rows), min_type_edges)
    per_type = per_type.rename(columns={"key": "cell_type"})[
        ["cell_type", "best_radius", "jaccard", "n_truth_edges", "missed_rate", "false_rate"]
    ].reset_index(drop=True)

    per_pair = _argmax_by(pd.DataFrame(pair_rows), min_pair_edges)
    per_pair = per_pair.rename(columns={"key": "pair"})[
        ["pair", "best_radius", "jaccard", "n_truth_edges"]
    ].reset_index(drop=True)

    keep = per_type[per_type["n_truth_edges"] >= min_type_edges]
    if keep.empty:
        keep = per_type
    radius_map = {str(r.cell_type): float(r.best_radius) for r in keep.itertuples(index=False)}

    if verbose:
        print(f"\n[radius] Global best: {global_best:g} um (Jaccard {global_jaccard:.3f})")
        print("[radius] Per cell type:")
        print(per_type.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if radius_map:
            lo, hi = min(radius_map.values()), max(radius_map.values())
            print(f"[radius] Fitted map spans {lo:g}-{hi:g} um across {len(radius_map)} types")

    best_type_jaccard = per_type["jaccard"].max() if len(per_type) else np.nan
    if np.isfinite(best_type_jaccard) and best_type_jaccard < WEAK_FIT_JACCARD:
        warnings.warn(
            f"Best per-type Jaccard is only {best_type_jaccard:.2f}: a centroid radius "
            "reproduces contact poorly in this tissue. Prefer the polygon contact "
            "graph (CCCConfig without juxtacrine_radius_map) where you can afford it.",
            UserWarning,
            stacklevel=2,
        )

    return RadiusOptimizationResult(
        radius_map=radius_map,
        per_type=per_type,
        per_pair=per_pair,
        sweep=sweep,
        global_best=global_best,
        global_jaccard=global_jaccard,
        truth_params={
            "buffer": truth_buffer,
            "coord_type": coord_type,
            "n_truth_edges": int(truth.size),
            "radii": list(radii),
            "group_col": group_col,
        },
    )


def build_typed_radius_graph(
    sp: spatioloji,
    radius_map: dict[str, float],
    group_col: str = "cell_type",
    coord_type: str = "global",
    combine: Literal["union", "intersection", "sender"] = "union",
    default_radius: float | None = None,
    verbose: bool = True,
) -> PointSpatialGraph:
    """Build a contact graph whose radius varies by cell type.

    Two cells are connected when their centroid distance satisfies the
    ``combine`` rule over their two type radii:

    - ``'union'`` (default): ``d <= max(r_i, r_j)`` — permissive; the larger
      cell's reach governs the pair.
    - ``'intersection'``: ``d <= min(r_i, r_j)`` — conservative.
    - ``'sender'``: ``d <= r_i`` — directed; each cell reaches its own radius.

    The graph carries a ``contact_frac_df`` so it drops straight into
    :func:`~spatioloji_s.ccc.scoring.score_edges` as ``graph_juxtacrine``.
    Contact fractions are **uniform 1.0**: a radius graph has distances, not
    membrane geometry, so every retained edge weighs the same and the
    juxtacrine score reduces to ``sqrt(L_i * R_j)``.

    Args:
        sp: spatioloji object with ``group_col`` in ``cell_meta``.
        radius_map: ``{cell_type: radius}``, e.g. from
            :func:`optimize_contact_radius`.
        group_col: Column in ``sp.cell_meta`` holding cell type labels.
        coord_type: ``'global'`` or ``'local'``.
        combine: How to combine the two endpoint radii (see above).
        default_radius: Radius for cell types absent from ``radius_map``.
            ``None`` (default) raises instead of guessing.
        verbose: Print a one-line graph summary.

    Returns:
        :class:`~spatioloji_s.spatial.point.graph.PointSpatialGraph` with
        ``contact_frac_df`` attached.

    Raises:
        ValueError: If ``group_col`` is missing, ``radius_map`` is empty,
            ``combine`` is unknown, or a present cell type has no radius and
            no ``default_radius`` is given.

    Example:
        >>> opt = optimize_contact_radius(sp_roi)
        >>> graph = build_typed_radius_graph(sp, opt.radius_map)
        >>> edges = score_edges(sp, pairs, graph_juxtacrine=graph)
    """
    if group_col not in sp.cell_meta.columns:
        raise ValueError(f"group_col '{group_col}' not found in cell_meta")
    if not radius_map:
        raise ValueError("radius_map must not be empty")
    if combine not in ("union", "intersection", "sender"):
        raise ValueError(f"combine must be 'union', 'intersection' or 'sender', got {combine!r}")

    from spatioloji_s.spatial.point.graph import PointSpatialGraph, build_radius_graph

    types = sp.cell_meta[group_col].astype(str).to_numpy()
    missing = sorted(set(types) - set(radius_map))
    if missing and default_radius is None:
        raise ValueError(f"No radius for cell type(s) {missing}. Add them to radius_map or pass default_radius=.")
    per_cell_radius = np.array([float(radius_map.get(t, default_radius)) for t in types], dtype=np.float64)

    # One sweep at the largest radius, then threshold each edge by its own rule.
    r_max = float(per_cell_radius.max())
    base = build_radius_graph(sp, radius=r_max, coord_type=coord_type)
    dist = base.distances.tocoo()

    ri = per_cell_radius[dist.row]
    rj = per_cell_radius[dist.col]
    if combine == "union":
        threshold = np.maximum(ri, rj)
    elif combine == "intersection":
        threshold = np.minimum(ri, rj)
    else:  # sender: row reaches its own radius
        threshold = ri
    keep = (dist.data <= threshold) & (dist.row != dist.col)

    n = sp.n_cells
    kept_dist = sparse.coo_matrix((dist.data[keep], (dist.row[keep], dist.col[keep])), shape=(n, n)).tocsr()
    if combine in ("union", "intersection"):
        # both rules are symmetric in the radii; make the stored graph symmetric too
        kept_dist = kept_dist.maximum(kept_dist.T.tocsr())
    adjacency = (kept_dist != 0).astype(np.float32)

    cell_ids = sp.cell_index
    graph = PointSpatialGraph(
        adjacency=adjacency,
        distances=kept_dist,
        cell_ids=cell_ids,
        method="typed_radius",
        params={
            "radius_map": dict(radius_map),
            "combine": combine,
            "default_radius": default_radius,
            "group_col": group_col,
            "symmetrize": combine != "sender",
        },
        coord_type=coord_type,
    )

    # Uniform contact fractions so juxtacrine scoring works unchanged (w = 1.0).
    coo = adjacency.tocoo()
    upper = coo.row < coo.col if combine != "sender" else np.ones(coo.row.size, dtype=bool)
    graph.contact_frac_df = pd.DataFrame(
        {
            "cell_a": cell_ids[coo.row[upper]],
            "cell_b": cell_ids[coo.col[upper]],
            "fraction_a": 1.0,
            "fraction_b": 1.0,
        }
    )

    if verbose:
        lo, hi = per_cell_radius.min(), per_cell_radius.max()
        print(
            f"  ✓ Typed radius graph ({combine}): radii {lo:g}-{hi:g}, "
            f"{graph.n_edges} edges, mean degree={graph.mean_degree:.1f}"
        )
    return graph
