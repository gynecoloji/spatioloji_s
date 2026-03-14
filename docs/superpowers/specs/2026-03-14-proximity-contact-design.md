# Proximity Contact Mode for Buffer Graphs — Design Spec

**Date:** 2026-03-14
**Status:** Approved
**Module:** `spatial/polygon/boundaries.py` (modify existing)

---

## 1. Goal

When `contact_length`, `contact_fraction`, and `free_boundary_fraction` receive a buffer graph, automatically use buffered neighbor polygons for intersection instead of raw polygons. This makes "contact" metrics meaningful for non-touching cell pairs connected via buffer distance.

## 2. Current Behavior

`_boundary_intersection_length(cell_A, cell_B)` computes `cell_A.boundary.intersection(cell_B)`. For buffer graph pairs that don't physically touch, this returns 0 — making contact_length/fraction useless for buffer graphs.

## 3. Design

### 3.1 Auto-detection

The graph's `.method` attribute is `"contact"`, `"buffer"`, or `"knn"`. When `graph.method == "buffer"`, enable proximity mode. The buffer distance is read from `graph.params["buffer_distance"]`.

### 3.2 Algorithm Change

**Before intersection, buffer the neighbor polygon:**

```python
# Existing (contact/knn graphs):
seg = cell_A.boundary.intersection(cell_B)

# Proximity mode (buffer graphs):
seg = cell_A.boundary.intersection(cell_B_buffered)
```

Where `cell_B_buffered = cell_B.buffer(buffer_distance)`.

### 3.3 Caching

Pre-compute buffered polygons once per cell to avoid redundant buffer operations:

```python
buffer_distance = graph.params["buffer_distance"]
buffered_polys = {cell_id: geom.buffer(buffer_distance) for cell_id, geom in gdf.geometry.items()}
```

This is O(n_cells), not O(n_pairs). The per-pair intersection cost is unchanged.

### 3.4 Functions Modified

**`_boundary_intersection_length`** — No change to its signature. Callers pass the buffered polygon as `geom_nbr` when in proximity mode.

**`contact_length`** — Detect buffer graph. If so, build `buffered_polys` cache. For each pair, pass `buffered_polys[cell_b]` instead of `geom.loc[cell_b]` to `_boundary_intersection_length`. Column names remain `contact_length_a`, `contact_length_b` — semantically these now represent "facing boundary length within buffer distance."

**`_collect_contact_segments`** — Same change: use buffered neighbor polys when graph is buffer type.

**`contact_fraction`** — No code change needed. It calls `contact_length` internally, which handles the buffering. The fraction computation (`contact_length / perimeter`) is still correct — it now represents "what fraction of my perimeter faces a neighbor within buffer distance."

**`free_boundary_fraction`** — No code change needed. It calls `_collect_contact_segments` internally. The free boundary fraction now represents "what fraction of my perimeter does NOT face any neighbor within buffer distance."

**`contact_summary`** — No code change needed. It calls `contact_length` internally.

### 3.5 Print Output

Update the `[Boundaries]` print messages to indicate proximity mode:

```
[Boundaries] Computing contact lengths (proximity mode, buffer=50.0)...
```

### 3.6 No New Functions, No New Columns, No API Changes

The existing API is unchanged. The behavior adapts based on the graph type. Users who already pass buffer graphs to these functions get meaningful results instead of zeros — no code changes needed on their side.

## 4. Edge Cases

| Condition | Behavior |
|-----------|----------|
| `graph.method == "contact"` or `"knn"` | No change — existing behavior |
| `graph.method == "buffer"` but `"buffer_distance"` not in params | Fall back to non-buffered mode with a `UserWarning` |
| Buffered polygon fully encloses neighbor | `contact_length` = full perimeter of the enclosed cell (correct) |
| Two cells already touching + buffer graph | Buffered intersection >= direct intersection (also correct — buffer expands the contact surface) |

## 5. Testing

- Test that buffer graph pairs with non-touching cells now get `contact_length > 0`
- Test that contact graph behavior is unchanged
- Test that `contact_fraction` values are in [0, 1]
- Test that `free_boundary_fraction` + `total_contact_fraction` = 1.0
- Test the edge case: buffer_distance missing from params → warning + fallback
