"""
conftest.py - Shared test fixtures for spatioloji

What is this file?
------------------
pytest automatically reads this file before running any test.
Anything defined here (called a "fixture") is available to ALL test files
without needing to import it — pytest injects it automatically.

Think of fixtures as reusable Lego blocks:
  - You build the block once here
  - Every test that needs it just asks for it by name
  - pytest hands it over automatically

How fixtures work:
------------------
    @pytest.fixture          ← decorator that marks a reusable setup block
    def my_fixture():
        return something_useful

    def test_something(my_fixture):   ← pytest sees the name and injects it
        assert my_fixture == expected
"""

import numpy as np
import pandas as pd
import pytest
from spatioloji_s.data.core import spatioloji

# ===========================================================================
# Constants — the size of our fake dataset
# ===========================================================================

N_CELLS = 60  # total fake cells (30 per FOV)
N_GENES = 25  # total fake genes
N_FOVS = 2  # two fake FOVs (field of views)


# ===========================================================================
# Fixture 1: minimal spatioloji object (no polygons, no images)
# ===========================================================================


@pytest.fixture
def sp_basic():
    """
    The simplest possible spatioloji object.

    Use this when a test only needs:
      - expression matrix
      - cell/gene metadata
      - spatial coordinates
      - FOV labels

    60 cells × 25 genes, split across 2 FOVs.
    """

    np.random.seed(42)  # fix random seed → reproducible

    # --- Expression matrix (cells × genes) ---
    # We use integers 0-9 so the matrix is sparse-ish (many zeros)
    expression = np.random.randint(0, 5, (N_CELLS, N_GENES)).astype(float)

    # --- Cell and gene identifiers ---
    cell_ids = [f"cell_{i}" for i in range(N_CELLS)]
    gene_names = [f"gene_{i}" for i in range(N_GENES)]

    # --- Cell metadata ---
    # fov: first 30 cells → FOV "1", next 30 → FOV "2"
    # cell_type: randomly assigned A or B
    cell_meta = pd.DataFrame(
        {
            "fov": ["1"] * 30 + ["2"] * 30,
            "cell_type": np.random.choice(["TypeA", "TypeB"], N_CELLS),
            "n_counts": expression.sum(axis=1),  # total counts per cell
        },
        index=cell_ids,
    )

    # --- Gene metadata ---
    # NegProbe: last 2 genes are negative control probes
    gene_meta = pd.DataFrame(
        {
            "NegProbe": [False] * (N_GENES - 2) + [True, True],
        },
        index=gene_names,
    )

    # --- Spatial coordinates ---
    # global x/y: scattered in a 1000×1000 space
    # local x/y:  within each FOV (0–500)
    spatial = {
        "x_global": np.random.uniform(0, 1000, N_CELLS),
        "y_global": np.random.uniform(0, 1000, N_CELLS),
        "x_local": np.random.uniform(0, 500, N_CELLS),
        "y_local": np.random.uniform(0, 500, N_CELLS),
    }

    # --- FOV positions ---
    fov_positions = pd.DataFrame(
        {
            "x_offset": [0, 500],
            "y_offset": [0, 500],
        },
        index=pd.Index(["1", "2"], name="fov"),
    )

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        gene_metadata=gene_meta,
        spatial_coords=spatial,
        fov_positions=fov_positions,
    )


# ===========================================================================
# Fixture 2: spatioloji object WITH polygon data
# ===========================================================================


@pytest.fixture
def sp_with_polygons(sp_basic):
    """
    Same as sp_basic but with fake cell boundary polygons attached.

    Each cell gets a tiny square polygon (4 vertices) around its centroid.
    Use this for any test that touches sp.polygons.

    Builds on sp_basic so we don't duplicate setup code — this is called
    a 'fixture that depends on another fixture'.
    """

    x = sp_basic.spatial.x_global
    y = sp_basic.spatial.y_global
    cell_ids = list(sp_basic.cell_index)

    rows = []
    for cid, cx, cy in zip(cell_ids, x, y, strict=False):
        # A tiny 5×5 square around the centroid — 4 corners + closing vertex
        for vx, vy in [
            (cx - 2, cy - 2),
            (cx + 2, cy - 2),
            (cx + 2, cy + 2),
            (cx - 2, cy + 2),
            (cx - 2, cy - 2),
        ]:  # close the polygon
            rows.append({"cell": cid, "x_global_px": vx, "y_global_px": vy})

    polygons = pd.DataFrame(rows)

    # Rebuild spatioloji with polygons attached
    np.random.seed(42)
    expression = np.random.randint(0, 5, (N_CELLS, N_GENES)).astype(float)
    cell_ids_list = [f"cell_{i}" for i in range(N_CELLS)]
    gene_names = [f"gene_{i}" for i in range(N_GENES)]

    cell_meta = pd.DataFrame(
        {
            "fov": ["1"] * 30 + ["2"] * 30,
            "cell_type": np.random.choice(["TypeA", "TypeB"], N_CELLS),
            "n_counts": expression.sum(axis=1),
        },
        index=cell_ids_list,
    )

    gene_meta = pd.DataFrame(
        {
            "NegProbe": [False] * (N_GENES - 2) + [True, True],
        },
        index=gene_names,
    )

    spatial = {
        "x_global": x,
        "y_global": y,
        "x_local": sp_basic.spatial.x_local,
        "y_local": sp_basic.spatial.y_local,
    }

    fov_positions = pd.DataFrame(
        {
            "x_offset": [0, 500],
            "y_offset": [0, 500],
        },
        index=pd.Index(["1", "2"], name="fov"),
    )

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids_list,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        gene_metadata=gene_meta,
        spatial_coords=spatial,
        polygons=polygons,
        fov_positions=fov_positions,
    )


# ===========================================================================
# Fixture 3: sparse-specific object (highly sparse expression matrix)
# ===========================================================================


@pytest.fixture
def sp_sparse():
    """
    spatioloji object where the expression matrix is highly sparse (>80% zeros).

    spatioloji auto-converts dense → sparse when sparsity > 50%.
    Use this to test that sparse storage works correctly.
    """

    np.random.seed(0)

    # Only ~15% of entries are non-zero → triggers auto-sparse conversion
    expression = np.random.choice(
        [0, 0, 0, 0, 0, 1, 2, 3],  # mostly zeros
        size=(N_CELLS, N_GENES),
    ).astype(float)

    cell_ids = [f"cell_{i}" for i in range(N_CELLS)]
    gene_names = [f"gene_{i}" for i in range(N_GENES)]

    cell_meta = pd.DataFrame(
        {
            "fov": ["1"] * 30 + ["2"] * 30,
        },
        index=cell_ids,
    )

    spatial = {
        "x_global": np.random.uniform(0, 1000, N_CELLS),
        "y_global": np.random.uniform(0, 1000, N_CELLS),
        "x_local": np.random.uniform(0, 500, N_CELLS),
        "y_local": np.random.uniform(0, 500, N_CELLS),
    }

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        spatial_coords=spatial,
    )
