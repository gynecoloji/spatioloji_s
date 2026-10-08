"""Tests for spatioloji.from_merscope.

The fixture writes a miniature MERSCOPE export rather than depending on a real one, so
these run without the 275 MB public dataset. The schema mirrors a genuine Vizgen export
(verified against the human fetal/pediatric colon dataset, Zenodo 10.5281/zenodo.19450420):
cell_by_gene.csv keyed on 'cell', cell_metadata.csv keyed on EntityID with center_x/center_y
and an fov column, and cell_boundaries.parquet carrying WKB MultiPolygons once per z-plane.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatioloji_s import spatioloji

shapely = pytest.importorskip("shapely")
from shapely import wkb  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

N_CELLS, N_GENES, N_BLANK, N_Z = 6, 4, 2, 8


def _square(cx, cy, half):
    return Polygon([(cx - half, cy - half), (cx + half, cy - half),
                    (cx + half, cy + half), (cx - half, cy + half)])


@pytest.fixture
def merscope_dir(tmp_path):
    ids = [f"{4057517800002100000 + i}" for i in range(N_CELLS)]
    genes = [f"GENE{i}" for i in range(N_GENES)]
    blanks = [f"Blank-{i}" for i in range(1, N_BLANK + 1)]

    rng = np.random.default_rng(0)
    cbg = pd.DataFrame(rng.integers(0, 20, size=(N_CELLS, N_GENES + N_BLANK)),
                       index=pd.Index(ids, name="cell"), columns=genes + blanks)
    cbg.to_csv(tmp_path / "cell_by_gene.csv")

    meta = pd.DataFrame({
        "fov": [0, 0, 1, 1, 2, 2],
        "volume": np.linspace(100, 200, N_CELLS),
        "center_x": np.arange(N_CELLS, dtype=float) * 10.0,
        "center_y": np.arange(N_CELLS, dtype=float) * 5.0,
    }, index=pd.Index(ids, name="EntityID"))
    meta.to_csv(tmp_path / "cell_metadata.csv")

    # One polygon per cell per z-plane. The half-width grows with z, so the
    # largest-area plane is unambiguously the last one (z = N_Z - 1).
    rows = []
    for zi in range(N_Z):
        for k, cid in enumerate(ids):
            poly = _square(k * 10.0, k * 5.0, 1.0 + 0.5 * zi)
            rows.append({"ID": zi * N_CELLS + k, "EntityID": int(cid), "ZIndex": zi,
                         "Geometry": wkb.dumps(MultiPolygon([poly])), "Type": "cell",
                         "ZLevel": 1.5 * (zi + 1), "Name": None,
                         "ParentID": None, "ParentType": None})
    pd.DataFrame(rows).to_parquet(tmp_path / "cell_boundaries.parquet")
    return tmp_path


def test_loads_cells_genes_and_fovs(merscope_dir):
    sp = spatioloji.from_merscope(str(merscope_dir))
    assert sp.n_cells == N_CELLS
    assert sp.n_genes == N_GENES, "Blank probes should be dropped by default"
    assert sp.cell_meta["fov"].nunique() == 3, "FOVs come from the data, not a dummy"


def test_cell_metadata_actually_joins(merscope_dir):
    """Regression: the default config names the cell column 'cell', so a loader that
    does not declare cell_id_col silently drops every metadata row and every polygon."""
    sp = spatioloji.from_merscope(str(merscope_dir))
    assert len(sp.cell_meta) == N_CELLS
    assert not sp.cell_meta["center_x"].isna().any()
    assert sp.polygons is not None and sp.polygons["cell_id"].nunique() == N_CELLS


def test_blank_probes_separated_and_counted(merscope_dir):
    sp = spatioloji.from_merscope(str(merscope_dir))
    assert "blank_counts" in sp.cell_meta.columns
    assert (sp.cell_meta["blank_counts"] >= 0).all()
    assert not any(str(g).lower().startswith("blank") for g in sp.gene_index)

    kept = spatioloji.from_merscope(str(merscope_dir), drop_blank_probes=False)
    assert kept.n_genes == N_GENES + N_BLANK
    assert bool(kept.gene_meta["is_blank"].sum() == N_BLANK)


def test_coordinates_are_microns_and_local_equals_global(merscope_dir):
    sp = spatioloji.from_merscope(str(merscope_dir))
    np.testing.assert_allclose(sp.spatial.x_global, np.arange(N_CELLS) * 10.0)
    np.testing.assert_allclose(sp.spatial.x_local, sp.spatial.x_global)
    np.testing.assert_allclose(sp.spatial.y_local, sp.spatial.y_global)


def test_z_plane_selection(merscope_dir):
    """Boundaries exist once per z-plane; exactly one plane must reach the object."""
    mid = spatioloji.from_merscope(str(merscope_dir))                      # 'middle'
    low = spatioloji.from_merscope(str(merscope_dir), z_index=0)
    big = spatioloji.from_merscope(str(merscope_dir), z_index="max_area")

    def width(sp):
        g = sp.polygons[sp.polygons.cell_id == sp.polygons.cell_id.iloc[0]]
        return g["x_global_px"].max() - g["x_global_px"].min()

    assert width(low) == pytest.approx(2.0)                 # half = 1.0
    assert width(mid) == pytest.approx(2.0 + 2 * 0.5 * (N_Z // 2))
    assert width(big) == pytest.approx(2.0 + 2 * 0.5 * (N_Z - 1)), "largest plane"
    # one polygon per cell, not N_Z of them
    for sp in (mid, low, big):
        assert len(sp.polygons) == N_CELLS * 5             # 4 corners + closing vertex


def test_bad_z_index_raises(merscope_dir):
    with pytest.raises(ValueError, match="not present"):
        spatioloji.from_merscope(str(merscope_dir), z_index=99)


def test_missing_files_raise_clearly(merscope_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="MERSCOPE directory not found"):
        spatioloji.from_merscope(str(tmp_path / "nope"))

    (merscope_dir / "cell_by_gene.csv").unlink()
    with pytest.raises(FileNotFoundError, match="cell_by_gene.csv"):
        spatioloji.from_merscope(str(merscope_dir))


def test_mismatched_cells_raise_rather_than_fill_nan(merscope_dir):
    """A metadata file from a different run must fail loudly, not produce NaN coords."""
    meta = pd.read_csv(merscope_dir / "cell_metadata.csv", index_col=0)
    meta.index = [str(int(i) + 999) for i in meta.index]
    meta.to_csv(merscope_dir / "cell_metadata.csv")
    with pytest.raises(ValueError, match="do not describe the same run"):
        spatioloji.from_merscope(str(merscope_dir))


def test_legacy_boundary_directory_is_reported(merscope_dir):
    (merscope_dir / "cell_boundaries.parquet").unlink()
    (merscope_dir / "cell_boundaries").mkdir()
    with pytest.raises(FileNotFoundError, match="legacy per-FOV boundary directory"):
        spatioloji.from_merscope(str(merscope_dir))


def test_load_boundaries_false_skips_polygons(merscope_dir):
    sp = spatioloji.from_merscope(str(merscope_dir), load_boundaries=False)
    assert sp.polygons is None
    assert sp.n_cells == N_CELLS


def test_largest_polygon_part_wins(merscope_dir):
    """A MultiPolygon with a fragment must yield the real cell, not the fragment."""
    b = pd.read_parquet(merscope_dir / "cell_boundaries.parquet")
    big, tiny = _square(0.0, 0.0, 5.0), _square(500.0, 500.0, 0.1)
    b.loc[b.index[0], "Geometry"] = wkb.dumps(MultiPolygon([tiny, big]))
    b.loc[b.index[0], "ZIndex"] = 0
    b.to_parquet(merscope_dir / "cell_boundaries.parquet")

    sp = spatioloji.from_merscope(str(merscope_dir), z_index=0)
    first = str(sorted(int(c) for c in sp.polygons.cell_id.unique())[0])
    g = sp.polygons[sp.polygons.cell_id == first]
    assert g["x_global_px"].max() - g["x_global_px"].min() == pytest.approx(10.0)
