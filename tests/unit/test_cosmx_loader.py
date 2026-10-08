"""Tests for spatioloji.from_cosmx.

The fixture writes a miniature CosMx SMI export rather than depending on a real one. The
schema mirrors a genuine NanoString flat-file export (verified against a Run5452_S2 ovarian
section): per-FOV cell_ID numbering starting at 1 with a cell_ID==0 background row per FOV,
NegPrb control features, and polygon columns already in spatioloji's own names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatioloji_s import spatioloji

N_FOV, N_PER_FOV, N_GENES, N_NEG = 3, 4, 5, 2


@pytest.fixture
def cosmx_dir(tmp_path):
    genes = [f"GENE{i}" for i in range(N_GENES)]
    negs = [f"NegPrb{i}" for i in range(1, N_NEG + 1)]
    rng = np.random.default_rng(0)

    expr_rows, meta_rows, poly_rows = [], [], []
    for fov in range(1, N_FOV + 1):
        # cell_ID 0 is CosMx background: present in exprMat, absent from metadata.
        for cid in range(0, N_PER_FOV + 1):
            counts = rng.integers(0, 30, size=N_GENES + N_NEG)
            expr_rows.append({"fov": fov, "cell_ID": cid,
                              **dict(zip(genes + negs, counts, strict=True))})
            if cid == 0:
                continue
            lx, ly = cid * 10.0, cid * 4.0
            gx, gy = lx + fov * 1000.0, ly + fov * 500.0
            meta_rows.append({"fov": fov, "cell_ID": cid, "Area": 100 + cid,
                              "CenterX_local_px": lx, "CenterY_local_px": ly,
                              "CenterX_global_px": gx, "CenterY_global_px": gy})
            for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                poly_rows.append({"fov": fov, "cell_ID": cid,
                                  "x_local_px": lx + dx, "y_local_px": ly + dy,
                                  "x_global_px": gx + dx, "y_global_px": gy + dy})

    pd.DataFrame(expr_rows).to_csv(tmp_path / "Run1_S1_exprMat_file.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(tmp_path / "Run1_S1_metadata_file.csv", index=False)
    pd.DataFrame(poly_rows).to_csv(tmp_path / "Run1_S1-polygons.csv", index=False)
    pd.DataFrame({"fov": range(1, N_FOV + 1),
                  "x_global_px": [f * 1000.0 for f in range(1, N_FOV + 1)],
                  "y_global_px": [f * 500.0 for f in range(1, N_FOV + 1)]}
                 ).to_csv(tmp_path / "Run1_S1_fov_positions_file.csv", index=False)
    return tmp_path


def test_loads_cells_genes_fovs(cosmx_dir):
    sp = spatioloji.from_cosmx(str(cosmx_dir), images_folder=None)
    assert sp.n_cells == N_FOV * N_PER_FOV
    assert sp.n_genes == N_GENES, "NegPrb controls should be dropped by default"
    assert sp.cell_meta["fov"].nunique() == N_FOV


def test_background_cell_id_zero_is_dropped(cosmx_dir):
    """cell_ID == 0 is CosMx background: one row per FOV in exprMat, not a cell."""
    sp = spatioloji.from_cosmx(str(cosmx_dir), images_folder=None)
    assert sp.n_cells == N_FOV * N_PER_FOV          # not N_FOV * (N_PER_FOV + 1)
    assert not any(str(c).endswith("_0") for c in sp.cell_index)


def test_cell_ids_are_fov_scoped(cosmx_dir):
    """cell_ID repeats across FOVs; the key must be (fov, cell_ID) or cells collide."""
    raw = pd.read_csv(cosmx_dir / "Run1_S1_metadata_file.csv")
    assert not raw["cell_ID"].is_unique, "fixture must exercise the collision"

    sp = spatioloji.from_cosmx(str(cosmx_dir), images_folder=None)
    assert len(set(sp.cell_index)) == sp.n_cells, "composite key must disambiguate"
    assert "1_1" in set(sp.cell_index) and "2_1" in set(sp.cell_index)


def test_local_and_global_coordinates_differ(cosmx_dir):
    """Unlike Xenium/MERSCOPE, CosMx tiles FOVs, so the two frames are not the same."""
    sp = spatioloji.from_cosmx(str(cosmx_dir), images_folder=None)
    assert not np.allclose(sp.spatial.x_local, sp.spatial.x_global)
    assert not np.allclose(sp.spatial.y_local, sp.spatial.y_global)


def test_negative_probes_separated_and_counted(cosmx_dir):
    sp = spatioloji.from_cosmx(str(cosmx_dir), images_folder=None)
    assert "negprobe_counts" in sp.cell_meta.columns
    assert not any(str(g).lower().startswith("negprb") for g in sp.gene_index)

    kept = spatioloji.from_cosmx(str(cosmx_dir), images_folder=None,
                                 drop_negative_probes=False)
    assert kept.n_genes == N_GENES + N_NEG
    assert int(kept.gene_meta["is_negative"].sum()) == N_NEG


def test_polygons_attach_to_every_cell(cosmx_dir):
    sp = spatioloji.from_cosmx(str(cosmx_dir), images_folder=None)
    assert sp.polygons is not None
    assert sp.polygons["cell"].nunique() == sp.n_cells
    assert len(sp.polygons) == sp.n_cells * 4


def test_fov_positions_loaded(cosmx_dir):
    sp = spatioloji.from_cosmx(str(cosmx_dir), images_folder=None)
    assert sp.fov_positions is not None and len(sp.fov_positions) == N_FOV


def test_files_discovered_regardless_of_run_prefix(cosmx_dir, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    for f in cosmx_dir.glob("Run1_S1*"):
        f.rename(other / f.name.replace("Run1_S1", "Run9999_S7"))
    sp = spatioloji.from_cosmx(str(other), images_folder=None)
    assert sp.n_cells == N_FOV * N_PER_FOV


def test_missing_directory_and_matrix_raise(cosmx_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="CosMx directory not found"):
        spatioloji.from_cosmx(str(tmp_path / "nope"))

    (cosmx_dir / "Run1_S1_exprMat_file.csv").unlink()
    with pytest.raises(FileNotFoundError, match="expression matrix"):
        spatioloji.from_cosmx(str(cosmx_dir), images_folder=None)


def test_two_exports_in_one_directory_raise(cosmx_dir):
    """Globbing must not silently pick one of two runs."""
    import shutil
    shutil.copy(cosmx_dir / "Run1_S1_exprMat_file.csv",
                cosmx_dir / "Run2_S1_exprMat_file.csv")
    with pytest.raises(ValueError, match="Multiple expression matrix files"):
        spatioloji.from_cosmx(str(cosmx_dir), images_folder=None)


def test_mismatched_metadata_raises(cosmx_dir):
    m = pd.read_csv(cosmx_dir / "Run1_S1_metadata_file.csv")
    m["fov"] = m["fov"] + 50
    m.to_csv(cosmx_dir / "Run1_S1_metadata_file.csv", index=False)
    with pytest.raises(ValueError, match="do not describe the same run"):
        spatioloji.from_cosmx(str(cosmx_dir), images_folder=None)


def test_load_boundaries_false_skips_polygons(cosmx_dir):
    sp = spatioloji.from_cosmx(str(cosmx_dir), images_folder=None, load_boundaries=False)
    assert sp.polygons is None
