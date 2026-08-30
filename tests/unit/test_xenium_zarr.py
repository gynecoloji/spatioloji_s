"""Tests for from_xenium's zarr reader.

Covers the REAL 10x Xenium Onboard Analysis layout::

    cell_feature_matrix.zarr.zip
    └── cell_features/            (group)
        ├── attrs: feature_ids, feature_keys, feature_types
        ├── cell_id   (n_cells, 2) uint32   [encoded id, version]
        ├── data      feature-major CSR values
        ├── indices   column (cell) indices
        └── indptr    (n_features + 1,)

and the legacy AnnData-style layout (``X/data`` + ``obs``/``var``) that the
pre-0.4.8 reader expected. Both must load through ``from_xenium`` on zarr 2
and zarr 3.
"""

import gzip

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

zarr = pytest.importorskip("zarr")

from spatioloji_s.data.core import _decode_xenium_cell_ids, spatioloji  # noqa: E402

# ── Fixture data ─────────────────────────────────────────────────────────────

# Real (id, version) → string pairs taken from a public Xenium dataset
# (Xenium Prime Human Lymph Node): 992 → "aaaaadoa", 11125 → "aaaaclhf".
KNOWN_IDS = np.array([[992, 1], [11125, 1], [21081, 1]], dtype=np.uint32)
KNOWN_STRS = ["aaaaadoa-1", "aaaaclhf-1", "aaaafcfj-1"]

GENES = ["GENE_A", "GENE_B", "NegControlProbe_00001", "GENE_C"]
GENE_IDS = ["ENSG000001", "ENSG000002", "NegControlProbe_00001", "ENSG000003"]
FEATURE_TYPES = ["Gene Expression", "Gene Expression", "Negative Control Probe", "Gene Expression"]

# cells × genes dense counts (3 cells, 4 features)
DENSE = np.array(
    [
        [5, 0, 1, 0],
        [0, 2, 0, 7],
        [3, 0, 0, 1],
    ],
    dtype=np.uint32,
)


def _new_zip_group(path):
    """Open a writable zarr group backed by a fresh zip file (zarr 2 + 3)."""
    try:  # zarr >= 2.13 exposes it here; zarr 3 only here
        from zarr.storage import ZipStore
    except ImportError:  # pragma: no cover - very old zarr 2
        ZipStore = zarr.ZipStore
    store = ZipStore(str(path), mode="w")
    return zarr.group(store=store), store


def _put_array(group, name, data):
    """Create an array in a zarr group across zarr 2/3 APIs."""
    data = np.asarray(data)
    if hasattr(group, "create_array"):  # zarr 3
        arr = group.create_array(name, shape=data.shape, dtype=data.dtype)
        arr[:] = data
    else:  # zarr 2
        group.array(name, data)


def _write_cells_csv(xenium_dir, cell_ids):
    cells = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "x_centroid": [10.0, 20.0, 30.0],
            "y_centroid": [1.0, 2.0, 3.0],
            "transcript_counts": DENSE.sum(axis=1),
            "cell_area": [50.0, 60.0, 70.0],
        }
    )
    with gzip.open(xenium_dir / "cells.csv.gz", "wt") as fh:
        cells.to_csv(fh, index=False)


@pytest.fixture
def xenium_10x_zarr_dir(tmp_path):
    """Minimal Xenium output dir whose matrix uses the real 10x zarr layout."""
    _write_cells_csv(tmp_path, KNOWN_STRS)

    root, store = _new_zip_group(tmp_path / "cell_feature_matrix.zarr.zip")
    cf = root.create_group("cell_features")
    # Real XOA zarr carries one extra synthetic 'aggregate_gene' feature
    # holding each cell's total gene counts — the loader must drop it.
    cf.attrs["feature_ids"] = GENE_IDS + ["aggregate_gene"]
    cf.attrs["feature_keys"] = GENES + ["aggregate_gene"]
    cf.attrs["feature_types"] = FEATURE_TYPES + ["aggregate_gene"]
    _put_array(cf, "cell_id", KNOWN_IDS)
    dense_with_agg = np.hstack([DENSE, DENSE.sum(axis=1, keepdims=True)])
    fm = sps.csr_matrix(dense_with_agg.T)  # feature-major, as 10x ships it
    _put_array(cf, "data", fm.data.astype(np.uint32))
    _put_array(cf, "indices", fm.indices.astype(np.uint32))
    _put_array(cf, "indptr", fm.indptr.astype(np.uint32))
    store.close()
    return tmp_path


@pytest.fixture
def xenium_legacy_zarr_dir(tmp_path):
    """Xenium dir whose zarr.zip uses the AnnData-style layout (pre-0.4.8 expectation)."""
    _write_cells_csv(tmp_path, KNOWN_STRS)

    root, store = _new_zip_group(tmp_path / "cell_feature_matrix.zarr.zip")
    x = root.create_group("X")
    cm = sps.csc_matrix(DENSE)  # cells × genes CSC ... transposed to (genes, cells)?
    # Legacy reader does: csc_matrix((data, indices, indptr), shape=attrs["shape"]).T.tocsr()
    # so store genes × cells CSC and shape (n_genes, n_cells).
    gm = sps.csc_matrix(DENSE.T)
    _put_array(x, "data", gm.data)
    _put_array(x, "indices", gm.indices)
    _put_array(x, "indptr", gm.indptr)
    x.attrs["shape"] = [DENSE.shape[1], DENSE.shape[0]]
    obs = root.create_group("obs")
    _put_array(obs, "cell_id", np.array(KNOWN_STRS, dtype="U16"))
    var = root.create_group("var")
    _put_array(var, "feature_name", np.array(GENES, dtype="U32"))
    _put_array(var, "gene_ids", np.array(GENE_IDS, dtype="U32"))
    store.close()
    del cm
    return tmp_path


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDecodeCellIds:
    def test_known_pairs(self):
        assert _decode_xenium_cell_ids(KNOWN_IDS) == KNOWN_STRS

    def test_roundtrip_alphabet(self):
        # id 0 → all-'a'; id 0xFFFFFFFF → all-'p'
        arr = np.array([[0, 1], [0xFFFFFFFF, 2]], dtype=np.uint32)
        assert _decode_xenium_cell_ids(arr) == ["aaaaaaaa-1", "pppppppp-2"]


class TestFromXenium10xZarr:
    def test_loads_matrix_and_ids(self, xenium_10x_zarr_dir):
        sp = spatioloji.from_xenium(
            str(xenium_10x_zarr_dir), load_boundaries=False, matrix_type="zarr"
        )
        assert sp.n_cells == 3
        assert sp.n_genes == 4
        assert list(sp.cell_index) == KNOWN_STRS
        assert list(sp.gene_index) == GENES
        np.testing.assert_array_equal(
            np.asarray(sp.get_expression()), DENSE
        )

    def test_gene_meta_has_ids_and_types(self, xenium_10x_zarr_dir):
        sp = spatioloji.from_xenium(
            str(xenium_10x_zarr_dir), load_boundaries=False, matrix_type="zarr"
        )
        assert list(sp.gene_meta["gene_id"]) == GENE_IDS
        assert list(sp.gene_meta["feature_type"]) == FEATURE_TYPES

    def test_auto_detect_prefers_zarr(self, xenium_10x_zarr_dir):
        sp = spatioloji.from_xenium(
            str(xenium_10x_zarr_dir), load_boundaries=False, matrix_type="auto"
        )
        assert sp.n_cells == 3
        assert sp.n_genes == 4


class TestFromXeniumLegacyZarr:
    def test_loads_anndata_style_layout(self, xenium_legacy_zarr_dir):
        sp = spatioloji.from_xenium(
            str(xenium_legacy_zarr_dir), load_boundaries=False, matrix_type="zarr"
        )
        assert sp.n_cells == 3
        assert list(sp.cell_index) == KNOWN_STRS
        assert list(sp.gene_index) == GENES
        np.testing.assert_array_equal(np.asarray(sp.get_expression()), DENSE)
