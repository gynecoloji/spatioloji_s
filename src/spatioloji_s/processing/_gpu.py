"""
_gpu.py - GPU dispatch helpers for processing pipelines.

Provides a small set of utilities used by clustering and dimensionality
reduction modules to optionally execute on GPU via RAPIDS (cuML / cuGraph
/ cuPy / cuDF). The convention across spatioloji_s is:

- Public functions accept ``device: Literal['auto', 'cpu', 'gpu'] = 'auto'``.
- ``'auto'`` tries GPU; on missing RAPIDS or runtime failure, warns and
  falls back to CPU.
- ``'gpu'`` raises ``ImportError`` if RAPIDS is unavailable.
- All public functions always return numpy / scipy.sparse / pandas
  objects, so downstream code is unaffected by backend choice.

RAPIDS is **not** declared in pyproject.toml. Users must install it
separately into the same conda env, e.g.::

    conda install -c rapidsai -c conda-forge -c nvidia \\
        cuml=26.04 cugraph=26.04 cudf=26.04 python=3.12

See ``docs/gpu.md`` for the LD_PRELOAD note for envs that also ship
PyTorch (which bundles its own CUDA).
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy import sparse

Device = Literal["auto", "cpu", "gpu"]


@lru_cache(maxsize=1)
def _gpu_available() -> bool:
    """Probe whether RAPIDS (cuML + cuPy) can be imported.

    Cached for the lifetime of the process — RAPIDS install state does not
    change at runtime. Returns False on any ImportError or runtime CUDA
    failure (no driver, no device, OOM at probe).
    """
    try:
        import cuml  # noqa: F401
        import cupy as cp

        # Probe that we can actually allocate on a device, not just import.
        cp.zeros(1)
        return True
    except Exception:
        return False


def _resolve_device(device: Device, op_name: str) -> str:
    """Return the concrete backend ('gpu' or 'cpu') for an operation.

    Args:
        device: One of ``'auto'``, ``'cpu'``, ``'gpu'``.
        op_name: Human-readable operation name for diagnostics
            (e.g. ``'cuML PCA'``).

    Returns:
        ``'gpu'`` if GPU should be used, otherwise ``'cpu'``.

    Raises:
        ValueError: If ``device`` is not a recognised value.
        ImportError: If ``device='gpu'`` but RAPIDS is unavailable.
    """
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"device must be 'auto', 'cpu', or 'gpu', got {device!r}")

    if device == "cpu":
        return "cpu"

    available = _gpu_available()

    if device == "gpu":
        if not available:
            raise ImportError(
                f"{op_name}: device='gpu' requested but RAPIDS is not importable. "
                "Install into the same conda env, e.g.: "
                "`conda install -c rapidsai -c conda-forge -c nvidia "
                "cuml cugraph cudf`. Use device='auto' to fall back silently."
            )
        return "gpu"

    # device == 'auto'
    return "gpu" if available else "cpu"


def _to_cupy(X):
    """Move a numpy / scipy.sparse array onto the GPU.

    Dense numpy arrays become ``cupy.ndarray``. Scipy CSR/CSC become the
    corresponding ``cupyx.scipy.sparse`` matrix. Other inputs pass through.

    ``cupyx.scipy.sparse`` only supports ``bool / float32 / float64 /
    complex64 / complex128`` dtypes, so integer-typed inputs (the typical
    Xenium / CosMx raw-counts matrix) are up-promoted to ``float32``
    before being moved to the device. Integer dense inputs are likewise
    promoted to keep downstream cupy reductions in floating point.
    """
    import cupy as cp

    _SUPPORTED_SPARSE = ("?", "f", "d", "F", "D")  # bool, f32, f64, c64, c128

    if sparse.issparse(X):
        from cupyx.scipy import sparse as cusparse

        if not sparse.isspmatrix_csr(X):
            X = X.tocsr()
        data = X.data
        if data.dtype.char not in _SUPPORTED_SPARSE:
            data = data.astype(np.float32, copy=False)
        return cusparse.csr_matrix(
            (cp.asarray(data), cp.asarray(X.indices), cp.asarray(X.indptr)),
            shape=X.shape,
        )

    arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32, copy=False)
    return cp.asarray(arr)


def _to_numpy(X):
    """Move a cupy / cupyx.sparse / numpy array to the host as numpy/scipy.

    Pass-through for objects already on the host.
    """
    if X is None:
        return None

    # cupyx sparse → scipy sparse via .get()
    try:
        from cupyx.scipy import sparse as cusparse

        if isinstance(X, cusparse.spmatrix):
            return X.get()
    except ImportError:
        pass

    # cupy ndarray exposes .get()
    if hasattr(X, "get") and not isinstance(X, np.ndarray):
        try:
            return X.get()
        except Exception:
            pass

    if sparse.issparse(X) or isinstance(X, np.ndarray):
        return X

    return np.asarray(X)


def _gpu_install_hint(component: str) -> str:
    """Standard install-hint string for missing RAPIDS components."""
    return (
        f"{component} requires RAPIDS in the same conda env. Install with: "
        "`conda install -c rapidsai -c conda-forge -c nvidia cuml cugraph cudf`. "
        "Pass device='cpu' or device='auto' to use the CPU backend instead."
    )


def _warn_fallback(op_name: str, exc: Exception) -> None:
    """Emit a one-line warning when an auto GPU attempt fell back to CPU."""
    warnings.warn(
        f"{op_name}: GPU backend unavailable ({type(exc).__name__}: {exc}). "
        "Falling back to CPU.",
        UserWarning,
        stacklevel=3,
    )
