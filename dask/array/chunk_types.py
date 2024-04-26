from __future__ import annotations

import numpy as np

# Start list of valid chunk types, to be added to with guarded imports
_HANDLED_CHUNK_TYPES = [np.ndarray, np.ma.MaskedArray]


def register_chunk_type(type):
    """Register the given type as a valid chunk and downcast array type

    Parameters
    ----------
    type : type
        Duck array type to be registered as a type Dask can safely wrap as a chunk and
        to which Dask does not defer in arithmetic operations and NumPy
        functions/ufuncs.

    Notes
    -----
    A :py:class:`dask.array.Array` can contain any sufficiently "NumPy-like" array in
    its chunks. These are also referred to as "duck arrays" since they match the most
    important parts of NumPy's array API, and so, behave the same way when relying on
    duck typing.

    However, for multiple duck array types to interoperate properly, they need to
    properly defer to each other in arithmetic operations and NumPy functions/ufuncs
    according to a well-defined type casting hierarchy (
    `see NEP 13 <https://numpy.org/neps/nep-0013-ufunc-overrides.html#type-casting-hierarchy>`__
    ). In an effort to maintain this hierarchy, Dask defers to all other duck array
    types except those in its internal registry. By default, this registry contains

    * :py:class:`numpy.ndarray`


    >>> da.register_chunk_type(FlaggedArray)
    >>> x = da.ones(5) - FlaggedArray(np.ones(5), True)
    >>> x
    dask.array<sub, shape=(5,), dtype=float64, chunksize=(5,), chunktype=dask.FlaggedArray>
    >>> x.compute()
    Flag: True, Array: array([0., 0., 0., 0., 0.])
    """
    _HANDLED_CHUNK_TYPES.append(type)


try:
    import cupy

    register_chunk_type(cupy.ndarray)
except ImportError:
    pass

try:
    from cupyx.scipy.sparse import spmatrix

    register_chunk_type(spmatrix)
except ImportError:
    pass

try:
    import sparse

    register_chunk_type(sparse.SparseArray)
except ImportError:
    pass

try:
    import scipy.sparse

    register_chunk_type(scipy.sparse.spmatrix)
except ImportError:
    pass


def is_valid_chunk_type(type):
    """Check if given type is a valid chunk and downcast array type"""
    try:
        return type in _HANDLED_CHUNK_TYPES or issubclass(
            type, tuple(_HANDLED_CHUNK_TYPES)
        )
    except TypeError:
        return False


def is_valid_array_chunk(array):
    """Check if given array is of a valid type to operate with"""
    return array is None or isinstance(array, tuple(_HANDLED_CHUNK_TYPES))
