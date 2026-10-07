"""Shared array and integer-domain contracts for MLFCS numerical code."""

import math
import operator

import numpy as np

INT64_MAX = (1 << 63) - 1
INTP_MAX = int(np.iinfo(np.intp).max)


def require_bound(name: str, value: int, limit: int = INT64_MAX) -> int:
    """Require a nonnegative integer bound to fit within a fixed-width limit.

    Parameters
    ----------
    name
        Description used in error messages.
    value
        Nonnegative bound to validate.
    limit
        Largest permitted value.

    Returns
    -------
    int
        The validated bound.

    Raises
    ------
    OverflowError
        If ``value`` is negative or exceeds ``limit``.
    """
    value = operator.index(value)
    if not 0 <= value <= limit:
        raise OverflowError(f"{name}: proven bound {value} exceeds allowed range 0..{limit}")
    return value


def require_allocation(name: str, shape: tuple[int, ...], itemsize: int = 8) -> None:
    """Require an array shape and byte extent to fit NumPy's indexing domain.

    This checks representability by ``np.intp`` only; it does not estimate
    available physical memory or allocate the array.

    Raises
    ------
    OverflowError
        If any extent or the total byte length exceeds the supported range.
    """
    for extent in shape:
        require_bound(name, int(extent), INTP_MAX)
    require_bound(f"{name} byte length", math.prod(shape) * itemsize, INTP_MAX)


def as_int64_array(values, *, name: str = "array") -> np.ndarray:
    """Return a readonly C-contiguous int64 representation of integer-valued input.

    Floating-point input is rejected even when numerically integral. Object
    entries must implement the integer index protocol. Values must lie in the
    symmetric range ``[-INT64_MAX, INT64_MAX]``, excluding ``INT64_MIN`` so
    that absolute-value bounds remain representable.

    Raises
    ------
    ValueError
        If the input is not explicitly integer-valued.
    OverflowError
        If any value or required allocation lies outside the supported range.
    """
    source = np.asarray(values)
    if source.dtype.kind not in "iubO":
        raise ValueError(f"{name} must contain declared integers, not {source.dtype}")
    if source.dtype.kind == "O":
        for value in source.flat:
            try:
                entry = operator.index(value)
            except TypeError as error:
                raise ValueError(f"{name} contains a value that is not an integer") from error
            if not -INT64_MAX <= entry <= INT64_MAX:
                raise OverflowError(f"{name} entry does not fit the symmetric int64 domain")
    elif source.size and (int(source.min()) < -INT64_MAX or int(source.max()) > INT64_MAX):
        raise OverflowError(f"{name} entry does not fit the symmetric int64 domain")
    require_allocation(name, source.shape)
    result = np.ascontiguousarray(source, dtype=np.int64)
    # Copy shared writable storage so the returned readonly array cannot freeze the caller's input.
    if result.flags.writeable:
        if np.shares_memory(result, source):
            result = result.copy()
        result.setflags(write=False)
    return result


def readonly(values, dtype):
    """Return a readonly C-contiguous array with the requested dtype.

    Writable shared storage is copied before freezing so the caller's input
    remains independently mutable. Shape and value validation are left to the
    calling domain operation.
    """
    result = np.ascontiguousarray(values, dtype=dtype)
    if result.flags.writeable:
        if np.shares_memory(result, np.asarray(values)):
            result = result.copy()
        result.setflags(write=False)
    return result
