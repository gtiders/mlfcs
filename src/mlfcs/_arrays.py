"""Exact array normalization, readonly ownership and machine-word bounds."""

import math
import operator

import numpy as np

INT64_MAX = (1 << 63) - 1
INTP_MAX = int(np.iinfo(np.intp).max)


def require_bound(name: str, value: int, limit: int = INT64_MAX) -> int:
    """Validate a nonnegative Python-integer bound and return it unchanged.

    Raises OverflowError if the bound exceeds ``limit``. Compute bounds before
    casting to fixed-width integers so validation itself cannot wrap.
    """
    value = operator.index(value)
    if not 0 <= value <= limit:
        raise OverflowError(f"{name}: proven bound {value} exceeds allowed range 0..{limit}")
    return value


def require_allocation(name: str, shape: tuple[int, ...], itemsize: int = 8) -> None:
    """Check array extents and byte length against the platform's np.intp limit.

    ``shape`` contains nonnegative extents; ``itemsize`` is bytes per element.
    This checks representability, not available RAM, and does not allocate.
    """
    for extent in shape:
        require_bound(name, int(extent), INTP_MAX)
    require_bound(f"{name} byte length", math.prod(shape) * itemsize, INTP_MAX)


def integer_array(values, *, name: str = "integer array") -> np.ndarray:
    """Normalize declared integers to a readonly, C-contiguous int64 array.

    Floating-point inputs are rejected, even if integral. Object entries must
    support the integer index protocol. Values must lie in the symmetric domain
    [-INT64_MAX, INT64_MAX], excluding INT64_MIN so later absolute values are safe.
    Writable caller-owned storage is copied; readonly storage may be shared.
    Invalid values raise ValueError; range or allocation failures raise OverflowError.
    """
    source = np.asarray(values)
    if source.dtype.kind not in "iubO":
        raise ValueError(f"{name} must contain declared integers, not {source.dtype}")
    if source.dtype.kind == "O":
        for value in source.flat:
            try:
                exact = operator.index(value)
            except TypeError as error:
                raise ValueError(f"{name} contains a value that is not an integer") from error
            if not -INT64_MAX <= exact <= INT64_MAX:
                raise OverflowError(f"{name} entry does not fit the symmetric int64 domain")
    elif source.size and (int(source.min()) < -INT64_MAX or int(source.max()) > INT64_MAX):
        raise OverflowError(f"{name} entry does not fit the symmetric int64 domain")
    require_allocation(name, source.shape)
    result = np.ascontiguousarray(source, dtype=np.int64)
    # A read-only view does not freeze an input owned by the caller.
    if result.flags.writeable:
        if np.shares_memory(result, source):
            result = result.copy()
        result.setflags(write=False)
    return result


def readonly(values, dtype):
    """Return a C-contiguous array of ``dtype`` without freezing caller-owned storage.

    Writable shared input is copied before marking the result readonly; already
    readonly contiguous input can be reused. No shape or finiteness check is made.
    """
    result = np.ascontiguousarray(values, dtype=dtype)
    if result.flags.writeable:
        if np.shares_memory(result, np.asarray(values)):
            result = result.copy()
        result.setflags(write=False)
    return result
