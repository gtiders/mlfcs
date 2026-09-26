"""Package logging configuration and domain errors."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TextIO

_NAME = "mlfcs"
_HANDLER_MARK = "_mlfcs_handler_kind"
_FILE_MARK = "_mlfcs_file_path"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure(
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
    file: str | os.PathLike[str] | None = None,
    file_mode: str = "a",
    format: str = _FORMAT,
) -> logging.Logger:
    """Configure package console/file logging without touching the root logger.

    Repeated calls update the package level and formatter. A supplied file is
    opened once per resolved path and remains active if a later call omits it.
    """
    if file_mode not in {"a", "w"}:
        raise ValueError("file_mode must be 'a' or 'w'")
    logger = logging.getLogger(_NAME)
    handlers = {
        getattr(handler, _HANDLER_MARK, None): handler
        for handler in logger.handlers
        if getattr(handler, _HANDLER_MARK, None) is not None
    }
    console = handlers.get("console")
    selected_stream = sys.stdout if stream is None else stream
    if console is None:
        console = logging.StreamHandler(selected_stream)
        setattr(console, _HANDLER_MARK, "console")
        logger.addHandler(console)
    elif stream is not None and console.stream is not stream:
        console.setStream(stream)
    console.setLevel(logging.NOTSET)
    console.setFormatter(logging.Formatter(format, datefmt=_DATE_FORMAT))

    if file is not None:
        path = Path(file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = handlers.get("file")
        if file_handler is None or getattr(file_handler, _FILE_MARK, None) != path:
            if file_handler is not None:
                logger.removeHandler(file_handler)
                file_handler.close()
            file_handler = logging.FileHandler(path, mode=file_mode, encoding="utf-8")
            setattr(file_handler, _HANDLER_MARK, "file")
            setattr(file_handler, _FILE_MARK, path)
            logger.addHandler(file_handler)
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(logging.Formatter(format, datefmt=_DATE_FORMAT))
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger."""
    if name == _NAME or name.startswith(f"{_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_NAME}.{name}")


def log_error(
    error: BaseException,
    *,
    logger: str | logging.Logger = "errors",
    level: int = logging.ERROR,
    message: str | None = None,
) -> None:
    """Write an already-handled exception and its traceback to MLFCS logging.

    Errors are not logged automatically when constructed or raised: callers
    choose the boundary where an exception becomes an operational failure.
    """
    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    target = logger if isinstance(logger, logging.Logger) else get_logger(logger)
    text = message or f"{type(error).__name__}: {error}"
    traceback = (type(error), error, error.__traceback__) if error.__traceback__ else False
    target.log(level, text, exc_info=traceback)


class MLFCSError(Exception):
    """Base class for errors with domain meaning in MLFCS."""

    def log(self, *, message: str | None = None) -> None:
        """Log this handled domain error through the package logging system."""
        log_error(self, message=message)


class AliasingError(MLFCSError):
    """A supercell cannot distinguish all requested primitive parameters."""


class UnobservedParameterError(MLFCSError):
    """Training structures do not excite every requested model parameter."""


class ConstraintProjectionError(MLFCSError):
    """A physical invariance projection did not reach its requested accuracy."""


class IntegerRangeError(MLFCSError, ArithmeticError):
    """An exact integer result cannot be represented by the requested dtype."""


class RankCertificateError(MLFCSError, ArithmeticError):
    """Exact modular arithmetic could not certify a matrix rank."""


__all__ = [
    "AliasingError",
    "ConstraintProjectionError",
    "IntegerRangeError",
    "MLFCSError",
    "RankCertificateError",
    "UnobservedParameterError",
    "configure",
    "get_logger",
    "log_error",
]
