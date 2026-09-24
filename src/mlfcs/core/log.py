"""Package logging policy, kept separate from scientific domain objects."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

_NAME = "mlfcs"
_HANDLER_MARK = "_mlfcs_stdout_handler"


def configure(*, level: int = logging.INFO, stream: TextIO | None = None) -> logging.Logger:
    """Configure the package logger once without touching the root logger."""
    logger = logging.getLogger(_NAME)
    handlers = [handler for handler in logger.handlers if getattr(handler, _HANDLER_MARK, False)]
    if handlers:
        handler = handlers[0]
    else:
        handler = logging.StreamHandler(sys.stdout if stream is None else stream)
        setattr(handler, _HANDLER_MARK, True)
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger."""
    if name == _NAME or name.startswith(f"{_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_NAME}.{name}")


__all__ = ["configure", "get_logger"]
