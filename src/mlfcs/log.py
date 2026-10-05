"""Package logging policy, kept separate from scientific domain objects."""

from __future__ import annotations

import logging
import sys

_NAME = "mlfcs"
_HANDLER_MARK = "_mlfcs_stdout_handler"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _StdoutHandler(logging.Handler):
    """Write to the current stdout without owning or retaining an external stream."""

    def emit(self, record: logging.LogRecord) -> None:
        """Format and flush a record through the task's current stdout redirection."""
        try:
            sys.stdout.write(self.format(record) + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - logging must not leak handler failures
            self.handleError(record)


def configure(*, level: int = logging.INFO) -> logging.Logger:
    """Configure the package logger without touching the root logger.

    Install one package-owned stdout handler with fixed timestamps; repeated
    calls only change the level. Bash or the task script owns redirection and
    files. Existing user handlers are untouched; the root logger is unchanged.
    """
    logger = logging.getLogger(_NAME)
    handlers = [handler for handler in logger.handlers if getattr(handler, _HANDLER_MARK, False)]
    if not handlers:
        handler = _StdoutHandler()
        setattr(handler, _HANDLER_MARK, True)
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
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
