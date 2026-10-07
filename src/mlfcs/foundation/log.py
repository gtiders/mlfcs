"""Configure and access the package logger used by MLFCS workflows."""

from __future__ import annotations

import logging
import sys

_NAME = "mlfcs"
_HANDLER_MARK = "_mlfcs_stdout_handler"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _StdoutHandler(logging.Handler):
    """Send records to the process's current standard output stream."""

    def emit(self, record: logging.LogRecord) -> None:
        """Write and flush one formatted record, reporting handler failures safely."""
        try:
            sys.stdout.write(self.format(record) + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - logging must not leak handler failures
            self.handleError(record)


def configure(*, level: int = logging.INFO) -> logging.Logger:
    """Configure package logging and return the mlfcs logger.

    Repeated calls reuse the package-owned stdout handler and update the
    logging level. The root logger and user-installed handlers are left
    unchanged; the caller manages output redirection and log files.
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
    """Return the logger named beneath the ``mlfcs`` logger hierarchy.

    A fully qualified ``mlfcs`` name is preserved; other names are prefixed
    with ``mlfcs.``.
    """
    if name == _NAME or name.startswith(f"{_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_NAME}.{name}")


__all__ = ["configure", "get_logger"]
