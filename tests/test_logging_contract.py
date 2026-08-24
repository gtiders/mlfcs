import importlib
import logging

import mlfcs


def test_package_logger_has_one_stdout_handler_and_reload_is_idempotent():
    logger = logging.getLogger("mlfcs")
    before = [
        handler for handler in logger.handlers if getattr(handler, "_mlfcs_stdout_handler", False)
    ]
    assert len(before) == 1
    assert logger.level == logging.INFO
    assert before[0].level == logging.NOTSET
    assert logger.propagate is False

    importlib.reload(mlfcs)
    after = [
        handler for handler in logger.handlers if getattr(handler, "_mlfcs_stdout_handler", False)
    ]
    assert after == before


def test_debug_is_opt_in():
    logger = logging.getLogger("mlfcs")
    assert not logger.isEnabledFor(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    try:
        assert logger.isEnabledFor(logging.DEBUG)
    finally:
        logger.setLevel(logging.INFO)
